"""Orchestrates corpus collection: run query terms against both sources, fold
duplicate raw hits (same candidate surfaced by multiple query terms) into one
row, score relevance, write the manifest.

This is NOT the Phase 4 dedup (arXiv versions / preprint-vs-conference / surveys)
— that happens later, deliberately, over this raw candidate pool. Here we only
avoid double-counting the *same* arXiv id or Semantic Scholar paper id when two
query terms both surface it.
"""

from __future__ import annotations

import json
import re

from tokrag.collect import acl_anthology, arxiv, relevance, semantic_scholar
from tokrag.collect.manifest import Candidate, read_manifest, write_manifest, write_manifest_summary
from tokrag.config import MANIFEST_PATH, PROJECT_ROOT

MANIFEST_SUMMARY_PATH = PROJECT_ROOT / "manifest_summary.md"

QUERY_TERMS = [
    "byte-level tokenization language model",
    "subword tokenization",
    "byte pair encoding tokenizer",
    "WordPiece tokenization",
    "SentencePiece tokenizer",
    "unigram language model tokenizer",
    "tokenizer-free language model",
    "character-level language model tokenization",
    "vocabulary size language model",
    "multilingual tokenizer fairness",
    "tokenizer fertility multilingual",
    "tokenization arithmetic reasoning",
    "tokenization code generation",
    "subword regularization",
    "tokenizer training algorithm",
    # Added 2026-10-06 after the canary check showed the fairness/arithmetic
    # themes weren't surfacing naturally: arXiv's quoted-phrase search needs a
    # literal substring match, and the original longer phrases above don't
    # appear verbatim in e.g. Petrov et al.'s or Singh & Strouse's abstracts.
    "unfairness between languages",
    "tokenizer fairness",
    "number tokenization arithmetic",
    "digit tokenization",
    "tokenization numeracy",
]

ARXIV_CATEGORY_FILTER = "(cat:cs.CL OR cat:cs.LG OR cat:cs.AI)"

# Canonical papers the corpus should contain no matter how the keyword queries
# rank them. Fetched directly by arXiv id (not search) so their presence is
# guaranteed; collect() still reports whether the keyword queries would have
# found them on their own, since that's the real signal for "the query set has
# a gap" per the 2026-10-06 DECISIONS.md entry.
CANARY_PAPERS = [
    ("1508.07909", "Sennrich et al. 2016 — BPE for NMT (the foundational method)"),
    ("1808.06226", "Kudo & Richardson 2018 — SentencePiece"),
    ("2105.13626", "Xue et al. 2021 — ByT5 (token-free)"),
    ("2103.06874", "Clark et al. 2021 — CANINE (tokenization-free encoder)"),
    ("2305.15425", "Petrov et al. 2023 — tokenizer fairness/fertility across languages"),
    ("2402.14903", "Singh & Strouse 2024 — tokenization's effect on arithmetic"),
]


def _arxiv_query(term: str) -> str:
    return f'all:"{term}" AND {ARXIV_CATEGORY_FILTER}'


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def _link_or_add(norm_title: str, new_candidate, key: str, candidates: dict, title_index: dict, group_counter: list) -> None:
    """Add `new_candidate` as its own row. If `norm_title` matches an existing
    candidate (same paper, different source — e.g. an arXiv preprint and its
    ACL-published version), link the two via a shared candidate_group_id
    instead of merging them into one row.

    Cross-source duplicates must stay as SEPARATE rows: Phase 4 needs them both
    present to measure dedup's before/after impact. Merging here (an earlier
    version of this function did) would silently do Phase 4's job during
    collection and destroy that signal — see the 2026-10-06 DECISIONS.md entry.
    """
    existing = title_index.get(norm_title)
    if existing is not None:
        if not existing.candidate_group_id:
            group_counter[0] += 1
            existing.candidate_group_id = f"grp{group_counter[0]:04d}"
        new_candidate.candidate_group_id = existing.candidate_group_id
    candidates[key] = new_candidate
    title_index[norm_title] = new_candidate


def collect(sample: bool = False, sample_n: int | None = None) -> list:
    """Run the full collection pass, or a small end-to-end sample pass.

    `sample_n` (if given) caps the collection to approximately that many
    INCLUDED candidates, using a handful of query terms and skipping the ACL
    Anthology bulk download and the canonical-paper canary injection (both
    network/time-heavy and unnecessary for a quick pipeline smoke test) — see
    the Phase 7 README section for the full --sample N quickstart. `sample`
    (bare bool, kept for backward compatibility) is equivalent to
    `sample_n=20`.
    """
    if sample_n is None and sample:
        sample_n = 20

    if sample_n is not None:
        query_terms = QUERY_TERMS[:5]
        max_per_query = max(10, sample_n * 2)  # headroom: relevance filtering rejects a chunk of raw hits
    else:
        query_terms = QUERY_TERMS
        max_per_query = 60

    candidates: dict = {}
    # arXiv id (no version) -> Candidate, kept in sync alongside `candidates` so
    # whichever source (arXiv or S2) finds a paper *second* merges into the
    # same row instead of creating a duplicate under its own key. Checking only
    # `candidates[f"arxiv:{id}"]` missed the case where S2 found a paper before
    # arXiv did within the query loop — fixed 2026-10-06, see DECISIONS.md.
    by_arxiv_id: dict = {}

    for term in query_terms:
        for entry in arxiv.search(_arxiv_query(term), max_results=max_per_query):
            existing = by_arxiv_id.get(entry.arxiv_id)
            if existing is not None:
                if term not in existing.query_matched:
                    existing.query_matched += f"; {term}"
                existing.arxiv_version = existing.arxiv_version or entry.version
                existing.source_url = existing.source_url or entry.abs_url
                existing.abstract = existing.abstract or entry.abstract
                continue

            included, reason = relevance.decide(entry.title, entry.abstract)
            c = Candidate(
                source="arxiv",
                source_id=f"{entry.arxiv_id}{entry.version}",
                arxiv_id=entry.arxiv_id,
                arxiv_version=entry.version,
                title=entry.title,
                authors="; ".join(entry.authors),
                year=entry.published[:4],
                venue="arXiv",
                abstract=entry.abstract,
                source_url=entry.abs_url,
                query_matched=term,
                relevance_score=relevance.score(entry.title, entry.abstract),
                included=included,
                rejection_reason="" if included else reason,
            )
            candidates[f"arxiv:{entry.arxiv_id}"] = c
            by_arxiv_id[entry.arxiv_id] = c

        try:
            s2_papers = semantic_scholar.search(term, max_results=max_per_query)
        except RuntimeError as e:
            print(f"  [semantic_scholar] skipping query {term!r}: {e}")
            s2_papers = []

        for paper in s2_papers:
            if not paper.title:
                continue

            arxiv_ext = (paper.external_ids or {}).get("ArXiv")
            existing = by_arxiv_id.get(arxiv_ext) if arxiv_ext else None
            if existing is not None:
                # Same paper we already have (from arXiv, or an earlier S2 hit) — enrich, don't duplicate.
                existing.semantic_scholar_id = existing.semantic_scholar_id or paper.paper_id
                existing.doi = existing.doi or (paper.external_ids or {}).get("DOI", "")
                existing.external_ids = existing.external_ids or json.dumps(paper.external_ids)
                if paper.venue and paper.venue.lower() != "arxiv.org" and existing.venue in ("", "arXiv"):
                    existing.venue = paper.venue
                if term not in existing.query_matched:
                    existing.query_matched += f"; {term}"
                continue

            key = f"s2:{paper.paper_id}"
            existing = candidates.get(key)
            if existing is not None:
                if term not in existing.query_matched:
                    existing.query_matched += f"; {term}"
                continue

            included, reason = relevance.decide(paper.title, paper.abstract)
            c = Candidate(
                source="semantic_scholar",
                source_id=paper.paper_id,
                arxiv_id=arxiv_ext or "",
                title=paper.title,
                authors="; ".join(paper.authors),
                year=paper.year,
                venue=paper.venue,
                abstract=paper.abstract or "",
                source_url=paper.url,
                doi=(paper.external_ids or {}).get("DOI", ""),
                semantic_scholar_id=paper.paper_id,
                external_ids=json.dumps(paper.external_ids),
                query_matched=term,
                relevance_score=relevance.score(paper.title, paper.abstract or ""),
                included=included,
                rejection_reason="" if included else reason,
            )
            candidates[key] = c
            if arxiv_ext:
                by_arxiv_id[arxiv_ext] = c

    # ACL Anthology isn't a per-query API (bulk bibliography, see acl_anthology.py)
    # so it runs once, not per query term — and only on the full run, since
    # parsing the ~40MB bulk export is wasted work for a 20-candidate dry run.
    # A single title_index (normalized title -> Candidate) is maintained across
    # both this ACL pass and the canary injection below, feeding _link_or_add so
    # a same-paper match anywhere in the pool gets linked via candidate_group_id
    # rather than merged into one row (see _link_or_add's docstring for why).
    if sample_n is None:
        title_index = {_normalize_title(c.title): c for c in candidates.values()}
        group_counter = [0]

        for paper in acl_anthology.iter_candidates():
            key = f"acl:{paper.anthology_id}"
            if key in candidates:
                continue
            included, reason = relevance.decide(paper.title, paper.abstract)
            c = Candidate(
                source="acl_anthology",
                source_id=paper.anthology_id,
                title=paper.title,
                authors="; ".join(paper.authors),
                year=paper.year,
                venue=paper.venue,
                abstract=paper.abstract,
                source_url=paper.url,
                doi=paper.doi,
                query_matched="acl_bulk_prefilter",
                relevance_score=relevance.score(paper.title, paper.abstract),
                included=included,
                rejection_reason="" if included else reason,
            )
            _link_or_add(_normalize_title(paper.title), c, key, candidates, title_index, group_counter)

        # Guarantee the canonical papers are present regardless of query
        # ranking. "Naturally found" = already in the pool (by exact arXiv id,
        # from the main query loop, or by title, from the ACL bulk pass) before
        # this injection runs — that's the real signal for "the query set has a
        # gap", separate from whether the paper ends up in the corpus either way.
        naturally_found = {}
        # arXiv's id_list API does NOT preserve request order (confirmed by
        # direct test: requesting [A, B, C] can return [C, A, B]) — zip()ing
        # the id list against fetch_by_ids() results by position silently
        # mispairs each canary's label with a *different* paper's actual
        # content. Index by the entry's own arxiv_id instead. Found
        # 2026-10-06 after a canary dedup check showed SentencePiece
        # duplicated and ByT5 missing its arXiv row — see DECISIONS.md.
        fetched_by_id = {e.arxiv_id: e for e in arxiv.fetch_by_ids([aid for aid, _ in CANARY_PAPERS])}
        for arxiv_id, _label in CANARY_PAPERS:
            entry = fetched_by_id.get(arxiv_id)
            if entry is None:
                continue
            key = f"arxiv:{arxiv_id}"
            norm_title = _normalize_title(entry.title)

            if key in candidates:
                # The exact same arXiv record the query loop (or an earlier
                # canary) already added — not a new row, just note it as found.
                naturally_found[arxiv_id] = candidates[key]
                continue

            existing_by_title = title_index.get(norm_title)
            if existing_by_title is not None:
                naturally_found[arxiv_id] = existing_by_title

            included, reason = relevance.decide(entry.title, entry.abstract)
            c = Candidate(
                source="arxiv",
                source_id=f"{entry.arxiv_id}{entry.version}",
                arxiv_id=entry.arxiv_id,
                arxiv_version=entry.version,
                title=entry.title,
                authors="; ".join(entry.authors),
                year=entry.published[:4],
                venue="arXiv",
                abstract=entry.abstract,
                source_url=entry.abs_url,
                query_matched="canary_injected",
                relevance_score=relevance.score(entry.title, entry.abstract),
                included=included,
                rejection_reason="" if included else reason,
            )
            _link_or_add(norm_title, c, key, candidates, title_index, group_counter)

    result = list(candidates.values())

    if sample_n is not None:
        # Cap to approximately sample_n INCLUDED candidates (highest relevance
        # score first), keeping all rejected rows found along the way for a
        # realistic manifest shape — downstream `parse --limit` isn't needed
        # since this already bounds the included count directly.
        included = sorted((c for c in result if c.included), key=lambda c: -c.relevance_score)
        rejected = [c for c in result if not c.included]
        keep_ids = {id(c) for c in included[:sample_n]}
        result = [c for c in result if c.included and id(c) in keep_ids] + rejected

    write_manifest(MANIFEST_PATH, result)

    if sample_n is None:
        _report_canaries(naturally_found, candidates)
        write_manifest_summary(MANIFEST_SUMMARY_PATH, result)
        n_groups = len({c.candidate_group_id for c in result if c.candidate_group_id})
        print(f"\nCross-source same-paper groups linked via candidate_group_id: {n_groups}")
        print(f"Manifest summary written to {MANIFEST_SUMMARY_PATH}")
    else:
        write_manifest_summary(MANIFEST_SUMMARY_PATH, result)
        n_included = sum(1 for c in result if c.included)
        print(f"\nSample collection: {n_included} included candidates (target {sample_n}), {len(result) - n_included} rejected.")

    return result


def _report_canaries(naturally_found: dict, candidates: dict) -> None:
    print("\nCanary check (canonical papers the corpus must contain):")
    for arxiv_id, label in CANARY_PAPERS:
        c = naturally_found.get(arxiv_id) or candidates.get(f"arxiv:{arxiv_id}")
        found_by_queries = "found by keyword queries/ACL" if arxiv_id in naturally_found else "NOT found naturally (injected directly)"
        if c is None:
            print(f"  MISSING   {label} ({arxiv_id})")
        elif c.included:
            print(f"  INCLUDED  {label} ({arxiv_id}) — {found_by_queries}")
        else:
            print(f"  REJECTED  {label} ({arxiv_id}) — score={c.relevance_score:.1f}, reason={c.rejection_reason}")


def add_arxiv_version_pairs(limit: int = 25) -> list:
    """Add the v1 (original) text of a sample of multi-version included arXiv
    papers as separate rows, linked by candidate_group_id to the existing
    latest-version row.

    arXiv search only ever returns a paper's LATEST version, so collection
    never naturally produces the "multiple arXiv versions of the same paper"
    duplicate case the brief names first for Phase 4. This deliberately
    introduces it for a small sample (not corpus breadth) so Phase 4 has real
    cases to detect and measure. Idempotent: skips a paper if its v1 row
    already exists (safe to rerun).
    """
    rows = read_manifest(MANIFEST_PATH)
    existing_source_ids = {c.source_id for c in rows}

    multi_version = [
        c
        for c in rows
        if c.included and c.source == "arxiv" and c.arxiv_version not in ("", "v1")
        and f"{c.arxiv_id}v1" not in existing_source_ids
    ]
    # Prefer more-versioned papers first (v3+) — more interesting for dedup testing than a plain v1->v2.
    multi_version.sort(key=lambda c: -int(c.arxiv_version.lstrip("v") or 1))
    selected = multi_version[:limit]

    fetched_v1 = {e.arxiv_id: e for e in arxiv.fetch_by_ids([f"{c.arxiv_id}v1" for c in selected])}

    existing_group_nums = [int(c.candidate_group_id[3:]) for c in rows if c.candidate_group_id]
    group_counter = [max(existing_group_nums, default=0)]

    new_rows = []
    for c in selected:
        entry = fetched_v1.get(c.arxiv_id)
        if entry is None:
            continue
        if not c.candidate_group_id:
            group_counter[0] += 1
            c.candidate_group_id = f"grp{group_counter[0]:04d}"

        included, reason = relevance.decide(entry.title, entry.abstract)
        new_rows.append(
            Candidate(
                source="arxiv",
                source_id=f"{entry.arxiv_id}{entry.version}",
                arxiv_id=entry.arxiv_id,
                arxiv_version=entry.version,
                title=entry.title,
                authors=c.authors,
                year=entry.published[:4],
                venue="arXiv",
                abstract=entry.abstract,
                source_url=entry.abs_url,
                query_matched="arxiv_version_pair",
                relevance_score=relevance.score(entry.title, entry.abstract),
                included=included,
                rejection_reason="" if included else reason,
                candidate_group_id=c.candidate_group_id,
            )
        )

    all_rows = rows + new_rows
    write_manifest(MANIFEST_PATH, all_rows)
    write_manifest_summary(MANIFEST_SUMMARY_PATH, all_rows)
    return new_rows
