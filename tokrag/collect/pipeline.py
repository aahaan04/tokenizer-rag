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
from tokrag.collect.manifest import Candidate, write_manifest
from tokrag.config import MANIFEST_PATH

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


def collect(sample: bool = False) -> list:
    """Run the full (or, if sample=True, a small ~20-candidate) collection pass."""
    if sample:
        query_terms = QUERY_TERMS[:3]
        max_per_query = 6
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
    # both this ACL pass and the canary injection below, so a canonical paper
    # found via ACL doesn't get a second, disconnected row when it's also
    # fetched directly by arXiv id — that duplication happened in an earlier
    # version of this function (see the 2026-10-06 DECISIONS.md entry).
    if not sample:
        title_index = {_normalize_title(c.title): c for c in candidates.values()}

        for paper in acl_anthology.iter_candidates():
            norm_title = _normalize_title(paper.title)
            existing = title_index.get(norm_title)
            if existing is not None:
                # Same paper we already have (usually its arXiv preprint) — this
                # IS useful dedup signal for Phase 4, so record it, don't just enrich silently.
                existing.doi = existing.doi or paper.doi
                if paper.venue and (not existing.venue or existing.venue == "arXiv"):
                    existing.venue = paper.venue
                existing.query_matched += f"; acl:{paper.anthology_id}"
                continue

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
            candidates[key] = c
            title_index[norm_title] = c

        # Guarantee the canonical papers are present regardless of query
        # ranking. "Naturally found" = already in the pool (by title, from
        # either the keyword queries or the ACL bulk pass) before this
        # injection runs — that's the real signal for "the query set has a
        # gap", separate from whether the paper ends up in the corpus either way.
        naturally_found = {}
        for arxiv_id, entry in zip(
            [aid for aid, _ in CANARY_PAPERS], arxiv.fetch_by_ids([aid for aid, _ in CANARY_PAPERS])
        ):
            norm_title = _normalize_title(entry.title)
            existing = title_index.get(norm_title)
            if existing is not None:
                naturally_found[arxiv_id] = existing
                existing.arxiv_id = existing.arxiv_id or entry.arxiv_id
                existing.arxiv_version = existing.arxiv_version or entry.version
                existing.source_url = existing.source_url or entry.abs_url
                existing.query_matched += "; canary_check"
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
                query_matched="canary_injected",
                relevance_score=relevance.score(entry.title, entry.abstract),
                included=included,
                rejection_reason="" if included else reason,
            )
            candidates[f"arxiv:{arxiv_id}"] = c
            title_index[norm_title] = c

    result = list(candidates.values())
    write_manifest(MANIFEST_PATH, result)

    if not sample:
        _report_canaries(naturally_found, candidates, title_index)

    return result


def _report_canaries(naturally_found: dict, candidates: dict, title_index: dict) -> None:
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
