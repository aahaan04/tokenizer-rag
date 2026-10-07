"""Orchestrates Phase 4: canonical_id assignment, survey flagging, and the
false-merge audit (pairwise similarity scores within every existing group,
plus a bounded scan for near-miss pairs that were NOT merged).

Does not change which rows belong to which candidate_group_id — that
grouping happened in Phase 1 (title-exact-match across sources, see
DECISIONS.md) and stays the dedup unit. This module picks a canonical
representative row per group, flags surveys, and audits the groups for
false merges.
"""

from __future__ import annotations

import json
from collections import defaultdict

from tokrag.collect.manifest import read_manifest, write_manifest
from tokrag.config import MANIFEST_PATH, PROJECT_ROOT
from tokrag.dedup import match, survey

AUDIT_PATH = PROJECT_ROOT / "dedup_audit.md"

# Fuzzy preprint-vs-published matching tier (2026-10-07, Checkpoint 4
# revision): catches cases Phase 1's exact-title match can't, because the
# title genuinely changed between preprint and publication. Deliberately
# conservative — see DECISIONS.md for why title_sim stays an AND condition
# (not an either/or with the author check), and for the one paper this
# threshold does NOT catch (MorphBPE) despite being the same paper, handled
# as an explicit, user-confirmed exception rather than loosening the general
# rule to fit one case.
FUZZY_TITLE_SIM_THRESHOLD = 0.85
FUZZY_MIN_AUTHORS_STRONG = 2  # title_sim + >=2 shared authors -> merge
FUZZY_MIN_AUTHORS_WEAK = 1  # title_sim + 1 shared author needs abstract support too
FUZZY_ABSTRACT_SIM_THRESHOLD = 0.8
FUZZY_MAX_YEAR_DIFF = 2

# Confirmed directly by the user at Checkpoint 4 ("MorphBPE should be
# caught; I agree it's the same paper") — title_sim is only 0.667
# (char-level) / 0.333 (token-Jaccard), well under the general 0.85 bar, so
# this is applied as a named exception, not a threshold change.
MANUAL_FUZZY_MERGE_PAIRS = [
    ("semantic_scholar", "b6b6ba84783b6133d03365ceb8d4a22a706dab49", "semantic_scholar", "0a17cab809062aeabf4f6e3710aec88945c98e04"),
]

# Preference for which group member becomes "the" canonical row: a
# peer-reviewed ACL copy is more authoritative than an arXiv preprint, which
# in turn is preferred over a bare Semantic Scholar record. Within arXiv,
# the highest version number (latest) wins. See DECISIONS.md.
SOURCE_PRIORITY = {"acl_anthology": 0, "arxiv": 1, "semantic_scholar": 2}


def _version_num(m) -> int:
    try:
        return int((m.arxiv_version or "v0").lstrip("v"))
    except ValueError:
        return 0


def _pick_canonical(members: list):
    return sorted(members, key=lambda m: (SOURCE_PRIORITY.get(m.source, 9), -_version_num(m)))[0]


def find_fuzzy_merges(rows: list, model, max_block_size: int = 50) -> list:
    """Scans SINGLETON included rows (no candidate_group_id yet) for fuzzy
    preprint-vs-published matches: title_sim >= 0.85 AND (>=2 shared authors
    OR (>=1 shared author AND abstract cosine >= 0.8)) AND year_diff <= 2.
    Blocked by shared-author surname (same approach as find_near_miss_pairs)
    to stay tractable. Returns a list of candidate-pair dicts; does not
    mutate anything."""
    included = [r for r in rows if r.included]
    singletons = [r for r in included if not r.candidate_group_id]

    by_author = defaultdict(list)
    for r in singletons:
        for name in match._author_last_names(r.authors):
            by_author[name].append(r)

    seen_pairs = set()
    candidates = []
    for _name, members in by_author.items():
        if len(members) < 2 or len(members) > max_block_size:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                key = tuple(sorted([f"{a.source}:{a.source_id}", f"{b.source}:{b.source_id}"]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                # Proceedings/front-matter volumes ("Proceedings of the Nth
                # Workshop on X"), not individual papers — their "authors"
                # are often the workshop's editors, so different YEARS'
                # volumes sharing an organizing committee can look like a
                # high-confidence fuzzy match despite being genuinely
                # different things. Found via a real false merge (two
                # different SCLeM workshop years) during Checkpoint 4
                # review — see DECISIONS.md.
                if a.title.lower().startswith("proceedings of") or b.title.lower().startswith("proceedings of"):
                    continue

                t_sim = match.title_similarity(a.title, b.title)
                if t_sim < FUZZY_TITLE_SIM_THRESHOLD:
                    continue

                try:
                    year_diff = abs(int(a.year) - int(b.year))
                except (ValueError, TypeError):
                    year_diff = 0
                if year_diff > FUZZY_MAX_YEAR_DIFF:
                    continue

                n_shared = len(match._author_last_names(a.authors) & match._author_last_names(b.authors))
                abstract_sim = None
                if n_shared >= FUZZY_MIN_AUTHORS_STRONG:
                    qualifies = True
                elif n_shared >= FUZZY_MIN_AUTHORS_WEAK:
                    abstract_sim = match.abstract_cosine_similarities([a.abstract, b.abstract], model=model)[0][1]
                    qualifies = abstract_sim >= FUZZY_ABSTRACT_SIM_THRESHOLD
                else:
                    qualifies = False
                if not qualifies:
                    continue

                candidates.append(
                    {
                        "paper_a": a,
                        "paper_b": b,
                        "title_sim": round(t_sim, 3),
                        "shared_authors": n_shared,
                        "year_diff": year_diff,
                        "abstract_sim": round(abstract_sim, 3) if abstract_sim is not None else None,
                    }
                )
    return candidates


def apply_fuzzy_merges(rows: list, candidates: list, manual_pairs: list = None) -> list:
    """Assigns a shared candidate_group_id to each confirmed fuzzy-merge pair
    (and the manually-confirmed exceptions), extending an existing group if
    one side already got one earlier in this same pass. Returns the list of
    merges actually applied (for the audit report)."""
    by_key = {f"{r.source}:{r.source_id}": r for r in rows}
    existing_group_nums = [int(r.candidate_group_id[3:]) for r in rows if r.candidate_group_id]
    group_counter = [max(existing_group_nums, default=0)]
    applied = []

    def _merge(a, b, reason: str):
        if a.candidate_group_id and b.candidate_group_id:
            if a.candidate_group_id != b.candidate_group_id:
                return False  # both already in different groups -> don't silently combine, skip
            return False  # already merged
        gid = a.candidate_group_id or b.candidate_group_id
        if not gid:
            group_counter[0] += 1
            gid = f"grp{group_counter[0]:04d}"
        a.candidate_group_id = gid
        b.candidate_group_id = gid
        applied.append({"group_id": gid, "paper_a": f"{a.source}:{a.source_id}", "title_a": a.title, "paper_b": f"{b.source}:{b.source_id}", "title_b": b.title, "reason": reason})
        return True

    for c in candidates:
        _merge(c["paper_a"], c["paper_b"], f"fuzzy: title_sim={c['title_sim']} shared_authors={c['shared_authors']} year_diff={c['year_diff']} abstract_sim={c['abstract_sim']}")

    for sa, sida, sb, sidb in manual_pairs or []:
        a, b = by_key.get(f"{sa}:{sida}"), by_key.get(f"{sb}:{sidb}")
        if a and b:
            _merge(a, b, "manual (user-confirmed at Checkpoint 4)")

    return applied


def assign_canonical_ids(rows: list) -> dict:
    by_group = defaultdict(list)
    for r in rows:
        if r.included and r.candidate_group_id:
            by_group[r.candidate_group_id].append(r)

    group_canonical = {}
    for gid, members in by_group.items():
        canonical_row = _pick_canonical(members)
        canonical_key = f"{canonical_row.source}__{canonical_row.source_id}"
        group_canonical[gid] = canonical_key
        for m in members:
            m.canonical_id = canonical_key

    for r in rows:
        if r.included and not r.candidate_group_id:
            r.canonical_id = f"{r.source}__{r.source_id}"

    return group_canonical


def assign_doc_types(rows: list) -> None:
    for r in rows:
        if r.included:
            r.doc_type = "survey" if survey.is_survey(r.title, r.abstract) else "paper"


def audit_groups(rows: list, model) -> list:
    by_group = defaultdict(list)
    for r in rows:
        if r.included and r.candidate_group_id:
            by_group[r.candidate_group_id].append(r)

    audit = []
    for gid, members in sorted(by_group.items()):
        abstracts = [m.abstract for m in members]
        sims = match.abstract_cosine_similarities(abstracts, model=model) if len(members) > 1 else [[1.0]]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                audit.append(
                    {
                        "group_id": gid,
                        "paper_a": f"{a.source}:{a.source_id}",
                        "paper_b": f"{b.source}:{b.source_id}",
                        "title_a": a.title,
                        "title_b": b.title,
                        "title_sim": round(match.title_similarity(a.title, b.title), 3),
                        "author_overlap": round(match.author_overlap(a.authors, b.authors), 3),
                        "abstract_cosine": round(sims[i][j], 3),
                    }
                )
    return audit


def find_near_miss_pairs(rows: list, title_sim_range=(0.4, 0.85), min_author_overlap=0.2, max_block_size=50) -> list:
    """Bounded by shared-author blocking (not O(n^2) over all included rows):
    only compares papers that share at least one author surname. Flags pairs
    with title similarity in a near-miss band AND meaningful author overlap
    — the brief's "same authors, similar titles" risky-case check, covering
    both near-misses below the merge threshold and a sanity check that no
    genuinely different paper (e.g. a paper and its follow-up) got merged.
    """
    included = [r for r in rows if r.included]
    by_author = defaultdict(list)
    for r in included:
        for name in match._author_last_names(r.authors):
            by_author[name].append(r)

    seen_pairs = set()
    near_misses = []
    for _name, members in by_author.items():
        if len(members) < 2 or len(members) > max_block_size:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a.candidate_group_id and a.candidate_group_id == b.candidate_group_id:
                    continue
                key = tuple(sorted([f"{a.source}:{a.source_id}", f"{b.source}:{b.source_id}"]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                t_sim = match.title_similarity(a.title, b.title)
                auth_overlap = match.author_overlap(a.authors, b.authors)
                if title_sim_range[0] <= t_sim <= title_sim_range[1] and auth_overlap >= min_author_overlap:
                    near_misses.append(
                        {
                            "paper_a": f"{a.source}:{a.source_id}",
                            "title_a": a.title,
                            "paper_b": f"{b.source}:{b.source_id}",
                            "title_b": b.title,
                            "title_sim": round(t_sim, 3),
                            "author_overlap": round(auth_overlap, 3),
                        }
                    )
    near_misses.sort(key=lambda x: -x["title_sim"])
    return near_misses


def group_breakdown(rows: list) -> dict:
    """One consistent group-count breakdown, reused by the dedup report and
    manifest_summary.md (see the 2026-10-07 DECISIONS.md reconciliation
    entry — this number changed across intermediate debugging runs; this
    function is the single source of truth for the final count)."""
    by_group = defaultdict(list)
    for r in rows:
        if r.included and r.candidate_group_id:
            by_group[r.candidate_group_id].append(r)

    version_only = cross_source_only = both = degenerate = other = 0
    for members in by_group.values():
        if len(members) < 2:
            degenerate += 1  # e.g. the group's other member was excluded by relevance filtering
            continue
        sources = {m.source for m in members}
        arxiv_id_counts = defaultdict(int)
        for m in members:
            if m.source == "arxiv" and m.arxiv_id:
                arxiv_id_counts[m.arxiv_id] += 1
        has_version_pair = any(v >= 2 for v in arxiv_id_counts.values())
        is_cross_source = len(sources) > 1
        if has_version_pair and is_cross_source:
            both += 1
        elif has_version_pair:
            version_only += 1
        elif is_cross_source:
            cross_source_only += 1
        else:
            other += 1  # e.g. two same-source ACL duplicate bib records
    return {
        "total_groups": len(by_group),
        "total_rows_in_groups": sum(len(v) for v in by_group.values()),
        "version_pair_only": version_only,
        "cross_source_only": cross_source_only,
        "both_version_and_cross_source": both,
        "degenerate_single_member": degenerate,
        "same_source_duplicate_other": other,
    }


def _write_audit_report(audit, near_misses, low_confidence, fuzzy_merges, breakdown, report) -> None:
    lines = [
        "# Dedup false-merge audit",
        "",
        "## Group breakdown (single source of truth, see DECISIONS.md)",
        "",
        f"- Total groups: {breakdown['total_groups']} ({breakdown['total_rows_in_groups']} rows)",
        f"  - Version-pair only (same source, multiple arXiv versions): {breakdown['version_pair_only']}",
        f"  - Cross-source only (preprint vs. published, no version pair): {breakdown['cross_source_only']}",
        f"  - Both version pair AND cross-source: {breakdown['both_version_and_cross_source']}",
        f"  - Degenerate (1 included member; other member excluded by relevance filtering): {breakdown['degenerate_single_member']}",
        f"  - Other (e.g. duplicate same-source ACL bib records): {breakdown['same_source_duplicate_other']}",
        "",
        f"Groups audited (pairwise): {report['n_groups_audited']}",
        f"Pairs audited: {report['n_pairs_audited']}",
        f"Low-confidence merges (abstract cosine < 0.7 OR zero author overlap): {report['n_low_confidence_merges']}",
        f"Surveys flagged: {report['n_surveys_flagged']}",
        f"Near-miss pairs (NOT merged, similar title + shared author): {report['n_near_miss_pairs']}",
        f"Fuzzy-tier merges applied (title_sim>=0.85 + author/abstract support, or manual exception): {report['n_fuzzy_merges']}",
        "",
        "## Fuzzy-tier merges applied this run",
        "",
    ]
    if not fuzzy_merges:
        lines.append("(none)")
    for m in fuzzy_merges:
        lines.append(f"- **{m['group_id']}** — {m['reason']}")
        lines.append(f"  - A: {m['title_a']} ({m['paper_a']})")
        lines.append(f"  - B: {m['title_b']} ({m['paper_b']})")
    lines += ["", "## Low-confidence merges (manual review)", ""]
    for a in low_confidence:
        lines.append(f"- **{a['group_id']}**: title_sim={a['title_sim']} author_overlap={a['author_overlap']} abstract_cosine={a['abstract_cosine']}")
        lines.append(f"  - A: {a['title_a']} ({a['paper_a']})")
        lines.append(f"  - B: {a['title_b']} ({a['paper_b']})")
    lines += ["", "## All merge-cluster pairwise scores", ""]
    for a in audit:
        lines.append(
            f"- {a['group_id']}: title_sim={a['title_sim']} author_overlap={a['author_overlap']} "
            f"abstract_cosine={a['abstract_cosine']} | {a['title_a'][:70]}"
        )
    lines += ["", "## Near-miss pairs (NOT merged), for manual spot-check", ""]
    for n in near_misses:
        lines.append(f"- title_sim={n['title_sim']} author_overlap={n['author_overlap']}")
        lines.append(f"  - A: {n['title_a']} ({n['paper_a']})")
        lines.append(f"  - B: {n['title_b']} ({n['paper_b']})")
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict:
    from sentence_transformers import SentenceTransformer

    rows = read_manifest(MANIFEST_PATH)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    fuzzy_candidates = find_fuzzy_merges(rows, model)
    fuzzy_merges = apply_fuzzy_merges(rows, fuzzy_candidates, manual_pairs=MANUAL_FUZZY_MERGE_PAIRS)

    assign_canonical_ids(rows)
    assign_doc_types(rows)
    write_manifest(MANIFEST_PATH, rows)

    audit = audit_groups(rows, model)
    near_misses = find_near_miss_pairs(rows)
    low_confidence = [a for a in audit if a["abstract_cosine"] < 0.7 or a["author_overlap"] == 0.0]
    breakdown = group_breakdown(rows)

    report = {
        "n_groups_audited": len({a["group_id"] for a in audit}),
        "n_pairs_audited": len(audit),
        "n_low_confidence_merges": len(low_confidence),
        "n_surveys_flagged": sum(1 for r in rows if r.included and r.doc_type == "survey"),
        "n_near_miss_pairs": len(near_misses),
        "n_fuzzy_merges": len(fuzzy_merges),
        "group_breakdown": breakdown,
    }
    _write_audit_report(audit, near_misses, low_confidence, fuzzy_merges, breakdown, report)
    print(json.dumps(report, indent=2))
    return report
