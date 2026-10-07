"""Retrieval-time result diversification — applied AFTER ranking over a deep
candidate pool (the caller should overfetch, e.g. top-50), not by merging
rows out of the index.

Two functions are kept, deliberately:

`diversify_cap1_naive` is the FIRST version (2026-10-06): caps one chunk per
canonical_paper_id, no near-duplicate-chunk awareness. Kept only as an
ablation baseline for WRITEUP.md — it measurably hurt Recall@5/MRR (see
DECISIONS.md) because capping by PAPER, not by PASSAGE, throws away distinct
relevant content whenever a paper's duplicate copies don't chunk identically:
if the gold passage only survives in a copy that isn't the highest-ranked
one, cap-1 discards it even though it was never actually redundant with what
it kept.

`diversify` is the fix (2026-10-07): collapses near-duplicate CHUNKS within a
group first (the same passage repeated across a paper's arXiv/ACL/S2 copies,
detected by shingle-Jaccard similarity on the chunk text — not just "same
paper"), preferring the canonical row's copy of a passage when one of the
near-duplicates belongs to it. Only after that does it cap how many
(now-distinct) chunks from one paper can fill the top-k, allowing up to 2 by
default rather than 1, so two genuinely different sections of the same paper
can both surface. Surveys are still demoted the same way in both.
"""

from __future__ import annotations

from collections import defaultdict


def _shingles(text: str, n: int = 3) -> set:
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def collapse_near_duplicate_chunks(ranked_chunks: list, canonical_ids: dict = None, threshold: float = 0.9) -> list:
    """Collapses chunks within the same canonical_paper_id group whose text
    is near-duplicate (word-trigram Jaccard >= threshold) — the same passage
    surfacing from more than one source/version copy of a paper. Keeps the
    canonical row's copy of the passage when one of the duplicates belongs to
    it (via `canonical_ids`: group_id -> canonical row's paper_id); otherwise
    keeps the highest-ranked copy. The surviving representative is placed at
    the best (lowest-index) rank any member of its cluster achieved — rank
    order is otherwise preserved.
    """
    canonical_ids = canonical_ids or {}
    clusters: list = []  # each: {"group": pid, "members": [chunk,...], "shingles": set, "min_rank": int}

    for idx, c in enumerate(ranked_chunks):
        pid = c.get("canonical_paper_id")
        sh = _shingles(c.get("text", ""))
        matched = None
        for cl in clusters:
            if cl["group"] == pid and _jaccard(sh, cl["shingles"]) >= threshold:
                matched = cl
                break
        if matched is None:
            matched = {"group": pid, "members": [], "shingles": sh, "min_rank": idx}
            clusters.append(matched)
        matched["members"].append(c)
        matched["min_rank"] = min(matched["min_rank"], idx)

    out = []
    for cl in clusters:
        canonical_key = canonical_ids.get(cl["group"])
        rep = next((c for c in cl["members"] if c.get("paper_id") == canonical_key), cl["members"][0])
        out.append((cl["min_rank"], rep))
    out.sort(key=lambda pair: pair[0])
    return [c for _, c in out]


def diversify(
    ranked_chunks: list,
    k: int = 5,
    survey_group_ids: set = frozenset(),
    canonical_ids: dict = None,
    max_per_group: int = 2,
    near_dup_threshold: float = 0.9,
) -> list:
    deduped = collapse_near_duplicate_chunks(ranked_chunks, canonical_ids=canonical_ids, threshold=near_dup_threshold)

    primaries = [c for c in deduped if c.get("canonical_paper_id") not in survey_group_ids]
    surveys = [c for c in deduped if c.get("canonical_paper_id") in survey_group_ids]

    result: list = []
    per_group_count: dict = defaultdict(int)
    for c in primaries:
        pid = c.get("canonical_paper_id")
        if per_group_count[pid] >= max_per_group:
            continue
        result.append(c)
        per_group_count[pid] += 1
        if len(result) >= k:
            return result[:k]

    for c in surveys:
        pid = c.get("canonical_paper_id")
        if per_group_count[pid] >= max_per_group:
            continue
        result.append(c)
        per_group_count[pid] += 1
        if len(result) >= k:
            break
    return result[:k]


def diversify_cap1_naive(ranked_chunks: list, k: int = 5, survey_group_ids: set = frozenset()) -> list:
    """The original (2026-10-06) diversification: one chunk per paper, no
    near-duplicate-chunk collapsing. Kept as an ablation baseline only — see
    module docstring and DECISIONS.md for why it's not used by default."""
    seen_papers: set = set()
    primaries: list = []
    deferred_surveys: list = []

    for c in ranked_chunks:
        pid = c.get("canonical_paper_id")
        if pid in seen_papers:
            continue
        if pid in survey_group_ids:
            deferred_surveys.append(c)
            continue
        seen_papers.add(pid)
        primaries.append(c)
        if len(primaries) >= k:
            return primaries[:k]

    result = list(primaries)
    for c in deferred_surveys:
        pid = c.get("canonical_paper_id")
        if pid in seen_papers:
            continue
        seen_papers.add(pid)
        result.append(c)
        if len(result) >= k:
            break
    return result[:k]
