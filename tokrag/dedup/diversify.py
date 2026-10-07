"""Retrieval-time result diversification — the actual fix for duplicate rows
(and surveys) crowding the top-k, applied AFTER ranking, not by merging rows
out of the index. Two rules, applied to an already-score-sorted chunk list:

1. Cap one chunk per canonical_paper_id in the top-k — stops 2-4 rows of the
   same paper (arXiv v1/latest, ACL copy, S2 record) from filling multiple
   slots that should go to different relevant papers.
2. Demote surveys: a survey chunk only fills a slot if there aren't enough
   non-survey ("primary source") chunks to fill k — a survey restating a
   finding shouldn't outrank the primary paper that established it.

This is a simpler, deliberate choice over full MMR (brief offers both) —
see DECISIONS.md for why, given the time budget.
"""

from __future__ import annotations


def diversify(ranked_chunks: list, k: int = 5, survey_group_ids: set = frozenset()) -> list:
    seen_papers = set()
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
