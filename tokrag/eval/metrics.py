"""Recall@k, MRR, and duplicate-rate@k, using the group-level relevance
definition in DECISIONS.md (2026-10-06 entries): a retrieved chunk is a hit
if its canonical_paper_id matches the gold paper's group id AND the chunk's
text contains the gold supporting span (normalized, with a fuzzy fallback on
significant-word overlap). Section is kept as citation metadata but is NOT
part of the hit criterion — PDF section headings are too noisy (see the
earlier Phase 2 parsing-quality entries) to use as a scoring signal."""

from __future__ import annotations

import re

_NORMALIZE_RE = re.compile(r"[^a-z0-9]")
_WORD_RE = re.compile(r"[a-z0-9]{4,}")
FUZZY_WORD_OVERLAP_THRESHOLD = 0.7


def _normalize_text(s: str) -> str:
    return _NORMALIZE_RE.sub("", (s or "").lower())


def _significant_words(s: str) -> set:
    return set(_WORD_RE.findall((s or "").lower()))


def normalize_section(title: str) -> str:
    """Still used for display/citation purposes, not for hit scoring."""
    t = re.sub(r"^[\d.]+\s*", "", (title or "").lower())
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return t.strip()


def is_hit(retrieved_chunk: dict, gold_group_id: str, gold_span: str) -> bool:
    if retrieved_chunk.get("canonical_paper_id") != gold_group_id:
        return False
    chunk_text = retrieved_chunk.get("text", "")
    if not chunk_text or not gold_span:
        return False

    norm_chunk = _normalize_text(chunk_text)
    norm_span = _normalize_text(gold_span)
    if norm_span and norm_span in norm_chunk:
        return True

    # Fuzzy fallback: most of the span's distinctive (len>=4) words appear in
    # the chunk — tolerates minor extraction/whitespace differences without
    # requiring a byte-exact substring match.
    span_words = _significant_words(gold_span)
    if not span_words:
        return False
    chunk_words = _significant_words(chunk_text)
    overlap = len(span_words & chunk_words) / len(span_words)
    return overlap >= FUZZY_WORD_OVERLAP_THRESHOLD


def recall_at_k(retrieved: list[dict], gold_group_id: str, gold_span: str, k: int = 5) -> int:
    return 1 if any(is_hit(c, gold_group_id, gold_span) for c in retrieved[:k]) else 0


def reciprocal_rank(retrieved: list[dict], gold_group_id: str, gold_span: str) -> float:
    for i, c in enumerate(retrieved, start=1):
        if is_hit(c, gold_group_id, gold_span):
            return 1.0 / i
    return 0.0


def duplicate_rate_at_k(retrieved: list[dict], k: int = 5) -> float:
    """Fraction of the top-k chunks whose canonical_paper_id repeats one
    already seen earlier in the top-k — "N rows of the same paper crowding
    the other slots," independent of whether any of them are relevant."""
    top = retrieved[:k]
    if not top:
        return 0.0
    seen = set()
    dup_count = 0
    for c in top:
        pid = c.get("canonical_paper_id")
        if pid in seen:
            dup_count += 1
        else:
            seen.add(pid)
    return dup_count / len(top)


def mean_recall_at_k(results: list[tuple[list[dict], str, str]], k: int = 5) -> float:
    if not results:
        return 0.0
    return sum(recall_at_k(r, g, s, k) for r, g, s in results) / len(results)


def mean_reciprocal_rank(results: list[tuple[list[dict], str, str]]) -> float:
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, g, s) for r, g, s in results) / len(results)


def mean_duplicate_rate_at_k(all_retrieved: list[list[dict]], k: int = 5) -> float:
    if not all_retrieved:
        return 0.0
    return sum(duplicate_rate_at_k(r, k) for r in all_retrieved) / len(all_retrieved)
