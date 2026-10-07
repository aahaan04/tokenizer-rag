"""Similarity scoring for the Phase 4 false-merge audit: title similarity,
author overlap, and abstract cosine similarity (via bge-small embeddings).

These scores do NOT drive merge decisions for the 105 groups Phase 1 already
formed (those were title-exact-match within collection, already conservative
— see DECISIONS.md). They exist to AUDIT those groups (flag any that look
wrong) and to surface near-miss pairs just below a merge threshold, per the
brief's false-merge-audit requirement.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]")


def normalize_title(title: str) -> str:
    t = _NORMALIZE_RE.sub("", (title or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_title(a), normalize_title(b)).ratio()


def _author_last_names(authors_field: str) -> set:
    names = set()
    for a in (authors_field or "").split(";"):
        a = a.strip()
        if not a:
            continue
        # "First Last" or "Last, First" -> take the longest token as a crude surname proxy
        parts = re.split(r"[,\s]+", a)
        parts = [p for p in parts if p]
        if parts:
            names.add(max(parts, key=len).lower())
    return names


def author_overlap(authors_a: str, authors_b: str) -> float:
    set_a, set_b = _author_last_names(authors_a), _author_last_names(authors_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def abstract_cosine_similarities(abstracts: list, model=None) -> "list[list[float]]":
    """Pairwise cosine similarity matrix for a list of abstracts, via a
    sentence-transformers model (bge-small by default, lazily loaded so
    modules that don't need it don't pay the import/load cost)."""
    import numpy as np

    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    texts = [a or "" for a in abstracts]
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    sims = emb @ emb.T
    return sims.tolist()
