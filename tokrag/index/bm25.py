"""BM25 sparse index over chunk texts (rank_bm25). Used both as a Phase 3
retrieval baseline and as the gold-passage search tool for building the eval
set, per the brief's instruction to find candidates with BM25 + reading, not
the dense embedder (keeps the eval set from being biased toward the baseline
model it's also used to score)."""

from __future__ import annotations

import json
import pickle
import re
import time

from rank_bm25 import BM25Okapi

from tokrag.config import INDEX_DIR
from tokrag.index.chunks import chunk_metadata, load_chunks

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_bm25_index() -> dict:
    chunks = load_chunks()
    corpus_tokens = [tokenize(c["text"]) for c in chunks]

    t0 = time.monotonic()
    bm25 = BM25Okapi(corpus_tokens)
    build_seconds = time.monotonic() - t0

    out_dir = INDEX_DIR / "bm25"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(chunk_metadata(c), ensure_ascii=False) + "\n")

    stats = {"n_chunks": len(chunks), "build_seconds": round(build_seconds, 1)}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def load_bm25_index():
    out_dir = INDEX_DIR / "bm25"
    with (out_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    with (out_dir / "metadata.jsonl").open(encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
    return bm25, metadata


def search(bm25: BM25Okapi, query: str, k: int = 5) -> list[tuple[int, float]]:
    scores = bm25.get_scores(tokenize(query))
    top_idx = scores.argsort()[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_idx]
