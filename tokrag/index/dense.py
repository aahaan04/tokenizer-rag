"""Dense vector index: embed chunks with a sentence-transformers model, store
as an L2-normalized numpy matrix + parallel metadata, brute-force cosine
search via a single matrix-vector product.

Brute force, not FAISS: corpus-scale is a few thousand to tens of thousands
of chunks, which a numpy matmul handles in well under a second on CPU — see
the 2026-10-06 DECISIONS.md entry on this from Phase 1 planning.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from tokrag.config import INDEX_DIR
from tokrag.index.chunks import chunk_metadata, load_chunks


def build_dense_index(
    model_name: str,
    out_name: str,
    batch_size: int = 64,
    checkpoint_every: int = 2000,
    num_threads: int | None = None,
) -> dict:
    """Embeds all chunks and writes an L2-normalized float32 matrix + metadata.

    Checkpointed every `checkpoint_every` chunks to `embeddings_partial.npy` +
    `progress.json` in the output dir, so a killed/interrupted run (e.g. an
    overnight SPECTER embed) resumes from the last checkpoint instead of
    restarting — see the 2026-10-06 DECISIONS.md entry on embedding speed.
    """
    torch.set_num_threads(num_threads or os.cpu_count() or 4)

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    n = len(texts)

    out_dir = INDEX_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / "embeddings_partial.npy"
    progress_path = out_dir / "progress.json"

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    if partial_path.exists() and progress_path.exists():
        embeddings = np.load(partial_path)
        start = json.loads(progress_path.read_text())["completed"]
        print(f"Resuming {model_name} embedding from {start}/{n}")
    else:
        embeddings = np.zeros((n, dim), dtype=np.float32)
        start = 0

    t0 = time.monotonic()
    i = start
    while i < n:
        end = min(i + checkpoint_every, n)
        batch = model.encode(
            texts[i:end], batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )
        embeddings[i:end] = batch
        np.save(partial_path, embeddings)
        progress_path.write_text(json.dumps({"completed": end, "total": n}), encoding="utf-8")

        elapsed = time.monotonic() - t0
        rate = (end - start) / elapsed if elapsed > 0 else 0
        eta_min = (n - end) / rate / 60 if rate > 0 else float("nan")
        print(f"  {end}/{n} chunks embedded ({rate:.1f} chunks/s, ETA {eta_min:.1f} min)")
        i = end

    embed_seconds = time.monotonic() - t0

    np.save(out_dir / "embeddings.npy", embeddings)
    partial_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(chunk_metadata(c), ensure_ascii=False) + "\n")

    stats = {
        "model": model_name,
        "n_chunks": n,
        "dim": int(dim),
        "embed_seconds": round(embed_seconds, 1),
        "chunks_per_second": round((n - start) / embed_seconds, 1) if embed_seconds > 0 else None,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def load_dense_index(out_name: str):
    out_dir = INDEX_DIR / out_name
    embeddings = np.load(out_dir / "embeddings.npy")
    with (out_dir / "metadata.jsonl").open(encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
    return embeddings, metadata


def search(query_embedding: np.ndarray, embeddings: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    sims = embeddings @ query_embedding
    top_idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(int(i), float(sims[i])) for i in top_idx]
