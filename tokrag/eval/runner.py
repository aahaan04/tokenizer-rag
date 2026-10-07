"""Runs eval/questions.jsonl against a retrieval configuration (dense or
BM25) and computes Recall@5, MRR, and duplicate-rate@5 — for three
diversification modes (none / cap1_naive / new, see dedup/diversify.py), per
the brief's before/after dedup measurement. Supports a per-question-type
breakdown and a gold-standalone-rewrite mode for follow-ups.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

from tokrag.config import MANIFEST_PATH, PROJECT_ROOT
from tokrag.dedup.diversify import diversify, diversify_cap1_naive
from tokrag.eval import metrics
from tokrag.index.chunks import load_chunks

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
OVERFETCH_K = 50


def load_questions() -> list:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _load_manifest_rows() -> list:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_survey_group_ids() -> set:
    """Chunk-metadata `canonical_paper_id` values (group id, or a singleton's
    own `source__source_id` key) for every row flagged as a survey — NOT
    Phase 4's `canonical_id` column, which names one representative row
    within a group rather than the group itself."""
    rows = _load_manifest_rows()
    return {r["candidate_group_id"] or f"{r['source']}__{r['source_id']}" for r in rows if r.get("doc_type") == "survey"}


def load_canonical_ids() -> dict:
    """group id -> Phase 4's chosen canonical representative row's paper_id,
    for collapse_near_duplicate_chunks' "prefer the canonical copy" rule."""
    rows = _load_manifest_rows()
    out = {}
    for r in rows:
        gid = r["candidate_group_id"]
        if gid and r.get("canonical_id"):
            out[gid] = r["canonical_id"]
    return out


class DenseSearcher:
    def __init__(self, index_name: str, model_name: str):
        from sentence_transformers import SentenceTransformer

        from tokrag.index.dense import load_dense_index

        self.embeddings, self.meta = load_dense_index(index_name)
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, k: int) -> list:
        from tokrag.index.dense import search as dense_search

        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        return [i for i, _score in dense_search(q_emb, self.embeddings, k=k)]


class HybridSearcher:
    """BM25 + a dense searcher, combined via Reciprocal Rank Fusion (RRF) —
    chosen over a raw score-weighted blend because BM25 and cosine-similarity
    scores aren't on comparable scales; RRF only needs each list's RANKING,
    sidestepping that entirely. Tokenizer research is full of exact
    terminology ("SentencePiece," "fertility") BM25 is naturally good at,
    which motivates combining it with a dense model rather than replacing it.
    """

    def __init__(self, dense_searcher, bm25_searcher, rrf_k: int = 60):
        self.dense = dense_searcher
        self.bm25 = bm25_searcher
        self.rrf_k = rrf_k
        self.meta = dense_searcher.meta  # same chunk ordering/index for both

    def search(self, query: str, k: int) -> list:
        # Overfetch both rankings generously so RRF has enough to fuse before truncating to k.
        fetch_k = max(k * 4, 50)
        dense_idxs = self.dense.search(query, fetch_k)
        bm25_idxs = self.bm25.search(query, fetch_k)

        scores: dict = defaultdict(float)
        for rank, idx in enumerate(dense_idxs, start=1):
            scores[idx] += 1.0 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_idxs, start=1):
            scores[idx] += 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.keys(), key=lambda i: -scores[i])
        return ranked[:k]


class BM25Searcher:
    def __init__(self):
        from tokrag.index.bm25 import load_bm25_index

        self.bm25, self.meta = load_bm25_index()

    def search(self, query: str, k: int) -> list:
        from tokrag.index.bm25 import search as bm25_search

        return [i for i, _score in bm25_search(self.bm25, query, k=k)]


def _diversify_mode_fn(mode: str, k: int, survey_group_ids: set, canonical_ids: dict):
    if mode == "none":
        return lambda retrieved: retrieved[:k]
    if mode == "cap1_naive":
        return lambda retrieved: diversify_cap1_naive(retrieved, k=k, survey_group_ids=survey_group_ids)
    if mode == "new":
        return lambda retrieved: diversify(retrieved, k=k, survey_group_ids=survey_group_ids, canonical_ids=canonical_ids, max_per_group=2)
    raise ValueError(f"unknown diversify mode: {mode}")


def run_eval(searcher, k: int = 5, diversify_mode: str = "none", use_gold_rewrite: bool = False) -> dict:
    """Returns overall metrics plus a per-question-type breakdown and
    query-latency stats (search time only, excludes model load)."""
    import time

    chunks = load_chunks()
    assert len(chunks) == len(searcher.meta), "chunks.jsonl and index metadata are out of sync"

    survey_group_ids = load_survey_group_ids()
    canonical_ids = load_canonical_ids()
    apply_diversify = _diversify_mode_fn(diversify_mode, k, survey_group_ids, canonical_ids)

    questions = load_questions()
    by_type: dict = {}  # type -> {"recall_results": [...], "same_paper": [...], "redundant": [...]}
    latencies_ms: list = []
    per_question: list = []  # [(question_id, retrieved, golds)], for cross-config paired bootstrap CIs

    for q in questions:
        if q["type"] == "unanswerable":
            continue
        query_text = q["question"]
        if use_gold_rewrite and q.get("gold_standalone_rewrite"):
            query_text = q["gold_standalone_rewrite"]

        t0 = time.perf_counter()
        idxs = searcher.search(query_text, OVERFETCH_K)
        retrieved = [{**chunks[i], **searcher.meta[i]} for i in idxs]
        retrieved = apply_diversify(retrieved)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        bucket = by_type.setdefault(q["type"], {"recall_results": [], "same_paper": [], "redundant": []})
        bucket["same_paper"].append(metrics.same_paper_share_at_k(retrieved, k=k))
        bucket["redundant"].append(metrics.redundant_copy_rate_at_k(retrieved, k=k))
        golds = [(g["group_id"], g["supporting_span"]) for g in q["gold"]]
        bucket["recall_results"].append((retrieved, golds))
        per_question.append((q["id"], retrieved, golds))

    latencies_ms.sort()
    n_lat = len(latencies_ms)
    latency_stats = {
        "mean_ms": round(sum(latencies_ms) / n_lat, 2) if n_lat else 0.0,
        "p50_ms": round(latencies_ms[n_lat // 2], 2) if n_lat else 0.0,
        "p95_ms": round(latencies_ms[min(n_lat - 1, int(n_lat * 0.95))], 2) if n_lat else 0.0,
    }

    def _summarize(bucket):
        return {
            "n": len(bucket["recall_results"]),
            "recall_at_k": round(metrics.mean_recall_at_k_multi(bucket["recall_results"], k=k), 4),
            "mrr": round(metrics.mean_reciprocal_rank_multi(bucket["recall_results"]), 4),
            "same_paper_share_at_k": round(sum(bucket["same_paper"]) / len(bucket["same_paper"]), 4) if bucket["same_paper"] else 0.0,
            "redundant_copy_rate_at_k": round(sum(bucket["redundant"]) / len(bucket["redundant"]), 4) if bucket["redundant"] else 0.0,
        }

    all_recall_results = [r for b in by_type.values() for r in b["recall_results"]]
    all_same_paper = [d for b in by_type.values() for d in b["same_paper"]]
    all_redundant = [d for b in by_type.values() for d in b["redundant"]]

    return {
        "diversify_mode": diversify_mode,
        "k": k,
        "gold_rewrite": use_gold_rewrite,
        "overall": {
            "n": len(all_recall_results),
            "recall_at_k": round(metrics.mean_recall_at_k_multi(all_recall_results, k=k), 4),
            "mrr": round(metrics.mean_reciprocal_rank_multi(all_recall_results), 4),
            "same_paper_share_at_k": round(sum(all_same_paper) / len(all_same_paper), 4) if all_same_paper else 0.0,
            "redundant_copy_rate_at_k": round(sum(all_redundant) / len(all_redundant), 4) if all_redundant else 0.0,
        },
        "by_type": {t: _summarize(b) for t, b in sorted(by_type.items())},
        "query_latency": latency_stats,
        "per_question": per_question,
    }


def paired_bootstrap_ci(per_question_a: list, per_question_b: list, k: int = 5, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap 95% CI for (metric_a - metric_b), resampling
    QUESTIONS with replacement (not individual chunk hits) so the pairing by
    question stays intact across the two configs being compared. Requires
    per_question_a/b (as returned by run_eval's "per_question") to list the
    SAME questions in the SAME order — true by construction, since both come
    from iterating eval/questions.jsonl in file order.

    Returns Recall@5 and MRR diffs; "significant" means the 95% CI excludes
    0 — report differences only when this holds, per the brief's "only claim
    differences the CIs support" instruction.
    """
    import random

    ids_a = [qid for qid, _, _ in per_question_a]
    ids_b = [qid for qid, _, _ in per_question_b]
    assert ids_a == ids_b, "per_question lists must cover the same questions in the same order to pair correctly"

    n = len(per_question_a)
    recall_a = [metrics.recall_at_k_multi(r, g, k=k) for _, r, g in per_question_a]
    recall_b = [metrics.recall_at_k_multi(r, g, k=k) for _, r, g in per_question_b]
    mrr_a = [metrics.reciprocal_rank_multi(r, g) for _, r, g in per_question_a]
    mrr_b = [metrics.reciprocal_rank_multi(r, g) for _, r, g in per_question_b]

    rng = random.Random(seed)
    recall_diffs, mrr_diffs = [], []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        recall_diffs.append(sum(recall_a[i] for i in idxs) / n - sum(recall_b[i] for i in idxs) / n)
        mrr_diffs.append(sum(mrr_a[i] for i in idxs) / n - sum(mrr_b[i] for i in idxs) / n)
    recall_diffs.sort()
    mrr_diffs.sort()
    lo_i, hi_i = int(0.025 * n_boot), int(0.975 * n_boot)

    def _result(diffs, observed):
        lo, hi = diffs[lo_i], diffs[hi_i]
        return {"observed_diff": round(observed, 4), "ci_95": [round(lo, 4), round(hi, 4)], "significant": not (lo <= 0 <= hi)}

    return {
        "n": n,
        "recall_at_k": _result(recall_diffs, sum(recall_a) / n - sum(recall_b) / n),
        "mrr": _result(mrr_diffs, sum(mrr_a) / n - sum(mrr_b) / n),
    }


def index_size_mb(path) -> float:
    from pathlib import Path

    p = Path(path)
    if p.is_file():
        return round(p.stat().st_size / (1024 * 1024), 2)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


def run_dense_eval(index_name: str, model_name: str, k: int = 5, use_diversify: bool = False, use_gold_rewrite: bool = False) -> dict:
    """Back-compat convenience wrapper used by the CLI's `tokrag eval`."""
    searcher = DenseSearcher(index_name, model_name)
    mode = "new" if use_diversify else "none"
    result = run_eval(searcher, k=k, diversify_mode=mode, use_gold_rewrite=use_gold_rewrite)
    flat = {"index": index_name, "model": model_name, **result["overall"], "diversify_mode": mode, "k": k}
    return flat
