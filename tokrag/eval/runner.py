"""Runs eval/questions.jsonl against a dense-index retrieval configuration
and computes Recall@5, MRR, and duplicate-rate@5 — with and without Phase 4's
diversification (cap 1 chunk/paper + survey demotion), to measure dedup's
before/after impact per the brief.
"""

from __future__ import annotations

import csv
import json

from tokrag.config import MANIFEST_PATH, PROJECT_ROOT
from tokrag.dedup.diversify import diversify as diversify_fn
from tokrag.eval import metrics
from tokrag.index.chunks import load_chunks
from tokrag.index.dense import load_dense_index, search as dense_search

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.jsonl"
OVERFETCH_K = 30  # retrieve more than k so diversification has room to drop duplicates/surveys


def load_questions() -> list:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_survey_group_ids() -> set:
    """Returns the chunk-metadata `canonical_paper_id` values (group id, or a
    singleton's own `source__source_id` key — matching how chunks.jsonl was
    built in Phase 2) for every row flagged as a survey. NOT the same as
    Phase 4's `canonical_id` column, which names one representative ROW
    within a group rather than the group itself."""
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = set()
    for r in rows:
        if r.get("doc_type") == "survey":
            ids.add(r["candidate_group_id"] or f"{r['source']}__{r['source_id']}")
    return ids


def run_dense_eval(index_name: str, model_name: str, k: int = 5, use_diversify: bool = False, use_gold_rewrite: bool = False) -> dict:
    from sentence_transformers import SentenceTransformer

    embeddings, meta = load_dense_index(index_name)
    chunks = load_chunks()
    assert len(chunks) == len(meta), "chunks.jsonl and dense index metadata are out of sync"

    model = SentenceTransformer(model_name)
    survey_group_ids = load_survey_group_ids() if use_diversify else set()
    questions = load_questions()

    recall_results = []
    dup_rates = []
    n_unanswerable = 0

    for q in questions:
        if q["type"] == "unanswerable":
            n_unanswerable += 1
            continue

        query_text = q["question"]
        if use_gold_rewrite and q.get("gold_standalone_rewrite"):
            query_text = q["gold_standalone_rewrite"]

        q_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        hits = dense_search(q_emb, embeddings, k=OVERFETCH_K)
        retrieved = [{**meta[i], "text": chunks[i]["text"]} for i, _score in hits]

        if use_diversify:
            retrieved = diversify_fn(retrieved, k=k, survey_group_ids=survey_group_ids)
        else:
            retrieved = retrieved[:k]

        dup_rates.append(metrics.duplicate_rate_at_k(retrieved, k=k))
        golds = [(g["group_id"], g["supporting_span"]) for g in q["gold"]]
        recall_results.append((retrieved, golds))

    return {
        "index": index_name,
        "model": model_name,
        "diversify": use_diversify,
        "gold_rewrite": use_gold_rewrite,
        "k": k,
        "n_answerable": len(recall_results),
        "n_unanswerable": n_unanswerable,
        "recall_at_k": round(metrics.mean_recall_at_k_multi(recall_results, k=k), 4),
        "mrr": round(metrics.mean_reciprocal_rank_multi(recall_results), 4),
        "duplicate_rate_at_k": round(sum(dup_rates) / len(dup_rates), 4) if dup_rates else 0.0,
    }
