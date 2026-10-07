"""Shared chunk loading for index builders."""

from __future__ import annotations

import json

from tokrag.config import PROCESSED_DIR

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

METADATA_FIELDS = (
    "paper_id",
    "canonical_paper_id",
    "source",
    "title",
    "year",
    "venue",
    "section_title",
    "chunk_index",
    "abstract_only",
    "fetch_method",
)


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_metadata(c: dict) -> dict:
    return {k: c[k] for k in METADATA_FIELDS}
