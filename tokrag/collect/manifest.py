"""The corpus manifest: one row per candidate paper, every column the brief asks for."""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Candidate:
    source: str  # "arxiv" or "semantic_scholar"
    source_id: str
    title: str = ""
    authors: str = ""  # "; "-joined
    year: str = ""
    venue: str = ""
    abstract: str = ""
    source_url: str = ""
    arxiv_id: str = ""
    arxiv_version: str = ""
    doi: str = ""
    semantic_scholar_id: str = ""
    external_ids: str = ""  # JSON-encoded
    license: str = ""
    query_matched: str = ""
    relevance_score: float = 0.0
    included: bool = False
    rejection_reason: str = ""
    # Filled in during Phase 4 (dedup); blank until then.
    canonical_id: str = ""
    doc_type: str = ""  # "paper" or "survey"


MANIFEST_FIELDS = [f.name for f in dataclasses.fields(Candidate)]


def write_manifest(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for c in sorted(candidates, key=lambda c: (-c.relevance_score, c.title)):
            writer.writerow(dataclasses.asdict(c))


def read_manifest(path: Path) -> list[Candidate]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["relevance_score"] = float(row["relevance_score"] or 0.0)
            row["included"] = row["included"] in ("True", "true", "1")
            rows.append(Candidate(**row))
        return rows
