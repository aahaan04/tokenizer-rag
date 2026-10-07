"""Orchestrates Phase 2: fetch raw text per included manifest row, extract
section-aware text, chunk it, and write chunks.jsonl + a parse report.

Every included row (not deduplicated by candidate_group_id) gets its own
fetch/parse/chunk pass — see fetch.py's docstring for why.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokrag.collect.manifest import read_manifest
from tokrag.config import MANIFEST_PATH, PROCESSED_DIR
from tokrag.parse import chunker, extract, fetch

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
PARSE_REPORT_PATH = PROCESSED_DIR / "parse_report.json"
MIN_USABLE_CHARS = 200  # extracted text shorter than this is treated as a failed parse


def run(limit: int | None = None) -> dict:
    rows = [r for r in read_manifest(MANIFEST_PATH) if r.included]
    if limit:
        rows = rows[:limit]

    print(f"Fetching raw text for {len(rows)} included rows...")
    status = fetch.fetch_all(rows)

    chunks_out = []
    report = {
        "total_rows": len(rows),
        "by_fetch_method": {},
        "full_text_parsed": 0,
        "abstract_only": 0,
        "no_text_at_all": 0,
        "total_chunks": 0,
    }

    for row in rows:
        key = fetch.safe_key(row)
        fetch_result = status.get(key, {"method": "none", "path": None, "status": "abstract_only"})
        method = fetch_result["method"]
        report["by_fetch_method"][method] = report["by_fetch_method"].get(method, 0) + 1

        sections: list[tuple[str, str]] = []
        if fetch_result["status"] == "ok" and fetch_result["path"]:
            try:
                sections = extract.extract_sections(Path(fetch_result["path"]))
            except Exception as e:
                print(f"  [parse error] {key}: {e}")
                sections = []
            if sum(len(t) for _, t in sections) < MIN_USABLE_CHARS:
                sections = []  # extraction produced too little to be useful -> fall back

        abstract_only = not sections
        if abstract_only:
            if row.abstract:
                sections = [("Abstract", row.abstract)]
            report["abstract_only"] += 1
        else:
            report["full_text_parsed"] += 1

        if not sections:
            report["no_text_at_all"] += 1
            continue

        for c in chunker.chunk_sections(sections):
            chunks_out.append(
                {
                    "paper_id": key,
                    "canonical_paper_id": row.candidate_group_id or key,
                    "source": row.source,
                    "title": row.title,
                    "year": row.year,
                    "venue": row.venue,
                    "abstract_only": abstract_only,
                    "fetch_method": method,
                    **c,
                }
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks_out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    report["total_chunks"] = len(chunks_out)
    PARSE_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
