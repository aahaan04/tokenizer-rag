"""Downloads raw paper content (arXiv HTML, ACL/S2 PDFs) per manifest row.

Every row gets its own fetch attempt — not deduplicated by candidate_group_id
— because Phase 4 needs each duplicate/version row to have its own indexable
text to measure dedup's effect on retrieval. See DECISIONS.md.

Resumable: a JSON status file records the outcome per row, so a rerun only
attempts rows that haven't been tried yet (success or failure alike — this is
a "don't redo work" resumability, not an automatic retry-on-failure).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from tokrag.collect.manifest import Candidate
from tokrag.config import CACHE_DIR, RAW_DIR

RAW_HTML_DIR = RAW_DIR / "html"
RAW_PDF_DIR = RAW_DIR / "pdf"
STATUS_PATH = CACHE_DIR / "fetch_status.json"
for _d in (RAW_HTML_DIR, RAW_PDF_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ARXIV_MIN_INTERVAL_S = 3.1
GENERIC_MIN_INTERVAL_S = 0.5
_last_arxiv_request = 0.0
_last_generic_request = 0.0

HEADERS = {"User-Agent": "tokrag-research-project/0.1 (educational; contact arsheth@wisc.edu)"}


def safe_key(row: Candidate) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{row.source}__{row.source_id}")


def _rate_limit(kind: str) -> None:
    global _last_arxiv_request, _last_generic_request
    now = time.monotonic()
    if kind == "arxiv":
        elapsed = now - _last_arxiv_request
        if elapsed < ARXIV_MIN_INTERVAL_S:
            time.sleep(ARXIV_MIN_INTERVAL_S - elapsed)
        _last_arxiv_request = time.monotonic()
    else:
        elapsed = now - _last_generic_request
        if elapsed < GENERIC_MIN_INTERVAL_S:
            time.sleep(GENERIC_MIN_INTERVAL_S - elapsed)
        _last_generic_request = time.monotonic()


def load_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {}


def save_status(status: dict) -> None:
    STATUS_PATH.write_text(json.dumps(status, indent=1), encoding="utf-8")


def _get(url: str, kind: str, timeout: int) -> requests.Response | None:
    _rate_limit(kind)
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException:
        return None


def _try_arxiv_html(arxiv_id: str, key: str) -> Path | None:
    path = RAW_HTML_DIR / f"{key}.html"
    if path.exists():
        return path
    resp = _get(f"https://arxiv.org/html/{arxiv_id}", "arxiv", 30)
    if resp is None or resp.status_code != 200 or len(resp.text) < 500:
        return None
    path.write_text(resp.text, encoding="utf-8")
    return path


def _try_arxiv_pdf(arxiv_id: str, key: str) -> Path | None:
    path = RAW_PDF_DIR / f"{key}.pdf"
    if path.exists():
        return path
    resp = _get(f"https://arxiv.org/pdf/{arxiv_id}", "arxiv", 60)
    if resp is None or resp.status_code != 200 or len(resp.content) < 1000:
        return None
    path.write_bytes(resp.content)
    return path


def _try_generic_pdf(url: str, key: str) -> Path | None:
    path = RAW_PDF_DIR / f"{key}.pdf"
    if path.exists():
        return path
    resp = _get(url, "generic", 60)
    if resp is None or resp.status_code != 200 or len(resp.content) < 1000:
        return None
    if not resp.content[:5].startswith(b"%PDF-"):
        return None
    path.write_bytes(resp.content)
    return path


_s2_open_access_index: dict | None = None


def _s2_open_access_url(paper_id: str) -> str | None:
    """Scans cached Semantic Scholar search responses for this paper's
    openAccessPdf.url. No network call — everything is already on disk from
    Phase 1 collection."""
    global _s2_open_access_index
    if _s2_open_access_index is None:
        _s2_open_access_index = {}
        s2_cache_dir = CACHE_DIR / "semantic_scholar"
        for f in s2_cache_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for p in data.get("data") or []:
                url = (p.get("openAccessPdf") or {}).get("url")
                if url and p.get("paperId"):
                    _s2_open_access_index[p["paperId"]] = url
    return _s2_open_access_index.get(paper_id)


def fetch_row(row: Candidate) -> dict:
    """Try arXiv HTML -> arXiv PDF -> source-specific PDF, in that order.
    Returns {"method": str, "path": str|None, "status": "ok"|"abstract_only"}.
    """
    key = safe_key(row)

    if row.arxiv_id:
        path = _try_arxiv_html(row.arxiv_id, key)
        if path:
            return {"method": "arxiv_html", "path": str(path), "status": "ok"}
        path = _try_arxiv_pdf(row.arxiv_id, key)
        if path:
            return {"method": "arxiv_pdf", "path": str(path), "status": "ok"}

    if row.source == "acl_anthology" and row.source_url:
        pdf_url = row.source_url.rstrip("/") + ".pdf"
        path = _try_generic_pdf(pdf_url, key)
        if path:
            return {"method": "acl_pdf", "path": str(path), "status": "ok"}

    if row.source == "semantic_scholar":
        oap_url = _s2_open_access_url(row.source_id)
        if oap_url:
            path = _try_generic_pdf(oap_url, key)
            if path:
                return {"method": "s2_pdf", "path": str(path), "status": "ok"}

    return {"method": "none", "path": None, "status": "abstract_only"}


def fetch_all(rows: list, progress_every: int = 20) -> dict:
    status = load_status()
    processed = 0
    for row in rows:
        key = safe_key(row)
        if key in status:
            continue
        status[key] = fetch_row(row)
        processed += 1
        if processed % progress_every == 0:
            save_status(status)
            print(f"  fetched {processed} new rows so far ({len(status)} total in status)...")
    save_status(status)
    return status
