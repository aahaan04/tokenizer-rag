"""Semantic Scholar Graph API client: search, disk-cache, conservative backoff on 429."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from tokrag.config import CACHE_DIR, load_config

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,venue,authors,externalIds,url,openAccessPdf"
S2_CACHE_DIR = CACHE_DIR / "semantic_scholar"
S2_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MIN_REQUEST_INTERVAL_S_UNAUTH = 3.0  # conservative for the shared unauthenticated pool
MIN_REQUEST_INTERVAL_S_AUTH = 1.0  # a key gets its own dedicated rate limit
_last_request_time = 0.0


@dataclass
class S2Paper:
    paper_id: str
    title: str
    abstract: str
    year: str
    venue: str
    authors: list
    external_ids: dict
    url: str
    open_access_pdf: str


def _cache_path(query: str, offset: int, limit: int) -> Path:
    key = hashlib.sha256(f"{query}|{offset}|{limit}".encode()).hexdigest()[:24]
    return S2_CACHE_DIR / f"{key}.json"


def _rate_limit(has_key: bool) -> None:
    global _last_request_time
    min_interval = MIN_REQUEST_INTERVAL_S_AUTH if has_key else MIN_REQUEST_INTERVAL_S_UNAUTH
    elapsed = time.monotonic() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.monotonic()


def _fetch_page(query: str, offset: int, limit: int) -> dict:
    cache_file = _cache_path(query, offset, limit)
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    headers = {}
    api_key = load_config().semantic_scholar_api_key
    if api_key:
        headers["x-api-key"] = api_key

    backoff = 5.0
    for _ in range(3):
        _rate_limit(has_key=bool(api_key))
        resp = requests.get(
            S2_SEARCH_URL,
            params={"query": query, "offset": offset, "limit": limit, "fields": FIELDS},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
        data = resp.json()
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data
    # The unauthenticated tier is a shared global pool (not just our own request
    # rate) and can be saturated by other users; callers should treat this as
    # "S2 unavailable right now" and continue with arXiv-only results rather than
    # aborting the whole collection run. See DECISIONS.md.
    raise RuntimeError(f"Semantic Scholar rate limit: giving up on query={query!r} offset={offset}")


def _parse_paper(raw: dict) -> S2Paper:
    authors = [a.get("name", "") for a in raw.get("authors") or []]
    open_access = raw.get("openAccessPdf") or {}
    return S2Paper(
        paper_id=raw.get("paperId", "") or "",
        title=raw.get("title", "") or "",
        abstract=raw.get("abstract", "") or "",
        year=str(raw.get("year", "") or ""),
        venue=raw.get("venue", "") or "",
        authors=authors,
        external_ids=raw.get("externalIds") or {},
        url=raw.get("url", "") or "",
        open_access_pdf=open_access.get("url", "") or "",
    )


def search(query: str, max_results: int = 100, page_size: int = 100) -> list:
    """Search Semantic Scholar for `query`. Disk-cached per (query, offset, limit)."""
    papers: list[S2Paper] = []
    offset = 0
    while len(papers) < max_results:
        batch = min(page_size, max_results - len(papers))
        data = _fetch_page(query, offset, batch)
        raw_papers = data.get("data", [])
        if not raw_papers:
            break
        papers.extend(_parse_paper(p) for p in raw_papers)
        offset += batch
        total = data.get("total", 0)
        if offset >= total:
            break
    return papers[:max_results]
