"""arXiv API client: query, parse, disk-cache, rate-limited (~1 request / 3s)."""

from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests

from tokrag.config import CACHE_DIR

ARXIV_API_URL = "http://export.arxiv.org/api/query"
MIN_REQUEST_INTERVAL_S = 3.1
ARXIV_CACHE_DIR = CACHE_DIR / "arxiv"
ARXIV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

_last_request_time = 0.0


@dataclass
class ArxivEntry:
    arxiv_id: str  # base id, e.g. "2104.12345" (no version suffix)
    version: str  # e.g. "v2"
    title: str
    authors: list
    abstract: str
    categories: list
    published: str  # ISO date this specific version was published
    updated: str
    pdf_url: str
    abs_url: str


def _cache_path(query: str, start: int, max_results: int) -> Path:
    key = hashlib.sha256(f"{query}|{start}|{max_results}".encode()).hexdigest()[:24]
    return ARXIV_CACHE_DIR / f"{key}.xml"


def _rate_limit() -> None:
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL_S:
        time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)
    _last_request_time = time.monotonic()


def _fetch_page(query: str, start: int, max_results: int) -> str:
    cache_file = _cache_path(query, start, max_results)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    _rate_limit()
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    cache_file.write_text(resp.text, encoding="utf-8")
    return resp.text


def _parse_entry(entry: ET.Element) -> ArxivEntry:
    raw_id = entry.findtext("atom:id", default="", namespaces=_ATOM_NS)
    ident = raw_id.rsplit("/", 1)[-1]  # e.g. "2104.12345v2"
    if "v" in ident and ident.rsplit("v", 1)[-1].isdigit():
        base_id, version = ident.rsplit("v", 1)
        version = f"v{version}"
    else:
        base_id, version = ident, "v1"

    title = (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").strip()
    title = " ".join(title.split())
    abstract = (entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").strip()
    abstract = " ".join(abstract.split())
    authors = [
        " ".join((a.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").split())
        for a in entry.findall("atom:author", _ATOM_NS)
    ]
    categories = [c.get("term", "") for c in entry.findall("atom:category", _ATOM_NS)]
    published = entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or ""
    updated = entry.findtext("atom:updated", default="", namespaces=_ATOM_NS) or ""

    pdf_url = ""
    for link in entry.findall("atom:link", _ATOM_NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")

    return ArxivEntry(
        arxiv_id=base_id,
        version=version,
        title=title,
        authors=authors,
        abstract=abstract,
        categories=categories,
        published=published,
        updated=updated,
        pdf_url=pdf_url,
        abs_url=f"https://arxiv.org/abs/{base_id}{version}",
    )


def fetch_by_ids(arxiv_ids: list) -> list:
    """Fetch specific papers by base arXiv id (latest version), via id_list
    rather than a search query. Used to guarantee known canonical papers are
    in the candidate pool regardless of whether the keyword queries rank them
    highly enough to surface within max_results.
    """
    if not arxiv_ids:
        return []
    id_list = ",".join(arxiv_ids)
    cache_file = ARXIV_CACHE_DIR / f"ids_{hashlib.sha256(id_list.encode()).hexdigest()[:24]}.xml"
    if cache_file.exists():
        xml_text = cache_file.read_text(encoding="utf-8")
    else:
        _rate_limit()
        resp = requests.get(
            ARXIV_API_URL, params={"id_list": id_list, "max_results": len(arxiv_ids)}, timeout=30
        )
        resp.raise_for_status()
        xml_text = resp.text
        cache_file.write_text(xml_text, encoding="utf-8")
    root = ET.fromstring(xml_text)
    return [_parse_entry(e) for e in root.findall("atom:entry", _ATOM_NS)]


def search(query: str, max_results: int = 100, page_size: int = 100) -> list:
    """Search arXiv for `query` (arXiv query syntax, e.g. 'all:"byte-level tokenization"').

    Paginates in chunks of `page_size`. Each page is disk-cached so reruns cost
    nothing and respect the rate limit only on a cold cache.
    """
    entries: list[ArxivEntry] = []
    start = 0
    while len(entries) < max_results:
        batch = min(page_size, max_results - len(entries))
        xml_text = _fetch_page(query, start, batch)
        root = ET.fromstring(xml_text)
        page_entries = [_parse_entry(e) for e in root.findall("atom:entry", _ATOM_NS)]
        if not page_entries:
            break
        entries.extend(page_entries)
        start += batch
        if len(page_entries) < batch:
            break
    return entries[:max_results]
