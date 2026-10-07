"""ACL Anthology client.

Unlike arXiv/Semantic Scholar, the Anthology has no per-query search API — it
publishes its entire bibliography (with abstracts) as one bulk BibTeX export.
We download that once (cached), parse it, and filter locally. This is also
what we want for dedup: it's the clean source of preprint-vs-conference-version
matches (arXiv preprint -> its ACL-published form), see Phase 4.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass

import requests

from tokrag.config import CACHE_DIR

BIB_URL = "https://aclanthology.org/anthology+abstracts.bib.gz"
ACL_CACHE_DIR = CACHE_DIR / "acl"
BIB_GZ_PATH = ACL_CACHE_DIR / "anthology+abstracts.bib.gz"
ACL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Entries are separated by a line that's just "}" followed by the next "@type{".
_ENTRY_RE = re.compile(r"@(\w+)\{([^,\n]+),\n(.*?)\n\}\n", re.DOTALL)
# Field values are either "quoted" or {brace-delimited} — BibTeX allows both,
# and exporters commonly switch to braces for a field containing diacritics
# needing their own nested brace protection (e.g. author = {...{\"u}...}).
# A regex-only parser only matching quotes silently drops any brace-delimited
# field, which is NOT rare for international author names — found via the
# Phase 4 dedup audit when a paper's "author" field (braced, due to '{\"U}'
# in "Üstün") was missing and the code fell back to the wrong field
# (conference "editor" names) entirely. One level of nested braces is
# handled, which covers the diacritic-escaping case; stop at the next
# ",\n    field =", a trailing "\n}", or end of string (the entry regex
# strips the body's trailing "\n}\n", so the last field ends at EOF).
_FIELD_RE = re.compile(
    r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\{((?:[^{}]|\{[^{}]*\})*)\})'
    r"(?=,\s*\n\s*\w+\s*=|\s*\n\}|\s*$)",
    re.DOTALL,
)

# Cheap substring pre-filter so we don't fully field-parse all ~130k entries —
# only the small fraction whose title or abstract mentions tokenization at all.
PREFILTER_TERMS = ("token", "bpe", "subword", "sentencepiece", "wordpiece", "byte-pair", "byte pair")


@dataclass
class AclPaper:
    anthology_id: str
    title: str
    authors: list
    year: str
    venue: str
    abstract: str
    url: str
    doi: str


def _strip_braces(s: str) -> str:
    return re.sub(r"[{}]", "", s or "")


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split())


def _download_if_needed() -> None:
    if BIB_GZ_PATH.exists():
        return
    resp = requests.get(BIB_URL, timeout=180)
    resp.raise_for_status()
    BIB_GZ_PATH.write_bytes(resp.content)


def _parse_entry(key: str, body: str):
    fields = {m.group(1): (m.group(2) if m.group(2) is not None else m.group(3)) for m in _FIELD_RE.finditer(body)}
    title = _clean_ws(_strip_braces(fields.get("title", "")))
    if not title:
        return None
    authors_raw = fields.get("author", "") or fields.get("editor", "")
    authors = [_clean_ws(_strip_braces(a)) for a in re.split(r"\s+and\s+", authors_raw) if a.strip()]
    venue = _clean_ws(_strip_braces(fields.get("booktitle", "") or fields.get("journal", "")))
    return AclPaper(
        anthology_id=key,
        title=title,
        authors=authors,
        year=fields.get("year", ""),
        venue=venue,
        abstract=_clean_ws(_strip_braces(fields.get("abstract", ""))),
        url=fields.get("url", ""),
        doi=fields.get("doi", ""),
    )


def iter_candidates():
    """Yield AclPaper for every bib entry that passes the cheap keyword pre-filter.

    Downloads + caches the ~40MB bulk export on first call. Parsing all ~130k
    entries takes a couple of seconds; most are discarded by the pre-filter
    before the (slower) field-extraction regex runs on them.
    """
    _download_if_needed()
    with gzip.open(BIB_GZ_PATH, "rt", encoding="utf-8") as f:
        text = f.read()

    for _etype, key, body in _ENTRY_RE.findall(text):
        body_l = body.lower()
        if not any(t in body_l for t in PREFILTER_TERMS):
            continue
        paper = _parse_entry(key, body)
        if paper is None:
            continue
        yield paper
