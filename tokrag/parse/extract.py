"""Turns raw HTML (arXiv's native LaTeXML rendering) or PDF text into a list
of (section_title, text) pairs.

Decisions (see DECISIONS.md for rationale): reference lists are dropped
(everything from a "References"/"Bibliography" heading onward); figure/table
captions are kept inline since they're picked up as ordinary text nodes;
tables and equations are kept as whatever linearized text PyMuPDF/the HTML
parser already extracts them as, not specially reconstructed.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz
from bs4 import BeautifulSoup

_HEADING_TAGS = ["h1", "h2", "h3", "h4"]
_CONTENT_TAGS = ["p", "figcaption", "caption", "td", "th", "li"]
_REFERENCES_RE = re.compile(r"^(references|bibliography)\s*$", re.IGNORECASE)
_LEADING_NUMBER_RE = re.compile(r"^[\d.]+\s*")


def extract_sections_from_html(html_text: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html_text, "lxml")
    root = soup.find("article") or soup.find("body") or soup

    sections: list[tuple[str, str]] = []
    current_heading = "Abstract"
    current_parts: list[str] = []

    def flush() -> None:
        text = re.sub(r"\s+", " ", " ".join(current_parts)).strip()
        if text:
            sections.append((current_heading, text))

    for el in root.find_all(_HEADING_TAGS + _CONTENT_TAGS):
        if el.name in _HEADING_TAGS:
            heading_text = el.get_text(" ", strip=True)
            heading_clean = _LEADING_NUMBER_RE.sub("", heading_text).strip()
            if _REFERENCES_RE.match(heading_clean):
                flush()
                return sections
            flush()
            current_heading = heading_clean or current_heading
            current_parts = []
        else:
            txt = el.get_text(" ", strip=True)
            if txt:
                current_parts.append(txt)
    flush()
    return sections


# Keyword headings (fallback when a line doesn't look numbered but names a
# common section) — crude but cheap, will occasionally misfire on body text
# that happens to be a short line matching one of these words.
_PDF_KEYWORD_HEADING_RE = re.compile(
    r"^\s*(\d{1,2}\.?\d{0,2}\.?\s*)?"
    r"(abstract|introduction|related work|background|method(ology)?|approach|"
    r"experiment(s)?|evaluation|result(s)?|discussion|conclusion(s)?|"
    r"limitations?|acknowledg(e)?ments?|appendix|references|bibliography)\s*$",
    re.IGNORECASE,
)
# Numbered headings like "3 Method" or "4.2 Results" — section number followed
# by a short, capitalized title (not a full sentence).
_PDF_NUMBERED_HEADING_RE = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,2})\.?\s+[A-Z][\w\-/ ]{1,60}$")
_REFERENCES_START_RE = re.compile(r"^(\d+\.?\s*)?(references|bibliography)", re.IGNORECASE)
# Running headers/footers: venue lines, bare page numbers, copyright notices.
_BOILERPLATE_LINE_RE = re.compile(
    r"^(proceedings of|page \s*\d+$|^\d{1,4}$|copyright \d|©|"
    r"association for computational linguistics)",
    re.IGNORECASE,
)


def _page_lines_with_fonts(doc) -> list[list[tuple[str, float, bool]]]:
    """Per page, a list of (line_text, font_size, is_bold) — one entry per text line."""
    pages = []
    for page in doc:
        d = page.get_text("dict")
        lines = []
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                s0 = spans[0]
                is_bold = bool(s0.get("flags", 0) & (1 << 4)) or "bold" in s0.get("font", "").lower()
                lines.append((text, s0.get("size", 10.0), is_bold))
        pages.append(lines)
    return pages


def _strip_running_headers_footers(pages: list[list[tuple[str, float, bool]]]) -> list[list[tuple[str, float, bool]]]:
    """Drops lines that repeat (near-)verbatim across a large fraction of pages
    — running headers/footers like the venue/proceedings line PyMuPDF can't
    otherwise distinguish from body text — plus explicit boilerplate patterns."""
    if len(pages) < 3:
        return [[l for l in page if not _BOILERPLATE_LINE_RE.search(l[0])] for page in pages]

    line_page_count = Counter()
    for page in pages:
        seen = {text.strip().lower() for text, _, _ in page}
        for key in seen:
            line_page_count[key] += 1

    threshold = max(2, int(len(pages) * 0.4))
    repeated = {k for k, v in line_page_count.items() if v >= threshold}

    return [
        [l for l in page if l[0].strip().lower() not in repeated and not _BOILERPLATE_LINE_RE.search(l[0])]
        for page in pages
    ]


def _body_font_size(pages: list[list[tuple[str, float, bool]]]) -> float:
    char_count_by_size: Counter = Counter()
    for page in pages:
        for text, size, _ in page:
            char_count_by_size[round(size, 1)] += len(text)
    if not char_count_by_size:
        return 10.0
    return char_count_by_size.most_common(1)[0][0]


def extract_sections_from_pdf(pdf_path: Path) -> list[tuple[str, str]]:
    doc = fitz.open(pdf_path)
    try:
        pages = _page_lines_with_fonts(doc)
    finally:
        doc.close()

    pages = _strip_running_headers_footers(pages)
    body_size = _body_font_size(pages)

    sections: list[tuple[str, str]] = []
    current_heading = "Abstract"
    current_parts: list[str] = []

    def flush() -> None:
        t = re.sub(r"\s+", " ", " ".join(current_parts)).strip()
        if t:
            sections.append((current_heading, t))

    for page in pages:
        for text, size, is_bold in page:
            heading_text = None
            if _PDF_NUMBERED_HEADING_RE.match(text):
                heading_text = text
            elif len(text) < 60 and _PDF_KEYWORD_HEADING_RE.match(text):
                heading_text = text
            elif len(text) < 70 and len(text.split()) <= 8 and (size > body_size + 0.5 or is_bold) and text[:1].isupper():
                heading_text = text

            if heading_text:
                if _REFERENCES_START_RE.match(heading_text):
                    flush()
                    return sections
                flush()
                current_heading = _LEADING_NUMBER_RE.sub("", heading_text).strip().title()
                current_parts = []
            else:
                current_parts.append(text)
    flush()
    return sections


def extract_sections(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".html":
        return extract_sections_from_html(path.read_text(encoding="utf-8"))
    if path.suffix == ".pdf":
        return extract_sections_from_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")
