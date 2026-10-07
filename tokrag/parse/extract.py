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


# Crude but cheap: a standalone short line matching a common paper section
# name. Will miss non-standard headers and occasionally misfire on body text
# that happens to be a short line matching one of these words — acceptable
# given the "working parse over perfect parse" time budget; see DECISIONS.md.
_PDF_HEADER_RE = re.compile(
    r"^\s*(\d{1,2}\.?\d{0,2}\.?\s*)?"
    r"(abstract|introduction|related work|background|method(ology)?|approach|"
    r"experiment(s)?|evaluation|result(s)?|discussion|conclusion(s)?|"
    r"limitations?|acknowledg(e)?ments?|appendix|references|bibliography)\s*$",
    re.IGNORECASE,
)


def extract_sections_from_pdf(pdf_path: Path) -> list[tuple[str, str]]:
    doc = fitz.open(pdf_path)
    try:
        text = "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()

    sections: list[tuple[str, str]] = []
    current_heading = "Abstract"
    current_parts: list[str] = []

    def flush() -> None:
        t = re.sub(r"\s+", " ", " ".join(current_parts)).strip()
        if t:
            sections.append((current_heading, t))

    for line in text.split("\n"):
        stripped = line.strip()
        m = _PDF_HEADER_RE.match(stripped) if len(stripped) < 60 else None
        if m:
            word = m.group(0).strip()
            if re.match(r"^(references|bibliography)", word, re.IGNORECASE):
                flush()
                return sections
            flush()
            current_heading = _LEADING_NUMBER_RE.sub("", word).strip().title()
            current_parts = []
        elif stripped:
            current_parts.append(stripped)
    flush()
    return sections


def extract_sections(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".html":
        return extract_sections_from_html(path.read_text(encoding="utf-8"))
    if path.suffix == ".pdf":
        return extract_sections_from_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")
