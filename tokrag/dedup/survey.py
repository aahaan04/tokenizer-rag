"""Survey/review detection: flags a document type without merging it into the
papers it discusses. Per the brief, surveys restate other papers' findings
and should be diversified at retrieval time (see diversify.py), not merged —
merging would conflate a survey's summary of a finding with the primary
source that established it."""

from __future__ import annotations

import re

_TITLE_RE = re.compile(r"\b(survey|systematic review|a review of|literature review)\b", re.IGNORECASE)
_ABSTRACT_RE = re.compile(
    r"\b(we survey|this survey|in this survey|we review the|systematic review of|we provide a (?:comprehensive |)overview)\b",
    re.IGNORECASE,
)


def is_survey(title: str, abstract: str) -> bool:
    if _TITLE_RE.search(title or ""):
        return True
    return bool(_ABSTRACT_RE.search(abstract or ""))
