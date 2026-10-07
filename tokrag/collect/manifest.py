"""The corpus manifest: one row per candidate paper, every column the brief asks for."""

from __future__ import annotations

import csv
import dataclasses
from collections import Counter
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
    # Opportunistic same-paper hint from collection (title match across sources,
    # e.g. an arXiv preprint and its ACL-published version) — rows sharing a
    # candidate_group_id are kept as SEPARATE rows deliberately, so Phase 4 can
    # measure dedup's before/after impact rather than the pair already being
    # silently collapsed during collection. See the 2026-10-06 DECISIONS.md entry.
    candidate_group_id: str = ""
    # Filled in during Phase 4 (dedup); blank until then. Distinct from
    # candidate_group_id above: this is Phase 4's actual merge decision (DOI/
    # external-id matching, similarity thresholds), not Phase 1's title hint.
    canonical_id: str = ""
    doc_type: str = ""  # "paper" or "survey"
    # Filled in by Phase 2 parsing: "full_text", "abstract_only", or
    # "no_text_not_indexed" (no full text AND no abstract -> zero chunks,
    # excluded from the index). Blank until parse runs.
    parse_status: str = ""


MANIFEST_FIELDS = [f.name for f in dataclasses.fields(Candidate)]


def write_manifest(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for c in sorted(candidates, key=lambda c: (-c.relevance_score, c.title)):
            writer.writerow(dataclasses.asdict(c))


def write_manifest_summary(path: Path, candidates: list[Candidate]) -> None:
    """Human-readable counts alongside manifest.csv. Corpus size for the
    write-up is UNIQUE PAPERS, not included rows — a paper found via two or
    three sources (arXiv/S2/ACL) is deliberately kept as separate linked rows
    (see candidate_group_id), so rows overcount distinct papers.
    """
    included = [c for c in candidates if c.included]
    grouped = [c for c in included if c.candidate_group_id]
    n_groups = len({c.candidate_group_id for c in grouped})
    n_singletons = len(included) - len(grouped)
    unique_papers = n_groups + n_singletons

    by_source = Counter(c.source for c in included)
    by_year = Counter(c.year for c in included)
    vetoed = [c for c in candidates if c.rejection_reason.startswith("veto:")]
    veto_by_category = Counter(c.rejection_reason.split(":")[1] for c in vetoed)
    below_threshold = sum(
        1 for c in candidates if not c.included and not c.rejection_reason.startswith("veto:")
    )

    lines = [
        "# Manifest summary",
        "",
        f"Total candidate rows: {len(candidates)}",
        f"Included rows: {len(included)}",
        f"**Unique papers (corpus size, reported in WRITEUP.md): {unique_papers}**",
        f"  - {n_groups} cross-source groups ({len(grouped)} rows) — same paper found via multiple sources/versions",
        f"  - {n_singletons} singleton rows — found via one source only",
        "",
        "## Included, by source",
    ]
    for src, n in by_source.most_common():
        lines.append(f"- {src}: {n}")
    lines += ["", "## Included, by year"]
    for year, n in sorted(by_year.items()):
        lines.append(f"- {year or '(unknown)'}: {n}")
    lines += [
        "",
        f"## Rejected: {len(candidates) - len(included)}",
        f"- below relevance threshold: {below_threshold}",
        f"- vetoed (off-topic domain or classic-pipeline): {len(vetoed)}",
    ]
    for cat, n in veto_by_category.most_common():
        lines.append(f"  - {cat}: {n}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[Candidate]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["relevance_score"] = float(row["relevance_score"] or 0.0)
            row["included"] = row["included"] in ("True", "true", "1")
            rows.append(Candidate(**row))
        return rows
