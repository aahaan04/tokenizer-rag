"""Orchestrates Phase 4: canonical_id assignment, survey flagging, and the
false-merge audit (pairwise similarity scores within every existing group,
plus a bounded scan for near-miss pairs that were NOT merged).

Does not change which rows belong to which candidate_group_id — that
grouping happened in Phase 1 (title-exact-match across sources, see
DECISIONS.md) and stays the dedup unit. This module picks a canonical
representative row per group, flags surveys, and audits the groups for
false merges.
"""

from __future__ import annotations

import json
from collections import defaultdict

from tokrag.collect.manifest import read_manifest, write_manifest
from tokrag.config import MANIFEST_PATH, PROJECT_ROOT
from tokrag.dedup import match, survey

AUDIT_PATH = PROJECT_ROOT / "dedup_audit.md"

# Preference for which group member becomes "the" canonical row: a
# peer-reviewed ACL copy is more authoritative than an arXiv preprint, which
# in turn is preferred over a bare Semantic Scholar record. Within arXiv,
# the highest version number (latest) wins. See DECISIONS.md.
SOURCE_PRIORITY = {"acl_anthology": 0, "arxiv": 1, "semantic_scholar": 2}


def _version_num(m) -> int:
    try:
        return int((m.arxiv_version or "v0").lstrip("v"))
    except ValueError:
        return 0


def _pick_canonical(members: list):
    return sorted(members, key=lambda m: (SOURCE_PRIORITY.get(m.source, 9), -_version_num(m)))[0]


def assign_canonical_ids(rows: list) -> dict:
    by_group = defaultdict(list)
    for r in rows:
        if r.included and r.candidate_group_id:
            by_group[r.candidate_group_id].append(r)

    group_canonical = {}
    for gid, members in by_group.items():
        canonical_row = _pick_canonical(members)
        canonical_key = f"{canonical_row.source}__{canonical_row.source_id}"
        group_canonical[gid] = canonical_key
        for m in members:
            m.canonical_id = canonical_key

    for r in rows:
        if r.included and not r.candidate_group_id:
            r.canonical_id = f"{r.source}__{r.source_id}"

    return group_canonical


def assign_doc_types(rows: list) -> None:
    for r in rows:
        if r.included:
            r.doc_type = "survey" if survey.is_survey(r.title, r.abstract) else "paper"


def audit_groups(rows: list, model) -> list:
    by_group = defaultdict(list)
    for r in rows:
        if r.included and r.candidate_group_id:
            by_group[r.candidate_group_id].append(r)

    audit = []
    for gid, members in sorted(by_group.items()):
        abstracts = [m.abstract for m in members]
        sims = match.abstract_cosine_similarities(abstracts, model=model) if len(members) > 1 else [[1.0]]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                audit.append(
                    {
                        "group_id": gid,
                        "paper_a": f"{a.source}:{a.source_id}",
                        "paper_b": f"{b.source}:{b.source_id}",
                        "title_a": a.title,
                        "title_b": b.title,
                        "title_sim": round(match.title_similarity(a.title, b.title), 3),
                        "author_overlap": round(match.author_overlap(a.authors, b.authors), 3),
                        "abstract_cosine": round(sims[i][j], 3),
                    }
                )
    return audit


def find_near_miss_pairs(rows: list, title_sim_range=(0.4, 0.85), min_author_overlap=0.2, max_block_size=50) -> list:
    """Bounded by shared-author blocking (not O(n^2) over all included rows):
    only compares papers that share at least one author surname. Flags pairs
    with title similarity in a near-miss band AND meaningful author overlap
    — the brief's "same authors, similar titles" risky-case check, covering
    both near-misses below the merge threshold and a sanity check that no
    genuinely different paper (e.g. a paper and its follow-up) got merged.
    """
    included = [r for r in rows if r.included]
    by_author = defaultdict(list)
    for r in included:
        for name in match._author_last_names(r.authors):
            by_author[name].append(r)

    seen_pairs = set()
    near_misses = []
    for _name, members in by_author.items():
        if len(members) < 2 or len(members) > max_block_size:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a.candidate_group_id and a.candidate_group_id == b.candidate_group_id:
                    continue
                key = tuple(sorted([f"{a.source}:{a.source_id}", f"{b.source}:{b.source_id}"]))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                t_sim = match.title_similarity(a.title, b.title)
                auth_overlap = match.author_overlap(a.authors, b.authors)
                if title_sim_range[0] <= t_sim <= title_sim_range[1] and auth_overlap >= min_author_overlap:
                    near_misses.append(
                        {
                            "paper_a": f"{a.source}:{a.source_id}",
                            "title_a": a.title,
                            "paper_b": f"{b.source}:{b.source_id}",
                            "title_b": b.title,
                            "title_sim": round(t_sim, 3),
                            "author_overlap": round(auth_overlap, 3),
                        }
                    )
    near_misses.sort(key=lambda x: -x["title_sim"])
    return near_misses


def _write_audit_report(audit, near_misses, low_confidence, report) -> None:
    lines = [
        "# Dedup false-merge audit",
        "",
        f"Groups audited: {report['n_groups_audited']}",
        f"Pairs audited: {report['n_pairs_audited']}",
        f"Low-confidence merges (abstract cosine < 0.7 OR zero author overlap): {report['n_low_confidence_merges']}",
        f"Surveys flagged: {report['n_surveys_flagged']}",
        f"Near-miss pairs (NOT merged, similar title + shared author): {report['n_near_miss_pairs']}",
        "",
        "## Low-confidence merges (manual review)",
        "",
    ]
    for a in low_confidence:
        lines.append(f"- **{a['group_id']}**: title_sim={a['title_sim']} author_overlap={a['author_overlap']} abstract_cosine={a['abstract_cosine']}")
        lines.append(f"  - A: {a['title_a']} ({a['paper_a']})")
        lines.append(f"  - B: {a['title_b']} ({a['paper_b']})")
    lines += ["", "## All merge-cluster pairwise scores", ""]
    for a in audit:
        lines.append(
            f"- {a['group_id']}: title_sim={a['title_sim']} author_overlap={a['author_overlap']} "
            f"abstract_cosine={a['abstract_cosine']} | {a['title_a'][:70]}"
        )
    lines += ["", "## Near-miss pairs (NOT merged), for manual spot-check", ""]
    for n in near_misses:
        lines.append(f"- title_sim={n['title_sim']} author_overlap={n['author_overlap']}")
        lines.append(f"  - A: {n['title_a']} ({n['paper_a']})")
        lines.append(f"  - B: {n['title_b']} ({n['paper_b']})")
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict:
    from sentence_transformers import SentenceTransformer

    rows = read_manifest(MANIFEST_PATH)
    assign_canonical_ids(rows)
    assign_doc_types(rows)
    write_manifest(MANIFEST_PATH, rows)

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    audit = audit_groups(rows, model)
    near_misses = find_near_miss_pairs(rows)
    low_confidence = [a for a in audit if a["abstract_cosine"] < 0.7 or a["author_overlap"] == 0.0]

    report = {
        "n_groups_audited": len({a["group_id"] for a in audit}),
        "n_pairs_audited": len(audit),
        "n_low_confidence_merges": len(low_confidence),
        "n_surveys_flagged": sum(1 for r in rows if r.included and r.doc_type == "survey"),
        "n_near_miss_pairs": len(near_misses),
    }
    _write_audit_report(audit, near_misses, low_confidence, report)
    print(json.dumps(report, indent=2))
    return report
