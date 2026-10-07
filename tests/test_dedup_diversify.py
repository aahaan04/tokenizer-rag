from tokrag.dedup.diversify import collapse_near_duplicate_chunks, diversify, diversify_cap1_naive


def _c(pid, text="distinct content here", paper_id=None):
    return {"canonical_paper_id": pid, "text": text, "paper_id": paper_id or pid}


# --- collapse_near_duplicate_chunks ---


def test_collapse_merges_near_identical_passages_in_same_group():
    same_text = "SentencePiece comprises four main components a b c d e f g h"
    ranked = [_c("A", same_text, "A_arxiv"), _c("A", same_text, "A_acl")]
    out = collapse_near_duplicate_chunks(ranked)
    assert len(out) == 1


def test_collapse_keeps_distinct_passages_in_same_group():
    ranked = [_c("A", "the introduction discusses motivation", "A_arxiv"), _c("A", "the results table shows numbers", "A_arxiv")]
    out = collapse_near_duplicate_chunks(ranked)
    assert len(out) == 2


def test_collapse_does_not_merge_across_different_groups():
    same_text = "identical passage text repeated verbatim here for testing"
    ranked = [_c("A", same_text), _c("B", same_text)]
    out = collapse_near_duplicate_chunks(ranked)
    assert len(out) == 2


def test_collapse_prefers_canonical_copy_when_duplicate_found():
    same_text = "SentencePiece comprises four main components a b c d e f g h"
    ranked = [_c("A", same_text, "A_s2"), _c("A", same_text, "A_acl")]
    out = collapse_near_duplicate_chunks(ranked, canonical_ids={"A": "A_acl"})
    assert out[0]["paper_id"] == "A_acl"


def test_collapse_survivor_takes_best_rank_position():
    same_text = "SentencePiece comprises four main components a b c d e f g h"
    ranked = [_c("X", "unrelated filler"), _c("A", same_text, "A_s2"), _c("A", same_text, "A_acl")]
    out = collapse_near_duplicate_chunks(ranked, canonical_ids={"A": "A_acl"})
    # the A-group survivor (canonical copy) should appear at rank index 1 (its cluster's best rank), before nothing changes for X
    assert [c.get("paper_id") for c in out] == ["X", "A_acl"]


# --- diversify (new, near-dup-aware, max 2/group by default) ---


def test_diversify_allows_up_to_two_distinct_chunks_per_paper():
    ranked = [_c("A", "intro text"), _c("A", "results text"), _c("A", "conclusion text"), _c("B", "b text")]
    out = diversify(ranked, k=5)
    assert [c["canonical_paper_id"] for c in out] == ["A", "A", "B"]


def test_diversify_collapses_duplicates_before_applying_cap():
    same_text = "SentencePiece comprises four main components a b c d e f g h"
    ranked = [_c("A", same_text), _c("A", same_text), _c("B", "distinct b text")]
    out = diversify(ranked, k=5)
    # both A chunks are near-duplicates -> collapse to 1, so A contributes only 1 slot despite max_per_group=2
    assert [c["canonical_paper_id"] for c in out] == ["A", "B"]


def test_diversify_survey_demoted_below_primaries():
    ranked = [_c("SURVEY", "survey text"), _c("A", "a text"), _c("B", "b text")]
    out = diversify(ranked, k=2, survey_group_ids={"SURVEY"})
    assert [c["canonical_paper_id"] for c in out] == ["A", "B"]


def test_diversify_survey_fills_slot_when_not_enough_primaries():
    ranked = [_c("SURVEY", "survey text"), _c("A", "a text")]
    out = diversify(ranked, k=3, survey_group_ids={"SURVEY"})
    assert [c["canonical_paper_id"] for c in out] == ["A", "SURVEY"]


def test_diversify_output_never_exceeds_k():
    ranked = [_c(f"P{i}", f"distinct text {i}") for i in range(10)]
    out = diversify(ranked, k=3)
    assert len(out) == 3


# --- diversify_cap1_naive (ablation baseline, preserved exactly) ---


def test_cap1_naive_allows_only_one_chunk_per_paper():
    ranked = [_c("A", "intro"), _c("A", "results"), _c("A", "conclusion"), _c("B", "b text"), _c("C", "c text")]
    out = diversify_cap1_naive(ranked, k=5)
    assert [c["canonical_paper_id"] for c in out] == ["A", "B", "C"]


def test_cap1_naive_survey_demoted():
    ranked = [_c("SURVEY", "x"), _c("A", "a"), _c("B", "b")]
    out = diversify_cap1_naive(ranked, k=2, survey_group_ids={"SURVEY"})
    assert [c["canonical_paper_id"] for c in out] == ["A", "B"]
