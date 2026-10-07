from tokrag.dedup.diversify import diversify


def _c(pid, rank):
    return {"canonical_paper_id": pid, "rank": rank}


def test_caps_one_chunk_per_paper():
    ranked = [_c("A", 1), _c("A", 2), _c("A", 3), _c("B", 4), _c("C", 5)]
    out = diversify(ranked, k=5)
    assert [c["canonical_paper_id"] for c in out] == ["A", "B", "C"]


def test_fills_remaining_slots_with_next_distinct_papers():
    ranked = [_c("A", 1), _c("A", 2), _c("B", 3), _c("C", 4), _c("D", 5), _c("E", 6)]
    out = diversify(ranked, k=5)
    assert [c["canonical_paper_id"] for c in out] == ["A", "B", "C", "D", "E"]


def test_survey_demoted_below_primaries():
    ranked = [_c("SURVEY", 1), _c("A", 2), _c("B", 3)]
    out = diversify(ranked, k=2, survey_group_ids={"SURVEY"})
    # SURVEY ranked #1 but there are >= k primaries, so it's excluded entirely
    assert [c["canonical_paper_id"] for c in out] == ["A", "B"]


def test_survey_fills_slot_when_not_enough_primaries():
    ranked = [_c("SURVEY", 1), _c("A", 2)]
    out = diversify(ranked, k=3, survey_group_ids={"SURVEY"})
    # Only 1 primary available -> survey fills the remaining slot
    assert [c["canonical_paper_id"] for c in out] == ["A", "SURVEY"]


def test_no_duplicate_papers_in_output_even_with_surveys():
    ranked = [_c("SURVEY", 1), _c("SURVEY", 2), _c("A", 3)]
    out = diversify(ranked, k=5, survey_group_ids={"SURVEY"})
    assert [c["canonical_paper_id"] for c in out] == ["A", "SURVEY"]


def test_output_never_exceeds_k():
    ranked = [_c(f"P{i}", i) for i in range(10)]
    out = diversify(ranked, k=3)
    assert len(out) == 3
