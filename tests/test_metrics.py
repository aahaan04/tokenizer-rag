from tokrag.eval.metrics import (
    duplicate_rate_at_k,
    is_hit,
    mean_duplicate_rate_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def _chunk(group_id, text, section="Introduction"):
    return {"canonical_paper_id": group_id, "text": text, "section_title": section}


def test_is_hit_requires_paper_match_and_span_in_text():
    c = _chunk("grp1", "SentencePiece comprises four main components: Normalizer, Trainer, Encoder, and Decoder.")
    assert is_hit(c, "grp1", "four main components")
    assert not is_hit(c, "grp2", "four main components")  # wrong paper
    assert not is_hit(c, "grp1", "vocabulary size of 50,000")  # right paper, span not present


def test_is_hit_normalizes_punctuation_and_number_formatting():
    c = _chunk("grp1", "The final shared mBERT vocabulary comprises a total of 119,547 subword tokens.")
    assert is_hit(c, "grp1", "a total of 119,547 subword tokens")
    assert is_hit(c, "grp1", "119547 subword tokens")  # comma difference tolerated


def test_is_hit_section_is_not_part_of_the_criterion():
    # Even a garbled/unrelated section_title shouldn't block a hit when the
    # span is genuinely present — PDF headings are too noisy to gate on.
    c = _chunk("grp1", "vocabulary size of 50,000", section="Regarding C2, For Computational Tractability We")
    assert is_hit(c, "grp1", "vocabulary size of 50,000")


def test_is_hit_fuzzy_fallback_tolerates_minor_wording_differences():
    c = _chunk("grp1", "we trained tokenizers with a target vocabulary size of 80000 tokens for the base experiments")
    # Not a verbatim substring (reordered / reworded) but shares most distinctive words.
    assert is_hit(c, "grp1", "target vocabulary size 80000 tokens base experiments")


def test_is_hit_fuzzy_fallback_rejects_unrelated_text():
    c = _chunk("grp1", "we study byte pair encoding for symbolic music generation using MIDI datasets")
    assert not is_hit(c, "grp1", "target vocabulary size 80000 tokens base experiments")


def test_recall_at_k_hit_within_top_k():
    retrieved = [_chunk("grpX", "irrelevant"), _chunk("grp1", "the answer span here"), _chunk("grpY", "irrelevant")]
    assert recall_at_k(retrieved, "grp1", "the answer span", k=5) == 1
    assert recall_at_k(retrieved, "grp1", "the answer span", k=1) == 0  # hit is at rank 2


def test_reciprocal_rank_scores_by_position():
    retrieved = [_chunk("grpX", "irrelevant"), _chunk("grp1", "the answer span here"), _chunk("grpY", "irrelevant")]
    assert reciprocal_rank(retrieved, "grp1", "the answer span") == 0.5
    assert reciprocal_rank(retrieved, "grpZ", "nothing matches") == 0.0


def test_duplicate_rate_counts_repeated_papers_in_top_k():
    retrieved = [_chunk("grpA", "a"), _chunk("grpA", "b"), _chunk("grpA", "c"), _chunk("grpB", "d"), _chunk("grpC", "e")]
    assert duplicate_rate_at_k(retrieved, k=5) == 2 / 5


def test_duplicate_rate_zero_when_all_distinct():
    retrieved = [_chunk(f"grp{i}", "x") for i in range(5)]
    assert duplicate_rate_at_k(retrieved, k=5) == 0.0


def test_mean_functions_average_across_questions():
    results = [
        ([_chunk("grp1", "the answer")], "grp1", "the answer"),  # hit at rank 1
        ([_chunk("grpX", "nope")], "grp2", "the answer"),  # miss
    ]
    assert mean_recall_at_k(results, k=5) == 0.5
    assert mean_reciprocal_rank(results) == 0.5

    all_retrieved = [[_chunk("grpA", "x"), _chunk("grpA", "y")], [_chunk("grpB", "z")]]
    assert mean_duplicate_rate_at_k(all_retrieved, k=5) == (0.5 + 0.0) / 2
