from tokrag.eval.runner import paired_bootstrap_ci


def _hit_chunk(gid):
    return {"canonical_paper_id": gid, "text": "the gold answer span is right here"}


def _miss_chunk(gid):
    return {"canonical_paper_id": gid, "text": "completely unrelated content"}


def test_bootstrap_detects_clear_difference():
    # config A always hits, config B always misses -> CI should exclude 0 and favor A.
    golds = [("grp1", "the gold answer span")]
    per_q_a = [(f"q{i}", [_hit_chunk("grp1")], golds) for i in range(20)]
    per_q_b = [(f"q{i}", [_miss_chunk("grpX")], golds) for i in range(20)]

    result = paired_bootstrap_ci(per_q_a, per_q_b, k=5, n_boot=500)
    assert result["recall_at_k"]["observed_diff"] == 1.0
    assert result["recall_at_k"]["significant"] is True
    assert result["recall_at_k"]["ci_95"][0] > 0  # CI entirely positive, favors A


def test_bootstrap_no_difference_when_configs_identical():
    golds = [("grp1", "the gold answer span")]
    per_q = [(f"q{i}", [_hit_chunk("grp1")], golds) for i in range(20)]

    result = paired_bootstrap_ci(per_q, per_q, k=5, n_boot=500)
    assert result["recall_at_k"]["observed_diff"] == 0.0
    assert result["recall_at_k"]["significant"] is False
    assert result["mrr"]["observed_diff"] == 0.0


def test_bootstrap_requires_matching_question_order():
    per_q_a = [("q1", [], [])]
    per_q_b = [("q2", [], [])]
    try:
        paired_bootstrap_ci(per_q_a, per_q_b, k=5, n_boot=10)
        assert False, "expected AssertionError for mismatched question order"
    except AssertionError:
        pass


def test_bootstrap_reports_sample_size():
    golds = [("grp1", "span")]
    per_q = [(f"q{i}", [_hit_chunk("grp1")], golds) for i in range(7)]
    result = paired_bootstrap_ci(per_q, per_q, k=5, n_boot=100)
    assert result["n"] == 7
