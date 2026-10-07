from tokrag.collect.manifest import Candidate
from tokrag.dedup.pipeline import apply_fuzzy_merges, find_fuzzy_merges


class _FakeModel:
    """Returns a fixed similarity matrix regardless of input, for tests that
    exercise the '1 shared author + abstract similarity' branch without a
    real embedding model."""

    def __init__(self, sim: float):
        self.sim = sim

    def encode(self, texts, **kwargs):
        import numpy as np

        n = len(texts)
        # a simple vector whose pairwise cosine similarity is roughly self.sim for all pairs
        return np.array([[1.0, 0.0]] * n) if self.sim >= 0.999 else np.array([[self.sim, (1 - self.sim**2) ** 0.5]] + [[1.0, 0.0]] * (n - 1))


def _row(source, source_id, title, authors, year, abstract=""):
    return Candidate(source=source, source_id=source_id, title=title, authors=authors, year=str(year), abstract=abstract, included=True)


def test_find_fuzzy_merges_catches_two_shared_authors_above_title_threshold():
    a = _row("semantic_scholar", "a1", "Tokenizer Method for Efficient Training", "Jane Doe; John Smith", 2024)
    b = _row("acl_anthology", "b1", "Tokenizer Method for Efficient LLM Training", "Jane Doe; John Smith", 2025)
    cands = find_fuzzy_merges([a, b], model=None)
    assert len(cands) == 1
    assert cands[0]["shared_authors"] == 2


def test_find_fuzzy_merges_rejects_below_title_threshold():
    a = _row("semantic_scholar", "a1", "Completely Different Title About Parsing", "Jane Doe; John Smith", 2024)
    b = _row("acl_anthology", "b1", "Tokenizer Method for Efficient LLM Training", "Jane Doe; John Smith", 2025)
    cands = find_fuzzy_merges([a, b], model=None)
    assert cands == []


def test_find_fuzzy_merges_rejects_large_year_gap():
    a = _row("semantic_scholar", "a1", "Tokenizer Method for Efficient Training", "Jane Doe; John Smith", 2018)
    b = _row("acl_anthology", "b1", "Tokenizer Method for Efficient LLM Training", "Jane Doe; John Smith", 2025)
    cands = find_fuzzy_merges([a, b], model=None)
    assert cands == []


def test_find_fuzzy_merges_one_shared_author_needs_abstract_support():
    a = _row("semantic_scholar", "a1", "Tokenizer Method for Efficient Training", "Jane Doe; Alice Brown", 2024, "we propose a new method")
    b = _row("acl_anthology", "b1", "Tokenizer Method for Efficient LLM Training", "Jane Doe; Bob White", 2025, "we propose a new method")
    cands_low = find_fuzzy_merges([a, b], model=_FakeModel(0.5))
    assert cands_low == []
    cands_high = find_fuzzy_merges([a, b], model=_FakeModel(1.0))
    assert len(cands_high) == 1


def test_find_fuzzy_merges_skips_rows_already_grouped():
    a = _row("semantic_scholar", "a1", "Tokenizer Method for Efficient Training", "Jane Doe; John Smith", 2024)
    a.candidate_group_id = "grp0001"
    b = _row("acl_anthology", "b1", "Tokenizer Method for Efficient LLM Training", "Jane Doe; John Smith", 2025)
    cands = find_fuzzy_merges([a, b], model=None)
    assert cands == []  # a is already grouped -> not a "singleton" candidate


def test_find_fuzzy_merges_skips_proceedings_volumes():
    # Real false merge found at Checkpoint 4: two different workshop YEARS'
    # proceedings volumes, "authored" by overlapping organizing-committee
    # editors, scored as a high-confidence fuzzy match. Must not merge.
    a = _row(
        "acl_anthology", "ws-2017-subword",
        "Proceedings of the First Workshop on Subword and Character Level Models in NLP",
        "Jane Doe; John Smith; Alice Brown", 2017,
    )
    b = _row(
        "acl_anthology", "ws-2018-subword",
        "Proceedings of the Second Workshop on Subword and Character Level Models",
        "Jane Doe; John Smith; Bob White", 2018,
    )
    cands = find_fuzzy_merges([a, b], model=None)
    assert cands == []


def test_apply_fuzzy_merges_assigns_shared_group_id():
    a = _row("semantic_scholar", "a1", "Title One", "Jane Doe", 2024)
    b = _row("acl_anthology", "b1", "Title Two", "Jane Doe", 2025)
    candidates = [{"paper_a": a, "paper_b": b, "title_sim": 0.9, "shared_authors": 2, "year_diff": 1, "abstract_sim": None}]
    applied = apply_fuzzy_merges([a, b], candidates)
    assert len(applied) == 1
    assert a.candidate_group_id == b.candidate_group_id
    assert a.candidate_group_id.startswith("grp")


def test_apply_fuzzy_merges_manual_exception():
    a = _row("semantic_scholar", "x1", "MorphBPE A", "Asgari", 2025)
    b = _row("semantic_scholar", "x2", "MorphBPE B", "Asgari", 2026)
    applied = apply_fuzzy_merges([a, b], candidates=[], manual_pairs=[("semantic_scholar", "x1", "semantic_scholar", "x2")])
    assert len(applied) == 1
    assert a.candidate_group_id == b.candidate_group_id
