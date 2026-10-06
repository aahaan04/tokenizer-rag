from tokrag.collect.manifest import Candidate
from tokrag.collect.pipeline import _link_or_add, _normalize_title


def test_normalize_title_ignores_case_and_punctuation():
    a = _normalize_title("CANINE: Pre-training an Efficient Tokenization-Free Encoder")
    b = _normalize_title("Canine: Pre-training an Efficient Tokenization-Free Encoder")
    assert a == b


def test_normalize_title_distinguishes_different_titles():
    a = _normalize_title("ByT5: Towards a Token-Free Future")
    b = _normalize_title("ByT5-Sanskrit, a Unified Model")
    assert a != b


def _fake_candidate(source, title):
    return Candidate(source=source, source_id=f"{source}-id", title=title)


def test_link_or_add_keeps_same_paper_as_two_rows_linked_by_group_id():
    # A preprint (arXiv) and its published version (ACL) must stay as SEPARATE
    # rows so Phase 4 can measure dedup's before/after impact — not merged.
    candidates = {}
    title_index = {}
    group_counter = [0]

    arxiv_c = _fake_candidate("arxiv", "CANINE: Pre-training an Efficient Tokenization-Free Encoder")
    _link_or_add(_normalize_title(arxiv_c.title), arxiv_c, "arxiv:1", candidates, title_index, group_counter)

    acl_c = _fake_candidate("acl_anthology", "Canine: Pre-training an Efficient Tokenization-Free Encoder")
    _link_or_add(_normalize_title(acl_c.title), acl_c, "acl:1", candidates, title_index, group_counter)

    assert len(candidates) == 2
    assert arxiv_c.candidate_group_id != ""
    assert arxiv_c.candidate_group_id == acl_c.candidate_group_id


def test_link_or_add_does_not_link_unrelated_titles():
    candidates = {}
    title_index = {}
    group_counter = [0]

    a = _fake_candidate("arxiv", "SentencePiece Tokenizer")
    _link_or_add(_normalize_title(a.title), a, "arxiv:1", candidates, title_index, group_counter)

    b = _fake_candidate("acl_anthology", "WordPiece Tokenizer")
    _link_or_add(_normalize_title(b.title), b, "acl:1", candidates, title_index, group_counter)

    assert a.candidate_group_id == ""
    assert b.candidate_group_id == ""


def test_link_or_add_three_way_group_shares_one_id():
    # arXiv preprint + ACL version + a canary-injected re-fetch of the same arXiv id.
    candidates = {}
    title_index = {}
    group_counter = [0]

    first = _fake_candidate("acl_anthology", "ByT5: Towards a Token-Free Future")
    _link_or_add(_normalize_title(first.title), first, "acl:1", candidates, title_index, group_counter)

    second = _fake_candidate("arxiv", "ByT5: Towards a Token-Free Future")
    _link_or_add(_normalize_title(second.title), second, "arxiv:1", candidates, title_index, group_counter)

    third = _fake_candidate("semantic_scholar", "ByT5: Towards a Token-Free Future")
    _link_or_add(_normalize_title(third.title), third, "s2:1", candidates, title_index, group_counter)

    assert len({first.candidate_group_id, second.candidate_group_id, third.candidate_group_id}) == 1
    assert first.candidate_group_id != ""
