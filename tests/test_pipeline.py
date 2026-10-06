from tokrag.collect.pipeline import _normalize_title


def test_normalize_title_ignores_case_and_punctuation():
    a = _normalize_title("CANINE: Pre-training an Efficient Tokenization-Free Encoder")
    b = _normalize_title("Canine: Pre-training an Efficient Tokenization-Free Encoder")
    assert a == b


def test_normalize_title_distinguishes_different_titles():
    a = _normalize_title("ByT5: Towards a Token-Free Future")
    b = _normalize_title("ByT5-Sanskrit, a Unified Model")
    assert a != b
