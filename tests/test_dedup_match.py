from tokrag.dedup.match import author_overlap, normalize_title, title_similarity


def test_normalize_title_strips_case_and_punctuation():
    assert normalize_title("CANINE: Pre-training an Efficient Encoder!") == normalize_title(
        "canine pretraining an efficient encoder"
    )


def test_title_similarity_identical_titles_is_one():
    assert title_similarity("SentencePiece: A Simple Tokenizer", "SentencePiece: A Simple Tokenizer") == 1.0


def test_title_similarity_unrelated_titles_is_low():
    assert title_similarity("SentencePiece: A Simple Tokenizer", "Convolutional Networks for Image Classification") < 0.4


def test_title_similarity_near_duplicate_titles_is_high():
    # Common preprint-vs-published title drift: punctuation/colon differences.
    s = title_similarity(
        "BPE-Dropout: Simple and Effective Subword Regularization",
        "BPE Dropout Simple and Effective Subword Regularization",
    )
    assert s > 0.95


def test_author_overlap_identical_author_lists_is_one():
    assert author_overlap("Jane Doe; John Smith", "Jane Doe; John Smith") == 1.0


def test_author_overlap_disjoint_author_lists_is_zero():
    assert author_overlap("Jane Doe; John Smith", "Alice Brown; Bob White") == 0.0


def test_author_overlap_partial_overlap_is_jaccard():
    # Shared surname "Doe" out of {doe, smith} union {doe, brown} = 1/3.
    assert author_overlap("Jane Doe; John Smith", "Jane Doe; Alice Brown") == 1 / 3


def test_author_overlap_empty_authors_is_zero():
    assert author_overlap("", "Jane Doe") == 0.0
    assert author_overlap("Jane Doe", "") == 0.0
