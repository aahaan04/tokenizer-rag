from tokrag.dedup.survey import is_survey


def test_survey_detected_by_title_keyword():
    assert is_survey("A Survey on Mathematical Reasoning and Optimization with Large Language Models", "")


def test_review_detected_by_title_keyword():
    assert is_survey("A Systematic Review of Subword Tokenization Methods", "")


def test_survey_detected_by_abstract_phrase_when_title_is_generic():
    assert is_survey(
        "Understanding Tokenization",
        "In this survey, we provide a comprehensive overview of subword tokenization algorithms developed over the last decade.",
    )


def test_non_survey_paper_not_flagged():
    assert not is_survey(
        "SentencePiece: A Simple and Language Independent Subword Tokenizer",
        "This paper describes SentencePiece, a language-independent subword tokenizer and detokenizer.",
    )


def test_paper_that_merely_cites_other_surveys_not_flagged():
    assert not is_survey(
        "BPE Gets Picky",
        "Building on prior survey work in tokenizer evaluation, we propose a new vocabulary refinement method.",
    )
