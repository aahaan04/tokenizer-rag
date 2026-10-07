from tokrag.chat.chatbot import Citation, detect_model_abstention, parse_used_citations


def _citations(n):
    return [Citation(index=i, paper_title=f"Paper {i}", year="2024", section="Intro", paper_id=f"p{i}") for i in range(1, n + 1)]


def test_parse_used_citations_ascii_brackets():
    citations = _citations(3)
    answer = "The vocabulary was 8192 [1]. This differs from [2]."
    used = parse_used_citations(answer, citations)
    assert [c.index for c in used] == [1, 2]


def test_parse_used_citations_fullwidth_brackets():
    # Real bug found via manual testing: some responses use full-width
    # brackets despite the prompt instructing ASCII only.
    citations = _citations(2)
    answer = "The authors trained three vocabulary sizes【1】."
    used = parse_used_citations(answer, citations)
    assert [c.index for c in used] == [1]


def test_parse_used_citations_drops_out_of_range_reference():
    citations = _citations(2)
    answer = "According to [5], the answer is X."  # source 5 was never provided
    used = parse_used_citations(answer, citations)
    assert used == []


def test_parse_used_citations_no_citations_in_answer():
    citations = _citations(3)
    answer = "This is an answer with no bracket citations at all."
    assert parse_used_citations(answer, citations) == []


def test_parse_used_citations_deduplicates_repeated_reference():
    citations = _citations(2)
    answer = "As shown in [1], and again in [1], the result holds."
    used = parse_used_citations(answer, citations)
    assert [c.index for c in used] == [1]


def test_detect_model_abstention_positive_cases():
    assert detect_model_abstention("The corpus doesn't contain enough information to answer this.")
    assert detect_model_abstention("I cannot answer this question based on the given sources.")
    assert detect_model_abstention("There is no information about this in the provided sources.")


def test_detect_model_abstention_negative_case():
    assert not detect_model_abstention("The vocabulary size was 8192, as reported in the paper [1].")
