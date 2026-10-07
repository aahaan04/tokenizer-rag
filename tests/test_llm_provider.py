from tokrag.chat.llm_provider import extract_text


def test_extract_text_returns_content():
    response = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
    assert extract_text(response) == "hello"


def test_extract_text_handles_missing_content():
    response = {"choices": [{"message": {"role": "assistant"}}]}
    assert extract_text(response) == ""


def test_extract_text_handles_malformed_response():
    assert extract_text({}) == ""
    assert extract_text({"choices": []}) == ""
