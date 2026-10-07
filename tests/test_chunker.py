from tokrag.parse.chunker import chunk_sections


class _FakeTokenizer:
    """Simulates a tokenizer where every word is `tokens_per_word` tokens —
    deterministic and fast, no real HF model needed for these tests."""

    def __init__(self, tokens_per_word: int = 2):
        self.tokens_per_word = tokens_per_word

    def encode(self, text, add_special_tokens=True):
        n_words = len(text.split())
        return list(range(n_words * self.tokens_per_word))


def test_short_section_becomes_one_chunk():
    chunks = chunk_sections([("Introduction", "one two three four five")], target_words=350, overlap_words=50)
    assert len(chunks) == 1
    assert chunks[0]["section_title"] == "Introduction"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["text"] == "one two three four five"


def test_long_section_splits_with_overlap():
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_sections([("Method", text)], target_words=40, overlap_words=10)

    # starts at 0, 30, 60 -> covers 0-40, 30-70, 60-100; the 3rd already
    # reaches the end (word 100) so the loop stops there, no 4th chunk.
    assert len(chunks) == 3
    assert all(c["section_title"] == "Method" for c in chunks)
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2]
    # consecutive chunks overlap by the configured amount
    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    assert first_words[-10:] == second_words[:10]


def test_empty_section_text_produces_no_chunks():
    chunks = chunk_sections([("Empty", "   ")])
    assert chunks == []


def test_multiple_sections_restart_chunk_index_per_section():
    chunks = chunk_sections([("A", "a b c"), ("B", "d e f")])
    assert [(c["section_title"], c["chunk_index"]) for c in chunks] == [("A", 0), ("B", 0)]


def test_token_cap_splits_oversized_chunk():
    # 300 words at 2 tokens/word = 600 tokens, over a 450-token cap -> must split.
    text = " ".join(f"w{i}" for i in range(300))
    tok = _FakeTokenizer(tokens_per_word=2)
    chunks = chunk_sections([("Method", text)], target_words=350, overlap_words=50, tokenizer=tok, max_tokens=450)
    assert len(chunks) > 1
    for c in chunks:
        n_tokens = len(tok.encode(c["text"]))
        assert n_tokens <= 450
    # splitting never drops or duplicates words
    rejoined = " ".join(c["text"] for c in chunks).split()
    assert rejoined == text.split()


def test_token_cap_leaves_small_chunk_untouched():
    text = "one two three four five"
    tok = _FakeTokenizer(tokens_per_word=2)
    chunks = chunk_sections([("Intro", text)], tokenizer=tok, max_tokens=450)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_token_cap_chunk_index_is_contiguous_after_split():
    text = " ".join(f"w{i}" for i in range(300))
    tok = _FakeTokenizer(tokens_per_word=2)
    chunks = chunk_sections([("Method", text)], target_words=350, overlap_words=50, tokenizer=tok, max_tokens=450)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
