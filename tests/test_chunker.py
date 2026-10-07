from tokrag.parse.chunker import chunk_sections


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
