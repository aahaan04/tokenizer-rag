"""Section-aware chunking: splits each (section_title, text) pair into
word-count-bounded chunks with a small overlap, preserving section metadata
per chunk. Word count is used as a cheap proxy for token count for the
primary split — not exact, but consistent across the corpus.

An optional hard token cap (checked with a real tokenizer) catches the cases
word-count alone misses: technical English running higher tokens/word than
casual text, and CJK/LaTeX-leak text where whitespace doesn't bound token
count at all. See the 2026-10-06 DECISIONS.md entries."""

from __future__ import annotations

TARGET_WORDS = 350
OVERLAP_WORDS = 50
MAX_TOKENS = 450  # headroom under bge-small/SPECTER's 512-token limit


def _split_to_token_cap(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Recursively halve `text` by word count until every piece tokenizes to
    <= max_tokens. Never splits mid-word. Recursion (rather than a single
    word-count estimate) handles pathological cases like CJK text, where a
    small number of words can still tokenize to far more than max_tokens."""
    words = text.split()
    if len(words) <= 1:
        return [text]
    if len(tokenizer.encode(text, add_special_tokens=True)) <= max_tokens:
        return [text]
    mid = len(words) // 2
    left = " ".join(words[:mid])
    right = " ".join(words[mid:])
    return _split_to_token_cap(left, tokenizer, max_tokens) + _split_to_token_cap(right, tokenizer, max_tokens)


def chunk_sections(
    sections: list[tuple[str, str]],
    target_words: int = TARGET_WORDS,
    overlap_words: int = OVERLAP_WORDS,
    tokenizer=None,
    max_tokens: int = MAX_TOKENS,
) -> list[dict]:
    step = max(target_words - overlap_words, 1)

    chunks: list[dict] = []
    for heading, text in sections:
        words = text.split()
        if not words:
            continue

        section_texts: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + target_words, len(words))
            section_texts.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += step

        if tokenizer is not None:
            expanded = []
            for t in section_texts:
                expanded.extend(_split_to_token_cap(t, tokenizer, max_tokens))
            section_texts = expanded

        for idx, t in enumerate(section_texts):
            chunks.append({"section_title": heading, "chunk_index": idx, "text": t})
    return chunks
