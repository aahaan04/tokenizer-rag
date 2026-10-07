"""Section-aware chunking: splits each (section_title, text) pair into
word-count-bounded chunks with a small overlap, preserving section metadata
per chunk. Word count is used as a cheap proxy for token count (no tokenizer
dependency here) — not exact, but consistent across the corpus."""

from __future__ import annotations

TARGET_WORDS = 350
OVERLAP_WORDS = 50


def chunk_sections(sections: list[tuple[str, str]], target_words: int = TARGET_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[dict]:
    chunks: list[dict] = []
    step = max(target_words - overlap_words, 1)

    for heading, text in sections:
        words = text.split()
        if not words:
            continue
        idx = 0
        start = 0
        while start < len(words):
            end = min(start + target_words, len(words))
            chunks.append(
                {
                    "section_title": heading,
                    "chunk_index": idx,
                    "text": " ".join(words[start:end]),
                }
            )
            idx += 1
            if end == len(words):
                break
            start += step
    return chunks
