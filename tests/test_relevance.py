from tokrag.collect.relevance import decide, score


def test_title_hit_outweighs_no_abstract_match():
    included, _ = decide("Byte-Level BPE for Multilingual Language Models", "")
    assert included


def test_passing_mention_is_excluded():
    # Tokenization mentioned once, in passing, in an otherwise unrelated abstract.
    included, reason = decide(
        "Scaling Instruction-Tuned Language Models",
        "We train a 70B model on a large corpus. We use BPE tokenization with a "
        "vocabulary of 128k and evaluate on standard benchmarks.",
    )
    assert not included
    assert "relevance_score" in reason


def test_unrelated_paper_scores_near_zero():
    s = score("Convolutional Networks for Image Classification", "We study ResNet variants on ImageNet.")
    assert s == 0.0


def test_off_topic_domain_without_llm_context_is_excluded():
    # High keyword score, but it's a speech-recognition tokenizer, not an LLM one.
    included, reason = decide(
        "BBPE16: UTF-16-based byte-level byte-pair encoding for improved multilingual speech recognition",
        "Multilingual automatic speech recognition (ASR) requires tokenization that "
        "efficiently covers many writing systems. Byte-level BPE (BBPE) using UTF-8 "
        "is widely adopted for its language-agnostic design.",
    )
    assert not included
    assert reason.startswith("veto:")
    assert "speech recognition" in reason


def test_symbolic_music_tokenizer_is_excluded_despite_transformer_mention():
    # "Transformer" alone isn't enough to rescue a non-LLM domain — Transformers
    # are used in music/vision/speech too. Needs an explicit LM/LLM mention.
    included, reason = decide(
        "From Words to Music: A Study of Subword Tokenization Techniques in Symbolic Music Generation",
        "Subword tokenization has been widely successful in text-based NLP tasks with "
        "Transformer-based models. We explore subword tokenization techniques, such as "
        "byte-pair encoding (BPE), in symbolic music generation using MIDI datasets.",
    )
    assert not included
    assert reason.startswith("veto:")


def test_visual_tokenizer_for_multimodal_llm_is_kept():
    included, _ = decide(
        "From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities",
        "Multimodal Large Language Models have made significant strides in integrating "
        "visual and textual information. We introduce a novel image tokenizer that "
        "applies Byte-Pair Encoding to visual data.",
    )
    assert included


def test_off_topic_domain_with_llm_context_is_kept():
    # Same off-topic-ish domain term, but explicit LLM framing should save it.
    included, _ = decide(
        "Byte-Pair Encoding for Multimodal Large Language Models on Molecular Data",
        "We study how a large language model tokenizer trained with byte-pair "
        "encoding represents molecular SMILES strings for downstream reasoning.",
    )
    assert included


def test_core_topic_paper_scores_high():
    s = score(
        "SentencePiece: A Simple and Language Independent Subword Tokenizer",
        "We describe SentencePiece, a language-independent subword tokenizer for "
        "neural text processing, covering vocabulary size and tokenization quality.",
    )
    assert s >= 10.0
