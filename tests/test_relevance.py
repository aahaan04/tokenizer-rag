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


def test_soft_domain_veto_is_rescued_by_llm_context():
    # Speech/music are "soft" veto terms — an explicit LLM mention still rescues them.
    included, _ = decide(
        "A Shared Tokenizer for Speech and Text in a Unified Language Model",
        "We study how a single large language model tokenizer trained with "
        "byte-pair encoding handles both speech recognition and text generation.",
    )
    assert included


def test_molecular_domain_never_rescued_by_llm_context():
    # Molecular/protein/genomic domains are a "hard" veto (2026-10-06, Checkpoint
    # 1b): unlike speech/music, an LLM mention doesn't rescue them — this is the
    # "Training Text-to-Molecule Models with Context-Aware Tokenization" case the
    # user flagged, which the old single-tier veto let through.
    included, reason = decide(
        "Byte-Pair Encoding for Multimodal Large Language Models on Molecular Data",
        "We study how a large language model tokenizer trained with byte-pair "
        "encoding represents molecular SMILES strings for downstream reasoning.",
    )
    assert not included
    assert reason.startswith("veto:domain-hard:")


def test_classic_word_segmentation_is_excluded():
    included, reason = decide(
        "Joint Dependency Parsing and Multiword Expression Tokenization",
        "We present a joint model for dependency parsing and word segmentation "
        "that identifies multiword expressions during tokenization for downstream "
        "part-of-speech tagging.",
    )
    assert not included
    assert reason.startswith("veto:classic-pipeline:")


def test_subword_segmentation_is_not_caught_by_classic_pipeline_veto():
    # "subword segmentation" contains "word segmentation" as a literal substring —
    # must not trip the classic-pipeline veto meant for the pre-neural sense.
    included, _ = decide(
        "Subword Segmentation Strategies for Multilingual Language Models",
        "We compare subword segmentation strategies for training a multilingual "
        "language model tokenizer.",
    )
    assert included


def test_classic_pipeline_rescued_by_explicit_subword_mention():
    included, _ = decide(
        "Dependency Parsing with Subword Tokenization for Low-Resource Languages",
        "We study how subword tokenization choices affect downstream dependency "
        "parsing accuracy for low-resource languages.",
    )
    assert included


def test_core_topic_paper_scores_high():
    s = score(
        "SentencePiece: A Simple and Language Independent Subword Tokenizer",
        "We describe SentencePiece, a language-independent subword tokenizer for "
        "neural text processing, covering vocabulary size and tokenization quality.",
    )
    assert s >= 10.0
