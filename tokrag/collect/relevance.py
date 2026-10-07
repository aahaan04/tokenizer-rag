"""Keyword-based relevance scoring: decides whether a candidate falls inside the
"tokenizer research for LLMs" topic boundary.

Deterministic and keyword-only — no LLM relevance call. See DECISIONS.md for why
(cost/time tradeoff; revisit if the keyword filter proves too noisy at Checkpoint 1b).

Topic boundary: papers centrally *about* tokenization (BPE, WordPiece, Unigram,
SentencePiece, byte-level/tokenizer-free models, vocabulary size, multilingual
fertility, tokenization's effect on arithmetic/code/reasoning). A paper that only
mentions "we use BPE with vocab 32k" in passing should score below threshold.
"""

from __future__ import annotations

import re

# STRONG_TERMS name a specific tokenization method or metric; a paper rarely uses
# one unless tokenization is a central subject, not a background detail.
STRONG_TERMS = [
    "wordpiece",
    "sentencepiece",
    "unigram language model",
    "unigram tokeniz",
    "byte-level",
    "byte level",
    "tokenizer-free",
    "tokenizer free",
    "token-free",
    "token free",
    "vocabulary size",
    "vocab size",
    "token merge",
    "merge rule",
    "fertility",
    "compression rate",
    "character-level language model",
    "token boundary",
    "vocabulary construction",
    "subword regularization",
    "tokenizer training",
    "detokeniz",
]

# WEAK_TERMS are generic enough that almost any LLM paper uses them once in
# passing ("we use BPE tokenization with vocab 32k") without studying tokenization.
WEAK_TERMS = [
    "byte-pair encoding",
    "bpe",
    "tokeniz",  # catches tokenize/tokenizer/tokenization/tokenizing
    "subword",
]

# Co-occurring with a core term, these raise confidence this is LLM-era tokenization
# research rather than, say, speech segmentation or pre-neural NLP tokenization.
CONTEXT_TERMS = [
    "language model",
    "llm",
    "large language model",
    "transformer",
    "arithmetic",
    "reasoning",
    "code generation",
    "numeracy",
    "multilingual",
    "cross-lingual",
]

# Used only to rescue an off-topic-domain match (below). Deliberately narrower
# than CONTEXT_TERMS — "multilingual" shows up in speech/MT papers just as often
# as LLM papers, and "transformer" is an architecture used in music/vision/speech
# models too, not a signal the tokenizer *feeds a language model*. Only an
# explicit "language model"/"LLM" mention does. See the 2026-10-06 topic-boundary
# rule in DECISIONS.md: in scope if the tokenizer feeds a (possibly multimodal)
# language model; out if it's a domain-specific sequence model with no LM.
STRONG_CONTEXT_TERMS = ["language model", "llm", "large language model"]

# BPE/subword vocabularies are reused well outside text LLMs (speech, chemistry,
# symbolic music). A paper in one of these domains is only in-scope if it also
# shows a strong LLM/text-model context term above — otherwise it's a different
# field that happens to reuse the same tokenization algorithm, not tokenizer
# research for LLMs. See DECISIONS.md for examples this caught in the Checkpoint
# 1a sample. Split into two tiers (2026-10-06, after the Checkpoint 1b review):
# SOFT terms can still be rescued by a strong LLM-context mention (e.g. a
# speech paper that's really about a shared text/speech LLM tokenizer); HARD
# terms never are — "Training Text-to-Molecule Models with Context-Aware
# Tokenization" survived the old single-tier veto by saying "language model"
# once, but it's bioinformatics/cheminformatics research borrowing LLM
# terminology, not tokenizer research for LLMs, and the user called this out
# explicitly at Checkpoint 1b.
OFF_TOPIC_DOMAIN_SOFT_TERMS = [
    "speech recognition",
    "automatic speech recognition",
    "symbolic music",
    "midi",
    "music generation",
]
OFF_TOPIC_DOMAIN_HARD_TERMS = [
    "molecular",
    "drug-like",
    "smiles string",
    "protein sequence",
    "genom",  # stem: catches genome/genomic/genomics — "genomic" isn't a
    # substring of "genome", so the literal term missed "genomic language
    # models" papers; found via the Phase 4 fuzzy-merge audit (2026-10-07)
    "dna sequence",
]

# Classic/pre-neural NLP uses "tokenization" to mean splitting text into words
# for a pipeline stage (parsing, tagging) — a different sense than subword
# vocabularies for neural LMs. ACL Anthology's bulk export goes back to the
# 1990s, so this sense shows up a lot. Added 2026-10-06 at Checkpoint 1b, after
# the user noted "Joint Dependency Parsing and Multiword Expression
# Tokenization" had been let through. Only excluded when NOT also about
# subword/neural tokenization, per the rescue terms below.
CLASSIC_PIPELINE_TERMS = [
    "dependency parsing",
    "treebank",
    "part-of-speech tagging",
    "pos tagging",
    "multiword expression",
    "chinese word segmentation",
    "sentence segmentation",
]
# "word segmentation" alone is classic-pipeline; "subword segmentation" is not
# (it's the neural/subword sense) — exclude matches immediately preceded by "sub".
_WORD_SEGMENTATION_RE = re.compile(r"(?<!sub)word segmentation")

# Rescues a classic-pipeline match: an explicit subword/neural-method or
# LLM-context signal means tokenization-for-neural-LMs is still the real
# subject, even if the paper also touches parsing/tagging.
CLASSIC_PIPELINE_RESCUE_TERMS = STRONG_TERMS + ["subword", "bpe", "byte-pair encoding"] + STRONG_CONTEXT_TERMS

TITLE_WEIGHT = 3.0  # applied per hit, strong or weak alike — a title mention is already a strong signal
ABSTRACT_STRONG_WEIGHT = 1.5
ABSTRACT_WEAK_WEIGHT = 0.5
ABSTRACT_CONTEXT_WEIGHT = 0.5

INCLUDE_THRESHOLD = 3.0


def _count_hits(text_lower: str, terms: list) -> int:
    return sum(1 for t in terms if t in text_lower)


def score(title: str, abstract: str) -> float:
    title_l, abstract_l = title.lower(), (abstract or "").lower()
    s = 0.0
    s += TITLE_WEIGHT * (_count_hits(title_l, STRONG_TERMS) + _count_hits(title_l, WEAK_TERMS))
    s += ABSTRACT_STRONG_WEIGHT * _count_hits(abstract_l, STRONG_TERMS)
    s += ABSTRACT_WEAK_WEIGHT * _count_hits(abstract_l, WEAK_TERMS)
    s += ABSTRACT_CONTEXT_WEIGHT * _count_hits(abstract_l, CONTEXT_TERMS)
    return s


def decide(title: str, abstract: str) -> tuple:
    """Return (included: bool, reason: str).

    A rejection reason starting with "veto:" means the keyword score alone
    would have included it — these are worth a manual look, since a veto is a
    blunt heuristic that can drop a genuinely relevant paper. The reason is
    tagged by category ("veto:domain-hard:", "veto:domain-soft:",
    "veto:classic-pipeline:") so they can be audited and reported separately.
    """
    s = score(title, abstract)
    text_l = f"{title} {abstract or ''}".lower()
    has_strong_context = _count_hits(text_l, STRONG_CONTEXT_TERMS) > 0

    matched_hard = [t for t in OFF_TOPIC_DOMAIN_HARD_TERMS if t in text_l]
    if matched_hard:
        return False, f"veto:domain-hard: {', '.join(matched_hard)} (relevance_score={s:.1f})"

    matched_soft = [t for t in OFF_TOPIC_DOMAIN_SOFT_TERMS if t in text_l]
    if matched_soft and not has_strong_context:
        return False, f"veto:domain-soft: {', '.join(matched_soft)} (relevance_score={s:.1f})"

    matched_classic = [t for t in CLASSIC_PIPELINE_TERMS if t in text_l]
    if _WORD_SEGMENTATION_RE.search(text_l):
        matched_classic.append("word segmentation")
    has_rescue = _count_hits(text_l, CLASSIC_PIPELINE_RESCUE_TERMS) > 0
    if matched_classic and not has_rescue:
        return False, f"veto:classic-pipeline: {', '.join(matched_classic)} (relevance_score={s:.1f})"

    if s >= INCLUDE_THRESHOLD:
        return True, f"relevance_score={s:.1f} >= {INCLUDE_THRESHOLD}"
    return False, f"relevance_score={s:.1f} < {INCLUDE_THRESHOLD} (below topic-boundary threshold)"
