# Failure analysis

Three real, distinct failure types found while building and evaluating the
chatbot (Phase 6 eval, [phase6_chat_eval_results.json](phase6_chat_eval_results.json),
and the Phase 7 transcript, [transcript.md](transcript.md)). Each case below
is reproduced exactly as the bot produced it — nothing edited for effect.

---

## Case 1 — Generation non-determinism (gold chunk WAS retrieved)

**Question (q03, specific_lookup):** "What is the total size of mBERT's
shared multilingual subword vocabulary, as reported by Rust et al.?"

**Actual output:** "The corpus doesn't contain enough information to
answer this." (abstained, `model_self_assessment`)

**Expected answer:** 119,547 tokens — the gold supporting span ("The final
shared mBERT vocabulary comprises a total of 119,547 subword tokens") is
from *How Good is Your Tokenizer? On the Monolingual Performance of
Multilingual Language Models* (Rust et al.), section "Background and
Related Work."

**Diagnosis:** confirmed the gold chunk WAS in the retrieved top-5 — the
literal text "119,547 subword tokens" was present in the sources block sent
to the model, well within the first 800 characters (no truncation issue
either). Ruled out retrieval and context-window causes directly. Called the
exact same prompt + sources 3 times, fresh (uncached, `temperature=0`):
1 correct answer, 2 identical false abstentions. This is genuine
**non-determinism in GPT-OSS-120B on Groq's infrastructure at
temperature=0** (likely MoE expert-routing or batched-inference variance),
not a logic bug in retrieval, prompting, or abstention detection.

**Mitigation attempted**: retry once (fresh) on self-abstention. Measured
effect: did NOT reduce the final false-positive count in the full eval run
(q03 still abstained after retry in that run) — a single retry raises the
*odds* of a correct draw but doesn't guarantee one. Documented honestly in
DECISIONS.md rather than claimed as fixed. **Residual risk if deployed**:
this specific question would non-deterministically abstain roughly 1 in 3
calls even with the retry in place; a production system would need either
a more deterministic model/provider or a higher retry budget, both outside
this project's scope.

---

## Case 2 — Retrieval miss on a follow-up (correct rewrite, wrong retrieval)

**Question (q25, follow_up, turn 2 of a conversation):** "How does that
compare to the vocabulary size where Beinborn & Pinter say WordPiece
tokenization quality plateaus?" (turn 1 established: BPE Gets Picky used
8192 for its EN-DE experiments)

**Actual output** (reproduced live in [transcript.md](transcript.md), Turn
2): "The corpus doesn't contain enough information to answer this."
(abstained, `model_self_assessment`)

**Expected answer:** per Beinborn & Pinter ("Analyzing Cognitive
Plausibility of Subword Tokenization"), WordPiece quality plateaus at
50,000 tokens for most languages — roughly 6x the 8192 BPE Gets Picky used.

**Diagnosis:** the query rewriter did its job correctly — the logged
rewrite ("How do the vocabulary sizes (16384, 32768, and 65536 tokens)...
compare to the vocabulary size at which WordPiece tokenization quality
plateaus, as reported by Beinborn and Pinter?") correctly names both papers
and resolves "that." Re-ran retrieval with this exact rewritten query and
checked whether Beinborn & Pinter's specific chunk (group `grp0111`,
section "Vocabulary Size and Morphology") was in the diversified top-5: it
was **not**. This is a genuine **retrieval miss**, not an abstention-logic
or rewriting error — abstaining was the correct, grounded response given
what retrieval actually returned. Of the 10 original Checkpoint-6 false
positives, 8 were this same class (see DECISIONS.md's per-question
diagnosis table); this project treats retrieval misses as a Phase 5
retrieval-coverage limitation, not a Phase 6 generation bug, and didn't
spend the capped prompt-tuning rounds chasing them.

**Likely cause** (not fully root-caused, flagged for future work): the
rewritten query packs in three numbers (16384/32768/65536) plus two paper
names, which may dilute the embedding/BM25 signal relative to a shorter,
more targeted query — consistent with the Phase 5 finding that follow-up
questions are hybrid retrieval's weakest category overall (Recall@5 0.71
hybrid vs 0.29/0.00 bge-small/SPECTER, still the best of a weak field).

---

## Case 3 — Correctly-cited facts with a fabricated author attribution

**Question (q08/transcript Turn 3, multi_paper):** "What is the core
trade-off of byte-level/character-level ('tokenizer-free') models like
ByT5 and CANINE compared to subword tokenization, and how does each paper
address the cost side of it?"

**Actual output** (verbatim, [transcript.md](transcript.md) Turn 3):
> *ByT5* (Jiang et al., 2022) argues that the standard Transformer can be
> applied to byte sequences "without a dramatically unfavorable increase in
> computational cost," but empirical results still show slower inference
> — about 1.5–2.6× slower on average and up to 9× on very long inputs【2】.
>
> *CANINE* (Zhang et al., 2023) acknowledges the same length-induced cost
> ... 【5】.

**Expected/correct attribution:** ByT5 is Xue et al. (2022); CANINE is
Clark et al. (2022) — and critically, **both correct names are literally
present in the retrieved source text itself.** Re-ran retrieval for this
exact query and inspected the actual chunks fed to the model: source [2]
reads "...ByT5 (Xue et al., 2022)... The training of ByT5 is based on the
subword-based multilingual mT5 (Xue et al., 2021) approach..." and source
[5] reads "...CANINE (Clark et al., 2022) operates directly on Unicode code
points, while ByT5 (Xue et al., 2022) processes raw UTF-8 bytes...".

**Diagnosis:** this is neither a retrieval miss nor plain non-determinism
— it's a **grounded-fact / fabricated-attribution split**: the model
correctly extracted and cited the *substance* of each claim (the slowdown
figures, the trade-off framing) with properly-formatted, in-range
citations pointing at the right chunks, but invented author names that
directly contradict text sitting right there in the same sources block.
The citation mechanism (bracket parsing, range validation) worked exactly
as designed — it has no way to verify the prose *around* a citation marker
against the source, only that the marker itself is valid. This is a real
limitation of citation-marker-based grounding: a citation being
structurally valid and pointing at real, relevant content does not
guarantee every claim attached to it is accurate. A stricter groundedness
check (e.g. asking the judge model to flag attribution details, not just
overall factual support) would likely catch this specific error class;
the current LLM-judge prompt asks only "are the claims supported," which
this answer would likely still pass since the *trade-off facts* ARE
supported — just not the names attached to them.

---

## What these three cases suggest about where to invest next

Case 1 (non-determinism) and Case 3 (fabricated attribution despite valid
citations) point the same direction: **the model's raw output needs a
cheap second pass that checks something narrower than full groundedness**
— e.g. verifying every named entity/author/year mentioned near a citation
actually appears in that citation's source text, which is a simpler,
more mechanical check than full claim-level groundedness and could run on
every answer without much added latency. Case 2 (retrieval miss) points
back to Phase 5: a reranker or a better-tuned hybrid weighting for
follow-up queries specifically, since that's consistently the weakest
question type across every retrieval configuration tested (see WRITEUP.md's
embedding comparison table).
