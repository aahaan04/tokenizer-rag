# Sample chat transcript

A real 5-turn session run through the actual bot (`Chatbot` in
[tokrag/chat/chatbot.py](tokrag/chat/chatbot.py), hybrid BM25+bge-small
retrieval with Phase 4 diversification), generated with
`reasoning_effort="low"` and `temperature=0`. Outputs below are **unedited**
— including Turn 2's abstention and Turn 3's citation error, both discussed
honestly rather than cleaned up. See [FAILURES.md](FAILURES.md) Case 3 for
the Turn 3 issue and [DECISIONS.md](DECISIONS.md) for the Turn 2 abstention
class (retrieval miss).

---

## Turn 1 — specific lookup

**Q:** What vocabulary size did Chizhov et al. use in the EN-DE machine
translation experiments in "BPE Gets Picky"?

**A:** In the EN‑DE experiments reported in "BPE Gets Picky," Chizhov et al.
trained models with three total vocabulary sizes: 16 384, 32 768 and
65 536 tokens (i.e., 16 k, 32 k and 64 k) [1].

**Citations:**
- [1] BPE Gets Picky: Efficient Vocabulary Refinement During Tokenizer Training (2024), section: Larger vocabularies.

**Abstained:** false | **top retrieval score:** 0.0323

---

## Turn 2 — follow-up (shows the logged query rewrite)

**Q:** How does that compare with WordPiece tokenization quality
plateauing, per Beinborn and Pinter?

**Logged rewrite** (fed to retrieval instead of the raw follow-up — this is
exactly what gets written to the chat log and is what the query-rewriter
diagnosis in DECISIONS.md is about):

> How do the vocabulary sizes (16 384, 32 768, and 65 536 tokens) used by
> Chizhov et al. in their EN‑DE machine‑translation experiments compare to
> the vocabulary size at which WordPiece tokenization quality plateaus, as
> reported by Beinborn and Pinter?

**A:** The corpus doesn't contain enough information to answer this.

**Abstained:** true (`model_self_assessment`) | **top retrieval score:** 0.0268

*Honest note*: the rewrite correctly resolved "that" and named both papers
— the abstention here is a genuine **retrieval miss** (Beinborn & Pinter's
specific plateau figure wasn't in the top-5 for this rewritten query), not
a rewriter or generation failure. This is the same class of false
abstention documented in DECISIONS.md's Checkpoint 6 diagnosis (8 of the
original 10 false positives were retrieval misses, not abstention-logic
errors) — included here deliberately rather than swapped for an easier
question, since the transcript is unedited.

---

## Turn 3 — multi-paper question

**Q:** What is the core trade-off of byte-level/character-level
("tokenizer-free") models like ByT5 and CANINE compared to subword
tokenization, and how does each paper address the cost side of it?

**A:** The key trade‑off is that "tokenizer‑free" byte‑ or character‑level
models eliminate vocabulary‑related bias and fertility disparity, but they
represent text with many more symbols, so input sequences become 3–5×
longer and the quadratic attention cost of the Transformer rises
sharply【5】.

*ByT5* (Jiang et al., 2022) argues that the standard Transformer can be
applied to byte sequences "without a dramatically unfavorable increase in
computational cost," but empirical results still show slower inference —
about 1.5–2.6× slower on average and up to 9× on very long inputs【2】.

*CANINE* (Zhang et al., 2023) acknowledges the same length‑induced cost and
frames the trade‑off as "computational: sequence lengths grow 3–5×,
increasing attention cost quadratically," accepting the higher compute in
exchange for equitable multilingual coverage【5】.

**Citations:**
- [2] Byte-Level Grammatical Error Correction Using Synthetic and Curated Corpora (2023), section: Sequence Segmentation For Gec
- [5] The Tokenizer Tax Across 25 European Languages: Domain Invariance, Cross-Lingual Few-Shot Effects, and the Ukrainian Penalty (2026), section: Tokenizer-free architectures.

**Abstained:** false | **top retrieval score:** 0.0301

*Honest note*: **"Jiang et al., 2022" and "Zhang et al., 2023" are wrong —
the correctly-grounded facts came with fabricated author names.** The
actual retrieved source text (both [2] and [5]) literally says "ByT5 (Xue
et al., 2022)" and "CANINE (Clark et al., 2022)." See
[FAILURES.md](FAILURES.md) Case 3 for the full diagnosis — this is kept
unedited because it's a genuine, instructive failure, not a cherry-picked
good run.

---

## Turn 4 — unanswerable question (abstention)

**Q:** What dropout rate is recommended when fine-tuning a vision
transformer on 3D medical images using BPE-style patch tokenization?

**A:** The corpus doesn't contain enough information to answer this.

**Abstained:** true (`model_self_assessment`) | **top retrieval score:** 0.0252

*Honest note*: this question is out-of-domain (vision transformers / 3D
medical imaging aren't in this corpus's scope), so abstention here is the
correct, intended behavior — a true negative, not a false positive.

---

## Turn 5 — specific lookup (partial grounded answer)

**Q:** What dataset does mT5 use, and how many languages does its
SentencePiece vocabulary cover?

**A:** The mT5 model was pretrained on the multilingual mC4 corpus [2]. The
sources do not specify how many languages are represented in the
SentencePiece vocabulary used for mT5.

**Citations:**
- [2] ByT5: Towards a token-free future with pre-trained byte-to-byte models (2021), section: ByT5 Design

**Abstained:** false | **top retrieval score:** 0.0301

*Honest note*: this is exactly the Checkpoint-6 partial-answer behavior
(see DECISIONS.md's "q10 partial-abstention" fix) working as intended —
the model answered the part the sources supported (the mC4 dataset) and
explicitly flagged the part they didn't (language count), instead of
abstaining on the whole question.
