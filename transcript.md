# Chat transcript

One real 5-turn session through the CLI (`python -m tokrag chat`: hybrid
BM25 + bge-small retrieval, Phase 4 diversification, `openai/gpt-oss-120b`
for answers, `openai/gpt-oss-20b` for query rewriting). Run once, on the
first attempt, after the citation-parser fix in DECISIONS.md. Questions
were piped to stdin, so the CLI doesn't echo them; each `You:` line below
shows the piped question, followed by the CLI's output **verbatim**,
including the `[rewritten query: ...]` lines it prints for every turn after
the first. An earlier session, kept for its retrieval-miss follow-up, is in
[transcript_earlier.md](transcript_earlier.md).

---

## Turn 1 — specific lookup

**You:** Which target vocabulary size did the AraToken paper pick for its base experiments?

```
Bot: The AraToken paper trained its tokenizers with a target vocabulary size of **80,000 tokens for the base experiments**【1】.

Sources:
  [1] AraToken: Optimizing Arabic Tokenization with Normalization Pipeline and Language Extension for Qwen3 (2025), section: Vocabulary Size
```

---

## Turn 2 — sets up the follow-up

**You:** What is BPE-Dropout, and how does it change BPE segmentation during training?

```
  [rewritten query: What is BPE‑Dropout, and how does it modify the Byte‑Pair Encoding (BPE) segmentation process during model training?]

Bot: BPE‑Dropout is a subword‑regularization technique that makes the normally deterministic Byte‑Pair Encoding (BPE) tokenizer stochastic.  Instead of always applying every merge rule in the learned merge table, BPE‑Dropout randomly discards each merge operation with a probability p at every merge step; the remaining merges are applied as usual.  Because some merges are dropped, the same word can be segmented in different ways across training iterations (e.g., “newest” may become “ne w e st” or “ne west”), giving the model exposure to multiple possible tokenizations and improving robustness to noise and segmentation errors【1†L1-L7】【2†L1-L5】.

Sources:
  [1] Bridging the Gap between Subword and Character Segmentation in Pretrained Language Models (2023), section: Subword Regularization For Bpe
  [2] BPE-Dropout: Simple and Effective Subword Regularization (2020), section: Introduction
```

*Note*: the model cited in the `【1†L1-L7】` format here. Before the fix
logged in DECISIONS.md, the parser missed this format and no Sources list
would have printed.

---

## Turn 3 — follow-up ("how does that compare with WordPiece?")

**You:** How does that compare with WordPiece?

```
  [rewritten query: How does the subword‑regularization technique BPE‑Dropout (which introduces stochasticity into the deterministic Byte‑Pair Encoding tokenizer) compare to the WordPiece tokenization method used in models such as BERT?]

Bot: BPE‑Dropout turns the normally deterministic Byte‑Pair Encoding (BPE) tokenizer into a stochastic one by randomly dropping merge operations during encoding, so the same word can be tokenized in several different ways (e.g., “example” → [8] or [6,7] or [0,1,2,3,4,5,0]) [1].  
WordPiece, the tokenizer used in BERT, is also deterministic, but stochasticity can be introduced with a regularization scheme called MaxMatch‑Dropout, which randomly drops vocabulary entries during the maximum‑matching tokenization step to produce alternative segmentations [2][3].  
Thus, both approaches aim to break the fixed tokenization of subword models, but BPE‑Dropout does so by perturbing the BPE merge process itself, whereas WordPiece requires an additional dropout‑based modification (MaxMatch‑Dropout) to achieve a similar stochastic effect.

Sources:
  [1] StochasTok: Improving Fine-Grained Subword Understanding in LLMs (2025), section: Stochastic Tokenization.
  [2] MaxMatch-Dropout: Subword Regularization for WordPiece (2022), section: Conclusion
  [3] MaxMatch-Dropout: Subword Regularization for WordPiece (2022), section: Conclusion
```

*Note*: the rewrite resolved "that" to BPE-Dropout, and retrieval found
the WordPiece counterpart paper (MaxMatch-Dropout). `[8]`, `[6,7]` and
`[0,...]` in the answer are token IDs quoted from the source, not
citations; the parser drops `[8]` as out of range. [2] and [3] are two
different chunks of the same paper (the cap allows 2 per paper).

---

## Turn 4 — multi-paper question

**You:** What do the Tokenizer Tax paper on 25 European languages and Petrov et al. report about tokenization cost differences between languages?

```
  [rewritten query: What do the studies “Tokenizer Tax: A Large-Scale Study of Tokenization Costs Across 25 European Languages” and “Petrov et al.’s analysis of tokenization cost differences between languages” report regarding how tokenization costs vary among different European languages?]

Bot: The “Tokenizer Tax” study measured token‑level “fertility” on parallel text for 25 European languages and found that tokenization can make API usage 2–15 times more expensive for some languages than for English, with certain morphologically rich languages (e.g., Ukrainian) incurring more than twice the cost per word [2][3].  Petrov et al. formalized this “language tax,” showing that non‑Latin‑script or highly inflected languages may require 2–15 × more tokens per semantic unit than English, leading to an order‑of‑magnitude disparity in commercial API costs across languages [3][4].

Sources:
  [2] The Tokenizer Tax Across 25 European Languages: Domain Invariance, Cross-Lingual Few-Shot Effects, and the Ukrainian Penalty (2026), section: Introduction
  [3] The Tokenizer Tax Across 25 European Languages: Domain Invariance, Cross-Lingual Few-Shot Effects, and the Ukrainian Penalty (2026), section: Tokenizer Fertility and Multilingual Fairness
  [4] Tokenizer Fertility and Zero-Shot Performance of Foundation Models on Ukrainian Legal Text: A Comparative Study (2026), section: Tokenizer Fertility and Multilingual Fairness
```

*Note*: Petrov et al.'s own paper is not among the sources. Its findings
are reported through two later papers that cite it ([3], [4]). This is a
secondary attribution, the kind of restated-finding problem the brief
raises about surveys.

---

## Turn 5 — unanswerable question

**You:** How many tokens are in the vocabulary of Google's Gemini 2.5 tokenizer?

```
  [rewritten query: How many tokens are included in the vocabulary of the tokenizer used by Google’s Gemini 2.5 language model?]

Bot: The corpus doesn't contain enough information to answer this.
```

*Note*: correct abstention. Gemini's tokenizer is not described in the
corpus.
