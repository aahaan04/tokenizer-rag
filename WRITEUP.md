# tokrag — Design Write-up

A RAG chatbot over a self-collected corpus of tokenizer-for-LLMs research.
This summarizes decisions and results; the complete dated reasoning behind
each one (false starts, bugs found/fixed, superseded numbers) is in
[DECISIONS.md](DECISIONS.md), the authoritative assumptions log.

## 1. Topic boundary and corpus composition

**Scope**: in scope if the tokenizer under study feeds a (possibly
multimodal) language model; out if it's a domain-specific sequence model
with no LM framing (music, molecules, speech-only ASR, genomics), or
classic/pre-neural tokenization. Enforced via an off-topic-domain veto: a
hit on a domain term (speech, molecular/drug, symbolic music, genome/
protein) is excluded unless it *also* contains an explicit LLM-context term
("language model," "LLM" — not bare "transformer," which off-topic papers
use just as often).

**Sources**: arXiv (category-filtered keyword search) + Semantic Scholar
(keyword search) + ACL Anthology (bulk bibliography, pre-filtered then
scored) — ACL added as a third source when Semantic Scholar's rate limits
proved too unreliable to supply the preprint-vs-conference-version links
dedup needs.

**Inclusion counts** (see [manifest_summary.md](manifest_summary.md)):

| | Count |
|---|---|
| Total candidates seen | 7,547 |
| Included | 843 (105 arXiv, 281 Semantic Scholar, 457 ACL Anthology) |
| **Unique papers** (de-duplicated) | **715** |
| Rejected — below relevance threshold | 6,338 |
| Rejected — vetoed (off-topic domain / classic pipeline) | 366 (251 classic-pipeline, 75 domain-soft, 40 domain-hard) |

A relevance score (title/abstract keyword hits, weighted by specificity)
gates inclusion at `>=3.0` — raised from 2.0 after that bar let through
papers where tokenization was just a passing setup detail ("we use BPE
with a 128k vocabulary"), which the brief asks to exclude. A hard-coded
canary check (6 canonical papers: Sennrich 2016 BPE, Kudo & Richardson
2018 SentencePiece, ByT5, CANINE, Petrov et al. 2023 fairness, Singh &
Strouse 2024 arithmetic) verifies the corpus contains the field's
foundational work regardless of query ranking, and reports whether each
was found *naturally* — a query-coverage gap signal, independent of final
inclusion.

**Known gap**: the off-topic veto matched "genome" but not "genomic";
fixed to a stem match (4 rows affected). A related gap — ACL
`@proceedings` front-matter entries entering the pool as if they were
papers — was caught once during fuzzy-merge dedup (§3) and guarded there
only, not exhaustively audited elsewhere.

## 2. Parsing and chunking

Each manifest row is fetched/parsed **independently**, not deduplicated by
cross-source group — a duplicate pair only earns its place in the corpus
so §3 can measure dedup's effect on retrieval, which needs both copies
indexed separately. Fetch priority: arXiv HTML (real heading tags) → arXiv
PDF → ACL PDF → Semantic Scholar open-access PDF → abstract-only fallback.
Section headers come from real HTML tags or a PDF regex heuristic
(occasionally misfires on running page headers). References are dropped;
figures/tables/equations are kept as whatever linearized text the
extractor produces.

Chunking targets ~350 words with 50-word overlap, restarting per section.
A **hard token cap** (`<=450` tokens under the target embedding model's
own tokenizer, recursively halving by word count) was added after 13.1% of
chunks exceeded the model's 512-token limit — driven mostly by CJK text
and LaTeX-extraction leakage, where word count doesn't bound token count
at all. After the cap: only 0.006% of 34,779 chunks still exceed it (two
pathological LaTeX-markup leaks with no whitespace to split on).

## 3. Dedup strategy

Three duplicate cases, handled separately:

1. **Multiple arXiv versions** (v1 vs. v3) — 7 version-only groups,
   deliberately seeded with 25 real pairs (arXiv search only ever returns
   the latest version, so this never occurs naturally).
2. **Preprint vs. published, cross-source** — the dominant case: 80
   cross-source-only groups + 18 both version-paired and cross-source.
3. **Near-duplicate chunks** within a paper's own multiple copies (e.g.
   arXiv HTML vs. ACL PDF extractions of the same section) — handled at
   retrieval time via word-trigram shingle Jaccard (>=0.9 collapses two
   chunks as the same passage).

Phase 1 links same-title candidates via `candidate_group_id` (kept as
*separate* rows deliberately, so Phase 4 can measure dedup's effect
instead of it being silently collapsed earlier). Phase 4 assigns the real
merge decision and ran a false-merge audit: **zero likely false merges**
among exact-title groups. A fuzzy tier (title_sim>=0.85 + shared-author/
abstract-cosine conditions) added 5 more real merges, plus one manual
exception — **MorphBPE**, two records with title_sim as low as 0.667 (same
authors, retitled for its 2026 publication) that the general rule
couldn't catch without lowering the bar for everything else.

**A real false merge was caught before reaching the manifest**: two
different years' workshop *proceedings volumes* scored title_sim=0.855
with 3 "shared authors" that were actually overlapping organizing
committees (`@proceedings` entries have no real author field). Fixed with
a guard skipping any pair where either title starts with "Proceedings of."

**The cap-1 ablation**: the first diversification design capped each
paper to 1 surviving chunk in the top-k. This looked right (dup rate hits
0.0) but was removing *distinct* relevant content — two different
sections of the same paper were treated as interchangeable, and only one
survived regardless of which answered the question:

| Config | Recall@5 | MRR | dup_rate@5 |
|---|---|---|---|
| bge-small, raw | 0.293 | 0.240 | 0.324 |
| **bge-small, cap-1 naive (flawed)** | 0.224 | 0.224 | 0.000 |
| bge-small, final (collapse dup chunks, then cap 2/paper) | 0.276 | 0.240 | 0.186 |
| BM25, raw | 0.431 | 0.389 | 0.359 |
| **BM25, cap-1 naive** | 0.379 | 0.371 | 0.000 |
| BM25, final | 0.431 | 0.389 | 0.200 |

Fix: collapse near-duplicate chunks first, then cap at 2/paper (not 1),
overfetching top-50 before diversifying. BM25 fully recovers its raw
Recall@5/MRR while still cutting dup rate 35.9%→20.0%; bge-small recovers
most of what cap-1 cost it. A stricter `redundant_copy_rate@5` (near-dup
*passages* specifically) confirms the fix drives literal repeats to
exactly 0.0 on both backends — the remaining ~19% `same_paper_share@5` is
legitimately two different sections, not duplication.

## 4. Embedding/retrieval comparison and recommendation

29 answerable eval questions, k=5, diversification on, follow-ups scored
on their gold standalone rewrite (§6):

| Config | Recall@5 | MRR | latency p50 (ms) | index size |
|---|---|---|---|---|
| bge-small-en-v1.5 | 0.241 | 0.231 | 18.0 | 51.0 MB |
| SPECTER (original) | 0.103 | 0.138 | 32.6 | 101.9 MB |
| BM25 | 0.397 | 0.372 | 180.9 | 28.3 MB |
| **Hybrid (BM25+bge-small, RRF)** | **0.466** | **0.428** | 297.9 | 79.3 MB |

**Paired bootstrap 95% CIs** (3,000 resamples, seed=42, n=29):

| Comparison | Recall@5 diff | 95% CI | Significant? |
|---|---|---|---|
| Hybrid vs. BM25 | +0.069 | [-0.069, 0.207] | No |
| Hybrid vs. bge-small | +0.224 | [0.017, 0.431] | **Yes** |

**Recommendation: hybrid (BM25 + bge-small via RRF, rrf_k=60).** Its edge
over dense-only is statistically supported; its edge over BM25 alone is
directionally consistent with the expected mechanism (BM25 for exact
terms, dense for paraphrase/continuity) but **not** significant at this
sample size — stated honestly rather than oversold. Hybrid never does
worse than either component alone, at an acceptable latency cost for a
CLI tool (298ms p50).

**SPECTER underperforms**: it's trained via citation-prediction on full
academic text; our queries are short conversational questions, unlike its
training distribution. Starkest on `follow_up` questions, where SPECTER
scores **0.0** Recall@5 — worse than plain BM25. Likely better suited to
a paper-similarity feature than conversational retrieval.

**SPECTER2, honestly reported as a limitation**: time-boxed to 15 minutes
per instruction. `pip install adapters` succeeded (forced a transformers/
huggingface-hub downgrade; verified the rest of the pipeline still works),
but loading either SPECTER2 adapter failed consistently — both attempt
legacy AdapterHub.ml resolution before falling back to HF Hub, which
errors out here regardless of invocation style. A genuine rough edge in
the `adapters` library, not a typo. **Not resolved within the box;
original SPECTER stands as the scientific-model ablation** — its 0.0
follow-up Recall@5 is already strong and well-explained, so this gap
likely wouldn't change the recommendation, but it's disclosed, not hidden.

## 5. Chat pipeline and metrics

Hybrid retrieval → Phase 4 diversification → citations mapped only to
retrieved chunks (paper + year + section) → dual-signal abstention
(score threshold + the model's own self-assessment) → query rewriting for
follow-ups via a smaller/faster model, every LLM call cached to disk and
retried with backoff on 429.

**Final 36-question chat eval** (after one round of fixes — DECISIONS.md
has the full diagnosis):

| Metric | Value |
|---|---|
| Abstention precision (raw) | 0.54 (tp=7, fp=6, fn=0, tn=23) |
| Abstention precision, excluding retrieval misses | **0.88** (1 real generation error of 8 excluded retrieval misses) |
| Abstention recall | **1.0** — zero hallucinated answers on unanswerable questions |
| Groundedness rate (LLM-judge) | 0.87 (20/23) |
| Citation rate | 0.91 (21/23) |
| Follow-up Recall@5, system rewrite vs. gold rewrite | 3/7 vs. 2/7 |

The raw precision gain mixes two things: a scorer fix (a hedge only counts
as abstention if the answer also has zero real citations) accounts for
about a quarter of it (0.41→0.44); the rest (0.44→0.54) is an actual
prompt change (allowing partial answers) plus an improved rewriter.

**The follow-up "win" (3/7 vs 2/7) is not evidence the system beats gold
rewriting** — n=7 is far too small, and every `gold_standalone_rewrite`
names the prior paper's exact title/figure, making it an oracle rewrite,
not a neutral baseline. The gold condition is a biased, optimistic
ceiling; the small edge is sample noise.

Full diagnosis of all 10 original false positives (8 retrieval misses, 1
non-determinism case, 1 detection-formula bug) is in DECISIONS.md; three
distinct real failure cases with full detail are in
[FAILURES.md](FAILURES.md).

## 6. Assumptions log

Every ambiguous-brief decision, bug, and eval-tuned threshold is logged
with rationale in [DECISIONS.md](DECISIONS.md). Two counter-intuitive
findings worth flagging: (1) resolving a follow-up with the *gold*,
maximally-specific rewrite can **hurt** lexical retrieval by over-
anchoring on the previous turn's named entities (§4); (2) a stricter
per-paper cap (1 chunk) looked strictly better than a looser one (2
chunks) but was removing distinct relevant content, not just redundancy
(§3).

## 7. What I'd do next with more time

1. **A cross-encoder reranker** over the hybrid RRF candidates — the
   brief's optional suggestion, not attempted given the time budget.
2. **A held-out eval set** — every threshold here was tuned by inspecting
   this same 36-question set; a genuinely held-out set is the correct fix.
3. **Resolve the SPECTER2 adapter-loading blocker** (§4) — likely fixable
   with more than a 15-minute box.
4. **Mechanical attribution-checking** (FAILURES.md Case 3): verify named
   entities/authors near a citation actually appear in that source —
   narrower and more catchable than full groundedness, which already
   passes answers whose facts are correct even with a fabricated name.
5. **Token-aware chunking by default**, not a post-hoc cap — the cap
   fixes truncation but the word-count chunk target still produces a long
   tail of oversized chunks for CJK/LaTeX-heavy papers.
