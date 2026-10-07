# tokrag — Design Write-up

A RAG chatbot over a self-collected corpus of tokenizer-for-LLMs research.
This document summarizes the design decisions and results; the complete,
dated reasoning behind every one of them (including false starts, bugs
found and fixed, and numbers superseded by later reruns) is in
[DECISIONS.md](DECISIONS.md), which this write-up treats as the
authoritative assumptions log rather than duplicating.

## 1. Topic boundary and corpus composition

**Scope rule**: a paper is in scope if the tokenizer under study feeds a
(possibly multimodal) language model; out of scope if it's a domain-
specific sequence model with no language-model framing (music, molecules,
speech-only ASR, genomics) — classic/pre-neural tokenization (pure CRF
word segmentation, etc.) is also excluded. Enforced via an off-topic-domain
veto: a title/abstract hit on a domain term (speech, molecular/drug,
symbolic music, genome/protein) is excluded unless it *also* contains an
explicit LLM-context term ("language model," "LLM" — deliberately not bare
"transformer," which off-topic papers use just as often).

**Sources**: arXiv (category-filtered, keyword search) + Semantic Scholar
(keyword search) + ACL Anthology (bulk bibliography export, pre-filtered
then scored), added as a third source when Semantic Scholar's
unauthenticated tier proved too rate-limited to reliably supply the
preprint-vs-conference-version links dedup needs.

**Inclusion counts** (full collection run, see [manifest_summary.md](manifest_summary.md)):

| | Count |
|---|---|
| Total candidate rows seen | 7,547 |
| Included | 843 (105 arXiv, 281 Semantic Scholar, 457 ACL Anthology) |
| **Unique papers** (de-duplicated by cross-source/version group) | **715** |
| Rejected — below relevance threshold | 6,338 |
| Rejected — vetoed (off-topic domain or classic pipeline) | 366 (251 classic-pipeline, 75 domain-soft, 40 domain-hard) |

A relevance score (title/abstract keyword hits, weighted by specificity)
gates inclusion at `>= 3.0` — tuned up from an initial 2.0 after that bar
let through papers where tokenization was only a passing setup detail
("we use BPE with a 128k vocabulary"), which the brief explicitly asks to
exclude; 3.0 requires either a title match or roughly two strong abstract
hits. A hard-coded canary check (6 canonical papers: Sennrich 2016 BPE,
Kudo & Richardson 2018 SentencePiece, ByT5, CANINE, Petrov et al. 2023
fairness, Singh & Strouse 2024 arithmetic) verifies the corpus contains
the field's foundational work regardless of how the keyword queries rank
them, and reports whether each was found *naturally* — a gap signal for
query coverage, independent of final inclusion.

**Known topic-boundary gap**: the off-topic-domain veto matches "genome"
but not "genomic" as a substring; found via the dedup audit (not the
original collection pass), affecting 4 rows / ~1 paper, fixed by switching
to a stem match. A related, broader gap — ACL `@proceedings` front-matter
entries (workshop volumes, not papers) can enter the candidate pool as if
they were papers — was caught once during fuzzy-merge dedup (see §3) and
guarded against there, but wasn't separately, exhaustively audited across
the full corpus given the time budget.

## 2. Parsing and chunking

Each manifest row is fetched and parsed **independently**, not
deduplicated by cross-source group — a cross-source/version duplicate only
earns its place in the corpus so Phase 4 can measure dedup's effect on
retrieval, which requires both copies to actually be indexed separately.
Fetch priority: arXiv HTML (LaTeXML-rendered, real heading tags) → arXiv
PDF → ACL PDF → Semantic Scholar open-access PDF → abstract-only fallback
(flagged, still indexed as a single chunk). Section headers come from real
HTML tags where available, or a PDF regex heuristic otherwise (crude —
known to occasionally capture a running page header as a false section).
References are dropped; figures/tables/equations are kept as whatever
linearized text the extractor produces, not specially reconstructed.

Chunking targets ~350 words with 50-word overlap, restarting per section.
A **hard token cap** (`<=450` tokens under the actual target embedding
model's own tokenizer, recursively halving by word count when exceeded)
was added after measuring that 13.1% of chunks exceeded the embedding
model's 512-token limit — driven mostly by CJK text and LaTeX-extraction
leakage, where word count doesn't bound token count at all. After the cap:
only 0.006% of 34,779 chunks still exceed it (two pathological single-
"word" LaTeX-markup leaks with no whitespace to split on).

## 3. Dedup strategy

Three distinct duplicate cases, found and handled separately:

1. **Multiple arXiv versions of the same paper** (v1 vs. v3, etc.) —
   7 version-only groups, deliberately seeded with 25 real version pairs
   (arXiv search only ever returns the latest version, so this case never
   occurs naturally during collection and had to be manually introduced
   for Phase 4 to have real cases to detect).
2. **Preprint vs. published version, cross-source** (arXiv + ACL/S2 record
   of the same paper) — the dominant case, 80 cross-source-only groups +
   18 groups that are both version-paired and cross-source.
3. **Near-duplicate chunks within a paper's own multiple copies** (e.g. an
   arXiv HTML extraction and an ACL PDF extraction of the same results
   section, differing slightly in wording/boundaries) — handled at
   retrieval time, not collection time, via word-trigram shingle Jaccard
   similarity (>=0.9 collapses two chunks as the same passage).

**Canonical IDs**: Phase 1 collection already links same-title candidates
via `candidate_group_id` (kept as *separate* rows deliberately, per an
early correction — collapsing them during collection would destroy the
signal Phase 4 needs to measure). Phase 4 assigns the actual merge
decision (`canonical_id`) and ran a full false-merge audit over all
resulting groups: **zero likely false merges** among the exact-title-match
groups. A fuzzy tier (title_sim >= 0.85 AND shared-author/abstract-cosine
conditions) added 5 more real cross-source merges, plus one manual,
user-confirmed exception — **MorphBPE**, two records with verified-by-hand
title_sim as low as 0.667 (same three core authors, a preprint retitled
for its 2026 publication) that the general 0.85 rule genuinely couldn't
catch without lowering the threshold for everything else.

**A real false merge was caught and fixed before it reached the
manifest**: two different years' workshop *proceedings volumes* ("First
Workshop on Subword and Character Level Models," 2017 vs. "Second
Workshop...," 2018) scored title_sim=0.855 with 3 "shared authors" — which
were actually the organizing committees' overlapping names (ACL
`@proceedings` entries have no real author field). Fixed with an explicit
guard: any pair where either title starts with "Proceedings of" is skipped
before fuzzy scoring, with a regression test.

**Diversification — the cap-1 ablation**: the first retrieval-side design
capped each paper to **1 surviving chunk** in the top-k. This looked right
(duplicate rate hits exactly 0.0) but a review caught that it was removing
*distinct* relevant content, not just duplicates — two genuinely different
sections of the same paper (Introduction vs. Results) were being treated
as interchangeable copies, and only one survived regardless of which
actually answered the question:

| Config | Recall@5 | MRR | dup_rate@5 |
|---|---|---|---|
| bge-small, raw (no diversify) | 0.293 | 0.240 | 0.324 |
| **bge-small, cap-1 naive (the flawed design)** | 0.224 | 0.224 | 0.000 |
| bge-small, final (collapse near-dup chunks first, then cap 2/paper) | 0.276 | 0.240 | 0.186 |
| BM25, raw (no diversify) | 0.431 | 0.389 | 0.359 |
| **BM25, cap-1 naive** | 0.379 | 0.371 | 0.000 |
| BM25, final | 0.431 | 0.389 | 0.200 |

The fix: collapse near-duplicate *chunks* first (so two different
sections of the same paper are confirmed non-duplicate, not just assumed
distinct), THEN cap at 2 chunks/paper (not 1), with retrieval overfetching
top-50 before diversifying to leave room to fill slots. BM25 fully
recovers its raw Recall@5/MRR under the fixed design while still cutting
duplicate rate from 35.9% to 20.0%; bge-small recovers most (not quite
all) of what cap-1 cost it. A stricter, content-aware
`redundant_copy_rate@5` metric (near-duplicate *passages* specifically,
the brief's actual "same results section 3 times" example) confirms the
fixed design drives literal repeated passages to exactly **0.0** on both
backends, while the remaining ~19% `same_paper_share@5` is legitimately
two different sections of the same paper, not duplication.

## 4. Embedding/retrieval comparison and recommendation

Compared on the 29 answerable eval questions (k=5, diversification on,
follow-ups scored on their **gold standalone rewrite** — see §6 for why
that matters):

| Config | Recall@5 | MRR | redundant_copy_rate@5 | latency p50 (ms) | index size |
|---|---|---|---|---|---|
| bge-small-en-v1.5 | 0.241 | 0.231 | 0.000 | 18.0 | 51.0 MB |
| SPECTER (original, no adapters) | 0.103 | 0.138 | 0.000 | 32.6 | 101.9 MB |
| BM25 | 0.397 | 0.372 | 0.000 | 180.9 | 28.3 MB |
| **Hybrid (BM25 + bge-small, RRF)** | **0.466** | **0.428** | 0.000 | 297.9 | 79.3 MB |

**Paired bootstrap 95% CIs** (3,000 resamples, seed=42, n=29 — too few
questions to bootstrap meaningfully by sub-type):

| Comparison | Recall@5 diff | 95% CI | Significant? |
|---|---|---|---|
| Hybrid vs. BM25 | +0.069 | [-0.069, 0.207] | No |
| Hybrid vs. bge-small | +0.224 | [0.017, 0.431] | **Yes** |

**Recommendation: hybrid (BM25 + bge-small via Reciprocal Rank Fusion,
rrf_k=60).** Its advantage over dense-only is statistically supported;
its advantage over BM25 alone is directionally consistent with the
expected mechanism (BM25 for exact terminology, dense for paraphrase/
semantic continuity) but is **not** statistically significant at this
sample size — stated honestly rather than oversold. Hybrid is the slowest
config (298ms p50, still far under any interactive-latency concern for a
CLI tool) and never does worse than either component alone on any metric
tested.

**SPECTER underperforms and why**: SPECTER is trained via citation-
prediction on full (title + abstract) academic text; our queries are
short, conversational natural-language questions, structurally unlike its
training distribution. This is starkest on `follow_up` questions — the
shortest, most pronoun-dependent queries — where SPECTER scores a
complete **0.0** Recall@5, worse than plain lexical BM25 (0.286 pre-gold-
rewrite). SPECTER would likely suit a paper-to-paper similarity feature
better than a conversational retrieval interface — a different use case
than this project's, not a fair fight.

**SPECTER2 attempt, framed honestly as a limitation**: per instruction,
time-boxed to 15 minutes. `pip install adapters` succeeded but forced a
`transformers`/`huggingface-hub` downgrade (verified the rest of the
pipeline still works). Loading either SPECTER2 adapter
(`allenai/specter2`, `allenai/specter2_adhoc_query`) failed consistently —
both attempt to resolve via the legacy AdapterHub.ml index before falling
back to HF Hub, and that index lookup errors out in this environment
regardless of invocation style. This reads like a genuine rough edge in
the `adapters` library's HF Hub resolution path for this model, not a
typo or transient blip. **Not resolved within the time-box; the original
SPECTER ablation stands as the scientific-embedding comparison point.**
Given SPECTER's 0.0 follow-up Recall@5 is already a strong, well-explained
result independent of which adapter variant is used, this gap likely
would not have changed the hybrid recommendation either way — but it's a
real, disclosed gap in the comparison, not swept under the rug.

## 5. Chat pipeline and metrics

Hybrid retrieval → Phase 4 diversification → citations mapped only to
retrieved chunks (paper + year + section) → dual-signal abstention
(retrieval-score threshold + the model's own instruction-level
self-assessment) → query rewriting for follow-ups using a smaller/faster
model (`openai/gpt-oss-20b` vs. `openai/gpt-oss-120b` for generation),
with every LLM call cached to disk and retried with backoff on 429.

**Final 36-question chat eval** (after one round of Checkpoint-6 fixes —
see DECISIONS.md for the full diagnosis):

| Metric | Value |
|---|---|
| Abstention precision (raw) | 0.54 (tp=7, fp=6, fn=0, tn=23) |
| Abstention precision, excluding retrieval misses | **0.88** (only 1 real generation error, out of 8 retrieval misses that were correctly excluded) |
| Abstention recall | **1.0** — zero hallucinated answers on genuinely unanswerable questions |
| Groundedness rate (LLM-judge) | 0.87 (20/23 non-abstained answers) |
| Citation rate | 0.91 (21/23) |
| Follow-up Recall@5, system rewrite vs. gold rewrite | 3/7 vs. 2/7 |

The raw abstention precision mixes two things: redefining the scorer so a
hedge only counts as abstention when the answer *also* has zero real
citations (fixing a case where the model gave a correctly-cited partial
answer but was scored as a full refusal) accounts for about a quarter of
the improvement (0.41→0.44); the rest (0.44→0.54) comes from an actual
prompt change (explicitly allowing partial answers on multi-part
questions) and an improved follow-up query rewriter.

**The follow-up recall "win" (3/7 vs. 2/7) should not be read as "the
system beats gold rewriting."** n=7 is far too small, and every
`gold_standalone_rewrite` in the eval set names the prior paper's exact
title (and sometimes an exact figure) — it's an oracle rewrite authored
with knowledge of the answer, not a neutral baseline a real rewriter could
match before retrieval happens. If anything the gold condition is a
biased, optimistic ceiling; the small observed edge is sample noise, not
evidence the heuristic rewriter is actually better.

Full diagnosis of all 10 original false-positive abstentions (8 retrieval
misses, 1 genuine temperature=0 non-determinism case, 1 detection-formula
bug) is in DECISIONS.md; three representative, distinct failure cases with
full before/after detail are in [FAILURES.md](FAILURES.md).

## 6. Assumptions log

Every ambiguous-brief decision, bug found and fixed, and threshold tuned
on the eval set is logged with its date and rationale in
[DECISIONS.md](DECISIONS.md) — not duplicated here. Two findings worth
flagging explicitly since they run counter to intuition: (1) resolving a
follow-up's pronoun with the *gold*, maximally-specific rewrite can
**hurt** lexical retrieval by over-anchoring on the previous turn's named
entities and crowding out the actual target paper (§4); (2) a stricter
per-paper cap (1 chunk) looked like a strictly better de-duplication
design than a looser one (2 chunks) but was actually removing distinct,
relevant content, not just redundancy (§3) — both are documented in full
in DECISIONS.md with the specific questions used to trace each mechanism.

## 7. What I'd do next with more time

1. **A cross-encoder reranker** over the hybrid RRF candidates — the
   brief's own optional suggestion; not attempted given the time budget,
   but the natural next lever now that hybrid retrieval and diversification
   are both in place and measured.
2. **A held-out eval set.** All thresholds (relevance cutoff, abstention
   score threshold, fuzzy-merge dedup parameters) were tuned by inspecting
   this same 36-question set's distribution — disclosed throughout
   DECISIONS.md, but a genuinely held-out set is the correct fix, not just
   the honest caveat.
3. **Resolve the SPECTER2 adapter-loading blocker** (§4) — likely a
   fixable library/environment issue given more than a 15-minute box, and
   the one piece of the embedding comparison left genuinely incomplete.
4. **Mechanical attribution-checking**, motivated directly by FAILURES.md
   Case 3: a cheap pass that verifies named entities/authors/years near a
   citation marker actually appear in that citation's source text — a
   narrower, more catchable check than full groundedness, which already
   passes answers whose *facts* are correct even when an attached name is
   fabricated.
5. **Token-aware chunking by default**, not a post-hoc hard cap — the cap
   fixes truncation but the underlying word-count-based chunk target still
   produces a long tail of oversized chunks for CJK/LaTeX-heavy papers.
