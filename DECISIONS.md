# Decisions log

Running log of assumptions and design choices made while the brief left something
unspecified. Each entry: date, decision, one-line rationale.

## 2026-10-06 — Timeline compression

Decision: treat this as a ~1.5-day build (not the brief's suggested 2.5 days) since
the deadline is 2026-10-07. Retrieval/dedup (25%) and evaluation rigour (25%) get
priority over corpus breadth (20%) if something has to give.
Rationale: those two categories are 50% of the grade and are the "challenging"
items the brief calls out explicitly.

## 2026-10-06 — Source selection: arXiv + Semantic Scholar only, ACL Anthology cut

Decision: collect from the arXiv API and Semantic Scholar API. Skip ACL Anthology.
Rationale: the brief lists ACL Anthology as optional ("if time allows"). Semantic
Scholar's externalIds (ACL ID, DOI, corpus ID) already link an arXiv preprint to
its published venue for most NLP papers, which covers the preprint-vs-conference
duplicate case the brief wants without a third scraper. Revisit if time remains
after Phase 4.

## 2026-10-06 — Dense index: brute-force numpy, not FAISS

Decision: use numpy matrix multiplication for dense vector search instead of FAISS.
Rationale: corpus is a few hundred papers (~10-20k chunks at few-hundred-token
chunks), so a 15k x 384 float32 matrix (~23MB) does exact cosine search in well
under a second on CPU. FAISS adds a dependency and an abstraction layer with no
speed benefit at this scale, and the brief explicitly allows numpy. Exact search
also makes eval numbers (Recall@5, MRR) deterministic, which matters for Phase 5.

## 2026-10-06 — LLM provider: Groq llama-3.3-70b-versatile

Decision: default generation/query-rewrite model is `llama-3.3-70b-versatile` on
Groq's free tier (1,000 req/day, 100k tokens/day as of Oct 2026 — verified via
Groq's docs, see console.groq.com/docs/models). Swappable to Gemini via
`LLM_PROVIDER=gemini` in `.env` through a small provider interface.
Rationale: brief requires a free/no-cost setup; Groq's free tier is generous
enough for eval runs (dozens of LLM calls) and a chat session.

## 2026-10-06 — Semantic Scholar failures are non-fatal

Decision: if Semantic Scholar's unauthenticated API returns 429 after retries
with backoff, log and skip that query term rather than aborting the whole
collection run; arXiv results for that term are kept.
Rationale: confirmed by hand that the unauthenticated tier is a shared global
pool (`TooManyRequestsException` on a cold first request, independent of our own
pacing) — this is an external capacity issue, not a bug in our rate limiting.
arXiv alone can carry primary collection; S2 is enrichment (external IDs for
Phase 4 dedup) and partial loss of it shouldn't block Phase 1. If this keeps
happening during the full run, get a free S2 API key (see `.env.example`) for a
dedicated rate limit instead of the shared pool.

## 2026-10-06 — Off-topic-domain veto in relevance scoring

Decision: the Checkpoint 1a sample run included a speech-recognition BPE paper
and a molecular-GAN BPE paper — both score high on tokenization keywords but
aren't about LLM text tokenization. Added a veto: if the title/abstract matches
an off-topic-domain term (speech recognition, molecular/drug, symbolic
music/MIDI, genome/protein) and does NOT also contain a *strong* LLM-context
term ("language model", "llm", "transformer" — deliberately not "multilingual",
which off-topic papers use just as often), the candidate is excluded regardless
of its keyword score.
Rationale: BPE/subword vocabularies are reused across many fields; "mentions
tokenization a lot" isn't sufficient, the paper has to be about tokenization
*for LLMs* specifically, per the brief's topic-boundary instruction. This is a
heuristic, not a solved problem — a symbolic-music tokenization paper that also
says "Transformer-based models" will still slip through; flagged for manual
review at Checkpoint 1a/1b rather than further hand-tuned.

## 2026-10-06 — Topic-boundary rule (user-approved at Checkpoint 1a)

Rule: a paper is in scope if the tokenizer under study feeds a (possibly
multimodal) language model; out if it's a domain-specific sequence model with
no language-model framing (music, molecules, speech-only ASR, genomics).
Concretely: "From Pixels to Tokens" (BPE for a multimodal LLM's image tokens)
is IN; "From Words to Music" (subword tokenization for MIDI generation, which
only mentions "Transformer-based models" in passing) is OUT. Implemented as
`STRONG_CONTEXT_TERMS` in `relevance.py` — deliberately excludes bare
"transformer" (used across music/vision/speech architectures too) and requires
an explicit "language model"/"LLM" mention to rescue an off-topic-domain hit.

## 2026-10-06 — Relevance score cutoff, spelled out

`INCLUDE_THRESHOLD = 3.0` against a per-candidate score built from:
- Title hit on any core term (strong or weak): +3.0 each. One clear title
  match (e.g. "SentencePiece", "WordPiece") is sufficient on its own.
- Abstract hit on a STRONG term (names a specific method/metric: SentencePiece,
  fertility, vocabulary size, token-free, ...): +1.5 each.
- Abstract hit on a WEAK term (generic: bpe, tokeniz, subword): +0.5 each.
- Abstract hit on a context term (language model, multilingual, arithmetic,
  ...): +0.5 each.

Why 3.0 and not 2.0 (the original value): a 2.0 bar let a single sentence like
"we use BPE tokenization with a vocabulary of 128k" (two weak-term hits, no
title match) past the gate — exactly the "passing mention" the brief says
should be excluded. 3.0 forces either a title match, or roughly 2 abstract
STRONG hits, or a title-adjacent combination — i.e., tokenization has to be
load-bearing in the abstract, not a setup detail. Caught and fixed via
`tests/test_relevance.py::test_passing_mention_is_excluded`.

## 2026-10-06 — Canary check for corpus completeness

Added a hard-coded list of 6 canonical tokenizer papers (Sennrich 2016 BPE,
Kudo & Richardson 2018 SentencePiece, Xue et al. 2021 ByT5, Clark et al. 2021
CANINE, Petrov et al. 2023 fairness/fertility, Singh & Strouse 2024
arithmetic) in `pipeline.CANARY_PAPERS`, fetched directly by arXiv id via
`arxiv.fetch_by_ids()` so their presence in the corpus doesn't depend on the
keyword queries ranking them in the top `max_per_query` results. `collect()`
also records whether each one was *naturally* found by the keyword queries
(before the direct fetch runs) and prints that at the end of a full run — a
canary not found naturally is a real signal the query set under-covers that
area, separate from whether the paper ends up in the corpus either way.

## 2026-10-06 — Added ACL Anthology as a third source

Reverses the earlier "ACL Anthology cut" decision. Trigger: Semantic Scholar's
unauthenticated tier is currently rate-limited hard enough (confirmed: 429 on
a cold first request) that it can't be relied on as the sole source of
preprint-vs-conference-version links, which Phase 4 dedup needs. ACL Anthology
has no per-query search API, so `acl_anthology.py` downloads+caches the bulk
`anthology+abstracts.bib.gz` export (~40MB, ~130k entries, all of
computational linguistics) once, pre-filters to entries mentioning
tokenization-adjacent keywords (~6.6k), then runs the same `relevance.decide`
scorer as the other sources. An ACL entry whose normalized title matches an
existing arXiv/S2 candidate is merged into that row rather than creating a
duplicate — which is itself useful signal for Phase 4's preprint-vs-published
matching, not just noise reduction.

## 2026-10-06 — Fixed two duplicate-row bugs found via the canary check

The first full run's canary report showed 5/6 canaries "not found naturally"
and injected directly by arXiv id. Checking the manifest revealed why, and
exposed two real bugs rather than confirming a query-coverage gap:

1. Canary injection only checked for an exact `arxiv:<id>` key before adding a
   row. If ACL Anthology had already added the same paper under an `acl:...`
   key (true for CANINE, ByT5, Sennrich — all three are ACL-published), the
   injection created a second, disconnected row for the same paper. Fixed by
   running ACL merging and canary injection against one shared `title_index`,
   so a title match anywhere in the pool is reused instead of re-added.
2. The main arXiv/S2 query loop had the same class of bug in the other
   direction: arXiv ingestion only checked `candidates[f"arxiv:{id}"]`, so if
   Semantic Scholar found a paper under an earlier-processed query term than
   the one where arXiv's own phrase search found it, arXiv created a second
   row (confirmed: Petrov et al.'s fairness paper, 2305.15425, had both an
   `s2:` and an `arxiv:` row). Fixed with a `by_arxiv_id` index that both
   ingestion branches check and update symmetrically.

After both fixes, a clean rerun found all 6 canaries naturally (via keyword
queries or the ACL pass) with exactly one manifest row each — zero injected.
Lesson for Phase 4: any "first writer wins, second writer merges" dedup logic
needs the check to run in both directions, not just the one that happened to
get tested first.

## 2026-10-06 — Dedup fixed properly: linked, not merged (user correction)

The user caught a real design flaw at Checkpoint 1b review: the ACL-merge and
canary-injection logic above was still *merging* a title-matched cross-source
row into the existing one (filling in doi/venue, no new row) rather than
keeping both. That's wrong — an arXiv preprint and its ACL-published version
(or a DOI-matched S2 record) are exactly the "preprint vs. conference version"
duplicate pair Phase 4 needs to detect and measure, not something Phase 1
should silently collapse. Fixed: added a `candidate_group_id` column
(`manifest.py`) and `pipeline._link_or_add()`, which adds a title-matched
candidate as its own row and links it to the existing one via a shared group
id, rather than merging fields into one row. 135 such groups exist in the
corpus after this fix (2-3 rows each, from arXiv/S2/ACL combinations of the
same paper). Distinct from the arXiv-id-exact-match dedup in the main query
loop (`by_arxiv_id`), which stays a true merge — that case is the literal same
arXiv record found twice via different queries, not a cross-source version.

While testing this fix, a THIRD bug turned up: `arxiv.fetch_by_ids()` does not
preserve request order (confirmed directly: requesting
`[1508.07909, 1808.06226, ...]` returned `[2103.06874, 2105.13626, ...]`), and
the canary-injection loop used `zip(canary_ids, fetch_by_ids(canary_ids))`,
silently pairing each canary's label with a *different* paper's actual fetched
content. This produced a duplicate SentencePiece row and a missing ByT5 row.
Fixed by indexing the fetch results by each entry's own `arxiv_id` instead of
trusting positional order.

## 2026-10-06 — Classic/pre-neural tokenization excluded from scope

Added rule: a paper about word segmentation, sentence segmentation, or
tokenization as a preprocessing step for parsing/tagging (dependency parsing,
treebanks, POS tagging, multiword-expression identification) is OUT of scope,
unless it's also about subword tokenization or tokenizers for neural LMs.
Trigger: the user noticed "Joint Dependency Parsing and Multiword Expression
Tokenization" had been included — ACL Anthology's bulk export goes back to the
1990s and "tokenization" meant something different (word-level text
segmentation for a pipeline stage) before subword/neural methods existed.
Implemented as `CLASSIC_PIPELINE_TERMS` + a rescue list (explicit
subword/BPE/LM mention) in `relevance.py`. Needed a negative-lookbehind regex
so "subword segmentation" (in scope) isn't caught by the "word segmentation"
(out of scope) substring match.

## 2026-10-06 — Off-topic-domain veto split into hard/soft tiers

The single-tier veto let "Training Text-to-Molecule Models with Context-Aware
Tokenization" through because it said "language model" once — but it's
cheminformatics research borrowing LLM terminology, not tokenizer research for
LLMs, per the user's explicit call-out. Split `OFF_TOPIC_DOMAIN_TERMS` into
`OFF_TOPIC_DOMAIN_SOFT_TERMS` (speech, music — still rescued by an explicit
LLM-context mention) and `OFF_TOPIC_DOMAIN_HARD_TERMS` (molecular, drug-like,
protein/DNA/genome — never rescued, excluded regardless of framing).

## 2026-10-06 — Corpus size is reported as unique papers, not included rows

822 included rows ≠ 723 unique papers — 197 of those rows are cross-source
duplicates of 98 other included rows (same paper, found via arXiv/S2/ACL,
kept as separate linked rows per the dedup-link fix above). WRITEUP.md and any
"corpus size" claim use the unique-paper count. `manifest.py` gained
`write_manifest_summary()`, called automatically at the end of `collect()` and
`add_arxiv_version_pairs()`, writing `manifest_summary.md` with this
breakdown so it can't silently go stale relative to `manifest.csv`.

## 2026-10-06 — Added a small sample of real arXiv multi-version duplicates

Collection's cross-source groups (preprint vs. ACL/S2 record) don't cover the
brief's first-named duplicate case: the same paper at multiple arXiv versions.
arXiv's search/id_list API only ever returns a paper's current version, so
collection can't produce this case by itself. Added
`pipeline.add_arxiv_version_pairs()`: for a sample of included, multi-version
arXiv papers (35 candidates had version > v1; took 25, preferring v3+ over v2
as more interesting dedup cases), fetches each one's v1 text via
`arxiv.fetch_by_ids(["<id>v1"])` and adds it as a new row sharing the existing
row's `candidate_group_id`. Idempotent (skips a paper if its v1 row is already
present). Three of the six canaries (Sennrich, CANINE, ByT5) now have both a
cross-source group and a version pair in the same group. Unique-paper count is
unaffected (723, unchanged) since these rows join existing groups.

## 2026-10-06 — Phase 2 parsing approach

Decision: fetch and parse text **per row, not deduplicated by
candidate_group_id**. A cross-source or multi-version duplicate pair only
exists in the corpus so Phase 4 can measure retrieval-level dedup impact
("duplicate rate in top-5," before/after Recall@5); that requires each
duplicate row to actually be indexable with its own chunks, not collapsed to
one shared text.

Per-row fetch priority: arXiv HTML (`arxiv.org/html/<id>`, native LaTeXML
rendering — clean section structure) → arXiv PDF fallback → ACL Anthology PDF
(`<source_url>.pdf`, standard pattern) → Semantic Scholar open-access PDF
(looked up from already-cached S2 search responses, no extra API calls) → if
none available, the abstract alone becomes the paper's one "Abstract" chunk
(flagged `abstract_only: true` in chunks.jsonl, so eval/retrieval code can
treat it differently if needed).

Section-header detection: HTML uses the actual `<h1>-<h4>` tags (reliable,
since arXiv's HTML is LaTeXML-generated with real heading elements). PDF uses
a regex heuristic matching common section-name lines (Introduction, Method,
Results, ...) under 60 characters — crude, will miss non-standard headers and
occasionally misfire, but cheap and "good enough" per the time budget. Known
noise: PDF extraction sometimes captures a page header/footer (venue/
proceedings line) as the start of the first section, since PyMuPDF doesn't
distinguish body text from running headers. Not fixed given time constraints.

References: dropped entirely — everything from a "References"/"Bibliography"
heading onward is discarded (both extractors). Figure/table captions: kept
inline, since they're ordinary text nodes in both the HTML and PDF text
stream — not specially tagged or separated. Tables: kept as whatever
linearized/reading-order text the extractor already produces; not specially
reconstructed into rows/columns (would need real table detection, out of
scope for the time available). Equations: same — kept as whatever inline text
(LaTeX source in HTML, or often-garbled glyph text in PDF) comes through.

Chunking: word count (not a real tokenizer) as a cheap proxy for token count,
~350 words/chunk with 50-word overlap, chunk index restarting per section.

## 2026-10-06 — Dropped all-MiniLM-L6-v2 for a 512-token embedding model

Measured actual chunk token lengths with the real `all-MiniLM-L6-v2` tokenizer
(its *effective* limit is 256 — `SentenceTransformer.max_seq_length`, not the
underlying BERT's raw 512 `model_max_length`, which is what you'd see if you
only checked the tokenizer config). Result: p50=468, p95=724, p99=993,
max=5869 tokens — 69.5% of chunks exceed 256 tokens, and 39% exceed even 512.

Root cause isn't just "chunks are too long in words": 750 of 821 papers (91%)
have at least one chunk over 512 tokens. Technical English alone runs higher
BPE tokens/word than casual text, but the extreme tail (chunks in the
thousands of tokens) comes from a few papers with CJK text or LaTeX-leakage
in extraction, where whitespace-delimited "word count" doesn't bound token
count at all (CJK has no inter-word spaces, so a 350-"word" chunk can be one
enormous run of individual-character tokens).

Decision: drop `all-MiniLM-L6-v2` from the Phase 5 comparison, use a
512-token-capacity general-purpose model instead (`bge-small-en-v1.5`) — kept
the 350-word chunk target as-is. Rationale ("less rework" test from the
brief): re-chunking the whole corpus at ~200 words wouldn't fix the real
problem (CJK/LaTeX-leak chunks would still blow past any word-count-based
target, since the unit itself doesn't bound token count for those cases) and
costs a full reparse; swapping the model is a planning-only change with no
reprocessing. The p50/p95 chunk (468/724 tokens) still exceeds bge-small's
512 limit at the p95 mark, so a meaningful tail is still silently truncated
at embedding time — accepted and documented here rather than fixed, given the
time budget. A real fix (token-aware chunking using the target model's own
tokenizer) is a natural follow-up noted in WRITEUP.md's limitations, not
attempted now.

## 2026-10-06 — Checkpoint 2 fixes: parse accounting, ACL PDF quality

User review of Checkpoint 2 caught a real reporting bug and asked for ACL PDF
quality improvements, timeboxed to ~45 minutes:

**Parse outcome accounting.** The original report double-counted: all 80
"abstract_only" rows included the 26 that had no abstract either (true hard
failures). Fixed to three mutually exclusive buckets (767 full_text / 54
abstract_only / 26 no_text_not_indexed, summing to 847). The 26 hard failures
are now written back into `manifest.csv` as `parse_status=no_text_not_indexed`
instead of only existing in a parse report — anyone reading the manifest can
see which rows aren't actually in the index.

**ACL PDF extraction**, rewritten in `extract.py`:
- Running header/footer removal: lines repeating verbatim across ≥40% of
  pages (venue/proceedings lines PyMuPDF can't otherwise tell apart from body
  text) are dropped, plus an explicit regex for "Proceedings of...", bare page
  numbers, and copyright lines.
- Numbered-heading pattern ("3 Method", "4.2 Results") added alongside the
  existing keyword-heading match.
- Font-size/bold heading detection via `page.get_text("dict")` (per-line font
  size + bold flag), rescuing non-numbered, non-keyword headings a body-text
  heuristic alone would miss.

Verified on the SentencePiece ACL PDF before/after: 4 sections -> 18, with
real subsections (Library Design, Lossless Tokenization) now correctly split
out instead of lumped into one 2000+ character "Introduction" blob. Some
front-matter noise remains (title/author/affiliation lines occasionally
detected as mini-headings, since they're also bold/large-font) — harmless
(small, correctly-placed fragments, not corrupted body content) and left as a
known limitation rather than further tuned, given the timebox.

Reran the full corpus: total_chunks went from 18,073 -> 28,285 (finer section
splitting -> more, shorter chunks), and this *also* substantially improved the
token-length problem from the entry above: against `bge-small-en-v1.5`'s
tokenizer, p50 dropped from 468 -> 182 tokens, and the fraction exceeding 512
tokens dropped from 39% -> 13.1%. The decision to use `bge-small-en-v1.5`
instead of `all-MiniLM-L6-v2` stands regardless (that entry's CJK/LaTeX-leak
root cause is unrelated to section-splitting granularity), but is now backed
by a noticeably healthier distribution.

**Clarification:** "chunks per paper" / "papers with >=1 chunk" stats are
always keyed by `paper_id` (one row = one source's copy of one paper), not
deduplicated by `candidate_group_id`. A paper found via arXiv, ACL, and S2 is
3 rows, each independently fetched/parsed/chunked, each counted separately in
these stats — consistent with the per-row fetching decision above. The
unique-paper corpus-size figure (723) is the one metric that *does*
deduplicate by group, and is reported separately for exactly this reason.

## 2026-10-06 — Hard token cap added to the chunker

13.1% of chunks still exceeded 512 tokens after the PDF-extraction fix above —
bge-small and SPECTER would silently truncate those. Added
`chunker._split_to_token_cap()`: after the normal word-count split, any chunk
is recursively halved by word count (never mid-word) until every piece
tokenizes to <= 450 tokens under the actual target model's tokenizer
(`bge-small-en-v1.5`), not a word-count estimate — recursion handles the CJK/
LaTeX-leak pathological cases a single size estimate would get wrong.

Reran the full corpus: total_chunks 28,285 -> 34,779. New distribution: p50
227, p95 394, p99 440, max 756 tokens; only 2 of 34,779 chunks (0.006%) still
exceed 450 — both single-"word" LaTeX-markup leakage with no whitespace to
split on (confirmed by inspection), an accepted edge case rather than a real
chunk. Fraction exceeding 512 is now ~0.003% (was 13.1%).

## 2026-10-06 — Group-level relevance definition for Recall@5/MRR

This governs every retrieval metric in Phase 3/4/5, especially the dedup
before/after numbers, so it's spelled out precisely here rather than left
implicit in eval code.

A retrieved chunk counts as a **hit** for a gold-labeled question if BOTH:
1. **Paper match**: the chunk's `canonical_paper_id` equals the gold paper's
   `candidate_group_id` (or, for a paper with no cross-source/version
   duplicates, its own singleton `paper_id` — `canonical_paper_id` is already
   set to one or the other for every chunk, see Phase 2). This means retrieving
   *any* row belonging to the gold paper's group — its arXiv v1, its latest
   version, its ACL-published copy, its S2 record — counts as finding the
   right paper. This is deliberate: before Phase 4 dedup, the index can
   legitimately return any duplicate row and should get credit for finding
   the paper; Phase 4's job is to stop 3 rows of the *same* paper crowding
   out 3 *different* relevant papers in the top-5, not to penalize retrieval
   for the duplication existing in the first place.
2. **Section match**: `normalize_section(chunk.section_title)` equals, or is a
   substring of, `normalize_section(gold.section)` (normalize = lowercase,
   strip leading section numbers and punctuation — the same chunk's section
   can read "1 Introduction" from one extractor and "Introduction" from
   another). A paper-level match in the wrong section is NOT a hit — gold
   labels are at paper+section granularity specifically so chunking or
   indexing changes don't silently inflate Recall@5 by rewarding "found the
   right paper, wrong part."

Consequence for the "duplicate rate in top-5" metric (Phase 4): computed
*before* this definition's paper-match collapsing — i.e., it counts how many
of the top-5 chunks share a `canonical_paper_id` with another chunk already
in that same top-5, which is exactly the "3 versions of Paper A crowding the
other 2 slots" case the brief's example illustrates. Recall@5/MRR use the hit
definition above; the duplicate-rate metric is a separate count over the same
top-5 list.
