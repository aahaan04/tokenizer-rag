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
