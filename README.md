# tokrag — Talk to the literature

A RAG chatbot over a self-collected corpus of research papers on
tokenizers for LLMs: BPE, WordPiece, Unigram, SentencePiece, byte-level /
tokenizer-free models, vocabulary size, multilingual fertility, and how
tokenization affects arithmetic/code/reasoning. Built end to end: corpus
collection → relevance filtering → parsing/chunking → dedup → hybrid
retrieval → grounded multi-turn chat with citations and abstention.

See [WRITEUP.md](WRITEUP.md) for the full design write-up (corpus, dedup,
retrieval comparison, chat metrics, what I'd do with more time),
[DECISIONS.md](DECISIONS.md) for the complete dated log of every
assumption/tuning decision, [transcript.md](transcript.md) for a real chat
session, and [FAILURES.md](FAILURES.md) for the failure analysis.

## Setup (Windows, PowerShell)

> **Windows long-path note**: clone into a short path (e.g. `C:\dev\tokrag`).
> `torch`'s package includes deeply-nested filenames that exceed Windows'
> default 260-character path limit once combined with a long clone path
> (found via testing a clone several directories deep under a long
> session-temp path) — `pip install` fails with an `OSError` naming a
> `torch\include\...` file. Either use a short path, or enable [Windows Long
> Path support](https://pip.pypa.io/warnings/enable-long-paths).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
# then edit .env and add your GROQ_API_KEY (free at https://console.groq.com/keys)
```

Everything downstream of `collect` needs that key (query rewriting,
chat generation, the groundedness judge). `collect`, `parse`, and `index`
don't need it.

## Quickstart — sample mode (minutes, not hours)

A full run embeds ~35,000 chunks, which takes **over 3 hours** (see the
timing table below) — not practical for a quick check. `--sample N` runs
the *entire* pipeline (collect → parse → index → eval → chat) end to end
on a small slice of real papers in a few minutes, exercising the same code
path as the full run, just scoped small. This is also what the fresh-clone
reproducibility check (see WRITEUP.md) actually runs.

```powershell
python -m tokrag collect --sample 30   # caps collection to ~30 included papers (~2-4 min, mostly network-bound)
python -m tokrag parse                  # parses + chunks whatever collect found (~1 min for 30 papers)
python -m tokrag index --model bm25
python -m tokrag index --model BAAI/bge-small-en-v1.5 --out bge_small
python -m tokrag eval --retrieval hybrid --diversify
python -m tokrag chat
```

**Expect poor retrieval/eval numbers in sample mode.** The 36-question eval
set ([eval/questions.jsonl](eval/questions.jsonl)) targets specific
papers from the *full* corpus; a 30-paper sample almost certainly won't
contain most of those papers, so `eval` will show near-zero Recall@5/MRR
and `chat` will abstain on most questions. That's expected — sample mode
is for *verifying the pipeline runs end to end*, not for judging retrieval
or chat quality (use the full run, or the numbers already reported in
[WRITEUP.md](WRITEUP.md), for that).

## Full run

```powershell
python -m tokrag collect          # full collection pass (~20 query terms x2 sources + ACL bulk + canaries)
python -m tokrag parse            # parse + chunk every included row
python -m tokrag index --model bm25
python -m tokrag index --model BAAI/bge-small-en-v1.5 --out bge_small
python -m tokrag dedup            # canonical ids, survey flagging, diversification inputs
python -m tokrag eval --retrieval hybrid --diversify
python -m tokrag chat
```

### Full-run timing (measured on this project's corpus: 847 included rows, 34,779 chunks, CPU-only)

| Stage | Time | Notes |
|---|---|---|
| `collect` (full) | not separately stopwatched; tens of minutes | Dominated by the ~40MB ACL Anthology bulk download (one-time) and Semantic Scholar rate-limit backoff, hit repeatedly across ~20 query terms. Measured directly in the Phase 7 fresh-clone test: `--sample 30` took ~3 min, almost entirely S2 backoff — the full run issues ~4x the query terms plus the one-time ACL bulk pass, so expect low tens of minutes, not hours. |
| `parse` (full, 847 rows) | ~65-70 min, extrapolated | Measured directly (fresh-clone test, cold fetch cache): 2m26s for 30 papers (4.85s/paper) — live PDF/HTML downloads, not local compute, dominate this. A warm cache (papers already fetched once) is much faster, ~1.4s/paper measured separately. |
| `index --model bm25` | 1.1s | Negligible — no embedding involved. |
| `index --model BAAI/bge-small-en-v1.5` | 4,757s (**~79 min**) | 7.3 chunks/s, CPU-only, 20 threads. Checkpointed every 2,000 chunks (resumable). |
| `index --model allenai-specter` | 7,617s (**~127 min**) | 4.6 chunks/s — slower AND performs worse than bge-small on this eval set (see WRITEUP.md); kept only as an ablation, not the recommended index. |
| `eval` / `chat` | seconds per question | Dominated by Groq API latency, not local compute; every LLM call is cached to disk so repeat runs are near-instant. |

**Total for a full from-scratch run, building both dense indexes**: on the
order of **4-5 hours** end to end (collection + a cold-cache parse + both
embedding passes), with the two embedding passes alone accounting for
~3.5 hours of that. This is exactly why sample mode exists and why the fresh-clone
reproducibility check uses it instead of a full rebuild.

## Running eval and chat

```powershell
# Retrieval-only eval (Recall@5, MRR, duplicate-rate), any backend:
python -m tokrag eval --retrieval hybrid --index bge_small --model BAAI/bge-small-en-v1.5 --diversify
python -m tokrag eval --retrieval bm25
python -m tokrag eval --retrieval dense --index specter --model allenai-specter

# Full chat pipeline eval (abstention precision/recall, groundedness, citation
# accuracy, follow-up rewrite comparison) -- see tokrag/chat/eval_chat.py:
python -c "
from tokrag.eval.runner import DenseSearcher, BM25Searcher, HybridSearcher
from tokrag.chat.eval_chat import run_chat_eval
hybrid = HybridSearcher(DenseSearcher('bge_small', 'BAAI/bge-small-en-v1.5'), BM25Searcher())
print(run_chat_eval(hybrid))
"

# Interactive multi-turn chat:
python -m tokrag chat
```

Type a question; `exit`/Ctrl+C to quit. Follow-up questions are
automatically rewritten against conversation history (printed when the
rewrite changes the query) before retrieval runs.

## File map

| Deliverable | File(s) |
|---|---|
| Corpus manifest (every candidate, included/rejected + why, license, indexed) | [manifest.csv](manifest.csv), [manifest_summary.md](manifest_summary.md) |
| Design write-up (corpus, dedup, retrieval comparison, chat metrics, limitations, next steps) | [WRITEUP.md](WRITEUP.md) |
| Dated assumptions/decisions log | [DECISIONS.md](DECISIONS.md) |
| Dedup false-merge audit | [dedup_audit.md](dedup_audit.md) |
| Real chat transcript (5+ turns, unedited) | [transcript.md](transcript.md) |
| Failure analysis (3+ distinct cases) | [FAILURES.md](FAILURES.md) |
| Chat eval results (abstention, groundedness, citation, follow-up recall) | [phase6_chat_eval_results.json](phase6_chat_eval_results.json) |
| Eval question set | [eval/questions.jsonl](eval/questions.jsonl) |
| Pipeline code | [tokrag/collect/](tokrag/collect/), [tokrag/parse/](tokrag/parse/), [tokrag/index/](tokrag/index/), [tokrag/dedup/](tokrag/dedup/), [tokrag/eval/](tokrag/eval/), [tokrag/chat/](tokrag/chat/) |
| Tests | [tests/](tests/) (`python -m pytest`) |

## CLI reference

```powershell
python -m tokrag collect [--sample N]   # run the corpus collection pipeline (optionally capped to ~N included papers)
python -m tokrag parse [--limit N]      # parse + chunk collected papers (optionally capped to the first N included rows)
python -m tokrag index --model <name|bm25> [--out NAME] [--batch-size N]
python -m tokrag dedup                  # canonical ids, survey flagging, false-merge audit, diversification inputs
python -m tokrag eval --retrieval {dense,bm25,hybrid} [--index NAME] [--model NAME] [--k N] [--diversify] [--gold-rewrite]
python -m tokrag chat                   # interactive multi-turn chat loop
```
