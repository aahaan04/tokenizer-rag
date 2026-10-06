# tokrag — Talk to the literature

A RAG chatbot over research papers on tokenizers for LLMs: BPE, WordPiece,
Unigram, SentencePiece, byte-level/tokenizer-free models, vocabulary size,
multilingual fertility, and how tokenization affects arithmetic/code/reasoning.

Status: under active development. This README will be filled in as each phase
lands; see `DECISIONS.md` for the running log of design choices.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
# then edit .env and add your GROQ_API_KEY (free at https://console.groq.com/keys)
```

## CLI

```powershell
python -m tokrag collect   # run the corpus collection pipeline
python -m tokrag parse     # parse + chunk collected papers
python -m tokrag index     # build BM25 / dense / hybrid indexes
python -m tokrag eval      # run the evaluation set against an index config
python -m tokrag chat      # start the multi-turn chat loop
```

(Subcommands are stubs until the corresponding phase lands.)
