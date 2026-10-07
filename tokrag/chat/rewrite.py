"""Query rewriting: turns a follow-up question into a standalone query using
chat history, via the fast model (simple task, doesn't need the full-size
model). Every rewrite is returned alongside the original so the caller can
log/print it — debugging a bad retrieval starts with checking whether the
rewrite or the retrieval is at fault (see the Phase 5 finding in DECISIONS.md
about rewrites that over-anchor on the prior turn).

Checkpoint-6 revision (round 1, see DECISIONS.md): the original version fed
the FULL prior answer text and the entire history, and only asked for
pronoun resolution. Measured Recall@5 of 0.14 (system rewrite) vs 0.29 (gold
rewrite) on the eval set's follow-ups showed this under-performs. Now uses
only the MOST RECENT turn (not full history — a 3-turn eval conversation
exists, and turn 3 should anchor on turn 2, not turn 1), a truncated
"summary" of the prior answer (first ~200 chars, cheaper than a second LLM
call and sufficient for topic context), and the titles of papers the prior
answer actually cited — then explicitly instructs the rewriter to NAME the
topic/entity, not just swap in a pronoun."""

from __future__ import annotations

from tokrag.chat.llm_provider import call_llm, extract_text
from tokrag.config import load_config

ANSWER_SUMMARY_CHARS = 200

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's follow-up question into a standalone question that "
    "can be understood WITHOUT the prior conversation. Use the previous "
    "question, a summary of its answer, and the papers that answer cited to "
    "figure out what the follow-up refers to.\n\n"
    "Critically: resolve every pronoun and vague reference (\"that\", \"it\", "
    "\"those\", \"the other paper\", \"that approach\") by NAMING the actual "
    "topic, paper title, or entity it refers to — do not just substitute one "
    "pronoun for another or lightly paraphrase. The rewritten question must "
    "be fully understandable to someone who has NOT seen the conversation. "
    "Keep all the specificity of the original follow-up, but do not "
    "introduce extra facts the follow-up doesn't actually need. Output ONLY "
    "the rewritten question, nothing else."
)


def rewrite_query(question: str, history: list, model: str = None) -> str:
    """history: list of {"question": str, "answer": str, "cited_papers": list[str]}
    prior turns, oldest first. Only the most recent turn is used (see module
    docstring). Returns the rewritten standalone question (falls back to the
    original question if history is empty or the LLM call fails)."""
    if not history:
        return question

    cfg = load_config()
    model = model or cfg.llm.groq_fast_model

    prior = history[-1]
    answer_summary = prior["answer"][:ANSWER_SUMMARY_CHARS]
    papers = prior.get("cited_papers") or []
    papers_line = f"Papers cited in that answer: {'; '.join(papers)}\n" if papers else ""
    context = f"Previous question: {prior['question']}\nSummary of its answer: {answer_summary}\n{papers_line}"

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{context}\nFollow-up question: {question}\n\nStandalone question:",
        },
    ]
    try:
        resp = call_llm(messages, model=model, max_tokens=150, reasoning_effort="low")
    except Exception:
        return question  # graceful degradation: retrieval still runs on the raw follow-up
    rewritten = extract_text(resp).strip().strip('"')
    return rewritten if rewritten else question
