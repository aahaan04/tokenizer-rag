"""Query rewriting: turns a follow-up question into a standalone query using
chat history, via the fast model (simple task, doesn't need the full-size
model). Every rewrite is returned alongside the original so the caller can
log/print it — debugging a bad retrieval starts with checking whether the
rewrite or the retrieval is at fault (see the Phase 5 finding in DECISIONS.md
about rewrites that over-anchor on the prior turn)."""

from __future__ import annotations

from tokrag.chat.llm_provider import call_llm, extract_text
from tokrag.config import load_config

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's follow-up question into a standalone question that "
    "can be understood WITHOUT the prior conversation, using the conversation "
    "history to resolve pronouns and references. Keep all the specificity of "
    "the original follow-up, but do not introduce extra facts or entities "
    "from the prior answer that the follow-up doesn't actually need — only "
    "resolve what the follow-up refers to. Output ONLY the rewritten "
    "question, nothing else."
)


def rewrite_query(question: str, history: list, model: str = None) -> str:
    """history: list of {"question": str, "answer": str} prior turns, oldest
    first. Returns the rewritten standalone question (falls back to the
    original question if history is empty or the LLM call fails)."""
    if not history:
        return question

    cfg = load_config()
    model = model or cfg.llm.groq_fast_model

    history_text = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history)
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Conversation so far:\n{history_text}\n\nFollow-up question: {question}\n\nStandalone question:",
        },
    ]
    try:
        resp = call_llm(messages, model=model, max_tokens=150, reasoning_effort="low")
    except Exception:
        return question  # graceful degradation: retrieval still runs on the raw follow-up
    rewritten = extract_text(resp).strip().strip('"')
    return rewritten if rewritten else question
