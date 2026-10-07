"""Grounded multi-turn chatbot: hybrid retrieval + Phase 4 diversification,
query rewriting for follow-ups, citations mapped to retrieved chunks only,
and abstention combining a retrieval-score threshold with an instruction-
level check (the LLM is told to say plainly when sources don't support an
answer, and its own refusal is also detected directly as a second signal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tokrag.chat.llm_provider import call_llm, extract_text
from tokrag.chat.rewrite import rewrite_query
from tokrag.config import load_config
from tokrag.dedup.diversify import diversify
from tokrag.eval.runner import load_canonical_ids, load_survey_group_ids
from tokrag.index.chunks import load_chunks

OVERFETCH_K = 50
TOP_K = 5

# Tuned against the eval set's answerable vs. unanswerable questions' top-1
# hybrid RRF score distributions (see DECISIONS.md for the full numbers —
# "tuned on the eval set" disclosure, per the brief). The two distributions
# overlap substantially (answerable: 0.023-0.033, unanswerable: 0.022-0.032)
# — RRF's rank-based fused score has no strong absolute "no real match"
# signal, so this threshold is set conservatively to catch only the single
# most extreme low-score case (q24, 0.0216) rather than trying to split the
# overlapping middle. The instruction-level check below (does the model
# itself say the sources don't answer the question) does the real work for
# everything else — exactly why the brief asks for both signals combined,
# not score alone.
ABSTAIN_SCORE_THRESHOLD = 0.0225

ABSTAIN_PHRASES = (
    "doesn't contain", "does not contain", "don't have enough information",
    "do not have enough information", "cannot answer", "can't answer",
    "not covered in the", "no information", "unable to answer",
    "corpus doesn't", "corpus does not",
)

ANSWER_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about tokenizer "
    "research for large language models, using ONLY the numbered source "
    "excerpts provided below.\n\n"
    "First, carefully read EVERY numbered source in full before deciding "
    "whether they answer the question — the answer may be in any one of "
    "them, not necessarily the first or highest-ranked. (Found via manual "
    "testing: without this instruction, the model sometimes concluded a "
    "clearly-present fact wasn't in the sources because it didn't check all "
    "of them — see DECISIONS.md.)\n\n"
    "Follow these rules strictly:\n"
    "1. Cite every factual claim using ASCII square brackets exactly like "
    "[1] or [2] — never full-width brackets, parentheses, or any other "
    "citation style. N is the source number it came from. Only cite sources "
    "that are actually provided — never invent a source number.\n"
    "2. Only say the sources don't contain enough information after you have "
    "checked ALL of them and genuinely found nothing relevant — say so "
    "plainly: \"The corpus doesn't contain enough information to answer "
    "this.\" Do not guess, and do not use outside knowledge not present in "
    "the sources.\n"
    "3. Be concise: 2-5 sentences unless the question clearly needs more."
)

# Matches both the requested ASCII [N] and full-width 【N】, which some
# Groq/GPT-OSS responses use despite the prompt's explicit instruction not to
# — found via manual testing (see DECISIONS.md). Parsing both is a safety
# net; the prompt fix is the primary mitigation.
_CITATION_RE = re.compile(r"[\[【](\d+)[\]】]")


def detect_model_abstention(answer: str) -> bool:
    """Instruction-level abstention check: did the model itself say the
    sources don't support an answer? (the second of the two signals the
    brief asks for, alongside the retrieval-score threshold)."""
    lower = answer.lower()
    return any(p in lower for p in ABSTAIN_PHRASES)


def parse_used_citations(answer: str, citations: list) -> list:
    """Citations map to retrieved chunks only: parses which [N] (or the
    full-width 【N】 some responses use despite the prompt) the answer
    actually cited, and drops any reference outside the provided source
    range rather than trusting the model blindly."""
    used_indices = {int(m) for m in _CITATION_RE.findall(answer)}
    return [c for c in citations if c.index in used_indices and 1 <= c.index <= len(citations)]


@dataclass
class Citation:
    index: int
    paper_title: str
    year: str
    section: str
    paper_id: str


@dataclass
class ChatTurn:
    question: str
    rewritten_question: str
    retrieved: list
    answer: str
    citations: list
    abstained: bool
    top_score: float
    abstain_reason: str = ""


class Chatbot:
    def __init__(self, searcher, k: int = TOP_K, model: str = None, abstain_threshold: float = ABSTAIN_SCORE_THRESHOLD):
        self.searcher = searcher
        self.k = k
        cfg = load_config()
        self.model = model or cfg.llm.groq_model
        self.abstain_threshold = abstain_threshold
        self.history: list = []  # [{"question":..., "answer":...}]
        self._chunks = load_chunks()
        self._survey_group_ids = load_survey_group_ids()
        self._canonical_ids = load_canonical_ids()

    def _retrieve(self, query: str) -> tuple:
        hits = self.searcher.search_with_scores(query, OVERFETCH_K)
        score_by_idx = {i: s for i, s in hits}
        retrieved = [{**self._chunks[i], **self.searcher.meta[i], "_score": s} for i, s in hits]
        diversified = diversify(
            retrieved, k=self.k, survey_group_ids=self._survey_group_ids, canonical_ids=self._canonical_ids, max_per_group=2
        )
        top_score = diversified[0]["_score"] if diversified else 0.0
        return diversified, top_score

    def _build_sources_block(self, retrieved: list) -> tuple:
        lines = []
        citations = []
        for i, c in enumerate(retrieved, start=1):
            title, year, section = c.get("title", ""), c.get("year", ""), c.get("section_title", "")
            lines.append(f"[{i}] {title} ({year}), section: {section}\n{c.get('text', '')[:800]}")
            citations.append(Citation(index=i, paper_title=title, year=year, section=section, paper_id=c.get("paper_id", "")))
        return "\n\n".join(lines), citations

    def ask(self, question: str) -> ChatTurn:
        rewritten = rewrite_query(question, self.history, model=load_config().llm.groq_fast_model) if self.history else question

        retrieved, top_score = self._retrieve(rewritten)
        sources_block, citations = self._build_sources_block(retrieved)

        if top_score < self.abstain_threshold:
            answer = "The corpus doesn't contain enough information to answer this."
            turn = ChatTurn(question, rewritten, retrieved, answer, [], True, top_score, abstain_reason="retrieval_score_below_threshold")
            self.history.append({"question": question, "answer": answer})
            return turn

        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{sources_block}\n\nQuestion: {rewritten}"},
        ]
        resp = call_llm(messages, model=self.model, max_tokens=500, reasoning_effort="low")
        answer = extract_text(resp).strip()
        model_abstained = detect_model_abstention(answer)

        # Retry once on self-abstention: manual testing found GPT-OSS-120B is
        # genuinely non-deterministic at temperature=0 for this task — 3
        # identical calls (same prompt, same sources, one containing the
        # answer verbatim) produced 1 correct answer and 2 false abstentions.
        # A single retry (fresh, uncached) meaningfully raises the odds of
        # getting the correct draw without unbounded cost — see DECISIONS.md.
        if model_abstained:
            resp = call_llm(messages, model=self.model, max_tokens=500, reasoning_effort="low", use_cache=False)
            retry_answer = extract_text(resp).strip()
            if not detect_model_abstention(retry_answer):
                answer = retry_answer
                model_abstained = False

        used_citations = parse_used_citations(answer, citations)

        turn = ChatTurn(
            question=question,
            rewritten_question=rewritten,
            retrieved=retrieved,
            answer=answer,
            citations=used_citations,
            abstained=model_abstained,
            top_score=top_score,
            abstain_reason="model_self_assessment" if model_abstained else "",
        )
        self.history.append({"question": question, "answer": answer})
        return turn
