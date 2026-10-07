"""Runs the full chatbot over eval/questions.jsonl — replaying follow-up
conversations turn-by-turn through one Chatbot instance (so history is the
model's OWN real prior answers, not the hand-authored prior_turns text) —
and computes:

- Abstention precision/recall against the true answerable/unanswerable label
- Groundedness rate: an LLM judge checks each non-abstained answer's claims
  against ONLY its cited sources
- Citation accuracy: fraction of non-abstained answers with >=1 valid
  citation, i.e. parse_used_citations found something (the mapping-to-
  retrieved-chunks-only guarantee is structural, enforced in chatbot.py, not
  re-verified here)
- Follow-up retrieval compared two ways: the system's own rewritten query vs
  the gold_standalone_rewrite, per the brief
"""

from __future__ import annotations

import json
from collections import defaultdict

from tokrag.chat.chatbot import Chatbot
from tokrag.chat.llm_provider import call_llm, extract_text
from tokrag.config import PROJECT_ROOT
from tokrag.eval import metrics
from tokrag.eval.runner import load_questions

TRANSCRIPT_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "chat_eval_log.jsonl"

GROUNDEDNESS_JUDGE_PROMPT = (
    "You are checking whether an AI assistant's answer is grounded in the "
    "sources it cited. You will see the sources (numbered) and the answer. "
    "Judge ONLY whether the answer's factual claims are actually supported "
    "by the numbered sources the answer cites — not whether the answer is "
    "complete or well-written.\n\n"
    "Respond with exactly one word on the first line: GROUNDED or "
    "UNGROUNDED. On the next line, give a one-sentence reason."
)


def _judge_groundedness(answer: str, sources_block: str, model: str) -> tuple:
    messages = [
        {"role": "system", "content": GROUNDEDNESS_JUDGE_PROMPT},
        {"role": "user", "content": f"Sources:\n{sources_block}\n\nAnswer to check:\n{answer}"},
    ]
    resp = call_llm(messages, model=model, max_tokens=150, reasoning_effort="low")
    text = extract_text(resp).strip()
    first_line = text.split("\n")[0].strip().upper()
    grounded = first_line.startswith("GROUNDED")
    return grounded, text


def _build_sources_block_for_judge(turn) -> str:
    lines = []
    for i, c in enumerate(turn.retrieved, start=1):
        lines.append(f"[{i}] {c.get('title', '')} ({c.get('year', '')}), section: {c.get('section_title', '')}\n{c.get('text', '')[:800]}")
    return "\n\n".join(lines)


def run_chat_eval(searcher, model: str = None, judge_model: str = None) -> dict:
    from tokrag.config import load_config

    cfg = load_config()
    model = model or cfg.llm.groq_model
    judge_model = judge_model or cfg.llm.groq_fast_model

    questions = load_questions()
    by_id = {q["id"]: q for q in questions}

    # Find each conversation's turn-1 "seed" standalone question by matching
    # the stored prior_turns[0] question text.
    conv_members = defaultdict(list)
    for q in questions:
        if q["conversation_id"]:
            conv_members[q["conversation_id"]].append(q)
    conv_seed_id = {}
    for cid, members in conv_members.items():
        turn1_text = sorted(members, key=lambda m: m["turn"])[0]["prior_turns"][0]["question"]
        seed = next((q for q in questions if q["question"] == turn1_text and not q["conversation_id"]), None)
        conv_seed_id[cid] = seed["id"] if seed else None

    handled_ids = set()
    log_entries = []
    abstain_predictions = []  # (question_id, true_unanswerable, predicted_abstain)
    groundedness_results = []  # (question_id, grounded: bool)
    citation_results = []  # (question_id, n_citations)
    followup_system_vs_gold = []  # (question_id, recall_system, recall_gold)

    def _log_turn(q, turn, bot_answer_is_abstain):
        entry = {
            "id": q["id"],
            "type": q["type"],
            "question": turn.question,
            "rewritten_question": turn.rewritten_question,
            "answer": turn.answer,
            "citations": [{"index": c.index, "paper_title": c.paper_title, "year": c.year, "section": c.section} for c in turn.citations],
            "abstained": turn.abstained,
            "top_score": turn.top_score,
            "abstain_reason": turn.abstain_reason,
        }
        log_entries.append(entry)
        return entry

    # --- Standalone questions (not part of a conversation) ---
    for q in questions:
        if q["conversation_id"] or q["id"] in handled_ids:
            continue
        bot = Chatbot(searcher, model=model)
        turn = bot.ask(q["question"])
        handled_ids.add(q["id"])
        _log_turn(q, turn, turn.abstained)

        is_unanswerable = q["type"] == "unanswerable"
        abstain_predictions.append((q["id"], is_unanswerable, turn.abstained))

        if not turn.abstained:
            citation_results.append((q["id"], len(turn.citations)))
            sources_block = _build_sources_block_for_judge(turn)
            grounded, _reason = _judge_groundedness(turn.answer, sources_block, judge_model)
            groundedness_results.append((q["id"], grounded))

        # If this question seeds a conversation, continue it with the SAME bot.
        seeded_convs = [cid for cid, seed_id in conv_seed_id.items() if seed_id == q["id"]]
        for cid in seeded_convs:
            for follow_q in sorted(conv_members[cid], key=lambda m: m["turn"]):
                turn = bot.ask(follow_q["question"])
                handled_ids.add(follow_q["id"])
                _log_turn(follow_q, turn, turn.abstained)
                abstain_predictions.append((follow_q["id"], False, turn.abstained))  # all follow-ups here are answerable
                if not turn.abstained:
                    citation_results.append((follow_q["id"], len(turn.citations)))
                    sources_block = _build_sources_block_for_judge(turn)
                    grounded, _reason = _judge_groundedness(turn.answer, sources_block, judge_model)
                    groundedness_results.append((follow_q["id"], grounded))

                # System-rewrite vs gold-rewrite retrieval comparison for this follow-up.
                gold_rewrite = follow_q.get("gold_standalone_rewrite")
                if gold_rewrite:
                    golds = [(g["group_id"], g["supporting_span"]) for g in follow_q["gold"]]
                    sys_retrieved, _ = bot._retrieve(turn.rewritten_question)
                    gold_retrieved, _ = bot._retrieve(gold_rewrite)
                    r_sys = metrics.recall_at_k_multi(sys_retrieved, golds, k=bot.k)
                    r_gold = metrics.recall_at_k_multi(gold_retrieved, golds, k=bot.k)
                    followup_system_vs_gold.append((follow_q["id"], r_sys, r_gold))

    # Abstention precision/recall
    tp = sum(1 for _, true_u, pred_a in abstain_predictions if true_u and pred_a)
    fp = sum(1 for _, true_u, pred_a in abstain_predictions if not true_u and pred_a)
    fn = sum(1 for _, true_u, pred_a in abstain_predictions if true_u and not pred_a)
    tn = sum(1 for _, true_u, pred_a in abstain_predictions if not true_u and not pred_a)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    groundedness_rate = sum(1 for _, g in groundedness_results if g) / len(groundedness_results) if groundedness_results else None
    citation_rate = sum(1 for _, n in citation_results if n > 0) / len(citation_results) if citation_results else None

    TRANSCRIPT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRANSCRIPT_LOG_PATH.open("w", encoding="utf-8") as f:
        for e in log_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return {
        "n_questions_evaluated": len(abstain_predictions),
        "abstention": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall},
        "groundedness_rate": groundedness_rate,
        "n_groundedness_judged": len(groundedness_results),
        "citation_rate": citation_rate,
        "n_citation_checked": len(citation_results),
        "followup_system_vs_gold_rewrite": followup_system_vs_gold,
        "followup_mean_recall_system": sum(r for _, r, _ in followup_system_vs_gold) / len(followup_system_vs_gold) if followup_system_vs_gold else None,
        "followup_mean_recall_gold": sum(r for _, _, r in followup_system_vs_gold) / len(followup_system_vs_gold) if followup_system_vs_gold else None,
    }
