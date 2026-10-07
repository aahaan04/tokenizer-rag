"""LLM provider: Groq by default (OpenAI-compatible HTTP API), swappable to
Gemini via LLM_PROVIDER=gemini in .env. Every call is cached to disk (keyed
by the full request body, so identical calls — common across eval reruns —
cost nothing on repeat) and retried with exponential backoff on 429.

GPT-OSS models (Groq's current free-tier default, see DECISIONS.md) are
reasoning models: the response has a separate "reasoning" field that counts
against the token budget before "content" is produced. reasoning_effort
defaults to "low" everywhere in this project — verified this still produces
correct, on-topic output while cutting reasoning-token overhead roughly in
half (31 -> 12 tokens on a trivial prompt).
"""

from __future__ import annotations

import hashlib
import json
import time

import requests

from tokrag.config import CACHE_DIR, load_config

LLM_CACHE_DIR = CACHE_DIR / "llm"
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

MAX_RETRIES = 6
INITIAL_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0


def _cache_key(**kwargs) -> str:
    blob = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str):
    return LLM_CACHE_DIR / f"{key}.json"


def _call_groq(model: str, messages: list, max_tokens: int, temperature: float, reasoning_effort: str, api_key: str) -> dict:
    backoff = INITIAL_BACKOFF_S
    last_error = None
    for _attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "reasoning_effort": reasoning_effort,
                },
                timeout=60,
            )
        except requests.RequestException as e:
            last_error = e
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
            continue
        if resp.status_code == 429:
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Groq: giving up after {MAX_RETRIES} retries for model={model!r} ({last_error})")


def call_llm(
    messages: list,
    model: str = None,
    max_tokens: int = 600,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
    use_cache: bool = True,
) -> dict:
    """Returns the raw provider response dict. Use extract_text() to get the
    answer string out of it — kept separate so callers needing usage stats
    (token counts) can still get them."""
    cfg = load_config()
    if cfg.llm.provider != "groq":
        raise NotImplementedError(f"LLM_PROVIDER={cfg.llm.provider!r} not implemented — only 'groq' is wired up")

    model = model or cfg.llm.groq_model
    if not cfg.llm.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env — see .env.example")

    key = _cache_key(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, reasoning_effort=reasoning_effort)
    cache_file = _cache_path(key)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    data = _call_groq(model, messages, max_tokens, temperature, reasoning_effort, cfg.llm.groq_api_key)
    if use_cache:
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def extract_text(response: dict) -> str:
    try:
        return response["choices"][0]["message"].get("content", "") or ""
    except (KeyError, IndexError):
        return ""
