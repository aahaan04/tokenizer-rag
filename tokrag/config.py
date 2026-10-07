"""Central config: paths and provider settings loaded from .env."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = PROJECT_ROOT / "eval"
MANIFEST_PATH = PROJECT_ROOT / "manifest.csv"


@dataclass
class LLMConfig:
    provider: str
    groq_api_key: str
    groq_model: str
    groq_fast_model: str
    gemini_api_key: str
    gemini_model: str


@dataclass
class AppConfig:
    llm: LLMConfig
    semantic_scholar_api_key: str


def load_config() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(
            provider=os.environ.get("LLM_PROVIDER", "groq"),
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            # llama-3.3-70b-versatile was removed from Groq's free tier
            # (Enterprise-only as of 2026-08-16) — verified current free-tier
            # production models via console.groq.com/docs/models on
            # 2026-10-07: openai/gpt-oss-120b and openai/gpt-oss-20b, both
            # 1,000 RPM / 250,000 TPM free. See DECISIONS.md.
            groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            groq_fast_model=os.environ.get("GROQ_FAST_MODEL", "openai/gpt-oss-20b"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        ),
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY", ""),
    )


for _dir in (RAW_DIR, CACHE_DIR, PROCESSED_DIR, INDEX_DIR, EVAL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
