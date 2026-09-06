"""Centralized configuration for the NIT Jamshedpur Document RAG prototype."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PAGES_DIR = PROCESSED_DIR / "pages"
TEXT_DIR = PROCESSED_DIR / "text"
METADATA_DIR = PROCESSED_DIR / "metadata"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EVALUATION_DIR = DATA_DIR / "evaluation"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Local retrieval models.
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VISUAL_EMBEDDING_MODEL = "vidore/colqwen2-v1.0"

# LLM Provider default
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")

# API-based multimodal generation models.
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-2-vision-128k")
XAI_BASE_URL = "https://api.x.ai/v1"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

KEYS_FILE = DATA_DIR / "keys.json"


def load_persistent_keys():
    global XAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, LLM_PROVIDER
    if KEYS_FILE.exists():
        try:
            import json

            data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
            if data.get("xai"):
                XAI_API_KEY = data["xai"]
                os.environ["XAI_API_KEY"] = data["xai"]
            if data.get("groq"):
                GROQ_API_KEY = data["groq"]
                os.environ["GROQ_API_KEY"] = data["groq"]
            if data.get("gemini"):
                GEMINI_API_KEY = data["gemini"]
                os.environ["GEMINI_API_KEY"] = data["gemini"]
            if data.get("provider"):
                LLM_PROVIDER = data["provider"]
        except Exception:
            pass


def get_llm_provider() -> str:
    if GROQ_API_KEY:
        return "groq"
    if XAI_API_KEY:
        return "xai"
    if GEMINI_API_KEY:
        return "gemini"
    return "groq"


load_persistent_keys()
if not LLM_PROVIDER:
    LLM_PROVIDER = get_llm_provider()

# Retrieval defaults.
TEXT_TOP_K = 5
VISUAL_TOP_K = 5
HYBRID_TOP_K = 5

# Naive Hybrid RRF baseline. Keep these fixed for reproducible comparisons.
HYBRID_RRF_TEXT_WEIGHT = 0.3
HYBRID_RRF_VISUAL_WEIGHT = 0.7
HYBRID_RRF_K = 60

# Improved Hybrid candidate union. App/demo can use 10; evaluation can use all pages.
HYBRID_CANDIDATE_UNION_SIZE = 10
VISUAL_RERANK_CANDIDATE_SIZE = 10

# Current text heuristic remains explicit rather than hidden.
TEXT_BM25_DOMINANCE_RATIO = 2.0
TEXT_BM25_DOMINANCE_MIN_SCORE = 5.0
TEXT_BM25_DOMINANCE_BONUS = 0.025

# Rendering. Existing 144-DPI corpus images are retained; high-res reranking uses cached
# in-memory resizing instead of rewriting the corpus on every query.
PAGE_IMAGE_DPI = 144
VISUAL_RERANK_DPI = 216

# Generation / transport.
GENERATION_MAX_TOKENS = 500
GENERATION_TEMPERATURE = 0.0
OPENAI_REQUEST_TIMEOUT_SECONDS = 60
GEMINI_REQUEST_TIMEOUT_SECONDS = 60

# A score-only rejection threshold is deliberately treated as a heuristic, not proof that
# evidence is absent. Do not label it as calibrated confidence.
RETRIEVAL_REJECTION_THRESHOLD = 0.0

VALID_MODES = (
    "text",
    "text_baseline",
    "visual",
    "visual_colqwen2",
    "hybrid",
    "hybrid_rrf_baseline",
    "hybrid_reranked",
)

VALID_PROVIDERS = ("xai", "groq", "gemini")
