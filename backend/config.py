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
LLM_PROVIDER = "groq"

# API-based multimodal generation models (GROQ).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

KEYS_FILE = DATA_DIR / "keys.json"


def load_persistent_keys():
    global GROQ_API_KEY
    if KEYS_FILE.exists():
        try:
            import json

            data = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
            if data.get("groq"):
                GROQ_API_KEY = data["groq"]
                os.environ["GROQ_API_KEY"] = data["groq"]
            elif data.get("api_key"):
                GROQ_API_KEY = data["api_key"]
                os.environ["GROQ_API_KEY"] = data["api_key"]
        except Exception:
            pass


load_persistent_keys()

# Retrieval defaults.
TEXT_TOP_K = 5
VISUAL_TOP_K = 5
HYBRID_TOP_K = 5

# Naive Hybrid RRF baseline. Keep these fixed for reproducible comparisons.
HYBRID_RRF_TEXT_WEIGHT = 0.3
HYBRID_RRF_VISUAL_WEIGHT = 0.7
HYBRID_RRF_K = 60

# Improved Hybrid candidate union.
HYBRID_CANDIDATE_UNION_SIZE = 10
VISUAL_RERANK_CANDIDATE_SIZE = 10

# Text BM25 dominance heuristics.
TEXT_BM25_DOMINANCE_RATIO = 2.0
TEXT_BM25_DOMINANCE_MIN_SCORE = 5.0
TEXT_BM25_DOMINANCE_BONUS = 0.025

# Rendering.
PAGE_IMAGE_DPI = 144
VISUAL_RERANK_DPI = 216

# Generation / transport.
GENERATION_MAX_TOKENS = 500
GENERATION_TEMPERATURE = 0.0
OPENAI_REQUEST_TIMEOUT_SECONDS = 60

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

VALID_PROVIDERS = ("groq",)

