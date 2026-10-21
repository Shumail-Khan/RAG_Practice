"""
Configuration Module for RAG Practice
=====================================
Centralizes all settings, environment variables, and connections.
Using a dedicated config module prevents hardcoding values throughout
the codebase and ensures security by keeping sensitive credentials in `.env`.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles to prevent charmap encoding errors
if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file in the project root
load_dotenv(BASE_DIR / ".env")

# ==============================================================================
# Google Gemini API Settings
# ==============================================================================
# The Gemini API key is obtained for free from Google AI Studio:
# https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Recommended Gemini generation model (e.g., gemini-2.5-flash or gemini-3.7-flash)
# "gemini-2.5-flash" is fast, accurate, and cost-effective for RAG synthesis tasks.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ==============================================================================
# PostgreSQL + pgvector Database Settings
# ==============================================================================
# These match the credentials defined in `docker-compose.yml`
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5433"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rag_password")

# Direct connection URI string
DATABASE_URI = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ==============================================================================
# Hugging Face Embedding Model Settings
# ==============================================================================
# "sentence-transformers/all-MiniLM-L6-v2" is a popular, lightweight open-source
# embedding model that runs completely locally on CPU/GPU without API costs or limits.
# Output dimension is 384 numbers (floats).
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# ==============================================================================
# Chunking Defaults
# ==============================================================================
# Chunk size: Maximum number of characters per text segment
# Chunk overlap: Characters shared between consecutive chunks to avoid breaking context
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def validate_gemini_key() -> None:
    """Validate that GEMINI_API_KEY is present before making LLM calls."""
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "your_gemini_api_key_here":
        raise ValueError(
            "\n[!] GEMINI_API_KEY is not configured.\n"
            "Please create a `.env` file (copy from `.env.example`) and add your key.\n"
            "Get a free Gemini API key at: https://aistudio.google.com/app/apikey"
        )
