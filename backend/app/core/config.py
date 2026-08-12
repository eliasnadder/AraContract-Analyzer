"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings
from typing import List
import secrets
from pathlib import Path

# المسار الجذر للـ backend
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "AraContract Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

    # File upload settings
    MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20 MB
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_EXTENSIONS: List[str] = [
        ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"
    ]

    # Model paths
    CLASSIFIER_MODEL_PATH: str = "../models/checkpoints/aracontract_classifier.pt"
    MAX_SEQUENCE_LENGTH: int = 512

    # ── RAG — Embedding ────────────────────────────────────────────────────
    EMBEDDING_MODEL_PATH: str = str(
        BASE_DIR / "models_local" / "paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_VECTOR_SIZE: int = 384

    # ── RAG — Chunking ─────────────────────────────────────────────────────
    CHUNK_MAX_CHARS: int = 450
    CHUNK_OVERLAP_CHARS: int = 50

    # ── RAG — Cross-Encoder ────────────────────────────────────────────────
    CROSS_ENCODER_MODEL_PATH: str = str(
        BASE_DIR / "models_local" / "cross-encoder-ms-marco-MiniLM-L-6-v2"
    )

    # ── RAG — Retrieval ────────────────────────────────────────────────────
    RETRIEVAL_SCORE_THRESHOLD: float = 0.45
    RETRIEVAL_INITIAL_TOP_K: int = 15
    RETRIEVAL_FINAL_TOP_K: int = 3

    # ── RAG — LLM ──────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "groq"           # groq | qwen
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    LLM_MODEL_PATH: str = str(
        BASE_DIR / "models_local" / "Qwen2.5-7B-Instruct"
    )

    # ── RAG — Qdrant ───────────────────────────────────────────────────────
    QDRANT_MODE: str = "memory"          # memory | url
    QDRANT_URL: str = "http://localhost:6333"

    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()