"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings
from typing import List
import secrets
from pydantic import ConfigDict


class Settings(BaseSettings):
    # model_config = ConfigDict(env_file=".env", case_sensitive=True)
    # Application settings
    APP_NAME: str = "AraContract Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000", "http://127.0.0.1:3000"]

    # File upload settings
    MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20 MB
    UPLOAD_DIR: str = "./uploads"
    ALLOWED_EXTENSIONS: List[str] = [
        ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

    # Model paths
    CLASSIFIER_MODEL_PATH: str = "../models/checkpoints/aracontract_classifier.pt"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"

    # Processing settings
    CHUNK_SIZE: int = 500  # For RAG text chunking
    CHUNK_OVERLAP: int = 50  # For RAG text chunking
    MAX_SEQUENCE_LENGTH: int = 512  # For classifier

    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
