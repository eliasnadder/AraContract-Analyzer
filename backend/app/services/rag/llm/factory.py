"""
LLM Factory — يقرأ LLM_PROVIDER من .env ويرجع الـ handler المناسب.

للتبديل:
    .env → LLM_PROVIDER=groq   (التطوير)
    .env → LLM_PROVIDER=qwen   (الإنتاج)
"""

import os
import logging
from app.services.rag.llm.base import BaseLLMHandler
from app.core.config import settings

logger = logging.getLogger(__name__)

_llm_instance = None


def get_llm() -> BaseLLMHandler:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = _create_handler()
    return _llm_instance


def _create_handler() -> BaseLLMHandler:
    provider = settings.LLM_PROVIDER.lower().strip()

    if provider == "groq":
        from app.services.rag.llm.groq_handler import GroqHandler
        logger.info("LLM Provider: Groq API")
        return GroqHandler()

    elif provider == "qwen":
        from app.services.rag.llm.qwen_handler import QwenHandler
        logger.info("LLM Provider: Qwen2.5-7B Local")
        return QwenHandler()

    elif provider == "ollama":
        from app.services.rag.llm.ollama_handler import OllamaHandler
        logger.info("LLM Provider: Ollama")
        return OllamaHandler()

    else:
        raise ValueError(
            f"LLM_PROVIDER غير معروف: '{provider}'\n"
            f"القيم المقبولة: 'groq' أو 'qwen' أو 'ollama'"
        )


def get_llm_status() -> dict:
    """Lightweight provider status check without forcing heavy local model loads."""
    provider = settings.LLM_PROVIDER.lower().strip()

    if provider == "groq":
        ready = bool(settings.GROQ_API_KEY)
        return {
            "provider": "groq",
            "ready": ready,
            "model": settings.GROQ_MODEL,
            "base_url": None,
            "details": "GROQ_API_KEY is set" if ready else "GROQ_API_KEY is missing",
        }

    if provider == "qwen":
        from pathlib import Path

        model_path = Path(settings.LLM_MODEL_PATH)
        ready = model_path.exists()
        return {
            "provider": "qwen",
            "ready": ready,
            "model": str(model_path),
            "base_url": None,
            "details": "Local model path exists" if ready else f"Missing local model path: {model_path}",
        }

    if provider == "ollama":
        from app.services.rag.llm.ollama_handler import OllamaHandler

        handler = OllamaHandler()
        ready = handler.is_ready()
        return {
            "provider": "ollama",
            "ready": ready,
            "model": settings.OLLAMA_MODEL,
            "base_url": settings.OLLAMA_BASE_URL,
            "details": (
                "Ollama is running and model is available"
                if ready
                else "Ollama unreachable or model is not pulled"
            ),
        }

    return {
        "provider": provider,
        "ready": False,
        "model": None,
        "base_url": None,
        "details": "Unsupported LLM provider",
    }