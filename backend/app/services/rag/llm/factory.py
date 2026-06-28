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

    else:
        raise ValueError(
            f"LLM_PROVIDER غير معروف: '{provider}'\n"
            f"القيم المقبولة: 'groq' أو 'qwen'"
        )