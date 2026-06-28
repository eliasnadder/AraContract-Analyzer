"""Groq API Handler — للتطوير والاختبار"""

import os
import logging
from typing import Optional
from app.services.rag.llm.base import BaseLLMHandler
from app.core.config import settings

logger = logging.getLogger(__name__)


class GroqHandler(BaseLLMHandler):

    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("شغّل: pip install groq")

        self._api_key = api_key or settings.GROQ_API_KEY
        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY غير موجود — أضفه في .env"
            )

        self._model = model or settings.GROQ_MODEL
        self._client = Groq(api_key=self._api_key)
        logger.info(f"Groq Handler جاهز — النموذج: {self._model}")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"خطأ في Groq API: {e}")
            raise RuntimeError(f"فشل توليد الإجابة عبر Groq: {e}")

    def is_ready(self) -> bool:
        return self._client is not None and self._api_key is not None