"""Ollama Handler — local LLM via Ollama HTTP API."""

import logging
from app.services.rag.llm.base import BaseLLMHandler
from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaHandler(BaseLLMHandler):
    def __init__(self):
        self._base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self._model = settings.OLLAMA_MODEL

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx package is required for Ollama integration") from exc

        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "")
            if not text:
                raise RuntimeError("Ollama returned an empty response")
            return text.strip()
        except Exception as exc:
            logger.error(f"Ollama generation failed: {exc}")
            raise RuntimeError(f"Failed to generate answer via Ollama: {exc}") from exc

    def is_ready(self) -> bool:
        try:
            import httpx
        except ImportError:
            return False

        tags_url = f"{self._base_url}/api/tags"

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(tags_url)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            available = {m.get("name", "") for m in models}
            return self._model in available or f"{self._model}:latest" in available
        except Exception:
            return False
