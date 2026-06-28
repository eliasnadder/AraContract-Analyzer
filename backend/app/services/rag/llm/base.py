"""Abstract interface — كل الكود خارج مجلد llm يتعامل مع هذا فقط"""

from abc import ABC, abstractmethod


class BaseLLMHandler(ABC):

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        يأخذ prompt ويرجع نص.

        Args:
            prompt: الـ prompt الجاهز من PromptBuilder
            max_tokens: الحد الأقصى للـ tokens

        Returns:
            نص الإجابة
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """يتحقق أن النموذج جاهز"""
        pass