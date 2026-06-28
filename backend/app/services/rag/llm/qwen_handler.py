"""Qwen2.5-7B Local Handler — للإنتاج النهائي"""

import logging
from pathlib import Path
from typing import Optional
from app.services.rag.llm.base import BaseLLMHandler
from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(settings.LLM_MODEL_PATH)



class QwenHandler(BaseLLMHandler):

    def __init__(self, model_path: Optional[str] = None):
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig
            )
            import torch
        except ImportError:
            raise ImportError("transformers أو torch غير مثبتة")

        self._path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not self._path.exists():
            raise FileNotFoundError(
                f"نموذج Qwen غير موجود في: {self._path}"
            )

        logger.info(f"تحميل Qwen من: {self._path} ...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._path),
            trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._path),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self._model.eval()
        logger.info("Qwen Handler جاهز")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            import torch

            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self._tokenizer(
                text,
                return_tensors="pt"
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(
                generated,
                skip_special_tokens=True
            ).strip()

        except Exception as e:
            logger.error(f"خطأ في Qwen inference: {e}")
            raise RuntimeError(f"فشل توليد الإجابة عبر Qwen: {e}")

    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None