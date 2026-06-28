"""
Prompt Builder module for AraContract RAG system.
Builds structured prompts for Q&A and summary generation.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PromptBuilder:

    def build_qa_prompt(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        question: str
    ) -> str:
        """
        يبني prompt للإجابة على سؤال المستخدم.
        يستخدم parent_text للحصول على السياق الكامل للبند.

        Args:
            retrieved_chunks: ناتج rag_store.search()
            question: سؤال المستخدم بالعربية

        Returns:
            prompt جاهز للـ LLM
        """
        if not retrieved_chunks:
            raise ValueError("لا توجد بنود مسترجعة لبناء الـ prompt")

        if not question or not question.strip():
            raise ValueError("السؤال فارغ")

        # بناء قسم السياق من parent_text
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"البند {i} (رقم البند في العقد: {chunk['clause_index'] + 1}):\n"
                f"{chunk['parent_text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""أنت مساعد قانوني متخصص في تحليل العقود التجارية العربية.
مهمتك هي الإجابة على سؤال المستخدم بناءً حصراً على البنود المستخرجة من العقد أدناه.

التعليمات:
- أجب بالعربية الفصحى بشكل واضح ومختصر.
- استند فقط إلى النص المقدم ولا تستخدم أي معلومات خارجية.
- إذا لم تجد الإجابة في البنود المقدمة، قل: "هذه المعلومة غير موجودة في العقد المقدم."
- لا تخترع أو تفترض معلومات غير موجودة في النص.

البنود ذات الصلة من العقد:
===
{context}
===

سؤال المستخدم: {question.strip()}

الإجابة:"""

        logger.info(
            f"تم بناء QA prompt — {len(retrieved_chunks)} بند، "
            f"طول الـ prompt: {len(prompt)} حرف"
        )
        return prompt

    def build_summary_prompt(
        self,
        clauses: List[str],
        analyzed_clauses: List[Dict[str, Any]] = None
    ) -> str:
        """
        يبني prompt لتوليد الملخص التنفيذي للعقد (FR-6).
        يستخدم كل البنود + نتائج التصنيف إن توفرت.

        Args:
            clauses: List[str] — كل بنود العقد
            analyzed_clauses: ناتج المصنّف (اختياري) — يحتوي على
                              predicted_type_clause, predicted_risk_level, warning

        Returns:
            prompt جاهز للـ LLM
        """
        if not clauses:
            raise ValueError("قائمة البنود فارغة")

        # بناء نص العقد الكامل
        full_contract = "\n\n".join(
            f"البند {i + 1}: {clause}"
            for i, clause in enumerate(clauses)
        )

        # إضافة البنود عالية الخطورة إن توفرت نتائج التصنيف
        high_risk_section = ""
        if analyzed_clauses:
            high_risk = [
                c for c in analyzed_clauses
                if c.get("predicted_risk_level") == "high"
            ]
            if high_risk:
                high_risk_lines = []
                for c in high_risk:
                    warning = c.get("warning", "")
                    text_preview = c["text"][:100] + "..." if len(c["text"]) > 100 else c["text"]
                    line = f"- {text_preview}"
                    if warning:
                        line += f"\n  ⚠️ {warning}"
                    high_risk_lines.append(line)

                high_risk_section = f"""
البنود عالية الخطورة المكتشفة:
===
{chr(10).join(high_risk_lines)}
===
"""

        prompt = f"""أنت مساعد قانوني متخصص في تحليل العقود التجارية العربية.
مهمتك هي كتابة ملخص تنفيذي واضح ومختصر للعقد التالي.

التعليمات:
- اكتب الملخص باللغة العربية الفصحى.
- يجب أن يكون الملخص من 3 إلى 5 جمل فقط.
- يجب أن يتضمن الملخص:
  1. أطراف العقد ودورهم
  2. موضوع العقد والالتزامات الرئيسية
  3. مدة العقد إن وُجدت
  4. أبرز البنود عالية الخطورة إن وُجدت
- لا تذكر تفاصيل غير ضرورية.
- لا تستخدم تعداداً أو نقاطاً — اكتب فقرة متكاملة.
{high_risk_section}
نص العقد:
===
{full_contract}
===

الملخص التنفيذي:"""

        logger.info(
            f"تم بناء Summary prompt — {len(clauses)} بند، "
            f"طول الـ prompt: {len(prompt)} حرف"
        )
        return prompt


# Singleton
_prompt_builder_instance = None

def get_prompt_builder() -> PromptBuilder:
    global _prompt_builder_instance
    if _prompt_builder_instance is None:
        _prompt_builder_instance = PromptBuilder()
    return _prompt_builder_instance