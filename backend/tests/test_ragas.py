"""
RAGAS Evaluation for AraContract RAG Pipeline — Non-LLM Version
لا يحتاج OpenAI API Key. يستخدم مقاييس تقليدية (BLEU, ROUGE, String Similarity).
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.extraction_service import extract_text_from_file
from app.routers.segment import segment_arabic_text
from app.services.rag.rag_pipeline import get_rag_pipeline

from ragas import evaluate
from ragas.metrics.collections import (
    BleuScore,
    RougeScore,
    NonLLMStringSimilarity,
    ExactMatch,
    DistanceMeasure,
)
from datasets import Dataset

CONTRACT_PATH = "tests/Contract-test.pdf"
DATASET_PATH  = "tests/rag_eval_dataset.json"


def build_ragas_dataset(pipeline, session_id: str, dataset: list) -> dict:
    user_inputs         = []
    responses           = []
    retrieved_contexts  = []
    references          = []

    for item in dataset:
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  ← السؤال: {question}")

        result = pipeline.ask(session_id, question, top_k=5)

        answer  = result["answer"]
        context = [c["parent_text"] for c in result["retrieved_clauses"]]

        user_inputs.append(question)
        responses.append(answer)
        retrieved_contexts.append(context)
        references.append(ground_truth)

        print(f"    الإجابة: {answer[:80]}...")
        print(f"    البنود المسترجعة: {len(context)}")

    return {
        "user_input":         user_inputs,
        "response":           responses,
        "retrieved_contexts": retrieved_contexts,
        "reference":          references,
    }


def run_ragas_evaluation():
    print("\n" + "="*60)
    print("RAGAS Evaluation — AraContract RAG Pipeline (Non-LLM)")
    print("="*60)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"\n✅ تم تحميل {len(dataset)} سؤال من الـ dataset")

    print("\n[1] استخراج النص وبناء الـ Vector Store...")
    extracted_text, _ = extract_text_from_file(CONTRACT_PATH)
    clauses           = segment_arabic_text(extracted_text)
    pipeline          = get_rag_pipeline()
    session_id        = pipeline.ingest(clauses)
    print(f"    ✅ session_id: {session_id}")

    print("\n[2] توليد الإجابات...")
    ragas_data = build_ragas_dataset(pipeline, session_id, dataset)

    print("\n[3] تشغيل RAGAS (Non-LLM Metrics)...")

    try:
        hf_dataset = Dataset.from_dict(ragas_data)

        # ✅ مقاييس Non-LLM — لا تحتاج OpenAI ولا أي API
        result = evaluate(
            dataset=hf_dataset,
            metrics=[
                BleuScore(),                                    # مقارنة n-gram
                RougeScore(rouge_type="rougeL", mode="fmeasure"),  # تداخل كلمات
                NonLLMStringSimilarity(
                    distance_measure=DistanceMeasure.LEVENSHTEIN
                ),                                              # تشابه نصي
                ExactMatch(),                                   # تطابق تام
            ],
        )

        print("\n" + "="*60)
        print("النتائج:")
        print("="*60)

        scores = result.to_pandas()

        metrics = {
            "bleu_score":              float(scores["bleu_score"].mean()),
            "rouge_score":             float(scores["rouge_score"].mean()),
            "non_llm_string_similarity": float(scores["non_llm_string_similarity"].mean()),
            "exact_match":             float(scores["exact_match"].mean()),
        }

        for metric, score in metrics.items():
            bar    = "█" * int(score * 20)
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
            print(f"  {status} {metric:<30} {score:.3f}  {bar}")

        overall = sum(metrics.values()) / len(metrics)
        print(f"\n  {'النتيجة الإجمالية':<30} {overall:.3f}")
        print("="*60)

        detail_cols = ["user_input", "bleu_score", "rouge_score",
                       "non_llm_string_similarity", "exact_match"]
        per_question = scores[detail_cols].rename(
            columns={"user_input": "question"}
        ).to_dict(orient="records")

        output = {
            "metrics":      metrics,
            "overall":      overall,
            "per_question": per_question
        }

        output_path = "tests/ragas_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n✅ النتائج محفوظة في: {output_path}")

    except Exception as e:
        print(f"\n❌ خطأ: {e}")
        print("\nنتائج بدون RAGAS (يدوية):")
        _manual_evaluation(ragas_data)

    finally:
        pipeline.cleanup(session_id)
        print("\n✅ تم تنظيف الجلسة")


def _manual_evaluation(ragas_data: dict):
    questions = ragas_data["user_input"]
    answers   = ragas_data["response"]
    contexts  = ragas_data["retrieved_contexts"]

    answered = sum(
        1 for a in answers
        if "غير موجودة" not in a and len(a) > 20
    )
    avg_ctx = sum(len(c) for c in contexts) / len(contexts)

    print(f"\n  الأسئلة الكلية:          {len(questions)}")
    print(f"  إجابات ذات محتوى:        {answered}/{len(questions)}")
    print(f"  متوسط البنود المسترجعة:  {avg_ctx:.1f}")
    print(f"  نسبة الإجابة:            {answered/len(questions)*100:.0f}%")

    print("\n--- تفاصيل كل سؤال ---")
    for i, (q, a, c) in enumerate(zip(questions, answers, contexts), 1):
        has_answer = "غير موجودة" not in a and len(a) > 20
        status     = "✅" if has_answer else "❌"
        print(f"\n  [{i}] {status} {q}")
        print(f"       الإجابة: {a[:100]}...")
        print(f"       البنود:  {len(c)}")


if __name__ == "__main__":
    run_ragas_evaluation()