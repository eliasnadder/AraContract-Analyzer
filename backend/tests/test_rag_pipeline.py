"""
اختبار RAG Pipeline كامل قبل بناء الـ API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.extraction_service import extract_text_from_file
from app.routers.segment import segment_arabic_text
from app.services.rag.rag_pipeline import get_rag_pipeline

# ── الملف التجريبي ─────────────────────────────────────────────────────────
CONTRACT_PATH = "tests/Contract-test.pdf"

def test_full_pipeline():

    print("\n" + "="*60)
    print("اختبار RAG Pipeline الكامل")
    print("="*60)

    # ── Step 1: Extraction + Segmentation ─────────────────────────────────
    print("\n[1] استخراج النص وتقسيمه...")
    extracted_text, is_scanned = extract_text_from_file(CONTRACT_PATH)
    clauses = segment_arabic_text(extracted_text)
    print(f"    ✅ عدد البنود: {len(clauses)}")
    print(f"    ✅ ممسوح: {is_scanned}")

    # ── Step 2: Ingestion ──────────────────────────────────────────────────
    print("\n[2] Ingestion في Qdrant...")
    pipeline = get_rag_pipeline()
    session_id = pipeline.ingest(clauses)
    print(f"    ✅ session_id: {session_id}")

    # ── Step 3: Q&A ────────────────────────────────────────────────────────
    print("\n[3] اختبار الأسئلة...")
    questions = [
        "ما هي مدة العقد؟",
        "من يتحمل مسؤولية الحوادث؟",
        "كيف تُقسم الأرباح؟",
        "ما هي شروط إنهاء العقد؟",
    ]

    for question in questions:
        print(f"\n    السؤال: {question}")
        result = pipeline.ask(session_id, question)
        print(f"    الإجابة: {result['answer']}")
        print(f"    البنود المستخدمة: {len(result['retrieved_clauses'])}")
        for clause in result['retrieved_clauses']:
            print(
                f"      - البند {clause['clause_index'] + 1} "
                f"(score: {clause['score']})"
            )

    # ── Step 4: Summary ────────────────────────────────────────────────────
    print("\n[4] اختبار الملخص التنفيذي...")
    summary = pipeline.summarize(clauses)
    print(f"    الملخص:\n    {summary}")

    # ── Step 5: Cleanup ────────────────────────────────────────────────────
    print("\n[5] تنظيف الجلسة...")
    deleted = pipeline.cleanup(session_id)
    print(f"    ✅ تم الحذف: {deleted}")

    print("\n" + "="*60)
    print("✅ اكتمل الاختبار بنجاح")
    print("="*60)

def test_raw_search_debug():
    print("\n" + "="*60)
    print("تشخيص: فحص نتائج similarity_search الخام (top 10)")
    print("="*60)

    extracted_text, is_scanned = extract_text_from_file(CONTRACT_PATH)
    clauses = segment_arabic_text(extracted_text)

    pipeline = get_rag_pipeline()
    session_id = pipeline.ingest(clauses)
    collection_name = f"contract_{session_id}"

    from app.services.rag.embedder import get_embedder
    embedder = get_embedder()

    questions = [
        "من يتحمل مسؤولية الحوادث؟",
        "ما هي شروط إنهاء العقد؟",
        "كيف تُقسم الأرباح؟",
    ]

    for question in questions:
        print(f"\n--- السؤال: {question} ---")
        query_vector = embedder.embed_query(question)

        raw_results = pipeline._retriever._service.similarity_search(
            collection_name, query_vector
        )

        for i, r in enumerate(raw_results, 1):
            preview = r["text"].replace("\n", " ")
            print(
                f"  [{i}] score={r['score']:.4f} | "
                f"clause_index={r['clause_index']} | {preview}..."
            )

    pipeline.cleanup(session_id)


if __name__ == "__main__":
    test_full_pipeline()
    test_raw_search_debug()
