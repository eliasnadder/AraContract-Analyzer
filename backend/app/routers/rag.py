"""
RAG Router for AraContract Analyzer.
Endpoints:
    POST /rag/ingest      → استيعاب العقد وبناء الـ Vector Store
    POST /rag/ask         → سؤال عن محتوى العقد
    POST /rag/summarize   → ملخص تنفيذي للعقد
    DELETE /rag/session   → حذف الجلسة
"""

from fastapi import APIRouter, HTTPException, status

import logging

from app.services.rag.rag_pipeline import get_rag_pipeline
from app.schemas.rag_schemas import *

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])

# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="استيعاب العقد",
    description="يأخذ البنود المقسّمة ويبني الـ Vector Store في Qdrant"
)
async def ingest_contract(request: IngestRequest):
    try:
        pipeline = get_rag_pipeline()
        session_id = pipeline.ingest(
            clauses=request.clauses,
            session_id=request.session_id
        )
        return IngestResponse(
            session_id=session_id,
            clauses_count=len(request.clauses),
            message="تم استيعاب العقد بنجاح وبناء الـ Vector Store"
        )

    except Exception as e:
        logger.error(f"خطأ في الـ Ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشل استيعاب العقد: {str(e)}"
        )


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="سؤال عن العقد",
    description="يجيب على سؤال المستخدم بناءً على محتوى العقد فقط"
)
async def ask_question(request: AskRequest):
    try:
        pipeline = get_rag_pipeline()
        result = pipeline.ask(
            session_id=request.session_id,
            question=request.question,
            top_k=request.top_k
        )
        return AskResponse(
            answer=result["answer"],
            retrieved_clauses=[
                RetrievedClause(**clause)
                for clause in result["retrieved_clauses"]
            ],
            session_id=result["session_id"]
        )

    except ValueError as e:
        # جلسة غير موجودة أو سؤال فارغ
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"خطأ في الـ Q&A: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشل توليد الإجابة: {str(e)}"
        )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="ملخص تنفيذي",
    description="يولّد ملخصاً تنفيذياً للعقد من 3 إلى 5 جمل"
)
async def summarize_contract(request: SummarizeRequest):
    try:
        pipeline = get_rag_pipeline()
        summary = pipeline.summarize(
            clauses=request.clauses,
            analyzed_clauses=request.analyzed_clauses
        )
        return SummarizeResponse(
            summary=summary,
            clauses_count=len(request.clauses)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"خطأ في الـ Summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشل توليد الملخص: {str(e)}"
        )


@router.delete(
    "/session/{session_id}",
    response_model=DeleteSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="حذف الجلسة",
    description="يحذف الـ Vector Store الخاص بالجلسة عند انتهائها"
)
async def delete_session(session_id: str):
    try:
        pipeline = get_rag_pipeline()
        deleted = pipeline.cleanup(session_id)
        return DeleteSessionResponse(
            session_id=session_id,
            deleted=deleted,
            message="تم حذف الجلسة بنجاح" if deleted else "الجلسة غير موجودة"
        )

    except Exception as e:
        logger.error(f"خطأ في حذف الجلسة: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشل حذف الجلسة: {str(e)}"
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="فحص حالة الـ RAG",
)
async def rag_health():
    return {"status": "ok", "service": "RAG Pipeline"}