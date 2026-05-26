"""
Question-Answering router for contracts using RAG.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List
from app.models.schemas import QARequest, QAResponse, ErrorResponse
from app.services.rag_service import get_rag_answer

router = APIRouter()


@router.post(
    "/ask",
    response_model=QAResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def ask_question(request: QARequest):
    """
    Ask a question about the contract using RAG.

    - **contract_id**: Unique identifier for the contract session
    - **question**: The Arabic question to ask about the contract
    """
    try:
        answer, sources = get_rag_answer(
            contract_id=request.contract_id,
            question=request.question
        )
        return QAResponse(
            answer=answer,
            sources=sources,
            message="Answer generated successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during question answering: {str(e)}",
        )