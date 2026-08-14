"""
Summarization router for contracts.
Generates executive summaries using an LLM.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from app.models.schemas import SummarizationRequest, SummarizationResponse, ErrorResponse
from app.services.summary_service import generate_contract_summary
from app.core.auth import get_current_user

router = APIRouter()


@router.post(
    "/summarize",
    response_model=SummarizationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def summarize_contract(
    request: SummarizationRequest,
    uid: str = Depends(get_current_user)
):
    """
    Generate an executive summary of the contract.

    - **text**: The full contract text
    - **classified_clauses**: List of classified clauses with types and risk levels
    """
    try:
        summary = generate_contract_summary(
            text=request.text,
            classified_clauses=request.classified_clauses
        )
        return SummarizationResponse(
            summary=summary,
            message="Executive summary generated successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during summarization: {str(e)}",
        )