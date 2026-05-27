"""
Unified Analysis router for contracts (FR-1 to FR-6).
Accepts a contract file, runs the entire pipeline, and returns the full analysis response.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import logging

from app.core.config import settings
from app.models.schemas import AnalysisResponse, AnalyzedClause, ErrorResponse
from app.services.extraction_service import extract_text_from_file
from app.routers.segment import segment_arabic_text
from app.routers.classify import get_model
from app.services.summary_service import generate_contract_summary
from app.models.labels import TYPE_DISPLAY_NAMES_AR, RISK_DISPLAY_NAMES_AR

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def analyze_contract(file: UploadFile = File(...)):
    """
    Accepts a contract file, runs extraction -> segmentation -> classification -> warnings -> summary,
    and returns a unified analysis response.
    """
    # Validate file size
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the limit of {settings.MAX_FILE_SIZE // (1024*1024)} MB",
        )

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    # Save the uploaded file temporarily
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        # 1. Extract text from the file (FR-1)
        extracted_text, is_scanned = extract_text_from_file(str(file_path))
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="لم يتم العثور على أي نص في الملف المرفوع. يرجى التأكد من جودة الملف أو اختيار ملف آخر.",
            )

        # 2. Segment text into clauses (FR-2)
        clauses = segment_arabic_text(extracted_text)
        if not clauses:
            # If no clauses could be segmented, fallback to treating the entire text as a single clause
            # (or splitting by paragraphs) to ensure we always return something.
            paragraphs = [p.strip() for p in extracted_text.split('\n') if len(p.strip()) >= 30]
            clauses = paragraphs if paragraphs else [extracted_text[:1000]]

        # 3. Classify clauses (FR-3 / FR-4)
        model = get_model()
        classification_results = model.predict_batch(clauses, return_probs=True)

        # 4. Map results to canonical schemas and add display names
        analyzed_clauses = []
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        type_counts = {}

        for text, res in zip(clauses, classification_results):
            pred_type = res["predicted_type_clause"]
            pred_risk = res["predicted_risk_level"]
            warning = res.get("warning", "")

            # Count statistics
            type_counts[pred_type] = type_counts.get(pred_type, 0) + 1
            if pred_risk == "high":
                high_risk_count += 1
            elif pred_risk == "medium":
                medium_risk_count += 1
            else:
                low_risk_count += 1

            analyzed_clauses.append(
                AnalyzedClause(
                    text=text,
                    predicted_type_clause=pred_type,
                    type_display_name=TYPE_DISPLAY_NAMES_AR.get(pred_type, pred_type),
                    predicted_risk_level=pred_risk,
                    risk_display_name=RISK_DISPLAY_NAMES_AR.get(pred_risk, pred_risk),
                    warning=warning
                )
            )

        # 5. Generate summary (FR-6)
        summary_clauses = [
            {
                "text": c.text,
                "predicted_type_clause": c.predicted_type_clause,
                "predicted_risk_level": c.predicted_risk_level,
                "warning": c.warning
            } for c in analyzed_clauses
        ]
        summary = generate_contract_summary(extracted_text, summary_clauses)

        # 6. Build stats dictionary
        stats = {
            "total_clauses": len(analyzed_clauses),
            "high_risk_clauses": high_risk_count,
            "medium_risk_clauses": medium_risk_count,
            "low_risk_clauses": low_risk_count,
            "type_distribution": type_counts
        }

        # Return the unified response
        response = AnalysisResponse(
            filename=file.filename,
            is_scanned=is_scanned,
            clauses=analyzed_clauses,
            summary=summary,
            stats=stats,
            message="تم تحليل العقد بنجاح واستخراج البنود والمخاطر."
        )

        return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())

    except Exception as e:
        logger.error(f"Error during contract analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشل تحليل العقد: {str(e)}",
        )
    finally:
        # Clean up temporary file
        if file_path.exists():
            file_path.unlink()
