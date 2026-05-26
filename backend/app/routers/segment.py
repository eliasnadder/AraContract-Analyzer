"""
Segmentation router for contract clauses.
"""

from fastapi import APIRouter, HTTPException, status
from app.models.schemas import SegmentationRequest, SegmentationResponse, ErrorResponse
import re

router = APIRouter()


def segment_arabic_text(text: str) -> List[str]:
    """
    Segment Arabic contract text into clauses using common markers.
    Markers: المادة, البند, الفصل, الباب, וכן מספור עשרoni.
    """
    if not text or not isinstance(text, str):
        return []

    # Normalize Arabic text
    text = text.strip()

    # Define segmentation patterns for Arabic
    # Patterns for article markers followed by optional number and punctuation
    patterns = [
        r'(?<=المادة\s*\d*[\s\d]*:)',  # المادة 1: or المادة:
        r'(?<=البند\s*\d*[\s\d]*:)',    # البند 2: or البند:
        r'(?<=الفصل\s*\d*[\s\d]*:)',    # الفصل
        r'(?<=الباب\s*\d*[\s\d]*:)',     # الباب
        r'(?<=^\d+\.\s*)',               # 1. 2. etc at start of line
        r'(?<=^\d+\)\s*)',               # 1) 2) etc at start of line
    ]

    # Combine patterns
    combined_pattern = '|'.join(f'({p})' for p in patterns)

    # Split by the patterns, keeping the delimiters
    parts = re.split(f'({combined_pattern})', text, flags=re.MULTILINE)

    # Recombine: each delimiter with the following text until next delimiter
    clauses = []
    i = 0
    while i < len(parts):
        if i == 0 and not re.match(combined_pattern, parts[i], flags=re.MULTILINE):
            # Leading text before first marker - treat as first clause if meaningful
            if parts[i].strip():
                clauses.append(parts[i].strip())
            i += 1
        elif re.match(combined_pattern, parts[i], flags=re.MULTILINE):
            # This is a delimiter, combine with next part if exists
            delimiter = parts[i]
            if i + 1 < len(parts):
                clause = delimiter + parts[i+1]
                clauses.append(clause.strip())
                i += 2
            else:
                # Delimiter at end, ignore
                i += 1
        else:
            # Regular text part, attach to previous clause if possible
            if clauses:
                clauses[-1] = clauses[-1] + ' ' + parts[i]
                clauses[-1] = clauses[-1].strip()
            else:
                clauses.append(parts[i].strip())
            i += 1

    # Filter out clauses that are too short (<30 characters)
    clauses = [clause for clause in clauses if len(clause) >= 30]

    return clauses


@router.post(
    "/segment",
    response_model=SegmentationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def segment_contract(request: SegmentationRequest):
    """
    Segment contract text into clauses.

    - **text**: The raw contract text to segment
    """
    try:
        clauses = segment_arabic_text(request.text)
        return SegmentationResponse(
            clauses=clauses,
            count=len(clauses),
            message=f"Text segmented into {len(clauses)} clauses",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during segmentation: {str(e)}",
        )