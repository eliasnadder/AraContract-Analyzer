"""
Segmentation router for contract clauses.
"""

from fastapi import APIRouter, HTTPException, status
from app.models.schemas import SegmentationRequest, SegmentationResponse, ErrorResponse
import re
from typing import List

router = APIRouter()


def segment_arabic_text(text: str) -> List[str]:
    """
    Segment Arabic contract text into clauses using common markers.
    Markers: المادة, البند, الفصل, الباب, أولاً, ثانياً, and numeric patterns (1., 1)).
    """
    if not text or not isinstance(text, str):
        return []

    text = text.strip()

    # Define the markers to split on
    # We match:
    # 1. المادة X or البند X or الفصل X or الباب X (where X can be number or word)
    # 2. أولاً, ثانياً, etc. optionally followed by a colon or space
    # 3. Numeric lists like 1. or 1- or 1) at the start of a line
    pattern = r'(\n(?:المادة|البند|الفصل|الباب)\s+\d+(?:\s*:\s*)?|\n(?:أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً|سابعاً|ثامناً|تاسعاً|عاشراً)(?:\s*:\s*)?|^\s*\d+[\.\-]\s+|^\s*\d+\)\s*|\n\s*\d+[\.\-]\s+|\n\s*\d+\)\s*)'

    # To handle markers at the very beginning of the text, we prepend a newline temporarily
    # to make matching at the start of line work consistently.
    text_to_split = "\n" + text
    parts = re.split(pattern, text_to_split)

    clauses = []
    current_clause = ""

    # Since we split on a capturing pattern, parts will alternate:
    # parts[0]: text before first marker (empty if text started with marker)
    # parts[1]: the first marker itself
    # parts[2]: text after first marker
    # etc.
    
    i = 0
    while i < len(parts):
        part = parts[i]
        if not part:
            i += 1
            continue
            
        # Check if this part matches our marker pattern (odd indices are the captured groups)
        is_marker = (i % 2 == 1)
        cleaned_part = part.strip()
        
        if is_marker:
            if current_clause.strip():
                clauses.append(current_clause.strip())
            current_clause = cleaned_part + " "
        else:
            current_clause += part
            
        i += 1
        
    if current_clause.strip():
        clauses.append(current_clause.strip())

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