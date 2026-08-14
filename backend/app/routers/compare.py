"""
Contract comparison router (optional/bonus feature).
Compares two contracts for differences in clauses, types, and risk levels.
"""

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
from typing import List

from app.services.comparison_service import compare_contracts
from app.models.schemas import ComparisonResponse, ErrorResponse
from app.core.config import settings
from app.core.auth import get_current_user

router = APIRouter()


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    responses={400: {"model": ErrorResponse}, 413: {
        "model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def compare_contract_endpoint(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    uid: str = Depends(get_current_user)
):
    """
    Compare two contract files for differences in clauses, types, and risk levels.

    - **file1**: First contract file (PDF or image)
    - **file2**: Second contract file (PDF or image)
    """
    # Validate file sizes
    contents1 = await file1.read()
    contents2 = await file2.read()

    if len(contents1) > settings.MAX_FILE_SIZE or len(contents2) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the limit of {settings.MAX_FILE_SIZE // (1024*1024)} MB per file",
        )

    # Validate file extensions
    file1_ext = Path(file1.filename).suffix.lower()
    file2_ext = Path(file2.filename).suffix.lower()
    allowed_extensions = settings.ALLOWED_EXTENSIONS

    if file1_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file1_ext} not allowed for first file. Allowed types: {', '.join(allowed_extensions)}",
        )

    if file2_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file2_ext} not allowed for second file. Allowed types: {', '.join(allowed_extensions)}",
        )

    # Save the uploaded files temporarily
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file1_path = upload_dir / file1.filename
    file2_path = upload_dir / file2.filename

    try:
        with open(file1_path, "wb") as buffer:
            buffer.write(contents1)
        with open(file2_path, "wb") as buffer:
            buffer.write(contents2)

        # Compare the contracts
        comparison_result = compare_contracts(str(file1_path), str(file2_path))

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            # content=comparison_result.dict(),
            content=comparison_result.model_dump()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during contract comparison: {str(e)}",
        )
    finally:
        # Clean up: remove the temporary files
        if file1_path.exists():
            file1_path.unlink()
        if file2_path.exists():
            file2_path.unlink()
