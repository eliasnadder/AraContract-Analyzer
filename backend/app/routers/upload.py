"""
Upload router for contract files.
Handles PDF and image uploads, text extraction, and preprocessing.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

from app.services.extraction_service import extract_text_from_file
from app.models.schemas import UploadResponse, ErrorResponse
from app.core.config import settings

router = APIRouter()


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def upload_contract(file: UploadFile = File(...)):
    """
    Upload a contract file (PDF or image) and extract text.

    - **file**: The contract file to upload (PDF, PNG, JPG, JPEG, TIFF, BMP)
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

        # Extract text from the file
        extracted_text, is_scanned = extract_text_from_file(str(file_path))

        # Prepare response
        response = UploadResponse(
            filename=file.filename,
            file_size=len(contents),
            is_scanned=is_scanned,
            extracted_text=extracted_text,
            message="File uploaded and text extracted successfully",
        )

        return JSONResponse(status_code=status.HTTP_200_OK, content=response.dict())

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}",
        )
    finally:
        # Clean up: remove the temporary file
        if file_path.exists():
            file_path.unlink()