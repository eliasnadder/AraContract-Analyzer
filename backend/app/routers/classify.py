"""
Classification router for contract clauses.
Loads the trained model and provides type and risk level predictions.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Union
import torch
import numpy as np
from pathlib import Path

from app.models.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    ErrorResponse
)
from app.models.inference import AraContractInference
from app.core.config import settings

router = APIRouter()

# Global model instance (loaded once at startup)
model_instance: AraContractInference = None


def get_model() -> AraContractInference:
    """Get or initialize the global model instance."""
    global model_instance
    if model_instance is None:
        model_path = Path(settings.CLASSIFIER_MODEL_PATH)
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Classification model not found. Please train the model first.",
            )
        model_instance = AraContractInference(str(model_path))
    return model_instance


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def classify_clause(request: ClassificationRequest):
    """
    Classify a single contract clause for type and risk level.

    - **text**: The clause text to classify
    """
    try:
        model = get_model()
        result = model.predict_single(request.text, return_probs=True)
        return ClassificationResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during classification: {str(e)}",
        )


@router.post(
    "/classify/batch",
    response_model=BatchClassificationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def classify_clauses_batch(request: BatchClassificationRequest):
    """
    Classify multiple contract clauses for type and risk level.

    - **texts**: List of clause texts to classify
    """
    try:
        model = get_model()
        results = model.predict_batch(request.texts, return_probs=True)
        return BatchClassificationResponse(results=results, count=len(results))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during batch classification: {str(e)}",
        )