"""
Pydantic models for request and response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ========================
# Upload Schemas
# ========================
class UploadResponse(BaseModel):
    filename: str = Field(..., description="Original filename of the uploaded file")
    file_size: int = Field(..., description="Size of the file in bytes")
    is_scanned: bool = Field(..., description="Whether the file was detected as scanned (requiring OCR)")
    extracted_text: str = Field(..., description="The text extracted from the file")
    message: str = Field(default="File uploaded and text extracted successfully")


# ========================
# Segmentation Schemas
# ========================
class SegmentationRequest(BaseModel):
    text: str = Field(..., description="Raw contract text to segment into clauses")


class SegmentationResponse(BaseModel):
    clauses: List[str] = Field(..., Description="List of segmented clauses")
    count: int = Field(..., description="Number of clauses identified")
    message: str = Field(default="Text segmented successfully")


# ========================
# Classification Schemas
# ========================
class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Clause text to classify")


class ClassificationResponse(BaseModel):
    predicted_type_clause: str = Field(..., description="Predicted clause type")
    predicted_risk_level: str = Field(..., description="Predicted risk level (low, medium, high)")
    type_clause_probabilities: Dict[str, float] = Field(..., description="Probability distribution over clause types")
    risk_level_probabilities: Dict[str, float] = Field(..., description="Probability distribution over risk levels")


class BatchClassificationRequest(BaseModel):
    texts: List[str] = Field(..., description="List of clause texts to classify")


class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse] = Field(..., description="List of classification results for each input text")
    count: int = Field(..., description="Number of texts processed")


# ========================
# Summarization Schemas
# ========================
class SummarizationRequest(BaseModel):
    text: str = Field(..., description="Full contract text")
    classified_clauses: List[Dict[str, Any]] = Field(..., description="List of classified clauses with type and risk")


class SummarizationResponse(BaseModel):
    summary: str = Field(..., description="Generated executive summary (3-5 sentences in Arabic)")
    message: str = Field(default="Executive summary generated successfully")


# ========================
# Question-Answering Schemas
# ========================
class QARequest(BaseModel):
    contract_id: str = Field(..., description="Unique identifier for the contract session")
    question: str = Field(..., description="Arabic question about the contract")


class QAResponse(BaseModel):
    answer: str = Field(..., description="Answer to the question based on contract content")
    sources: List[str] = Field(..., description="List of text chunks used to generate the answer")
    message: str = Field(default="Answer generated successfully")


# ========================
# Comparison Schemas
# ========================
class ComparisonResponse(BaseModel):
    contract1_summary: Dict[str, Any] = Field(..., description="Summary of statistics for contract 1")
    contract2_summary: Dict[str, Any] = Field(..., description="Summary of statistics for contract 2")
    differences: List[Dict[str, Any]] = Field(..., description="List of differences found between the two contracts")
    message: str = Field(default="Contract comparison completed successfully")