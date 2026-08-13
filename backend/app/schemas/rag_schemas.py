from pydantic import BaseModel, Field
from typing import Optional, List

class IngestRequest(BaseModel):
    clauses: List[str] = Field(
        ...,
        description="قائمة البنود من segment_arabic_text()",
        min_length=1
    )
    session_id: Optional[str] = Field(
        None,
        description="معرف الجلسة — يُنشأ تلقائياً إذا لم يُمرر"
    )


class IngestResponse(BaseModel):
    session_id: str
    clauses_count: int
    message: str


class AskRequest(BaseModel):
    session_id: str = Field(..., description="من IngestResponse")
    question: str = Field(
        ...,
        description="سؤال المستخدم بالعربية",
        min_length=3
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="عدد البنود المسترجعة"
    )


class RetrievedClause(BaseModel):
    clause_index: int
    parent_text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    retrieved_clauses: List[RetrievedClause]
    session_id: str


class SummarizeRequest(BaseModel):
    clauses: List[str] = Field(
        ...,
        description="كل بنود العقد",
        min_length=1
    )
    analyzed_clauses: Optional[List[dict]] = Field(
        None,
        description="ناتج المصنّف — اختياري، يُحسّن الملخص بإضافة البنود الخطرة"
    )


class SummarizeResponse(BaseModel):
    summary: str
    clauses_count: int


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool
    message: str


class RAGStatusResponse(BaseModel):
    status: str
    service: str
    provider: str
    ready: bool
    model: Optional[str] = None
    base_url: Optional[str] = None
    details: Optional[str] = None
