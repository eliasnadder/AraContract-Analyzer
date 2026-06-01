from app.routers.segment import segment_arabic_text
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
import io

client = TestClient(app)


def test_segment_article_markers():
    text = (
        "المادة 1 يلتزم الطرف الأول بتسليم كافة الوثائق المطلوبة خلال المدة المحددة في هذا العقد.\n"
        "المادة 2 يحق للطرف الأول فسخ العقد تلقائياً في حال تأخر الطرف الثاني عن السداد."
    )
    clauses = segment_arabic_text(text)
    assert len(clauses) == 2
    assert "المادة 1" in clauses[0]
    assert "المادة 2" in clauses[1]


def test_segment_arabic_numerals():
    """New: step4.py supports Arabic-Indic numerals like المادة ١"""
    text = (
        "المادة ١ يلتزم الطرف الأول بتسليم كافة الوثائق المطلوبة خلال المدة المحددة في هذا العقد.\n"
        "المادة ٢ يحق للطرف الأول فسخ العقد تلقائياً في حال تأخر الطرف الثاني عن السداد."
    )
    clauses = segment_arabic_text(text)
    assert len(clauses) == 2


def test_min_length_filter():
    # Now 50 chars minimum (was 30)
    text = "المادة 1 قصير جداً"
    clauses = segment_arabic_text(text)
    assert len(clauses) == 0


def test_ordinal_markers_fallback():
    text = (
        "أولاً: يلتزم الطرف الأول بكافة البنود الواردة في هذا العقد وعدم الإخلال بأي منها.\n"
        "ثانياً: يلتزم الطرف الثاني بدفع المستحقات المالية في وقتها المحدد وفق الجدول المرفق."
    )
    clauses = segment_arabic_text(text)
    assert len(clauses) == 2


def test_paragraph_fallback():
    """When no article markers exist, falls back to paragraph splitting."""
    text = (
        "يلتزم الطرف الأول بتسليم الوثائق المطلوبة خلال المدة المحددة في هذا العقد وبدون تأخير.\n\n"
        "يحق للطرف الثاني فسخ العقد في حال إخلال الطرف الأول بالتزاماته المنصوص عليها في هذه الاتفاقية."
    )
    clauses = segment_arabic_text(text)
    assert len(clauses) == 2

def test_segment_file_endpoint():
    with patch("app.routers.segment.extract_text_from_file") as mock_extract:
        mock_extract.return_value = (
            "المادة 1 يلتزم الطرف الأول بتسليم كافة الوثائق المطلوبة في الموعد المحدد.\n"
            "المادة 2 يلتزم الطرف الثاني بدفع المبالغ المستحقة وفق الجدول الزمني المتفق عليه.",
            False
        )

        dummy_pdf = io.BytesIO(b"%PDF-1.4 mock")
        response = client.post(
            "/api/contract/segment/file",
            files={"file": ("contract.pdf", dummy_pdf, "application/pdf")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["is_scanned"] == False
    assert "filename" in data
    assert "extracted_text_preview" in data