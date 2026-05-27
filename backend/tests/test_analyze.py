# test_analyze.py
from fastapi.testclient import TestClient
from unittest.mock import patch
import pytest
import io

from app.main import app

client = TestClient(app)

@patch("app.routers.analyze.extract_text_from_file")
def test_analyze_endpoint(mock_extract):
    # Setup mock to return text that will trigger our segmenter and analyzer
    mock_extract.return_value = (
        "الطرف الأول: شركة الاتحاد للخدمات العامة.\n"
        "الطرف الثاني: السيد محمد بن عبد الله.\n"
        "المادة 1: يلتزم الطرف الأول بتسليم كافة الوثائق المطلوبة في الموعد المحدد.\n"
        "المادة 2: يحق للطرف الأول فسخ العقد تلقائياً في حال تأخر الطرف الثاني عن السداد لمدة تتجاوز 10 أيام دون الحاجة لأي إنذار مسبق.\n"
        "المادة 3: يلتزم الطرف الثاني بدفع مبلغ مليون ليرة سورية كدفعة أولى عند توقيع العقد."
    , False)
    
    # Create a dummy file to upload
    dummy_pdf = io.BytesIO(b"%PDF-1.4 mock pdf data")
    
    response = client.post(
        "/api/contract/analyze",
        files={"file": ("contract.pdf", dummy_pdf, "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["filename"] == "contract.pdf"
    assert data["is_scanned"] == False
    assert "clauses" in data
    assert len(data["clauses"]) >= 3
    assert "summary" in data
    assert "stats" in data
    
    # Verify that display names are populated
    first_clause = data["clauses"][0]
    assert "type_display_name" in first_clause
    assert "risk_display_name" in first_clause
    
    # Verify stats structure
    assert "total_clauses" in data["stats"]
    assert "high_risk_clauses" in data["stats"]
    
    # Verify high risk clause warning
    high_risk_clauses = [c for c in data["clauses"] if c["predicted_risk_level"] == "high"]
    if high_risk_clauses:
        assert len(high_risk_clauses[0]["warning"]) > 0
