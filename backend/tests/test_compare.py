# test_compare.py
from fastapi.testclient import TestClient
from unittest.mock import patch
import pytest
import io

from app.main import app

client = TestClient(app)

@patch("app.services.comparison_service.extract_text_from_file")
def test_compare_endpoint(mock_extract):
    # Setup mock_extract to return different text on subsequent calls
    # Call 1 (contract 1): 2 clauses, 1 high-risk termination
    # Call 2 (contract 2): 1 clause, 0 high-risk
    mock_extract.side_effect = [
        (
            "الطرف الأول: شركة الاتحاد. الطرف الثاني: شركة الأمل.\n"
            "المادة 1: يلتزم الطرف الثاني بدفع مبلغ مليون ليرة.\n"
            "المادة 2: يحق للطرف الأول فسخ العقد تلقائياً في أي وقت دون إشعار."
        , False),
        (
            "الطرف الأول: شركة الاتحاد. الطرف الثاني: شركة الأمل.\n"
            "المادة 1: يلتزم الطرف الثاني بدفع مبلغ مليون ليرة."
        , False)
    ]
    
    # Create two dummy files to upload
    dummy_pdf1 = io.BytesIO(b"%PDF-1.4 mock pdf data 1")
    dummy_pdf2 = io.BytesIO(b"%PDF-1.4 mock pdf data 2")
    
    response = client.post(
        "/api/contract/compare",
        files={
            "file1": ("contract1.pdf", dummy_pdf1, "application/pdf"),
            "file2": ("contract2.pdf", dummy_pdf2, "application/pdf")
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "contract1_summary" in data
    assert "contract2_summary" in data
    assert "differences" in data
    assert len(data["differences"]) > 0
    
    # Verify values inside summaries
    assert data["contract1_summary"]["total_clauses"] == 2
    assert data["contract2_summary"]["total_clauses"] == 1
