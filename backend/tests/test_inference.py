# test_inference.py
from app.models.inference import AraContractInference

def test_inference_fallback():
    # Instantiate inference with non-existent path to trigger fallback
    inference = AraContractInference("./non_existent_checkpoint.pt")
    assert inference.is_fallback == True
    
    # Test a high-risk termination clause
    text = "يحق للطرف الأول فسخ هذا العقد في أي وقت ودون إخطار الطرف الثاني مسبقاً."
    result = inference.predict_single(text)
    
    assert "predicted_type_clause" in result
    assert "predicted_risk_level" in result
    assert "type_clause_probabilities" in result
    assert "risk_level_probabilities" in result
    
    # Check that warning is populated for high risk
    if result["predicted_risk_level"] == "high":
        assert len(result["warning"]) > 0
