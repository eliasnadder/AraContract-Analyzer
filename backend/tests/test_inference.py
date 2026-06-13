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


def test_legal_override_flags_full_party_b_responsibility_as_high():
    inference = AraContractInference("./non_existent_checkpoint.pt")
    text = "يتحمل الفريق الثاني كامل المسؤولية المدنية والجزائية عن الحوادث التي تقع معه أياً كان سببها ومسببها."

    result = inference.predict_single(text)

    assert result["predicted_type_clause"] == "party_obligations_b"
    assert result["predicted_risk_level"] == "high"
    assert "تحذير التزامات الطرف الثاني" in result["warning"]


def test_legal_override_flags_automatic_termination_as_high():
    inference = AraContractInference("./non_existent_checkpoint.pt")
    text = "يعتبر الفريق الثاني معذراً بمجرد حلول أجل الالتزامات دون الحاجة لإعذار أو الحصول على حكم قضائي ويجوز للفريق الأول اعتبار العقد مفسوخاً من تلقاء نفسه."

    result = inference.predict_single(text)

    assert result["predicted_type_clause"] == "party_obligations_b"
    assert result["predicted_risk_level"] == "high"
    assert "تحذير التزامات الطرف الثاني" in result["warning"]
