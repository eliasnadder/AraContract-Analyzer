# test_summary.py
from app.services.summary_service import generate_contract_summary, extract_parties

def test_extract_parties():
    text = """
    عقد تقديم خدمات استشارية
    إنه في يوم الأربعاء، تم الاتفاق بين كل من:
    الطرف الأول: شركة القدس للتطوير العقاري ويمثلها السيد أحمد المدير العام.
    الطرف الثاني: السيد عمر بن الخطاب المقيم في دمشق سوري الجنسية.
    تمهيد: يلتزم الطرفان بالشروط التالية...
    """
    party_a, party_b = extract_parties(text)
    assert "شركة القدس للتطوير العقاري" in party_a
    assert "عمر بن الخطاب" in party_b

def test_generate_summary():
    text = "الطرف الأول: شركة الاتحاد والطرف الثاني: شركة الأمل"
    clauses = [
        {
            "text": "يلتزم الطرف الثاني بدفع مبلغ مليون ليرة سورية شهرياً للطرف الأول.",
            "predicted_type_clause": "payment_financial",
            "predicted_risk_level": "low",
            "warning": ""
        },
        {
            "text": "يحق للطرف الأول فسخ العقد في أي وقت دون إشعار مسبق أو تعويض.",
            "predicted_type_clause": "termination",
            "predicted_risk_level": "high",
            "warning": "تحذير فسخ العقد: يمنح هذا البند حق الفسخ الأحادي الفوري دون تعويض عادل."
        }
    ]
    summary = generate_contract_summary(text, clauses)
    assert "شركة الاتحاد" in summary
    assert "شركة الأمل" in summary
    assert "بنداً مالياً" in summary
    assert "بنداً عالي الخطورة" in summary
    assert "تحذير فسخ العقد" in summary
