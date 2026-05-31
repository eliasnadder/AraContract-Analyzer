"""
Canonical label mappings and taxomony converters for AraContract Analyzer.
Resolves the mismatch between the 7-class model dataset and the 8-class SRS requirement.
"""

# SRS Canonical labels (8 classes)
TYPE_LABELS = [
    "payment_financial",
    "duration_expiration",
    "termination",
    "penalties_damages",
    "party_obligations_a",
    "party_obligations_b",
    "dispute_resolution",
    "general_provisions",
]

# Display names in Arabic for UI
TYPE_DISPLAY_NAMES_AR = {
    "payment_financial": "مالي / دفع",
    "duration_expiration": "مدة / انتهاء",
    "termination": "فسخ / إنهاء",
    "penalties_damages": "غرامات / تعويضات",
    "party_obligations_a": "التزامات الطرف الأول",
    "party_obligations_b": "التزامات الطرف الثاني",
    "dispute_resolution": "تسوية نزاعات",
    "general_provisions": "أحكام عامة",
}

# Raw model classes (legacy 7 classes)
MODEL_TYPE_LABELS_7 = [
    "general_provisions", 
    "payment_financial", 
    "party_obligations",
    "duration_expiration", 
    "termination", 
    "penalties_damages", 
    "dispute_resolution",
]

# Raw model classes (new 8 classes)
MODEL_TYPE_LABELS_8 = [
    "general_provisions",
    "payment_financial",
    "party_obligations_a",
    "party_obligations_b",
    "duration_expiration",
    "termination",
    "penalties_damages",
    "dispute_resolution",
]

# Default (for backwards compatibility)
MODEL_TYPE_LABELS = MODEL_TYPE_LABELS_7

MODEL_TYPE_LABEL_TO_IDX = {label: i for i, label in enumerate(MODEL_TYPE_LABELS)}
MODEL_TYPE_IDX_TO_LABEL = {i: label for i, label in enumerate(MODEL_TYPE_LABELS)}

# Risk levels
RISK_LABELS = ["low", "medium", "high"]
RISK_LABEL_TO_IDX = {label: i for i, label in enumerate(RISK_LABELS)}
RISK_IDX_TO_LABEL = {i: label for i, label in enumerate(RISK_LABELS)}

RISK_DISPLAY_NAMES_AR = {
    "low": "منخفض",
    "medium": "متوسط",
    "high": "مرتفع",
}

def map_model_output_to_canonical(model_type: str, text: str) -> str:
    """
    Maps 7-class raw model output to the canonical 8-class SRS taxonomy.
    Heuristically splits 'party_obligations' into Party A or Party B.
    
    Args:
        model_type: The predicted label from the 7-class model.
        text: The text content of the clause.
        
    Returns:
        One of the 8 canonical TYPE_LABELS.
    """
    if model_type != "party_obligations":
        # Other types map directly if they are in the canonical set
        if model_type in TYPE_LABELS:
            return model_type
        return "general_provisions" # Default fallback
        
    # Heuristic for party obligations:
    # Party A is typically 'الطرف الأول'
    # Party B is typically 'الطرف الثاني'
    text_normalized = text.replace("أ", "ا").replace("إ", "ا")
    
    has_party_a = "الطرف الاول" in text_normalized
    has_party_b = "الطرف الثاني" in text_normalized or "الطرف التاني" in text_normalized
    
    if has_party_a and not has_party_b:
        return "party_obligations_a"
    elif has_party_b and not has_party_a:
        return "party_obligations_b"
    elif "المستاجر" in text_normalized or "المشتري" in text_normalized or "الموظف" in text_normalized:
        # In leasing/sales/employment contracts, tenant/buyer/employee is usually Party B
        return "party_obligations_b"
    elif "المؤجر" in text_normalized or "البائع" in text_normalized or "صاحب العمل" in text_normalized:
        # Lessor/Seller/Employer is usually Party A
        return "party_obligations_a"
    else:
        # Default fallback if ambiguous
        return "party_obligations_a"
