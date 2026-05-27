"""
Contract comparison service for comparing two contracts (FR-8).
Processes both contracts and identifies differences in clause counts, risk levels, and specific clauses.
"""

from typing import Dict, Any, List
import logging

from app.services.extraction_service import extract_text_from_file
from app.routers.segment import segment_arabic_text
from app.routers.classify import get_model
from app.services.summary_service import generate_contract_summary, extract_parties
from app.models.schemas import ComparisonResponse
from app.models.labels import TYPE_DISPLAY_NAMES_AR

logger = logging.getLogger(__name__)

def _analyze_file_for_compare(file_path: str) -> Dict[str, Any]:
    """Helper to run the analysis pipeline on a single file for comparison."""
    # 1. Extract
    text, is_scanned = extract_text_from_file(file_path)
    
    # 2. Segment
    clauses = segment_arabic_text(text)
    if not clauses:
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) >= 30]
        clauses = paragraphs if paragraphs else [text[:1000]]
        
    # 3. Classify
    model = get_model()
    classification_results = model.predict_batch(clauses, return_probs=True)
    
    # 4. Compile stats and details
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0
    type_counts = {}
    
    analyzed_clauses = []
    for c_text, res in zip(clauses, classification_results):
        pred_type = res["predicted_type_clause"]
        pred_risk = res["predicted_risk_level"]
        
        type_counts[pred_type] = type_counts.get(pred_type, 0) + 1
        if pred_risk == "high":
            high_risk_count += 1
        elif pred_risk == "medium":
            medium_risk_count += 1
        else:
            low_risk_count += 1
            
        analyzed_clauses.append({
            "text": c_text,
            "type": pred_type,
            "risk": pred_risk
        })
        
    party_a, party_b = extract_parties(text)
    
    return {
        "party_a": party_a,
        "party_b": party_b,
        "total_clauses": len(clauses),
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "low_risk_count": low_risk_count,
        "type_counts": type_counts,
        "clauses": analyzed_clauses
    }

def compare_contracts(file1_path: str, file2_path: str) -> ComparisonResponse:
    """
    Compares two contracts and returns a ComparisonResponse.
    """
    # Analyze both contracts
    res1 = _analyze_file_for_compare(file1_path)
    res2 = _analyze_file_for_compare(file2_path)
    
    # Identify differences
    differences = []
    
    # 1. Compare party differences
    if res1["party_a"] != res2["party_a"] or res1["party_b"] != res2["party_b"]:
        differences.append({
            "type": "parties",
            "title": "اختلاف في أطراف التعاقد",
            "description": (
                f"العقد الأول يضم أطرافاً مختلفة ({res1['party_a']} و {res1['party_b']}) "
                f"مقارنة بالعقد الثاني ({res2['party_a']} و {res2['party_b']})."
            ),
            "severity": "info"
        })
        
    # 2. Compare high risk clause counts
    high_diff = res1["high_risk_count"] - res2["high_risk_count"]
    if high_diff != 0:
        more_risk_contract = "العقد الأول" if high_diff > 0 else "العقد الثاني"
        differences.append({
            "type": "risk_profile",
            "title": "اختلاف في خطورة العقود",
            "description": (
                f"يحتوي {more_risk_contract} على عدد أكبر من البنود عالية الخطورة بمقدار {abs(high_diff)} بنداً. "
                f"إجمالي بنود الخطورة العالية: العقد الأول ({res1['high_risk_count']})، العقد الثاني ({res2['high_risk_count']})."
            ),
            "severity": "warning" if abs(high_diff) > 1 else "info"
        })
        
    # 3. Compare clause type distributions
    all_types = set(list(res1["type_counts"].keys()) + list(res2["type_counts"].keys()))
    for ctype in all_types:
        count1 = res1["type_counts"].get(ctype, 0)
        count2 = res2["type_counts"].get(ctype, 0)
        
        if count1 != count2:
            display_name = TYPE_DISPLAY_NAMES_AR.get(ctype, ctype)
            differences.append({
                "type": "clause_distribution",
                "title": f"اختلاف في توزيع بنود {display_name}",
                "description": (
                    f"يحتوي العقد الأول على {count1} بنداً من نوع '{display_name}' "
                    f"بينما يحتوي العقد الثاني على {count2} بنداً من نفس النوع."
                ),
                "severity": "info"
            })
            
    # Assemble response summaries
    summary1 = {
        "parties": f"{res1['party_a']} & {res1['party_b']}",
        "total_clauses": res1["total_clauses"],
        "high_risk_clauses": res1["high_risk_count"],
        "medium_risk_clauses": res1["medium_risk_count"],
        "low_risk_clauses": res1["low_risk_count"],
        "type_distribution": res1["type_counts"]
    }
    
    summary2 = {
        "parties": f"{res2['party_a']} & {res2['party_b']}",
        "total_clauses": res2["total_clauses"],
        "high_risk_clauses": res2["high_risk_count"],
        "medium_risk_clauses": res2["medium_risk_count"],
        "low_risk_clauses": res2["low_risk_count"],
        "type_distribution": res2["type_counts"]
    }
    
    return ComparisonResponse(
        contract1_summary=summary1,
        contract2_summary=summary2,
        differences=differences,
        message="تمت مقارنة العقدين واستخراج الاختلافات في البنود والخطورة بنجاح."
    )
