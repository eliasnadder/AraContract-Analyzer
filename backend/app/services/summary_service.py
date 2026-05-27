"""
Summary service for generating executive summaries of contracts using heuristics.
Extracts parties, duration, and counts risk levels and key clauses.
"""

import re
from typing import List, Dict, Any

def extract_parties(text: str) -> tuple[str, str]:
    """
    Heuristically extracts the names of the first and second party from the contract text.
    
    Returns:
        Tuple of (party_a_name, party_b_name)
    """
    # Clean the first 2000 characters where parties are typically introduced
    header = text[:2000]
    
    # Normalize alef and spaces for easier matching
    header_clean = re.sub(r"[أإآ]", "ا", header)
    header_clean = re.sub(r"\s+", " ", header_clean)
    
    party_a = "الطرف الأول"
    party_b = "الطرف الثاني"
    
    # Try common patterns:
    # 1. الطرف الاول (المشار اليه بـ...) السيد/شركة ...
    # 2. الطرف الاول: ...
    # 3. بين كل من: الطرف الاول ... والطرف الثاني ...
    
    # Regex search for party names
    # Match everything after "الطرف الأول" or "الطرف الأول:" or "الطرف الأول /" up to 15-40 chars until punctuation or "المشار إليه"
    pattern_a = r"الطرف الاو[لى](?:\s*[:/]\s*|\s+هو\s+|\s+)(السيد|السيده|الاستاذ|الدكتور|المواطن|شركة|المؤسسة)?\s*([^\n\.\،\-\(]{2,50})"
    pattern_b = r"الطرف الثاني(?:\s*[:/]\s*|\s+هو\s+|\s+)(السيد|السيده|الاستاذ|الدكتور|المواطن|شركة|المؤسسة)?\s*([^\n\.\،\-\(]{2,50})"
    
    match_a = re.search(pattern_a, header_clean)
    match_b = re.search(pattern_b, header_clean)
    
    party_a_name = "الطرف الأول"
    party_b_name = "الطرف الثاني"
    
    if match_a:
        name = match_a.group(2).strip()
        prefix = match_a.group(1) or ""
        if name and len(name) > 3:
            # Clean up trailing words like "المشار" or "بصفته"
            name = re.split(r"(المشار|بصفته|المقيم|وجنسيته|سوري|من|المسمى)", name)[0].strip()
            party_a_name = f"{prefix} {name}".strip()
            
    if match_b:
        name = match_b.group(2).strip()
        prefix = match_b.group(1) or ""
        if name and len(name) > 3:
            name = re.split(r"(المشار|بصفته|المقيم|وجنسيته|سوري|من|المسمى)", name)[0].strip()
            party_b_name = f"{prefix} {name}".strip()
            
    return party_a_name, party_b_name


def generate_contract_summary(text: str, classified_clauses: List[Dict[str, Any]]) -> str:
    """
    Generate a 3-5 sentence Arabic executive summary of the contract.
    Highlights parties, total clauses, risk counts, and key warnings.
    
    Args:
        text: The full contract text.
        classified_clauses: List of dictionaries, each with 'text', 'predicted_type_clause', 
                            'predicted_risk_level', and optional 'warning'.
                            
    Returns:
        A string containing the summary in Arabic.
    """
    if not classified_clauses:
        return "لم يتم تحديد أي بنود في هذا العقد لإجراء التلخيص."
        
    # 1. Extract parties
    party_a, party_b = extract_parties(text)
    
    # 2. Risk breakdown
    total_clauses = len(classified_clauses)
    high_risk_count = sum(1 for c in classified_clauses if c.get("predicted_risk_level", "").lower() == "high")
    medium_risk_count = sum(1 for c in classified_clauses if c.get("predicted_risk_level", "").lower() == "medium")
    
    # 3. Type counts
    type_counts = {}
    for c in classified_clauses:
        ctype = c.get("predicted_type_clause", "")
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
        
    payment_count = type_counts.get("payment_financial", 0)
    termination_count = type_counts.get("termination", 0)
    
    # 4. Synthesize sentences
    sentences = []
    
    # Sentence 1: Introduction of parties
    sentences.append(f"تم تحليل هذا العقد الذي يجمع بين كل من {party_a} (كالطرف الأول) و {party_b} (كالطرف الثاني).")
    
    # Sentence 2: Structure
    financial_str = f" منها {payment_count} بنداً مالياً" if payment_count > 0 else ""
    termination_str = f" و {termination_count} بنداً لإنهاء العقد" if termination_count > 0 else ""
    sentences.append(f"يتكون العقد من {total_clauses} بنداً رئيسياً تم فحصها{financial_str}{termination_str}.")
    
    # Sentence 3: Risk summary
    if high_risk_count > 0 or medium_risk_count > 0:
        risk_detail = []
        if high_risk_count > 0:
            risk_detail.append(f"{high_risk_count} بنداً عالي الخطورة")
        if medium_risk_count > 0:
            risk_detail.append(f"{medium_risk_count} بنداً متوسط الخطورة")
        sentences.append(f"أظهر التحليل وجود {' و '.join(risk_detail)} تتطلب المراجعة والانتباه.")
    else:
        sentences.append("لم يتم العثور على أي بنود عالية أو متوسطة الخطورة، وتعتبر بنود العقد متوازنة ومستقرة.")
        
    # Sentence 4: Key high-risk warnings if any
    high_risk_clauses = [c for c in classified_clauses if c.get("predicted_risk_level", "").lower() == "high"]
    if high_risk_clauses:
        warnings = []
        # Get up to 2 unique types of high-risk warnings
        seen_types = set()
        for c in high_risk_clauses:
            ctype = c.get("predicted_type_clause", "")
            warning_text = c.get("warning", "")
            if ctype not in seen_types and warning_text:
                seen_types.add(ctype)
                warnings.append(warning_text)
            if len(warnings) >= 2:
                break
                
        if warnings:
            sentences.append("أبرز المخاطر المكتشفة: " + " بالإضافة إلى " .join(warnings))
            
    # Sentence 5: Call to action
    sentences.append("يوصى بمراجعة البنود المظللة باللون الأحمر بدقة ومناقشة تعديلها مع الطرف الآخر قبل توقيع العقد رسمياً.")
    
    return " ".join(sentences)