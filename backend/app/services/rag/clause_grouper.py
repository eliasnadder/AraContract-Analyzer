"""
Clause Grouper module for AraContract RAG system.
يجمع الـ subclauses التي تنتمي لنفس المادة الأم في نص واحد متكامل.
"""

import re
from typing import List
import logging

logger = logging.getLogger(__name__)

MIN_CLAUSE_LENGTH = 100

# Pattern يستخرج عنوان المادة من بداية النص
# مثال: "المادة 7 - (المسؤولية)" أو "المادة - (الصيانة)"
_HEADER_PATTERN = re.compile(
    r'^(المادة\s*[\d١٢٣٤٥٦٧٨٩٠]*\s*[-–]?\s*\([^)]+\))',
    re.UNICODE
)


def _extract_header(clause: str) -> str:
    """
    يستخرج عنوان المادة من بداية النص.
    إذا لم يجد نمط المادة — يرجع أول 50 حرف كمعرّف.
    """
    clause = clause.strip()
    match = _HEADER_PATTERN.match(clause)
    if match:
        return match.group(1).strip()
    # fallback: أول سطر أو أول 50 حرف
    first_line = clause.split("\n")[0].strip()
    return first_line[:50]


def group_by_header(clauses: List[str]) -> List[str]:
    """
    يجمع الـ subclauses التي تشترك بنفس عنوان المادة
    في نص واحد متكامل، بترتيب ظهورها الأصلي في العقد.

    Args:
        clauses: ناتج segment_arabic_text()

    Returns:
        قائمة جديدة — كل عنصر يمثل مادة كاملة مُجمّعة
    """
    if not clauses:
        return []

    groups: dict[str, List[str]] = {}
    order: List[str] = []

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        header = _extract_header(clause)

        if header not in groups:
            groups[header] = [clause]
            order.append(header)
        else:
            # تحقق أن النص ليس مكرراً حرفياً
            if clause not in groups[header]:
                groups[header].append(clause)

    grouped_clauses = []
    for header in order:
        texts = groups[header]

        if len(texts) == 1:
            # بند واحد — لا حاجة للدمج
            full_text = texts[0]
        else:
            # دمج: أول نص كاملاً + باقي النصوص بدون تكرار العنوان
            full_text = texts[0]
            for extra in texts[1:]:
                # احذف العنوان المكرر من بداية كل نص إضافي
                body = _HEADER_PATTERN.sub('', extra).strip()
                # احذف أيضاً أي شريطة أو نقطتين في البداية
                body = re.sub(r'^[\s\-–:]+', '', body).strip()
                if body:
                    full_text += "\n" + body

        full_text = full_text.strip()

        if len(full_text) >= MIN_CLAUSE_LENGTH:
            grouped_clauses.append(full_text)
        else:
            logger.info(
                f"استبعاد بند قصير ({len(full_text)} حرف): {full_text[:50]}"
            )

    logger.info(
        f"Clause Grouping: {len(clauses)} subclause → {len(grouped_clauses)} مادة مُجمّعة"
    )
    return grouped_clauses