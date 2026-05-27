# test_segmentation.py
from app.routers.segment import segment_arabic_text

def test_segment_basic_markers():
    text = "المادة 1: نص طويل جدا يتجاوز الحد الأدنى المسموح به لتقسيم البنود. البند 2: نص طويل جدا يتجاوز الحد الأدنى المسموح به لتقسيم البنود."
    clauses = segment_arabic_text(text)
    assert len(clauses) >= 2
    assert "المادة 1" in clauses[0]
    assert "البند 2" in clauses[1]

def test_min_length_filter():
    text = "المادة 1: قصير"  # < 30 chars
    clauses = segment_arabic_text(text)
    assert len(clauses) == 0

def test_words_markers():
    text = "أولاً: يلتزم الطرف الأول بكافة البنود الواردة في هذا العقد. ثانياً: يلتزم الطرف الثاني بدفع المستحقات المالية في وقتها المحدد."
    clauses = segment_arabic_text(text)
    assert len(clauses) == 2
    assert "أولاً" in clauses[0]
    assert "ثانياً" in clauses[1]
