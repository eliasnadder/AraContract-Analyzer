"""
Text extraction service for contract files.
Handles PDF (digital and scanned) and image files.
"""

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import os
from typing import Tuple

import re


_LEGIT_SHORT_ARABIC_WORDS = {
    'من', 'في', 'لا', 'ما', 'لم', 'لن', 'قد', 'بل', 'هل', 'أن', 'إن',
    'كي', 'لو', 'أو', 'يا', 'لك', 'له', 'بك', 'به', 'لي', 'بي', 'عن',
    'إذ', 'ثم', 'مع', 'كل', 'أي', 'إذا', 'و', 'ف', 'ب', 'ل', 'ك', 'س',
}


def _is_short_fragment(tok: str) -> bool:
    """جزء عربي قصير (1-2 حرف) وليس كلمة شرعية معروفة — على الأرجح تشوه استخراج."""
    return bool(re.fullmatch(r'[\u0621-\u064A]{1,2}', tok or '')) \
        and tok not in _LEGIT_SHORT_ARABIC_WORDS


def _garbledness_score(text: str) -> float:
    """
    مقياس تقريبي لدرجة "التشوّه" في نص عربي مستخرج من صفحة PDF: نسبة
    الشظايا القصيرة جداً (1-2 حرف) غير الشرعية إلى إجمالي عدد الكلمات
    العربية في النص.

    نستخدمه للمفاضلة بين طريقتين مختلفتين لاستخراج نفس الصفحة (الترتيب
    الافتراضي لـ PyMuPDF مقابل sort=True) واختيار الأقل تشوهاً، لأن بعض
    الأسطر ذات الأقواس/الفواصل المتداخلة تُربك خوارزمية ترتيب القراءة
    الداخلية لـ PyMuPDF وتُنتج كلمات مبتورة ومبعثرة (تحقّقنا من هذا فعلياً
    عبر مقارنة النص الخام لملف PDF حقيقي مع الخرج المشوّه).
    """
    tokens = re.findall(r'[\u0621-\u064A]+', text)
    if not tokens:
        return 0.0
    short = sum(1 for t in tokens if len(t) <=
                2 and t not in _LEGIT_SHORT_ARABIC_WORDS)
    return short / len(tokens)


def fix_broken_arabic_words(text: str) -> str:
    """
    يدمج الحروف العربية المفصولة خطأً بمسافات وهمية أثناء استخراج PyMuPDF.
    مثال: 'أ ي' -> 'أي'، 'اس تالمها' -> 'استالمها'، 'موافق ة' -> 'موافقة'
    """
    tokens = text.split(' ')
    i = 0
    while i < len(tokens) - 1:
        cur, nxt = tokens[i], tokens[i + 1]
        nxt2 = tokens[i + 2] if i + 2 < len(tokens) else None

        if _is_short_fragment(cur) and _is_short_fragment(nxt):
            tokens[i + 1] = cur + nxt
            del tokens[i]
            continue
        if _is_short_fragment(nxt) and _is_short_fragment(nxt2):
            i += 1
            continue
        if _is_short_fragment(cur) and re.match(r'^[\u0621-\u064A]{3,}', nxt or ''):
            tokens[i + 1] = cur + nxt
            del tokens[i]
            continue
        if re.fullmatch(r'[\u0621-\u064A]{3,}', cur or '') and _is_short_fragment(nxt):
            tokens[i] = cur + nxt
            del tokens[i + 1]
            continue
        i += 1
    return ' '.join(tokens)


# نمط "النقطة المنتقلة": كلمة قصيرة تظهر بعد نقطة نهاية الجملة بدل قبلها،
# بسبب تشوه في ترتيب القراءة (bidi). مثال حقيقي: "للعمل. عليها كما" يجب أن
# تكون "للعمل عليها. كما". لا يكفي افتراض أن هذا يحدث فقط في نهاية سطر PDF
# (اختبرنا هذا على نص حقيقي ووجدنا حالات تحدث في منتصف سطر متصل أيضاً)، لذا
# نقبل أيضاً الحالة التي يتبع فيها الكلمةَ المتأخرة رابطٌ جملة شائع (و/كما/
# حيث...)، وهو إشارة موثوقة لبداية جملة جديدة تالية. نُبقي الفحص بالكامل
# داخل lookahead غير مستهلِك حتى لا نحذف المسافة الأصلية الفاصلة بينهما.
_TRAILING_WORD_AFTER_PERIOD_RE = re.compile(
    r'\.[ \t]+([\u0621-\u064A]{1,20})'
    r'(?=[ \t]*(?:\n|$|[ \t]+(?:و[\u0621-\u064A]|كما|حيث|إذ|بينما|ثم|لكن|علماً|مع)))',
)


def _fix_trailing_word_after_period(text: str) -> str:
    """ينقل الكلمة القصيرة المتأخرة عن النقطة إلى ما قبلها."""
    return _TRAILING_WORD_AFTER_PERIOD_RE.sub(lambda m: f' {m.group(1)}.', text)


# نمط "قوس العنوان المقلوب": عنوان مادة مثل "(إلزامية )المقدمة" يجب أن يكون
# "(إلزامية المقدمة)" — كلمة العنوان الأخيرة انزاحت لتصبح بعد القوس المغلق
# بدل قبله بسبب نفس تشوه bidi. الإشارة المميزة (لتفادي المساس بنص سليم):
# مسافة قبل القوس المغلق، وكلمة ملتصقة به مباشرة بلا مسافة بعده — النص
# السليم يكون عادة عكس ذلك تماماً (بلا مسافة قبل القوس، ومسافة بعده).
_TITLE_PAREN_REORDER_RE = re.compile(
    r'\(([\u0621-\u064A][\u0621-\u064A ]{0,45}?)\s+\)([\u0621-\u064A]{1,25})\b'
)


def _fix_title_paren_reorder(text: str) -> str:
    """يعيد ترتيب عنوان مادة انزاحت آخر كلماته إلى خارج القوس المغلق."""
    return _TITLE_PAREN_REORDER_RE.sub(r'(\1 \2)', text)


def _normalize_arabic_pdf_text(text: str) -> str:
    """
    إصلاح التشوهات الشائعة في نص PDF العربي المستخرج بـ PyMuPDF.
    يُطبَّق مرة واحدة على النص الكامل قبل إرساله للـ segmenter.
    """
    # ① Hamza Presentation Forms — أكثر تشوه شيوعاً في PDF العربي
    #    "اإلقامة" → "الإقامة"   "اإلخالل" → "الإخلال"
    text = text.replace('اإل', 'الإ')
    text = text.replace('اإل', 'الإ')   # run twice: some PDFs double-encode
    text = text.replace('األ', 'الأ')
    text = text.replace('اآل', 'الآ')

    # ⑤ إصلاح كلمة "المادة" حين تُقسَّم بمسافة وهمية بسبب تبرير نص PDF
    #    (kashida / text-justification يُدرج مسافات وهمية داخل الكلمة نفسها)
    #    "الم ادة" / "الما دة" / "المادة" (سليمة أصلاً) ← "المادة"
    #    نطبّقه قبل fix_broken_arabic_words لأن تلك الدالة تُصلح فقط شظايا من 1-2 حرف،
    #    بينما "الم" هنا 3 أحرف فتفلت من الإصلاح العام.
    text = re.sub(r'ا\s?ل\s?م\s?ا\s?د\s?ة', 'المادة', text)

    # ② النقطتان في بداية السطر (RTL artifact)
    #    "\n:الأول السيد" → "\nالأول: السيد"
    text = re.sub(r'(?m)^:([^\s:،؛\n]{1,40})', r'\1:', text)

    # ③ الأقواس المعكوسة الكاملة: ")(المالك)" → "(المالك)"
    text = re.sub(r'\)\(([^)(،\n]{1,50})\)', r'(\1)', text)          # مع )
    # بدون )
    text = re.sub(r'\)\(([^)(،\n]{1,50})(?=[،\s\n●]|$)', r'(\1)', text)
    # ④ كلمة متصقة بنقطة بدون مسافة: "نهاية.كلمة" → "نهاية. كلمة"
    #    يجب أن يسبق إصلاح "النقطة المنتقلة" (⑥) لضمان وجود مسافة موحّدة
    #    قبل الكلمة المتأخرة بغض النظر عن حالة النص الخام.
    text = re.sub(r'([.،؛:])([^\s\d\n])', r'\1 \2', text)

    # ⑥ إصلاح نمط "النقطة المنتقلة" (انظر التعليق أعلى الدالة). يجب أن
    #    يُطبَّق هنا وليس في segment.py لأنه يعتمد على أسطر PDF الأصلية
    #    (\n) قبل أن يدمجها clean_clause_text في فقرات متصلة.
    text = _fix_trailing_word_after_period(text)

    # ⑦ إصلاح "قوس العنوان المقلوب" (انظر التعليق أعلى الدالة)
    text = _fix_title_paren_reorder(text)

    text = fix_broken_arabic_words(text)
    return text


def extract_text_from_file(file_path: str) -> Tuple[str, bool]:
    """
    Extract text from a contract file (PDF or image).
    Detects if the file is digital PDF or scanned/image and uses appropriate method.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (extracted_text: str, is_scanned: bool)
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        return _extract_text_from_pdf(file_path)
    elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return _extract_text_from_image(file_path), True
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


def _extract_page_text_best_effort(page) -> str:
    """
    يستخرج نص الصفحة بطريقتين مختلفتين ويختار الأقل تشوهاً.

    PyMuPDF's get_text("text") الافتراضي يعتمد خوارزمية ترتيب قراءة داخلية
    قد تفشل مع أسطر معقدة (أقواس متداخلة + نقطتان + نص RTL طويل)، فتُنتج
    كلمات مبتورة ومبعثرة. تحققنا من هذا فعلياً على عقد حقيقي: النص الكامل
    كان موجوداً في الـ PDF لكن get_text("text") فشل في إعادة بنائه بالترتيب
    الصحيح. get_text("text", sort=True) يرتّب المقاطع حسب الإحداثيات بدل
    الاعتماد على الاستدلال الداخلي، وغالباً ما يُصلح هذه الحالات — لكنه قد
    يكون أسوأ في صفحات أخرى (أعمدة متعددة مثلاً)، لذلك نقارن ونختار الأفضل
    لكل صفحة على حدة بدل التبديل الشامل غير المشروط.
    """
    default_text = page.get_text("text")
    try:
        sorted_text = page.get_text("text", sort=True)
    except TypeError:
        # إصدارات أقدم من PyMuPDF قد لا تدعم sort=
        sorted_text = default_text

    if not default_text.strip():
        return sorted_text
    if not sorted_text.strip():
        return default_text

    return sorted_text if _garbledness_score(sorted_text) < _garbledness_score(default_text) \
        else default_text


def _extract_text_from_pdf(file_path: str) -> Tuple[str, bool]:
    """
    Extract text from PDF. Attempts digital extraction first, falls back to OCR if needed.

    Returns:
        Tuple of (text: str, is_scanned: bool)
    """
    # Try to extract text digitally first
    doc = fitz.open(file_path)
    text = ""
    is_scanned = False

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = _extract_page_text_best_effort(page)
        if page_text.strip():
            text += page_text + "\n"
        else:
            # If a page has no text, it might be scanned
            is_scanned = True

    doc.close()

    # If we got text and it's not mostly whitespace, assume digital
    if text.strip() and not is_scanned:
        return _normalize_arabic_pdf_text(text.strip()), False

    # Otherwise, fall back to OCR
    ocr_text = _extract_text_via_ocr(file_path)
    return _normalize_arabic_pdf_text(ocr_text), True


def _extract_text_from_image(file_path: str) -> str:
    """
    Extract text from an image file using OCR.

    Args:
        file_path: Path to the image file

    Returns:
        Extracted text string
    """
    image = Image.open(file_path)
    # Use Tesseract with Arabic language
    custom_config = r'-l ara --oem 1 --psm 3'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text.strip()


def _extract_text_via_ocr(file_path: str) -> str:
    """
    Extract text from PDF using OCR (convert pages to images then OCR).

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text string
    """
    # Convert PDF to images
    images = convert_from_path(file_path, dpi=300)
    text = ""

    for image in images:
        # Use Tesseract with Arabic language
        custom_config = r'-l ara --oem 1 --psm 3'
        page_text = pytesseract.image_to_string(image, config=custom_config)
        text += page_text + "\n"

    return text.strip()
