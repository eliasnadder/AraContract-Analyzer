"""
Segmentation router for contract clauses.
Uses article-based extraction logic ported from the dataset pipeline (step4.py).
Handles المادة N patterns, Arabic numerals, multi-contract files, and paragraph fallback.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from app.models.schemas import (
    SegmentationRequest,
    SegmentationResponse,
    ErrorResponse,
    FileSegmentationResponse
)

import re
import unicodedata
from typing import List

from app.services.extraction_service import extract_text_from_file
from app.core.config import settings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Patterns ──────────────────────────────────────────────────────────────────

# Detects a new contract boundary inside a file (headings like # عقد / # صيغة)
_CONTRACT_BOUNDARY = re.compile(
    r'^#{1,2}\s+\*{0,2}\s*(?:عقد|صيغة|نموذج|إنذار)',
    re.MULTILINE | re.UNICODE
)

# Detects article headers: المادة N (supports both Western and Arabic-Indic numerals)
_ARTICLE_PATTERN = re.compile(
    r'(?:المادة\s*[\d١٢٣٤٥٦٧٨٩٠]+\s*[-–—]?\s*)',
    re.UNICODE
)

_SUBCLAUSE_LETTERS = "أبجدهوزحط"
# Regex يدعم النصوص النظيفة (أ-) والنصوص المشوهة من الـ PDF (-أ) مع تجاهل النقاط (Bullets)
_SUBCLAUSE_PATTERN = re.compile(
    # تجاهل الأسطر والنقاط والمسافات الوهمية
    r"(?:^|\n)[\s●○\u200b]*"
    r"(?:"
    r"([أبجدهوزحط])[\s\u200b]*[-–—\.]"  # الحالة الأولى: حرف ثم شريطة (نص نظيف)
    r"|"
    # الحالة الثانية: شريطة ثم حرف (تشوه PDF)
    r"[-–—\.][\s\u200b]*([أبجدهوزحط])"
    r")"
    r"[\s\u200b]+",                     # يجب أن يتبعه مسافة ليبدأ النص
    re.UNICODE | re.MULTILINE
)

# Ordinal markers fallback: أولاً:, ثانياً:, ثالثاً:, etc.
_ORDINAL_PATTERN = re.compile(
    r'(?:^|\n)\s*(أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً|سابعاً|ثامناً|تاسعاً|عاشراً)\s*[:\.\s]',
    re.UNICODE | re.MULTILINE,
)

# Pattern لعلامات الخيار في نماذج العقود
_OPTION_HEADER_RE = re.compile(
    r'\[\s*(?:الخيار\s+(?:الأول|الثاني|الثالث|الرابع|[١-٩\d]+)[^\]]*)\]',
    re.UNICODE,
)


def _clean_preamble(raw: str) -> str:
    """
    ينظف نص الـ preamble من:
    1. علامات اختيار الخيارات  → [ الخيار الأول: ... ]
    2. الأسطر المشوهة ذات نسبة عربية منخفضة جداً
    """
    # ① إزالة option headers
    text = _OPTION_HEADER_RE.sub('', raw).strip()

    # ② فلترة الأسطر المشوهة سطراً بسطر
    clean_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        total = len(re.sub(r'\s', '', line))
        if total == 0:
            continue
        arabic = len(re.findall(r'[\u0600-\u06FF]', line))
        # احتفظ بالسطر فقط إذا كان 60%+ من محتواه عربي
        if arabic / total >= 0.60:
            clean_lines.append(line)

    return ' '.join(clean_lines).strip()

# ── Text Helpers ───────────────────────────────────────────────────────────────


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text: remove diacritics, unify hamza/alef forms."""
    text = unicodedata.normalize("NFKC", text)
    # Remove tashkeel (harakat)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize alef variants
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    # Normalize alef maqsura
    text = text.replace('ى', 'ي')
    return text

# # ── Core Extraction ────────────────────────────────────────────────────────────


def split_subclauses(article_text: str, article_num: str) -> list[dict]:
    """
    Hybrid sub-clause splitting exactly matching step4.py's formatting.
    """
    matches = list(_SUBCLAUSE_PATTERN.finditer(article_text))

    if not matches:
        return [
            {
                "article_num": article_num,
                "parent_article_num": None,
                "sub_clause": "",
                "text": article_text,
            }
        ]

    # التقاط المقدمة وتنظيفها من نقاط الـ PDF الوهمية
    preamble = article_text[: matches[0].start()].strip()
    preamble = re.sub(r'[\u200b●○]+', '', preamble).strip()

    # ↓↓↓ السطر الجديد: تنظيف بدل الاستخدام المباشر
    clean_pre = _clean_preamble(preamble)

    results = []
    for i, m in enumerate(matches):
        # التقاط الحرف الأبجدي سواء كان بالترتيب الصحيح (أ-) أو المقلوب (-أ)
        sub_letter = m.group(1) or m.group(2)

        body_start = m.end()
        body_end = matches[i + 1].start() if i + \
            1 < len(matches) else len(article_text)
        sub_body = article_text[body_start:body_end].strip()

        # تنظيف جسم البند
        sub_body = re.sub(r'[\u200b●○]+', '', sub_body).strip()

        # استخدام الـ \n لربط المقدمة بالبند الفرعي كما يفعل step4.py تماماً
        if clean_pre and len(clean_pre) >= 10:
            full_text = f"{clean_pre}\n{sub_letter}- {sub_body}"
        else:
            full_text = f"{sub_letter}- {sub_body}"

        results.append(
            {
                "article_num": f"{article_num}_{sub_letter}",
                "parent_article_num": article_num,
                "sub_clause": sub_letter,
                "text": full_text.strip(),
            }
        )

    return results


def clean_clause_text(text: str) -> str:
    """Remove markdown artifacts, fix PDF RTL glitches, and normalize whitespace (step4.py style)."""
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\*{1,2}([^*]*)\*{1,2}', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[—–\-]{5,}', '', text)

    # 1. تنظيف رصاصات الـ PDF الوهمية
    text = re.sub(r'[\u200b●○]+', '', text)

    # 2. دمج الأسطر المكسورة مع الحفاظ على هيكلية الـ Hybrid
    text = re.sub(r'(?<!\n)\n(?![ \t]*[أبجدهوزحط][ \t]*[-–—\.])', ' ', text)

    # 3. التحسين الجديد: إصلاح التصاق الكلمات بعلامات الترقيم (إضافة مسافة بعد الفاصلة/النقطة إذا تبعها حرف)
    text = re.sub(r'([،.؛:])(?=[أ-يa-zA-Z])', r'\1 ', text)

    # 4. التحسين الجديد: إصلاح تشوه الأقواس المقلوبة بسبب الـ RTL (مثل: )(المالك -> (المالك))
    text = re.sub(r'\)\((.*?)(?=[\s،.]|$)', r'(\1)', text)

    # 5. إزالة المسافات الشاذة قبل علامات الترقيم (مثل: العقد . -> العقد.)
    text = re.sub(r'\s+([،.؛:])', r'\1', text)

    # 6. توحيد المسافات
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ── إضافات جديدة ──────────────────────────────────────────────────────

    # ⑦ إزالة علامات اختيار الخيارات التي تصل أحياناً لجسم البند
    text = _OPTION_HEADER_RE.sub('', text)

    # ⑧ إصلاح الأقواس المعكوسة المتبقية (حالات لا يعالجها الـ extractor)
    text = re.sub(r'\)\(([^)(]{1,60})\)', r'(\1)', text)

    # ⑩ إزالة ○ و ● المتبقية في منتصف الجملة
    text = re.sub(r'(?<=[^\n])[●○■◆]', ' ', text)

    return text.strip()


def _extract_articles_single(text: str) -> list[dict]:
    """
    Extract individual articles from a single contract block.
    """
    matches = list(_ARTICLE_PATTERN.finditer(text))
    if not matches:
        return []

    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()
        num_match = re.search(r"[\d١٢٣٤٥٦٧٨٩٠]+", m.group())
        num = num_match.group() if num_match else str(i + 1)
        articles.append({"article_num": num, "text": article_text})

    return articles


def _extract_from_single_contract(text: str) -> List[str]:
    """
    Segment a single contract using the hybrid sub-clause strategy.
    """
    # 1. Extract articles
    articles = _extract_articles_single(text)

    # 2. Fallback if no article patterns found
    if not articles:
        # Check for ordinal markers (أولاً:, ثانياً:, etc.) as an alternative split
        ordinal_matches = list(_ORDINAL_PATTERN.finditer(text))
        if len(ordinal_matches) > 1:
            # Split by ordinal markers
            clauses = []
            for i, m in enumerate(ordinal_matches):
                start = m.start()
                end = ordinal_matches[i + 1].start() if i + \
                    1 < len(ordinal_matches) else len(text)
                clause_text = text[start:end].strip()
                clause_text = clean_clause_text(clause_text)
                if len(clause_text) >= 50:
                    clauses.append(clause_text)
            return clauses

        # Standard paragraph fallback
        paragraphs = [p.strip()
                      for p in text.split("\n\n") if len(p.strip()) > 60]
        articles = [{"article_num": str(i + 1), "text": p}
                    for i, p in enumerate(paragraphs)]

    clauses = []
    for art in articles:
        # 3. Apply hybrid sub-clause splitting
        splits = split_subclauses(art["text"].strip(), art["article_num"])
        for sc in splits:
            art_text = clean_clause_text(sc["text"])
            if len(art_text) >= 50:
                clauses.append(art_text)

    return clauses


def segment_arabic_text(text: str) -> List[str]:
    """
    Segments Arabic contract text into clauses, matching the step4.py pipeline logic.
    """
    if not text or not isinstance(text, str):
        return []

    text = text.strip()

    boundary_matches = list(_CONTRACT_BOUNDARY.finditer(text))
    if len(boundary_matches) > 1:
        all_clauses = []
        if boundary_matches[0].start() > 0:
            preamble = text[:boundary_matches[0].start()]
            all_clauses.extend(_extract_from_single_contract(preamble))

        for i, bm in enumerate(boundary_matches):
            end = boundary_matches[i + 1].start() if i + \
                1 < len(boundary_matches) else len(text)
            sub_text = text[bm.start():end]
            all_clauses.extend(_extract_from_single_contract(sub_text))

        return all_clauses

    return _extract_from_single_contract(text)


# ── Router ─────────────────────────────────────────────────────────────────────

@router.post(
    "/segment",
    response_model=SegmentationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def segment_contract(request: SegmentationRequest):
    """
    Segment contract text into clauses.

    - **text**: The raw contract text to segment
    """
    try:
        clauses = segment_arabic_text(request.text)
        return SegmentationResponse(
            clauses=clauses,
            count=len(clauses),
            message=f"Text segmented into {len(clauses)} clauses",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during segmentation: {str(e)}",
        )


@router.post(
    "/segment/file",
    response_model=FileSegmentationResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
)
async def segment_contract_file(file: UploadFile = File(...)):
    """
    Upload a contract file (PDF or image), extract text, then segment into clauses.

    - **file**: The contract file (PDF, PNG, JPG, JPEG, TIFF, BMP)
    """
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the limit of {settings.MAX_FILE_SIZE // (1024*1024)} MB",
        )

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not allowed. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        extracted_text, is_scanned = extract_text_from_file(str(file_path))

        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the file.",
            )

        clauses = segment_arabic_text(extracted_text)

        return FileSegmentationResponse(
            filename=file.filename,
            is_scanned=is_scanned,
            extracted_text_preview=extracted_text[:500],
            clauses=clauses,
            count=len(clauses),
            message=f"File processed and segmented into {len(clauses)} clauses",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error segmenting file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}",
        )
    finally:
        if file_path.exists():
            file_path.unlink()
