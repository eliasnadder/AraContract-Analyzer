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

    # ② النقطتان في بداية السطر (RTL artifact)
    #    "\n:الأول السيد" → "\nالأول: السيد"
    text = re.sub(r'(?m)^:([^\s:،؛\n]{1,40})', r'\1:', text)

    # ③ الأقواس المعكوسة الكاملة: ")(المالك)" → "(المالك)"
    text = re.sub(r'\)\(([^)(،\n]{1,50})\)', r'(\1)', text)          # مع )
    text = re.sub(r'\)\(([^)(،\n]{1,50})(?=[،\s\n●]|$)', r'(\1)', text)  # بدون )
    # ④ كلمة متصقة بنقطة بدون مسافة: "نهاية.كلمة" → "نهاية. كلمة"
    text = re.sub(r'([.،؛:])([^\s\d\n])', r'\1 \2', text)

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
        page_text = page.get_text()
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
