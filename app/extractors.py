# app/extractors.py
import os
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
from .ocr import extract_text_from_image


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in reader.pages)

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".docx":
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)

        elif ext in {".png", ".jpg", ".jpeg"}:
            ocr_text = extract_text_from_image(file_path)
            return f"[OCR] {ocr_text}" if ocr_text else "[OCR] No text found"

        else:
            return "[ERROR] Unsupported file type"

    except Exception as e:
        return f"[EXTRACT ERROR] {e}"
    
    
# ====================================================================
# Format,Library,Robustness
# PDF,PyPDF2,Handles encrypted
# TXT,open(),"errors=""ignore"""
# DOCX,python-docx,Paragraphs only
# Image,TrOCR,Tagged with [OCR]