"""Optional OCR for scanned / image-only PDFs and image files.

Two backends are auto-detected, in order of preference:
  1. RapidOCR (rapidocr-onnxruntime): self-contained, no system packages.
  2. Tesseract (pytesseract + the `tesseract` binary): the classic engine.

If neither is installed, OCR is simply unavailable and the parser reports a
clear, actionable message. Rasterising PDF pages needs `pdf2image` plus the
`pdftoppm` binary from poppler-utils (already present on most Linux systems).
"""

from __future__ import annotations

import io
import os
import shutil
from functools import lru_cache

from briefcase.config import get_settings


def _enabled() -> bool:
    from briefcase import settings_store

    value = settings_store.get("ocr") or os.getenv("BRIEFCASE_OCR") or get_settings().ocr
    return value.lower() not in ("0", "off", "false", "none")


@lru_cache
def _backend() -> str | None:
    if not _enabled():
        return None
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return "rapidocr"
    except Exception:
        pass
    try:
        import pytesseract  # noqa: F401
        if shutil.which("tesseract"):
            return "tesseract"
    except Exception:
        pass
    return None


def _can_rasterize_pdf() -> bool:
    try:
        import pdf2image  # noqa: F401
    except Exception:
        return False
    return shutil.which("pdftoppm") is not None or shutil.which("pdftocairo") is not None


def available_for_pdf() -> bool:
    return _backend() is not None and _can_rasterize_pdf()


def available_for_image() -> bool:
    return _backend() is not None


def status() -> dict:
    return {
        "backend": _backend(),
        "pdf": available_for_pdf(),
        "image": available_for_image(),
    }


@lru_cache
def _rapidocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _ocr_pil_image(image) -> str:
    backend = _backend()
    if backend == "rapidocr":
        import numpy as np

        result, _ = _rapidocr_engine()(np.asarray(image.convert("RGB")))
        if not result:
            return ""
        return "\n".join(line[1] for line in result)
    if backend == "tesseract":
        import pytesseract

        return pytesseract.image_to_string(image)
    return ""


def ocr_image_bytes(data: bytes) -> str:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        return _ocr_pil_image(img).strip()


def ocr_pdf_bytes(data: bytes, *, dpi: int | None = None, max_pages: int | None = None) -> str:
    from pdf2image import convert_from_bytes

    cfg = get_settings()
    dpi = dpi or cfg.ocr_dpi
    max_pages = max_pages or cfg.ocr_max_pages

    pages = convert_from_bytes(data, dpi=dpi)
    texts: list[str] = []
    for image in pages[:max_pages]:
        page_text = _ocr_pil_image(image).strip()
        if page_text:
            texts.append(page_text)
    return "\n\n".join(texts)
