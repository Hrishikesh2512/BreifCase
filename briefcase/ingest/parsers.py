from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from briefcase.text_utils import normalize_whitespace

# File extensions that are parsed as plain text / source code as-is.
_CODE_AND_TEXT = {
    ".txt", ".text", ".log", ".rst", ".tex",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cc", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
    ".bash", ".zsh", ".sql", ".r", ".m", ".pl", ".lua", ".dart", ".vue",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".properties",
    ".csv", ".tsv", ".json", ".jsonl", ".xml", ".svg",
}
_MARKDOWN = {".md", ".markdown", ".mdx"}
_HTML = {".html", ".htm", ".xhtml"}
_PDF = {".pdf"}
_DOCX = {".docx"}
_PPTX = {".pptx"}
_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


@dataclass
class ParsedDocument:
    text: str
    title: str
    source_type: str
    metadata: dict = field(default_factory=dict)


class ParserError(RuntimeError):
    pass


def supported_extensions() -> set[str]:
    exts = _CODE_AND_TEXT | _MARKDOWN | _HTML | _PDF | _DOCX | _PPTX
    from briefcase.ingest import ocr

    if ocr.available_for_image():
        exts = exts | _IMAGE
    return exts


def _title_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or None


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return "Untitled"


def _parse_pdf(payload: bytes) -> str:
    import io

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(payload))
        if reader.is_encrypted:
            # Many PDFs are encrypted with an empty owner password; that decrypts
            # transparently. A real user password cannot be recovered.
            try:
                decrypted = reader.decrypt("")
            except Exception:
                decrypted = 0
            if not decrypted:
                raise ParserError(
                    "This PDF is password-protected. Open it, remove the password "
                    "(or 'Print to PDF' an unlocked copy), then upload again."
                )
        pages = [page.extract_text() or "" for page in reader.pages]
    except ParserError:
        raise
    except Exception as exc:  # malformed / unsupported / truncated PDFs
        raise ParserError(
            f"Could not read this PDF; it may be corrupted or an unsupported "
            f"variant ({type(exc).__name__})."
        ) from exc

    text = "\n\n".join(pages)
    if text.strip():
        return text

    # No text layer: this is a scanned / image-only PDF. Try OCR if available.
    from briefcase.ingest import ocr

    if ocr.available_for_pdf():
        ocr_text = ocr.ocr_pdf_bytes(payload)
        if ocr_text.strip():
            return ocr_text
        raise ParserError(
            "OCR ran on this PDF but found no readable text. The scan may be too "
            "low-resolution or not text at all."
        )
    raise ParserError(
        "This PDF has no selectable text; it looks like a scanned or image-only "
        "document. Reading it needs OCR, which is not installed. Install it with: "
        "pip install rapidocr-onnxruntime pdf2image  (poppler-utils must also be present)."
    )


def _parse_docx(payload: bytes) -> str:
    import io

    from docx import Document

    doc = Document(io.BytesIO(payload))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _parse_pptx(payload: bytes) -> str:
    import io

    from pptx import Presentation

    prs = Presentation(io.BytesIO(payload))
    parts: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        parts.append(f"# Slide {i}")
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text)
    return "\n".join(parts)


def _parse_markdown(raw: str) -> tuple[str, str | None]:
    from bs4 import BeautifulSoup
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark")
    tokens = md.parse(raw)
    title = next(
        (t.content for t in tokens if t.type == "inline" and t.content.strip()), None
    )
    html = md.render(raw)
    text = BeautifulSoup(html, "lxml").get_text("\n")
    return text, title


def _parse_html(raw: str) -> tuple[str, str | None]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    text = soup.get_text("\n")
    return text, title


def parse(payload: bytes, *, filename: str | None = None) -> ParsedDocument:
    """Parse raw bytes into text based on the file extension."""
    ext = Path(filename or "").suffix.lower()

    if ext in _PDF:
        text = _parse_pdf(payload)
        source_type = "pdf"
        title = _title_from_filename(filename) or _first_line(text)
    elif ext in _DOCX:
        text = _parse_docx(payload)
        source_type = "docx"
        title = _title_from_filename(filename) or _first_line(text)
    elif ext in _PPTX:
        text = _parse_pptx(payload)
        source_type = "pptx"
        title = _title_from_filename(filename) or _first_line(text)
    elif ext in _MARKDOWN:
        text, md_title = _parse_markdown(payload.decode("utf-8", errors="replace"))
        source_type = "markdown"
        title = _title_from_filename(filename) or md_title or _first_line(text)
    elif ext in _HTML:
        text, html_title = _parse_html(payload.decode("utf-8", errors="replace"))
        source_type = "html"
        title = html_title or _title_from_filename(filename) or _first_line(text)
    elif ext in _IMAGE:
        from briefcase.ingest import ocr

        if not ocr.available_for_image():
            raise ParserError(
                "Reading text from images needs OCR, which is not installed. "
                "Install it with: pip install rapidocr-onnxruntime"
            )
        text = ocr.ocr_image_bytes(payload)
        source_type = "image"
        if not text.strip():
            raise ParserError("OCR found no readable text in this image.")
        title = _title_from_filename(filename) or _first_line(text)
    elif ext in _CODE_AND_TEXT or ext == "":
        text = payload.decode("utf-8", errors="replace")
        source_type = "code" if ext not in ("", ".txt", ".text", ".log") else "text"
        title = _title_from_filename(filename) or _first_line(text)
    else:
        raise ParserError(f"Unsupported file type: {ext or 'unknown'}")

    text = normalize_whitespace(text)
    if not text:
        raise ParserError("No extractable text found in file")
    return ParsedDocument(text=text, title=title or "Untitled", source_type=source_type)


def parse_html_string(raw: str, *, url: str) -> ParsedDocument:
    text, title = _parse_html(raw)
    text = normalize_whitespace(text)
    if not text:
        raise ParserError("No extractable text found at URL")
    return ParsedDocument(
        text=text,
        title=title or url,
        source_type="url",
        metadata={"url": url},
    )
