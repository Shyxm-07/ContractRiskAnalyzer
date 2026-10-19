"""
document_processor.py – Document processing pipeline for contract analysis.

Phase 2 implementation:
  • Typed data structures for documents and pages (Pydantic models)
  • PDF text extraction via PyMuPDF with page tracking
  • DOCX text extraction via python-docx with paragraph and table support
  • File extension, file header (magic bytes), and size validation
  • Scanned PDF detection reporting OCR requirement
  • Whitespace normalization and extraction artifact removal
  • Clear custom exception hierarchy for corrupt, empty, unsupported, and scanned files
"""

from __future__ import annotations

import hashlib
import io
import logging
import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pymupdf as fitz
from docx import Document as DocxDocument
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


# ── Exception Hierarchy ──────────────────────────────────────────────

class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""


class EmptyDocumentError(DocumentProcessingError):
    """Raised when an uploaded document contains no text or is 0 bytes."""


class CorruptDocumentError(DocumentProcessingError):
    """Raised when a file cannot be opened or is structurally corrupted."""


class UnsupportedDocumentError(DocumentProcessingError):
    """Raised when a file format or MIME type is not supported."""


class ScannedDocumentError(DocumentProcessingError):
    """Raised when a PDF is a scanned document with images but no selectable text."""


# ── Typed Data Structures ─────────────────────────────────────────────

class PageContent(BaseModel):
    """Represents the extracted text and metadata of a single page."""
    page_number: int  # 1-indexed
    text: str
    char_count: int = 0
    word_count: int = 0
    has_images: bool = False


class DocumentMetadata(BaseModel):
    """Comprehensive metadata for an ingested contract document."""
    document_id: str
    filename: str
    file_type: str  # "pdf", "docx", "raw"
    file_size_bytes: int
    page_count: int = 1
    char_count: int = 0
    word_count: int = 0
    mime_type: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_scanned: bool = False
    ocr_needed: bool = False


class ExtractedDocument(BaseModel):
    """Container holding the raw, normalized, and page-level document contents."""
    metadata: DocumentMetadata
    raw_text: str
    normalized_text: str
    pages: List[PageContent] = Field(default_factory=list)


# ── Validation Functions ──────────────────────────────────────────────

def validate_file(
    filename: str,
    file_size_bytes: int,
    file_bytes: Optional[bytes] = None,
) -> Optional[str]:
    """
    Validate file extension, file size, and file signature (magic bytes).

    Returns an error message string if invalid, or None if valid.
    """
    if file_size_bytes == 0:
        return "The uploaded file is empty (0 bytes)."

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        return (
            f"File exceeds the {settings.max_upload_size_mb} MB upload limit "
            f"({file_size_bytes / (1024 * 1024):.1f} MB)."
        )

    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        return (
            f"Unsupported file type '{ext}'. "
            f"Allowed types: {', '.join(settings.allowed_extensions)}"
        )

    # Validate header magic bytes if byte stream is available
    if file_bytes:
        if ext == ".pdf":
            if not file_bytes.startswith(b"%PDF"):
                return "Invalid PDF header: file does not start with standard '%PDF' signature."
        elif ext == ".docx":
            # DOCX is a zip archive (starts with PK\x03\x04 or PK\x05\x06)
            if not (file_bytes.startswith(b"PK\x03\x04") or file_bytes.startswith(b"PK\x05\x06")):
                return "Invalid DOCX header: file is not a valid Microsoft Word OpenXML archive."

    return None


def generate_document_id(filename: str, file_bytes: bytes) -> str:
    """Generate a deterministic 12-char SHA-256 hash document ID."""
    content_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
    safe_name = re.sub(r"[^\w]", "_", Path(filename).stem)[:10]
    return f"doc_{safe_name}_{content_hash}"


# ── Text Normalization and Artifact Cleaning ──────────────────────────

def clean_extraction_artifacts(text: str) -> str:
    """
    Clean extraction artifacts and normalize whitespace without altering legal meaning.

    Operations:
      1. Normalize line endings (\r\n -> \n, \r -> \n).
      2. Remove form feed (\\x0c), soft hyphens (\\xad), and null bytes (\\x00).
      3. Rejoin words split across line breaks by hyphens (e.g. 'indemni-\\nfication' -> 'indemnification')
         while preserving intra-line compound words (e.g. 'non-compete', 'third-party').
      4. Remove standalone page number lines (e.g. 'Page 3 of 12', '- 4 -', '— 5 —').
      5. Collapse consecutive horizontal whitespace (tabs, multiple spaces) to a single space.
      6. Strip trailing and leading whitespace from each line.
      7. Collapse 3+ consecutive newlines to 2 (standard paragraph break).
    """
    if not text:
        return ""

    # 1. Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove control characters & extraction symbols
    text = text.replace("\x0c", "\n").replace("\xad", "").replace("\x00", "")

    # 3. Rejoin hyphenated line breaks (word broken across newline)
    # Match letters, hyphen, newline, whitespace, letters
    text = re.sub(r"(\b[a-zA-Z]{2,})-\n\s*([a-zA-Z]{2,}\b)", r"\1\2", text)

    # 4. Remove standalone page-number artifact lines
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    page_num_pattern = re.compile(
        r"^(?:page\s+\d+(?:\s+of\s+\d+)?|[-—–]\s*\d+\s*[-—–]|\d+\s*/\s*\d+)$",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        if page_num_pattern.match(stripped):
            continue
        # 5 & 6. Collapse horizontal whitespace
        normalized_line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(normalized_line)

    cleaned_text = "\n".join(cleaned_lines)

    # 7. Collapse excessive consecutive blank lines (max 2 newlines = 1 blank line)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


# ── Scanned PDF Detection ─────────────────────────────────────────────

def detect_scanned_pdf(doc: fitz.Document) -> tuple[bool, int, int]:
    """
    Inspect a PyMuPDF document to determine if it is a scanned PDF requiring OCR.

    Returns:
      (is_scanned, total_chars, total_images)
    """
    total_chars = 0
    total_images = 0

    for page in doc:
        text = page.get_text("text").strip()
        total_chars += len(text)
        images = page.get_images(full=True)
        total_images += len(images)

    page_count = max(doc.page_count, 1)
    chars_per_page = total_chars / page_count

    # If document has zero extractable characters or very low character count
    # and has images, it is a scanned document requiring OCR.
    if total_chars == 0 and total_images > 0:
        return True, total_chars, total_images

    if chars_per_page < 15 and total_images >= doc.page_count:
        return True, total_chars, total_images

    return False, total_chars, total_images


# ── Extraction Engines ────────────────────────────────────────────────

def extract_pdf_document(filename: str, file_bytes: bytes) -> ExtractedDocument:
    """Extract text, page metadata, and detect scanned status from a PDF byte stream."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise CorruptDocumentError(f"Corrupt or unreadable PDF: {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise EmptyDocumentError("The PDF document contains 0 pages.")

    is_scanned, total_chars, total_images = detect_scanned_pdf(doc)

    if is_scanned:
        doc.close()
        raise ScannedDocumentError(
            f"The PDF '{filename}' appears to be a scanned document containing "
            f"{total_images} image(s) but no selectable text. OCR is required to process it."
        )

    pages: List[PageContent] = []
    raw_page_texts: List[str] = []

    for i, page in enumerate(doc):
        page_raw = page.get_text("text")
        cleaned_page = clean_extraction_artifacts(page_raw)
        images = page.get_images()

        pages.append(
            PageContent(
                page_number=i + 1,
                text=cleaned_page,
                char_count=len(cleaned_page),
                word_count=len(cleaned_page.split()),
                has_images=len(images) > 0,
            )
        )
        raw_page_texts.append(page_raw)

    doc.close()

    raw_text = "\n\n".join(raw_page_texts).strip()
    normalized_text = clean_extraction_artifacts(raw_text)

    if not normalized_text:
        raise EmptyDocumentError(
            f"The PDF '{filename}' contains no extractable text."
        )

    doc_id = generate_document_id(filename, file_bytes)
    metadata = DocumentMetadata(
        document_id=doc_id,
        filename=filename,
        file_type="pdf",
        file_size_bytes=len(file_bytes),
        page_count=len(pages),
        char_count=len(normalized_text),
        word_count=len(normalized_text.split()),
        mime_type="application/pdf",
        is_scanned=False,
        ocr_needed=False,
    )

    return ExtractedDocument(
        metadata=metadata,
        raw_text=raw_text,
        normalized_text=normalized_text,
        pages=pages,
    )


def extract_docx_document(filename: str, file_bytes: bytes) -> ExtractedDocument:
    """Extract text from a DOCX byte stream, including paragraphs and tables."""
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
    except Exception as exc:
        raise CorruptDocumentError(f"Corrupt or unreadable DOCX file: {exc}") from exc

    raw_paragraphs: list[str] = []

    # 1. Extract paragraphs
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            raw_paragraphs.append(text)

    # 2. Extract table cell contents (important for contract payment schedules & definitions)
    for table in doc.tables:
        for row in table.rows:
            cell_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cell_texts:
                # Deduplicate merged cells
                row_str = " | ".join(dict.fromkeys(cell_texts))
                raw_paragraphs.append(row_str)

    raw_text = "\n\n".join(raw_paragraphs).strip()
    if not raw_text:
        raise EmptyDocumentError(
            f"The DOCX file '{filename}' contains no text or is empty."
        )

    normalized_text = clean_extraction_artifacts(raw_text)
    doc_id = generate_document_id(filename, file_bytes)

    # DOCX files do not have explicit page break objects by default
    pages = [
        PageContent(
            page_number=1,
            text=normalized_text,
            char_count=len(normalized_text),
            word_count=len(normalized_text.split()),
            has_images=False,
        )
    ]

    metadata = DocumentMetadata(
        document_id=doc_id,
        filename=filename,
        file_type="docx",
        file_size_bytes=len(file_bytes),
        page_count=1,
        char_count=len(normalized_text),
        word_count=len(normalized_text.split()),
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        is_scanned=False,
        ocr_needed=False,
    )

    return ExtractedDocument(
        metadata=metadata,
        raw_text=raw_text,
        normalized_text=normalized_text,
        pages=pages,
    )


# ── Public High-Level Dispatchers ─────────────────────────────────────

def process_document(filename: str, file_bytes: bytes) -> ExtractedDocument:
    """
    High-level document processing pipeline.

    Validates file headers and extension, extracts text, removes artifacts,
    normalizes whitespace, and packages the result into an ExtractedDocument.

    Raises:
      - EmptyDocumentError
      - CorruptDocumentError
      - UnsupportedDocumentError
      - ScannedDocumentError
      - DocumentProcessingError
    """
    val_err = validate_file(filename, len(file_bytes), file_bytes)
    if val_err:
        if "Unsupported file type" in val_err:
            raise UnsupportedDocumentError(val_err)
        elif "empty" in val_err.lower():
            raise EmptyDocumentError(val_err)
        elif "Invalid" in val_err:
            raise CorruptDocumentError(val_err)
        else:
            raise DocumentProcessingError(val_err)

    ext = Path(filename).suffix.lower()
    logger.info("Processing document '%s' (%d bytes)", filename, len(file_bytes))

    if ext == ".pdf":
        return extract_pdf_document(filename, file_bytes)
    elif ext == ".docx":
        return extract_docx_document(filename, file_bytes)
    else:
        raise UnsupportedDocumentError(f"No extractor available for '{ext}'.")


def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Backward-compatible convenience wrapper returning the normalized plain text.
    """
    doc = process_document(filename, file_bytes)
    return doc.normalized_text


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Backward-compatible PDF plain text extractor."""
    doc = extract_pdf_document("document.pdf", file_bytes)
    return doc.normalized_text


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Backward-compatible DOCX plain text extractor."""
    doc = extract_docx_document("document.docx", file_bytes)
    return doc.normalized_text
