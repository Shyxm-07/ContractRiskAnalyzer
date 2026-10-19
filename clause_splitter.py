"""
clause_splitter.py – Segment contract documents into structured clauses.

Phase 2 implementation:
  • Typed Clause data structure with rich source metadata
  • Robust heading, numbering, Roman numeral, and structural clue detection
  • Extraction of clause numbers (e.g. '1.1', 'Section 2', 'Article IV')
  • Preservation of original clause text alongside normalized text
  • Stable, deterministic clause IDs referencing parent documents
  • Page-number mapping across multi-page documents
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Heading and Clause Number Patterns ────────────────────────────────

_HEADING_PATTERNS: list[re.Pattern[str]] = [
    # 1. Numbered headings: "1. Definitions", "1.1 Service Level", "Section 1 Overview", "Article 2 Term"
    re.compile(
        r"^(?:Section|Article|Clause)?\s*\d+(?:\.\d+)*\.?\s+[A-Z]",
        re.IGNORECASE,
    ),
    # 2. Section/Article prefix with number only: "Section 1", "Article II", "Clause 4:"
    re.compile(
        r"^(?:Section|Article|Clause)\s+(?:\d+(?:\.\d+)*|[IVXLCDM]+)[\.:]?(?:\s+[A-Z].*)?$",
        re.IGNORECASE,
    ),
    # 3. Roman numeral headings: "Article I. Term", "II. Payment"
    re.compile(
        r"^(?:Article|Section)?\s*[IVXLCDM]+\.\s+[A-Z]",
        re.IGNORECASE,
    ),
    # 4. Lettered or parenthesized subsections: "A. Scope", "(a) Indemnification"
    re.compile(
        r"^(?:[A-Z]|\([a-z\d]+\))\.\s+[A-Z]",
    ),
    # 5. ALL-CAPS headings (8 to 80 chars, e.g. "LIMITATION OF LIABILITY")
    re.compile(r"^[A-Z][A-Z\s\d\.:\-_]{6,80}$"),
    # 6. Title-Case headings (short lines, <= 80 chars)
    re.compile(r"^(?:[A-Z][a-z]+(?:\s+(?:of|and|the|in|to|for|by|on|or|an?|with)\s+)?){1,6}[A-Z][a-z]+\.?$"),
    # 7. Standard legal transition markers
    re.compile(r"^(?:RECITALS|WITNESSETH|NOW,?\s+THEREFORE,?\s+.*)$", re.IGNORECASE),
]

_SECTION_KEYWORDS = {
    "indemnification", "indemnity", "limitation of liability",
    "termination", "confidentiality", "non-compete", "noncompete",
    "governing law", "intellectual property", "payment", "payment terms",
    "warranty", "warranties", "force majeure", "data protection", "privacy",
    "dispute resolution", "arbitration", "assignment", "amendment",
    "representations", "covenants", "definitions", "recitals",
    "notices", "severability", "entire agreement", "waiver", "compliance",
    "insurance", "independent contractor", "remedies",
}


# ── Typed Data Structure ──────────────────────────────────────────────

@dataclass
class Clause:
    """
    Represents a structured contract clause with source metadata.

    Attributes:
      index: Document order (0-indexed).
      heading: The detected heading or section title.
      text: The normalized clause text for NLP and risk analysis.
      clause_id: Stable, deterministic identifier (e.g. 'doc_1234_c001').
      clause_number: Extracted clause number string (e.g. '1', '2.1', 'Section 3').
      original_text: Preserved original un-normalized text slice.
      page_number: First page number where clause appears (1-indexed), if available.
      page_numbers: List of all pages spanned by this clause.
      document_id: Reference ID to the parent document.
      char_count: Length of normalized text.
      word_count: Word count of normalized text.
    """
    index: int
    heading: str
    text: str
    clause_id: str = ""
    clause_number: Optional[str] = None
    original_text: str = ""
    page_number: Optional[int] = None
    page_numbers: List[int] = field(default_factory=list)
    document_id: str = ""
    char_count: int = 0
    word_count: int = 0

    def __post_init__(self):
        if not self.original_text:
            self.original_text = self.text
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.clause_id:
            # Generate deterministic stable ID
            prefix = self.document_id[:10] if self.document_id else "cl"
            snippet_hash = hashlib.sha256(self.text[:100].encode("utf-8")).hexdigest()[:6]
            self.clause_id = f"{prefix}_c{self.index + 1:03d}_{snippet_hash}"
        if not self.clause_number:
            self.clause_number = extract_clause_number(self.heading)

    def __repr__(self) -> str:
        snippet = self.text[:60].replace("\n", " ")
        page_info = f", page={self.page_number}" if self.page_number else ""
        return (
            f"Clause({self.clause_id}, #{self.index}, num={self.clause_number!r}, "
            f"heading={self.heading!r}{page_info}, text={snippet!r}…)"
        )


# ── Heading and Numbering Helpers ─────────────────────────────────────

def _is_heading(line: str) -> bool:
    """Return True if the given line represents a section or clause heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    for pat in _HEADING_PATTERNS:
        if pat.match(stripped):
            return True
    lower = stripped.lower().rstrip(":. ")
    if lower in _SECTION_KEYWORDS:
        return True
    return False


def extract_clause_number(heading: str) -> Optional[str]:
    """
    Extract the clause number or identifier from a heading string.

    Examples:
      "1. Definitions"        -> "1"
      "1.2 Scope of Services" -> "1.2"
      "Section 3. Payment"    -> "Section 3"
      "Article IV. Term"      -> "Article IV"
      "(a) Indemnity"         -> "(a)"
      "A. General"            -> "A"
    """
    if not heading:
        return None
    stripped = heading.strip()

    # 1. Section/Article/Clause with number or Roman numeral
    m = re.match(r"^((?:Section|Article|Clause)\s+(?:\d+(?:\.\d+)*|[IVXLCDM]+))", stripped, re.IGNORECASE)
    if m:
        return m.group(1)

    # 2. Numbered prefix: "1.", "1.1", "1.1.2"
    m = re.match(r"^(\d+(?:\.\d+)*)\.?", stripped)
    if m:
        return m.group(1)

    # 3. Roman numerals alone: "IV.", "II."
    m = re.match(r"^([IVXLCDM]+)\.", stripped)
    if m:
        return m.group(1)

    # 4. Lettered or parenthesized: "(a)", "A."
    m = re.match(r"^(\([a-z\d]+\)|[A-Z]\.)", stripped)
    if m:
        return m.group(1).rstrip(".")

    return None


# ── Core Segmentation Logic ───────────────────────────────────────────

def split_into_clauses(
    text: str,
    *,
    min_clause_chars: int = 20,
    document_id: str = "",
) -> List[Clause]:
    """
    Split contract text into individual structured ``Clause`` objects.

    Strategy:
      1. Traverse text line-by-line detecting headings, numbering, and keywords.
      2. Accumulate clause body text while preserving original text.
      3. Fall back to paragraph-level splitting if no headings are present.
      4. Merge insignificant fragments (< min_clause_chars) into preceding clauses.
      5. Assign stable clause IDs, sequence indices, and clause numbers.
    """
    if not text or not text.strip():
        return []

    lines = text.splitlines()
    raw_clauses: list[dict[str, str]] = []
    current_heading = "Preamble"
    current_lines: list[str] = []

    for line in lines:
        if _is_heading(line):
            body = "\n".join(current_lines).strip()
            if body:
                raw_clauses.append({"heading": current_heading, "text": body})
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush final clause
    body = "\n".join(current_lines).strip()
    if body:
        raw_clauses.append({"heading": current_heading, "text": body})

    # ── Fallback: paragraph split if no headings were detected ────────
    if len(raw_clauses) <= 1:
        logger.info("No section headings detected – splitting by paragraphs.")
        paragraphs = re.split(r"\n{2,}", text.strip())
        raw_clauses = [
            {"heading": f"Paragraph {i + 1}", "text": p.strip()}
            for i, p in enumerate(paragraphs)
            if p.strip()
        ]

    # ── Merge tiny fragments into previous clause ─────────────────────
    merged: list[dict[str, str]] = []
    for clause in raw_clauses:
        if merged and len(clause["text"]) < min_clause_chars:
            merged[-1]["text"] += "\n\n" + clause["text"]
        else:
            merged.append(clause)

    clauses: List[Clause] = []
    for i, c in enumerate(merged):
        clause_obj = Clause(
            index=i,
            heading=c["heading"],
            text=c["text"],
            original_text=c["text"],
            document_id=document_id,
            clause_number=extract_clause_number(c["heading"]),
        )
        clauses.append(clause_obj)

    logger.info("Segmented contract into %d clauses.", len(clauses))
    return clauses


# ── Multi-Page Document Segmenter ─────────────────────────────────────

def segment_document(
    extracted_doc,  # ExtractedDocument
    *,
    min_clause_chars: int = 20,
) -> List[Clause]:
    """
    Segment an ExtractedDocument into Clause objects, preserving page numbers
    and document references.

    Parameters:
      extracted_doc: ExtractedDocument instance from document_processor.
      min_clause_chars: Minimum character threshold before merging fragments.
    """
    text = extracted_doc.normalized_text
    doc_id = extracted_doc.metadata.document_id
    clauses = split_into_clauses(text, min_clause_chars=min_clause_chars, document_id=doc_id)

    # Correlate clauses with page boundaries if pages are available
    if extracted_doc.pages and len(extracted_doc.pages) > 0:
        # Build page text lookup offsets
        for clause in clauses:
            search_snippet = clause.text[:60].strip()
            matching_pages: list[int] = []

            for page in extracted_doc.pages:
                if search_snippet and search_snippet in page.text:
                    matching_pages.append(page.page_number)

            if matching_pages:
                clause.page_number = matching_pages[0]
                clause.page_numbers = matching_pages
            else:
                # Default to page 1 if not located
                clause.page_number = 1
                clause.page_numbers = [1]

    return clauses
