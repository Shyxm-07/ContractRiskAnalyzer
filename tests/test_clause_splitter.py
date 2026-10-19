"""
test_clause_splitter.py – Tests for clause_splitter.py

Covers Phase 2 requirements:
  • Numbered clauses (e.g. '1.', '1.1', 'Section 2', 'Article IV')
  • Unnumbered paragraphs (preamble, recitals, standard text fallback)
  • Stable, deterministic clause IDs referencing the document
  • Extraction of clause numbers (clause_number attribute)
  • Preservation of original clause text
  • Multi-page document segmentation with page number tracking
  • Sequential document order indexing
"""

import pymupdf as fitz
import pytest

from clause_splitter import (
    Clause,
    extract_clause_number,
    segment_document,
    split_into_clauses,
)
from document_processor import extract_pdf_document


class TestSplitIntoClauses:
    def test_empty_text(self):
        assert split_into_clauses("") == []
        assert split_into_clauses("   ") == []

    def test_numbered_headings(self):
        text = (
            "1. Definitions\n"
            "This section defines key terms of the agreement.\n\n"
            "2. Payment Terms\n"
            "Client shall pay within 30 days of invoice.\n\n"
            "3. Termination\n"
            "Either party may terminate with 30 days notice."
        )
        clauses = split_into_clauses(text)
        assert len(clauses) >= 3
        assert all(isinstance(c, Clause) for c in clauses)
        assert clauses[0].clause_number == "1"
        assert clauses[1].clause_number == "2"
        assert clauses[2].clause_number == "3"

    def test_subsections_and_decimal_numbering(self):
        text = (
            "1.1 Service Scope\n"
            "Provider will deliver web development services.\n\n"
            "1.2 Deliverables\n"
            "Deliverables will be reviewed within 10 business days.\n\n"
            "2.1 Invoicing\n"
            "Invoices will be sent on the first of each month."
        )
        clauses = split_into_clauses(text)
        assert len(clauses) >= 3
        assert clauses[0].clause_number == "1.1"
        assert clauses[1].clause_number == "1.2"
        assert clauses[2].clause_number == "2.1"

    def test_article_and_section_prefixes(self):
        text = (
            "Section 1. Confidentiality\n"
            "Confidential Information will not be disclosed.\n\n"
            "Article II. Warranties\n"
            "Provider warrants the software functions as described.\n\n"
            "Clause 5. Indemnification\n"
            "Provider indemnifies Client against third-party claims."
        )
        clauses = split_into_clauses(text)
        assert len(clauses) >= 3
        assert clauses[0].clause_number == "Section 1"
        assert clauses[1].clause_number == "Article II"
        assert clauses[2].clause_number == "Clause 5"

    def test_paragraph_fallback(self):
        """Without headings, the splitter should fall back to paragraphs."""
        text = (
            "This is the first unnumbered paragraph with enough content to be a clause.\n\n"
            "This is the second unnumbered paragraph with different content for testing.\n\n"
            "This is a third unnumbered paragraph that should also be split out properly."
        )
        clauses = split_into_clauses(text)
        assert len(clauses) >= 2
        # Fallback paragraphs have no formal clause numbers
        assert clauses[0].heading.startswith("Paragraph")

    def test_clause_indices_are_sequential(self):
        text = (
            "Section 1 Overview\nContent for section one.\n\n"
            "Section 2 Details\nContent for section two.\n\n"
            "Section 3 Conclusion\nContent for section three."
        )
        clauses = split_into_clauses(text)
        for i, clause in enumerate(clauses):
            assert clause.index == i

    def test_stable_clause_ids_generated(self):
        text = (
            "1. Scope\nContent for scope.\n\n"
            "2. Term\nContent for term."
        )
        clauses_a = split_into_clauses(text, document_id="doc_xyz123")
        clauses_b = split_into_clauses(text, document_id="doc_xyz123")

        # Stable IDs must be consistent across multiple runs for identical documents
        assert len(clauses_a) == len(clauses_b)
        for ca, cb in zip(clauses_a, clauses_b):
            assert ca.clause_id == cb.clause_id
            assert "doc_xyz123" in ca.clause_id
            assert ca.document_id == "doc_xyz123"

    def test_preserves_original_text(self):
        text = (
            "1. Definitions\n"
            "Original line 1.\n"
            "Original line 2 with specific legal words."
        )
        clauses = split_into_clauses(text)
        assert len(clauses) >= 1
        assert clauses[0].original_text is not None
        assert "Original line 1" in clauses[0].original_text


class TestExtractClauseNumber:
    @pytest.mark.parametrize(
        "heading,expected",
        [
            ("1. Definitions", "1"),
            ("1.2 Scope of Work", "1.2"),
            ("1.2.3 Detailed Specs", "1.2.3"),
            ("Section 4. Payment Terms", "Section 4"),
            ("Article III. Governing Law", "Article III"),
            ("Clause 7. Liability", "Clause 7"),
            ("(a) Direct Damages", "(a)"),
            ("A. General Provisions", "A"),
            ("CONFIDENTIALITY", None),
            ("Preamble", None),
        ],
    )
    def test_extract_clause_number(self, heading, expected):
        assert extract_clause_number(heading) == expected


class TestSegmentMultiPageDocument:
    def test_multi_page_clause_page_attribution(self):
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 50), "1. Definitions\nFirst page definitions text.")
        p2 = doc.new_page()
        p2.insert_text((50, 50), "2. Payment Terms\nSecond page payment terms text.")
        pdf_bytes = doc.tobytes()
        doc.close()

        extracted = extract_pdf_document("multipage.pdf", pdf_bytes)
        clauses = segment_document(extracted)

        assert len(clauses) >= 2
        assert clauses[0].heading == "1. Definitions"
        assert clauses[0].page_number == 1
        assert 1 in clauses[0].page_numbers

        assert clauses[1].heading == "2. Payment Terms"
        assert clauses[1].page_number == 2
        assert 2 in clauses[1].page_numbers


class TestSampleContract:
    def test_sample_contract_splits_and_metadata(self):
        from sample_contract import SAMPLE_CONTRACT_TEXT

        clauses = split_into_clauses(SAMPLE_CONTRACT_TEXT, document_id="doc_sample_msa")
        assert len(clauses) >= 10
        # Check that numbered clauses in the sample contract got numbers
        numbered = [c for c in clauses if c.clause_number is not None]
        assert len(numbered) >= 8
        # Check stable IDs and doc references
        for c in clauses:
            assert c.document_id == "doc_sample_msa"
            assert c.clause_id.startswith("doc_sample")
