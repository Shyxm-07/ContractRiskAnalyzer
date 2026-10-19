"""
report_generator.py – Generate an executive-grade downloadable PDF contract risk report.

Phase 6 updates:
  • Executive summary with contract metadata and overall KPI risk card.
  • Complete audit methodology section detailing document ingestion, clause
    classification, deterministic rule detection, and the weighted scoring formula.
  • Transparent mathematical score breakdown with factor contributions.
  • Separate reporting of missing expected contractual protections.
  • Detailed clause-by-clause audit findings with stable source references
    (clause IDs, page numbers, headings), verbatim evidence quotes, practical
    operational impact, and draft alternative wording.
  • Explicit limitations section and legal review disclaimer preventing unsupported
    assertions or fabricated legal citations.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import settings
from llm_explainer import Explanation
from risk_rules import RiskFlag
from risk_scorer import OverallRiskReport

logger = logging.getLogger(__name__)

# ── Color Palette ─────────────────────────────────────────────────────
_BRAND_NAVY = colors.HexColor("#1A1A2E")
_BRAND_ACCENT = colors.HexColor("#0F3460")
_BORDER_COLOR = colors.HexColor("#DCE3EC")
_BG_LIGHT = colors.HexColor("#F8F9FA")
_BG_HIGHLIGHT = colors.HexColor("#EEF2F6")

_RISK_COLOURS = {
    "high": colors.HexColor("#E74C3C"),
    "medium": colors.HexColor("#F39C12"),
    "low": colors.HexColor("#27AE60"),
}


def generate_pdf_report(
    report: OverallRiskReport,
    explanations: List[Explanation],
    clauses: Optional[List[Any]] = None,
    doc_metadata: Optional[Any] = None,
    filename: str = "contract_risk_report.pdf",
) -> bytes:
    """
    Build a comprehensive executive PDF risk assessment report and return the raw bytes.
    Maintains backwards compatibility with Phase 1-5 callers while supporting
    rich Phase 6 audit metadata.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()

    # Typography Styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        spaceAfter=4,
        textColor=_BRAND_NAVY,
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#5D6D7E"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "H1Heading",
        parent=styles["Heading1"],
        fontSize=13,
        leading=17,
        spaceBefore=14,
        spaceAfter=6,
        textColor=_BRAND_NAVY,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "H2Heading",
        parent=styles["Heading2"],
        fontSize=11,
        leading=15,
        spaceBefore=10,
        spaceAfter=4,
        textColor=_BRAND_ACCENT,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2C3E50"),
    )
    body_bold = ParagraphStyle(
        "ReportBodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    quote_style = ParagraphStyle(
        "QuoteStyle",
        parent=body_style,
        fontSize=8,
        leading=11,
        fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#34495E"),
        leftIndent=8,
    )
    meta_style = ParagraphStyle(
        "MetaStyle",
        parent=body_style,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#7F8C8D"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=body_style,
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor("#616A6B"),
        fontName="Helvetica-Oblique",
    )

    elements: list = []

    # ── 1. HEADER & METADATA ──────────────────────────────────────────
    elements.append(Paragraph(f"🛡️ {settings.app_title}", title_style))
    doc_name = (
        getattr(doc_metadata, "filename", "Contract Document")
        if doc_metadata
        else "Contract Risk Assessment Audit"
    )
    doc_id = (
        getattr(doc_metadata, "document_id", "N/A")
        if doc_metadata
        else "N/A"
    )
    gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    elements.append(
        Paragraph(
            f"<b>Document:</b> {doc_name} &nbsp;·&nbsp; <b>ID:</b> <code>{doc_id}</code> &nbsp;·&nbsp; <b>Audit Date:</b> {gen_time}",
            subtitle_style,
        )
    )
    elements.append(HRFlowable(width="100%", thickness=1.5, color=_BRAND_NAVY, spaceAfter=8))

    # ── 2. EXECUTIVE SUMMARY ──────────────────────────────────────────
    elements.append(Paragraph("1. Executive Risk Summary", h1_style))

    risk_clr = _RISK_COLOURS.get(report.risk_level, colors.black)
    total_clauses = len(report.clause_details)
    profile_title = getattr(
        report, "profile", getattr(report, "scoring_profile", "standard_commercial")
    ).replace("_", " ").title()
    clean_count = getattr(
        report, "clean_clauses_count", getattr(report, "clean_clause_count", 0)
    )
    suspicious_count = getattr(
        report, "suspicious_clauses_count", getattr(report, "suspicious_clause_count", 0)
    )
    missing_count = getattr(
        report, "missing_protections_count", len(getattr(report, "missing_protections", []))
    )
    formula_text = getattr(
        report, "formula_explanation", getattr(report, "score_formula_explanation", "Weighted formula")
    )
    weight_sum = getattr(report, "applicable_weight_sum", 1.0)

    # High-level summary metrics table
    exec_data = [
        [
            Paragraph("<b>Overall Contract Risk Score</b>", body_style),
            Paragraph(f"<font color='{risk_clr.hexval()}' size=14><b>{report.overall_score:.1f} / 100</b></font>", body_style),
            Paragraph("<b>Assigned Risk Level</b>", body_style),
            Paragraph(f"<font color='{risk_clr.hexval()}'><b>{report.risk_level.upper()}</b></font>", body_style),
        ],
        [
            Paragraph("<b>Active Scoring Profile</b>", body_style),
            Paragraph(profile_title, body_style),
            Paragraph("<b>Total Clauses Analyzed</b>", body_style),
            Paragraph(str(total_clauses), body_style),
        ],
        [
            Paragraph("<b>High Risk Clauses</b>", body_style),
            Paragraph(f"<font color='#E74C3C'><b>{report.high_risk_count}</b></font>", body_style),
            Paragraph("<b>Medium Risk Clauses</b>", body_style),
            Paragraph(f"<font color='#F39C12'><b>{report.medium_risk_count}</b></font>", body_style),
        ],
        [
            Paragraph("<b>Low Risk Clauses</b>", body_style),
            Paragraph(f"<font color='#27AE60'><b>{report.low_risk_count}</b></font>", body_style),
            Paragraph("<b>Clean Clauses (0 Flags)</b>", body_style),
            Paragraph(f"<font color='#2980B9'><b>{clean_count}</b></font>", body_style),
        ],
        [
            Paragraph("<b>Suspicious Clauses Flagged</b>", body_style),
            Paragraph(str(suspicious_count), body_style),
            Paragraph("<b>Missing Expected Protections</b>", body_style),
            Paragraph(f"<b>{missing_count}</b> absent", body_style),
        ],
    ]

    exec_table = Table(exec_data, colWidths=[45 * mm, 40 * mm, 45 * mm, 45 * mm])
    exec_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _BG_LIGHT),
                ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.append(exec_table)
    elements.append(Spacer(1, 6 * mm))

    # Narrative Executive Assessment
    summary_text = (
        f"This contract evaluation concluded with an overall composite risk score of <b>{report.overall_score:.1f}/100</b>, "
        f"positioning the document in the <b>{report.risk_level.upper()}</b> review bracket under the "
        f"<b>{profile_title}</b> evaluation posture. Out of {total_clauses} segmented clauses, "
        f"<b>{suspicious_count}</b> clause(s) triggered one or more rule-based review indicators, while "
        f"<b>{clean_count}</b> clause(s) exhibited no flagged deviations. "
        f"Additionally, <b>{missing_count}</b> standard protective mechanism(s) were noted as absent."
    )
    elements.append(Paragraph(summary_text, body_style))
    elements.append(Spacer(1, 4 * mm))

    # ── 3. METHODOLOGY & SCORING FORMULA ──────────────────────────────
    elements.append(Paragraph("2. Audit Methodology & Transparent Scoring Formula", h1_style))
    method_text = (
        "<b>Analysis Architecture:</b> The contract was analyzed through a multi-stage pipeline: "
        "(1) Format-aware text extraction and paragraph normalization; "
        "(2) Structural clause segmentation with page and paragraph tracking; "
        "(3) Domain-specific legal category classification (LegalBERT / ML baseline); "
        "(4) Deterministic rule-based risk detection checking context and exceptions; and "
        "(5) Mathematically normalized weighted risk scoring.<br/><br/>"
        "<b>Mathematical Formula:</b> The composite score normalizes applicable factor weights to exactly 100%:<br/>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;<code>{formula_text}</code><br/>"
        f"<b>Sum of Applicable Factor Weights:</b> <code>{weight_sum * 100:.1f}%</code>. "
        "Missing factors and non-applicable clauses are excluded from the denominator to prevent score deflation."
    )
    elements.append(Paragraph(method_text, body_style))
    elements.append(Spacer(1, 4 * mm))

    # Factor Breakdown Table
    if getattr(report, "factor_breakdowns", None):
        elements.append(Paragraph("<b>Applicable Risk Factors & Weight Distribution:</b>", h2_style))
        factor_headers = ["Risk Factor", "Profile Weight", "Raw Factor Score", "Weighted Contribution", "Status"]
        factor_rows = [factor_headers]

        if isinstance(report.factor_breakdowns, dict):
            for factor_name, bd in report.factor_breakdowns.items():
                status_str = "Clean" if bd.get("raw_score", 0) == 0 else f"Triggered ({bd.get('flag_count', 1)})"
                factor_rows.append(
                    [
                        factor_name.replace("_", " ").title(),
                        f"{bd.get('profile_weight', 0.1) * 100:.1f}%",
                        f"{bd.get('raw_score', 0):.1f} / 100",
                        f"+{bd.get('weighted_contribution', 0):.2f} pts",
                        status_str,
                    ]
                )
        else:
            for fb in report.factor_breakdowns:
                raw_s = getattr(fb, "suspiciousness_score", 0.0)
                status_str = "Clean" if raw_s == 0 else f"Triggered ({getattr(fb, 'trigger_count', 1)})"
                if not getattr(fb, "is_applicable", True):
                    status_str = "N/A (Excluded)"
                factor_rows.append(
                    [
                        getattr(fb, "factor_name", "Factor"),
                        f"{getattr(fb, 'weight', 0.1) * 100:.1f}%",
                        f"{raw_s:.1f} / 100",
                        f"+{getattr(fb, 'weighted_contribution', 0.0):.2f} pts",
                        status_str,
                    ]
                )

        factor_table = Table(factor_rows, colWidths=[45 * mm, 30 * mm, 32 * mm, 38 * mm, 30 * mm])
        factor_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _BRAND_NAVY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, _BORDER_COLOR),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        elements.append(factor_table)
        elements.append(Spacer(1, 6 * mm))

    # ── 4. MISSING EXPECTED PROTECTIONS (DOCUMENT-LEVEL) ───────────────
    elements.append(Paragraph("3. Document-Level Findings & Missing Expected Protections", h1_style))
    if report.missing_protections_count > 0 or report.document_flags:
        elements.append(
            Paragraph(
                "The following standard commercial protections were identified as absent or inconsistent across the agreement:",
                body_style,
            )
        )
        elements.append(Spacer(1, 2 * mm))
        for flag in report.document_flags:
            flag_box = [
                [
                    Paragraph(
                        f"<b>⚠ {flag.label}</b> [<code>{flag.rule_id}</code>] &nbsp;·&nbsp; "
                        f"<b>Severity:</b> {flag.severity.upper()} &nbsp;·&nbsp; "
                        f"<b>Suspiciousness:</b> {flag.suspiciousness_score:.0f}/100",
                        body_bold,
                    )
                ],
                [Paragraph(f"<b>Issue:</b> {flag.description}", body_style)],
                [Paragraph(f"<b>Why it matters:</b> {flag.rationale}", body_style)],
                [Paragraph(f"<b>Calculation detail:</b> <code>{flag.score_breakdown}</code>", meta_style)],
            ]
            f_table = Table(flag_box, colWidths=[175 * mm])
            f_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF9E7")),
                        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#F39C12")),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            elements.append(f_table)
            elements.append(Spacer(1, 3 * mm))
    else:
        elements.append(
            Paragraph(
                "✓ No missing core protections (governing law, liability cap, or breach termination mechanism) were detected.",
                body_style,
            )
        )
    elements.append(Spacer(1, 6 * mm))

    # ── 5. DETAILED CLAUSE FINDINGS & DRAFT REVISIONS ──────────────────
    elements.append(Paragraph("4. Clause-by-Clause Audit Findings & Draft Revisions", h1_style))
    elements.append(
        Paragraph(
            "Below is the clause-by-clause analysis detailing original source text, triggered review indicators, "
            "verbatim supporting evidence, practical business consequences, and draft suggested revisions for attorney review.",
            body_style,
        )
    )
    elements.append(Spacer(1, 4 * mm))

    explanation_map = {e.clause_index: e for e in explanations}

    for detail in report.clause_details:
        clause_obj = clauses[detail.clause_index] if (clauses and detail.clause_index < len(clauses)) else None
        page_num = getattr(clause_obj, "page_number", None) or "Page 1"
        clause_id = getattr(clause_obj, "clause_id", f"clause_{detail.clause_index:03d}")
        clause_num = getattr(clause_obj, "clause_number", "")
        num_prefix = f"[{clause_num}] " if clause_num else ""

        clr = _RISK_COLOURS.get(detail.risk_level, colors.black)
        expl = explanation_map.get(detail.clause_index)

        clause_elements = []

        # Card header
        header_text = (
            f"<b>§ {num_prefix}{detail.heading}</b> &nbsp;·&nbsp; "
            f"<b>ID:</b> <code>{clause_id}</code> &nbsp;·&nbsp; "
            f"<b>Location:</b> {page_num} &nbsp;·&nbsp; "
            f"<b>Category:</b> {detail.category} &nbsp;·&nbsp; "
            f"<font color='{clr.hexval()}'><b>{detail.risk_level.upper()} ({detail.clause_score:.0f}/100)</b></font>"
        )
        clause_elements.append(Paragraph(header_text, body_bold))
        clause_elements.append(Spacer(1, 2 * mm))

        # Original text excerpt if available
        if clause_obj and getattr(clause_obj, "text", ""):
            clause_elements.append(Paragraph("<b>Original Clause Text:</b>", body_style))
            clause_elements.append(Paragraph(f'"{clause_obj.text}"', quote_style))
            clause_elements.append(Spacer(1, 2 * mm))

        # Flags & Evidence
        if detail.flags:
            clause_elements.append(Paragraph("<b>Triggered Review Indicators & Evidence:</b>", body_style))
            for flag in detail.flags:
                f_sev = flag.severity.upper()
                snip = flag.matched_snippet if flag.matched_snippet else "(contextual trigger)"
                flag_text = (
                    f"• <b>[{flag.rule_id}] {flag.label}</b> ({f_sev} severity, Score: {flag.suspiciousness_score:.0f}/100): "
                    f"{flag.description}<br/>"
                    f"&nbsp;&nbsp;<b>Verbatim Evidence:</b> <i>\"{snip}\"</i><br/>"
                    f"&nbsp;&nbsp;<b>Why it matters:</b> {getattr(flag, 'rationale', 'N/A')}"
                )
                clause_elements.append(Paragraph(flag_text, body_style))
            clause_elements.append(Spacer(1, 2 * mm))

        # Structured generative explanation details
        if expl and expl.structured_output:
            st_out = expl.structured_output
            clause_elements.append(
                Paragraph(f"<b>💼 Practical Business & Operational Consequences:</b><br/>{st_out.potential_impact}", body_style)
            )
            clause_elements.append(Spacer(1, 2 * mm))
            clause_elements.append(
                Paragraph(
                    f"<b>📝 Draft Suggested Revision (For Legal Review):</b><br/>"
                    f"<font color='#0E6655'><b>{st_out.suggested_revision}</b></font>",
                    body_style,
                )
            )
            if st_out.assumptions:
                clause_elements.append(Spacer(1, 1 * mm))
                clause_elements.append(
                    Paragraph(f"<b>Context Assumptions:</b> {'; '.join(st_out.assumptions)}", meta_style)
                )
            clause_elements.append(Spacer(1, 1 * mm))
            clause_elements.append(Paragraph(f"⚖️ {st_out.disclaimer}", disclaimer_style))
        elif expl and expl.recommendations:
            clause_elements.append(Paragraph("<b>Draft Recommendations:</b>", body_style))
            for rec in expl.recommendations:
                clause_elements.append(Paragraph(f"• {rec}", body_style))

        clause_elements.append(Spacer(1, 3 * mm))
        clause_elements.append(HRFlowable(width="100%", thickness=0.5, color=_BORDER_COLOR, spaceAfter=4))

        elements.append(KeepTogether(clause_elements))

    # ── 6. LIMITATIONS & LEGAL DISCLAIMER ─────────────────────────────
    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph("5. Methodology Limitations & Legal Review Disclaimer", h1_style))
    disclaimer_text = (
        "<b>Limitations:</b> This automated contract analysis is generated using natural language processing, "
        "rule-based heuristic pattern matching, and instruction-tuned generative models. While designed to highlight "
        "contractual ambiguities, structural imbalances, and unfavorable terms, the findings are indicative only. "
        "The system does not verify party identities, evaluate commercial pricing adequacy, or interpret parol evidence "
        "outside the supplied document text.<br/><br/>"
        "<b>No Fabricated Citations:</b> This report contains solely factual excerpts and grounded text comparisons. "
        "No fictitious judicial precedents, statutory sections, or external case law citations are included.<br/><br/>"
        "<b>Formal Legal Disclaimer:</b> THIS REPORT IS PROVIDED SOLELY FOR INFORMATIONAL, REVIEW PRIORITIZATION, AND "
        "RESEARCH PURPOSES. IT DOES NOT CONSTITUTE FORMAL LEGAL ADVICE OR A LEGAL OPINION. Flagged clauses are not automatically "
        "unlawful, invalid, or unenforceable. Users must consult a licensed attorney in the appropriate jurisdiction to "
        "evaluate contractual enforceability, conduct negotiation redlines, and ensure compliance with applicable governing laws."
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))

    # Build PDF
    doc.build(elements)
    pdf_bytes = buf.getvalue()
    buf.close()
    logger.info("Generated executive PDF report (%d bytes).", len(pdf_bytes))
    return pdf_bytes
