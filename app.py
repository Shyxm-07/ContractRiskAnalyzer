"""
app.py – Streamlit front-end for the AI Contract Risk Analyser.

Phase 2 updates:
  • Integrated document processing pipeline (PyMuPDF & python-docx)
  • Explicit scanned PDF detection & OCR requirement warning
  • Pre-analysis document extraction view & clause preview
  • Preservation and display of source metadata (page numbers, clause numbers, stable IDs)
  • In-memory processing without persistent storage or external leakage
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Configure logging ─────────────────────────────────────────────────
from config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Local imports ─────────────────────────────────────────────────────
from clause_splitter import Clause, segment_document, split_into_clauses
from classifier import ClassificationResult, classify_clauses
from document_processor import (
    CorruptDocumentError,
    DocumentMetadata,
    DocumentProcessingError,
    EmptyDocumentError,
    ExtractedDocument,
    PageContent,
    ScannedDocumentError,
    UnsupportedDocumentError,
    process_document,
    validate_file,
)
from llm_explainer import Explanation, explain_flags, get_sample_explanations
from report_generator import generate_pdf_report
from risk_rules import RiskFlag, evaluate_clause, evaluate_document
from risk_scorer import ClauseRiskDetail, OverallRiskReport, score_contract
from sample_contract import SAMPLE_CONTRACT_TEXT

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.app_title,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F0C29 0%, #1A1A2E 50%, #16213E 100%);
}
section[data-testid="stSidebar"] * {
    color: #E0E0E0 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #00D2FF !important;
}

/* ── Risk badges ─────────────────────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.risk-high   { background: #FDEDEC; color: #E74C3C; border: 1px solid #E74C3C; }
.risk-medium { background: #FEF9E7; color: #F39C12; border: 1px solid #F39C12; }
.risk-low    { background: #EAFAF1; color: #27AE60; border: 1px solid #27AE60; }

/* ── Metadata badges ─────────────────────────────────────── */
.meta-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px 4px 2px 0;
    background: #EEF2F6;
    color: #334E68;
    border: 1px solid #DCE3EC;
}

/* ── Score card ──────────────────────────────────────────── */
.score-card {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    color: #fff;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
}
.score-card .score-value {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.1;
}
.score-card .score-label {
    font-size: 0.9rem;
    opacity: 0.75;
    margin-top: 4px;
}

/* ── Stat cards ──────────────────────────────────────────── */
.stat-card {
    background: #fff;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #f0f0f0;
}
.stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
}
.stat-card .stat-label {
    font-size: 0.8rem;
    color: #888;
    margin-top: 2px;
}

/* ── Clause card ─────────────────────────────────────────── */
.clause-card {
    background: #FAFBFC;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
    border-left: 4px solid #DEE2E6;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.clause-card.high   { border-left-color: #E74C3C; }
.clause-card.medium { border-left-color: #F39C12; }
.clause-card.low    { border-left-color: #27AE60; }

/* ── Flag chip ───────────────────────────────────────────── */
.flag-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px 4px 2px 0;
}
.flag-chip.high   { background: #FDEDEC; color: #C0392B; }
.flag-chip.medium { background: #FEF9E7; color: #D68910; }
.flag-chip.low    { background: #EAFAF1; color: #1E8449; }

/* ── Upload area ─────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed #00D2FF44;
    border-radius: 12px;
    padding: 8px;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00D2FF;
}

/* ── Comparison diff box ──────────────────────────────────── */
.diff-container {
    display: flex;
    gap: 14px;
    margin-top: 8px;
    margin-bottom: 12px;
}
.diff-box {
    flex: 1;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.84rem;
    line-height: 1.45;
}
.diff-orig {
    background: #FDEDEC;
    border: 1px solid #F5B7B1;
    color: #78281F;
}
.diff-rev {
    background: #EAFAF1;
    border: 1px solid #A9DFBF;
    color: #145A32;
}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🛡️ Contract Risk Analyser")
    st.caption("AI-powered contract risk analysis · Phase 3")
    st.divider()

    st.markdown("### ⚙️ Machine Learning Models")
    classifier_option = st.selectbox(
        "Clause Classifier Engine",
        options=[
            "TF-IDF + Logistic Regression (Local ML Baseline - Recommended)",
            "LegalBERT (AmirLGass/clauseguard-legalbert - Fine-Tuned)",
            "Zero-Shot Transformer (facebook/bart-large-mnli)",
            "Keyword Heuristic Baseline",
        ],
        index=0,
        help="Choose the model used to classify clauses into legal categories.",
    )
    backend_mapping = {
        "TF-IDF + Logistic Regression (Local ML Baseline - Recommended)": "tfidf",
        "LegalBERT (AmirLGass/clauseguard-legalbert - Fine-Tuned)": "legalbert",
        "Zero-Shot Transformer (facebook/bart-large-mnli)": "zeroshot",
        "Keyword Heuristic Baseline": "keyword",
    }
    selected_backend = backend_mapping[classifier_option]

    st.markdown("### ⚖️ Risk Scoring Profile")
    profile_option = st.selectbox(
        "Risk Weighting Perspective",
        options=[
            "Standard Commercial (Balanced)",
            "Vendor-Protective (Payment, IP & Indemnity)",
            "Buyer-Protective (Caps, Termination & Discretion)",
        ],
        index=0,
        help="Adjusts relative weights of risk factors based on contract negotiation posture.",
    )
    profile_mapping = {
        "Standard Commercial (Balanced)": "standard_commercial",
        "Vendor-Protective (Payment, IP & Indemnity)": "vendor_protective",
        "Buyer-Protective (Caps, Termination & Discretion)": "buyer_protective",
    }
    selected_profile = profile_mapping[profile_option]

    st.markdown("### 🤖 Generative AI Explainer")
    use_llm_explain = st.toggle(
        "Enable AI Generative Explanations",
        value=False,
        help="Generate AI explanations, practical impact analysis, and draft revisions.",
    )

    selected_llm_provider = "template"
    llm_api_key_input = ""
    llm_endpoint_input = ""

    if use_llm_explain:
        llm_provider_option = st.selectbox(
            "AI Model Provider",
            options=[
                "Mock LLM (Offline Demonstration & Testing)",
                "Template Baseline (Offline / Rule-Based)",
                "Hugging Face Inference API / Dedicated Endpoint",
                "OpenAI-Compatible (Ollama, vLLM, LM Studio)",
                "Local Transformers Pipeline (On-Device)",
            ],
            index=0,
            help="Select the instruction-following model provider for clause risk explanations.",
        )
        provider_map = {
            "Mock LLM (Offline Demonstration & Testing)": "mock",
            "Template Baseline (Offline / Rule-Based)": "template",
            "Hugging Face Inference API / Dedicated Endpoint": "hf_inference",
            "OpenAI-Compatible (Ollama, vLLM, LM Studio)": "openai_compatible",
            "Local Transformers Pipeline (On-Device)": "local_pipeline",
        }
        selected_llm_provider = provider_map[llm_provider_option]

        if selected_llm_provider in ("hf_inference", "openai_compatible"):
            st.info(
                "🔒 **Privacy & Data Minimization:**\n"
                "Only the isolated clause text and triggered risk indicators are sent. "
                "Entire documents, metadata, and party names are never transmitted."
            )
            if selected_llm_provider == "hf_inference":
                llm_api_key_input = st.text_input(
                    "Hugging Face Token",
                    value=settings.hf_token,
                    type="password",
                    help="Loaded from HF_TOKEN in .env if available.",
                )
                llm_endpoint_input = st.text_input(
                    "Dedicated Endpoint URL (optional)",
                    value=settings.llm_endpoint_url,
                    placeholder="https://...endpoints.huggingface.cloud",
                )
            elif selected_llm_provider == "openai_compatible":
                llm_endpoint_input = st.text_input(
                    "API Endpoint URL",
                    value=settings.llm_endpoint_url or "http://localhost:11434/v1",
                    placeholder="http://localhost:11434/v1",
                )
                llm_api_key_input = st.text_input(
                    "API Key (if required)",
                    value=settings.llm_api_key,
                    type="password",
                )

    st.divider()
    st.markdown("### 📋 Workflow")
    st.markdown(
        """
1. **Upload** PDF or DOCX (or Load Sample)
2. **Review** Extracted Text & Clause Preview
3. **Inspect** Risk Dashboard & Breakdown
4. **Download** Official PDF Report
"""
    )

    st.divider()
    load_sample = st.button("📄 Load Sample Contract", use_container_width=True)

    st.divider()
    st.caption(
        f"v0.5.0 · Phase 5 (Generative AI Explanations)\n\n"
        f"ML Backend: `{selected_backend}`\n\n"
        f"Scoring Profile: `{selected_profile}`\n\n"
        f"Explainer: `{'LLM (' + selected_llm_provider + ')' if use_llm_explain else 'Template Offline'}`\n\n"
        f"Local & in-memory processing"
    )


# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center; color:#1A1A2E;'>🛡️ AI Contract Risk Analyser</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#666; margin-top:-8px;'>"
    "Intelligent document extraction, clause segmentation, and automated legal risk analysis."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("")


# ══════════════════════════════════════════════════════════════════════
#  PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════
def _run_pipeline(
    text: str,
    clauses: List[Clause],
    *,
    backend: str = "tfidf",
    use_llm_explain: bool = False,
    profile: str = "standard_commercial",
    llm_provider: str = "template",
    llm_api_key: Optional[str] = None,
    llm_endpoint_url: Optional[str] = None,
) -> dict:
    """Run classification, risk rule detection, and scoring on extracted clauses."""
    progress = st.progress(0, text=f"Classifying clauses via {backend}…")

    # 1. Classify
    classifications: List[ClassificationResult] = classify_clauses(
        clauses, backend=backend
    )
    progress.progress(40, text="Evaluating legal risk rules…")

    # 2. Risk rules (clause-level)
    clause_flags: Dict[int, List[RiskFlag]] = {}
    for clause in clauses:
        flags = evaluate_clause(clause.text)
        if flags:
            clause_flags[clause.index] = flags

    # 3. Risk rules (document-level)
    document_flags: List[RiskFlag] = evaluate_document(text)
    progress.progress(70, text="Computing risk scores…")

    # 4. Score
    report: OverallRiskReport = score_contract(
        classifications, clause_flags, document_flags, profile_name=profile
    )
    progress.progress(85, text="Generating explanations & recommendations…")

    # 5. Explanations
    explanations: List[Explanation] = []
    for detail in report.clause_details:
        clause_obj = clauses[detail.clause_index] if detail.clause_index < len(clauses) else None
        expl = explain_flags(
            detail.clause_index,
            detail.heading,
            detail.flags,
            clause_text=clause_obj.text if clause_obj else "",
            use_llm=use_llm_explain,
            clause_id=clause_obj.clause_id if clause_obj else f"clause_{detail.clause_index:03d}",
            provider_type=llm_provider,
            api_key=llm_api_key,
            endpoint_url=llm_endpoint_url,
        )
        explanations.append(expl)
    progress.progress(100, text="Risk analysis complete ✓")

    return {
        "clauses": clauses,
        "classifications": classifications,
        "clause_flags": clause_flags,
        "document_flags": document_flags,
        "report": report,
        "explanations": explanations,
    }


# ══════════════════════════════════════════════════════════════════════
#  UI RENDERING HELPERS
# ══════════════════════════════════════════════════════════════════════
def _badge(level: str) -> str:
    return f'<span class="risk-badge risk-{level}">{level}</span>'


def _render_document_metadata_card(doc: ExtractedDocument, clauses_count: int) -> None:
    """Render metadata statistics regarding the processed document."""
    st.markdown("### 📑 Document Ingestion & Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Document Type", doc.metadata.file_type.upper())
    with c2:
        st.metric("Total Pages", str(doc.metadata.page_count))
    with c3:
        st.metric("Total Words", f"{doc.metadata.word_count:,}")
    with c4:
        st.metric("Total Characters", f"{doc.metadata.char_count:,}")
    with c5:
        st.metric("Clauses Segmented", str(clauses_count))

    st.caption(
        f"**ID:** `{doc.metadata.document_id}` &nbsp;·&nbsp; "
        f"**Filename:** {doc.metadata.filename} &nbsp;·&nbsp; "
        f"**Size:** {doc.metadata.file_size_bytes / 1024:.1f} KB"
    )


def _render_clause_preview_section(clauses: List[Clause], doc: ExtractedDocument) -> None:
    """Display extracted text, clause count, and a clause preview before/with analysis."""
    st.markdown("---")
    st.markdown("### 🔍 Document Extraction & Clause Preview")

    col_preview, col_text = st.columns([1, 1])

    with col_preview:
        st.markdown(f"#### Identified Clauses ({len(clauses)})")
        st.caption("Review extracted clauses, numbers, and source page numbers:")
        for clause in clauses:
            num_str = f"[{clause.clause_number}] " if clause.clause_number else ""
            page_str = f"Page {clause.page_number}" if clause.page_number else "Page 1"

            with st.expander(f"§ {clause.index + 1}: {num_str}{clause.heading} ({page_str})"):
                st.markdown(
                    f'<span class="meta-chip">ID: {clause.clause_id}</span>'
                    f'<span class="meta-chip">{page_str}</span>'
                    f'<span class="meta-chip">{clause.word_count} words</span>',
                    unsafe_allow_html=True,
                )
                st.text_area(
                    "Normalized Clause Content",
                    value=clause.text,
                    height=90,
                    disabled=True,
                    key=f"prev_norm_{clause.clause_id}",
                )

    with col_text:
        st.markdown("#### Full Extracted Text")
        st.caption("Cleaned & normalized plain text without formatting artifacts:")
        st.text_area(
            "Extracted plain text",
            value=doc.normalized_text,
            height=420,
            disabled=True,
            key="full_extracted_view",
        )


def _render_dashboard(report: OverallRiskReport) -> None:
    """Render the KPI dashboard cards."""
    col_score, col_high, col_med, col_low, col_clean, col_total = st.columns(6)

    score_colour = {
        "low": "#27AE60",
        "medium": "#F39C12",
        "high": "#E74C3C",
    }.get(report.risk_level, "#888")

    with col_score:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-value" style="color:{score_colour}">{report.overall_score:.1f}</div>
                <div class="score-label">Overall Risk Score</div>
                <div style="margin-top:8px">{_badge(report.risk_level)}</div>
                <div style="font-size:0.72rem; opacity:0.8; margin-top:4px;">Profile: {report.profile.replace('_', ' ').title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_high:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value" style="color:#E74C3C">{report.high_risk_count}</div>
                <div class="stat-label">High Risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_med:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value" style="color:#F39C12">{report.medium_risk_count}</div>
                <div class="stat-label">Medium Risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_low:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value" style="color:#27AE60">{report.low_risk_count}</div>
                <div class="stat-label">Low Risk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_clean:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value" style="color:#2980B9">{report.clean_clauses_count}</div>
                <div class="stat-label">Clean Clauses</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_total:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value" style="color:#1A1A2E">{len(report.clause_details)}</div>
                <div class="stat-label">Total Clauses</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_score_breakdown(report: OverallRiskReport) -> None:
    """Render transparent mathematical score breakdown and factor contributions."""
    st.markdown("#### 📐 Transparent Scoring Formula & Factor Breakdown")
    formula_str = (
        report.formula_explanation
        or "Weighted Risk Score = sum(Suspiciousness_i * Weight_i) / sum(Applicable_Weights)"
    )
    st.info(
        f"**Mathematical Formula:** `{formula_str}`\n\n"
        f"• **Sum of Applicable Factor Weights:** `{report.applicable_weight_sum * 100:.1f}%`\n"
        f"• **Active Scoring Profile:** `{report.profile.replace('_', ' ').title()}`\n"
        f"• **Suspicious Clauses Flagged:** `{report.suspicious_clauses_count}` of `{len(report.clause_details)}`\n"
        f"• **Deterministic Calculation:** `{report.is_deterministic}` (identically reproducible)"
    )

    if report.factor_breakdowns:
        st.markdown("##### 📊 Risk Factor Weights & Weighted Contributions")
        table_rows = []
        if isinstance(report.factor_breakdowns, dict):
            for factor, bd in report.factor_breakdowns.items():
                raw_score = bd.get("raw_score", 0.0)
                profile_weight = bd.get("profile_weight", 0.0)
                weighted_contribution = bd.get("weighted_contribution", 0.0)
                flag_count = bd.get("flag_count", 0)
                if raw_score == 0:
                    status_badge = "🟢 Clean"
                elif raw_score >= 70:
                    status_badge = f"🔴 High ({flag_count} flag{'s' if flag_count > 1 else ''})"
                else:
                    status_badge = f"🟡 Moderate ({flag_count} flag{'s' if flag_count > 1 else ''})"

                table_rows.append(
                    {
                        "Risk Factor / Category": str(factor).replace("_", " ").title(),
                        "Profile Weight": f"{profile_weight * 100:.1f}%",
                        "Raw Score (0-100)": f"{raw_score:.1f}",
                        "Weighted Contribution": f"+{weighted_contribution:.2f} pts",
                        "Status": status_badge,
                    }
                )
        elif isinstance(report.factor_breakdowns, list):
            for bd in report.factor_breakdowns:
                raw_score = getattr(bd, "suspiciousness_score", 0.0)
                profile_weight = getattr(bd, "weight", 0.0)
                weighted_contribution = getattr(bd, "weighted_contribution", 0.0)
                flag_count = getattr(bd, "trigger_count", 0)
                factor_name = getattr(bd, "factor_name", getattr(bd, "category", "factor"))
                if raw_score == 0:
                    status_badge = "🟢 Clean"
                elif raw_score >= 70:
                    status_badge = f"🔴 High ({flag_count} flag{'s' if flag_count > 1 else ''})"
                else:
                    status_badge = f"🟡 Moderate ({flag_count} flag{'s' if flag_count > 1 else ''})"

                table_rows.append(
                    {
                        "Risk Factor / Category": str(factor_name).replace("_", " ").title(),
                        "Profile Weight": f"{profile_weight * 100:.1f}%",
                        "Raw Score (0-100)": f"{raw_score:.1f}",
                        "Weighted Contribution": f"+{weighted_contribution:.2f} pts",
                        "Status": status_badge,
                    }
                )
        if table_rows:
            st.dataframe(table_rows, use_container_width=True)

    st.markdown("---")
    st.caption(
        "⚖️ **Legal Disclaimer & Review Indicator Notice:**\n\n"
        "Antigravity Contract Risk Analyser highlights risk factors, ambiguities, and structural imbalances solely as "
        "review indicators to support qualified legal professionals. These indicators do not constitute formal legal advice, "
        "and flagged clauses are not automatically void, illegal, or unenforceable."
    )


def _render_risk_labels_breakdown(report: OverallRiskReport) -> None:
    """Render a breakdown table of all triggered risk labels across the contract."""
    st.markdown("#### 🏷️ Breakdown of Triggered Risk Labels")
    label_counts: Dict[str, Dict[str, Any]] = {}

    # Collect flags from clause details
    for detail in report.clause_details:
        for flag in detail.flags:
            key = flag.rule_id
            if key not in label_counts:
                label_counts[key] = {
                    "rule_id": flag.rule_id,
                    "label": flag.label,
                    "severity": flag.severity,
                    "count": 0,
                    "description": flag.description,
                }
            label_counts[key]["count"] += 1

    # Collect flags from document level
    for flag in report.document_flags:
        key = flag.rule_id
        if key not in label_counts:
            label_counts[key] = {
                "rule_id": flag.rule_id,
                "label": flag.label,
                "severity": flag.severity,
                "count": 0,
                "description": flag.description,
            }
        label_counts[key]["count"] += 1

    if not label_counts:
        st.success("🟢 No risk rule triggers detected across this agreement.")
        return

    rows = []
    for item in sorted(label_counts.values(), key=lambda x: (-x["count"], x["severity"])):
        sev = item["severity"].upper()
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(sev, "⚪")
        rows.append(
            {
                "Rule ID": item["rule_id"],
                "Risk Label": item["label"],
                "Severity": f"{icon} {sev}",
                "Trigger Count": item["count"],
                "Description": item["description"],
            }
        )
    st.dataframe(rows, use_container_width=True)


def _render_missing_protections_card(report: OverallRiskReport) -> None:
    """Show missing expected clauses separately from risks found in existing clauses."""
    st.markdown("#### 🛡️ Expected Standard Contractual Protections")
    st.caption(
        "Commercial agreements customarily include standard protective provisions. "
        "The analyser audits the contract for missing baseline protections separately from risks identified within existing clauses."
    )

    missing_items = []
    if report.missing_protections:
        for mp in report.missing_protections:
            missing_items.append(mp)
    else:
        for flag in report.document_flags:
            if flag.category == "missing_protections":
                missing_items.append(flag)

    if not missing_items:
        st.success(
            "✅ All core standard expected protections (Governing Law, Limitation of Liability, "
            "Termination for Cause, Confidentiality) were detected in this agreement."
        )
        return

    st.warning(
        f"⚠️ **{len(missing_items)} Standard Protective Mechanism(s) Absent from Agreement:**"
    )
    for item in missing_items:
        rule_id = getattr(item, "rule_id", "RULE-PROT-MISSING")
        label = getattr(item, "label", "Missing Expected Protection")
        sev = getattr(item, "severity", "medium").upper()
        desc = getattr(item, "description", "")
        rationale = getattr(
            item, "rationale", "Standard commercial safeguard is not expressly stated."
        )
        recommendation = getattr(item, "recommendation", "")

        icon = "🔴" if sev == "HIGH" else "🟡"
        with st.container():
            st.markdown(
                f"""
                <div style="border:1px solid #F5C6CB; border-left:4px solid #E74C3C; border-radius:8px; padding:12px 16px; margin-bottom:10px; background:#FDF7F7;">
                    <div style="font-weight:600; font-size:0.95rem; color:#721C24;">
                        {icon} [{rule_id}] {label} &nbsp;·&nbsp; <span class="risk-badge risk-{sev.lower()}">{sev}</span>
                    </div>
                    <div style="font-size:0.88rem; color:#495057; margin-top:4px;"><b>Observation:</b> {desc}</div>
                    <div style="font-size:0.86rem; color:#6C757D; margin-top:4px;"><b>Why it matters:</b> {rationale}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if recommendation:
                st.info(f"💡 **Suggested Model Clause to Insert:** {recommendation}")


def _render_category_distribution(report: OverallRiskReport) -> None:
    if not report.category_distribution:
        return
    st.markdown("#### 📊 Clause Category Distribution")
    cols = st.columns(min(len(report.category_distribution), 4))
    for i, (cat, count) in enumerate(
        sorted(report.category_distribution.items(), key=lambda x: -x[1])
    ):
        with cols[i % len(cols)]:
            st.metric(label=cat, value=count)


def _render_document_flags(flags: List[RiskFlag]) -> None:
    if not flags:
        return
    st.markdown("#### ⚠️ Document-Level Flags")
    for flag in flags:
        st.warning(f"**{flag.label}** ({flag.severity}): {flag.description}")


def _render_clause_details(
    report: OverallRiskReport,
    explanations: List[Explanation],
    clauses: List[Clause],
    classifications: Optional[List[ClassificationResult]] = None,
) -> None:
    """Render per-clause assessments with multi-dimensional filtering and side-by-side comparison."""
    st.markdown("#### 📑 Clause-by-Clause Review & Wording Comparison")
    expl_map = {e.clause_index: e for e in explanations}

    # ── Multi-Dimensional Filters ───────────────────────────────────────
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    # 1. Risk Level Filter
    with f_col1:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            options=["All Levels", "High Risk", "Medium Risk", "Low Risk", "Clean"],
            index=0,
            key="filter_risk_level",
        )

    # 2. Category Filter
    all_categories = sorted(list(set(d.category for d in report.clause_details if d.category)))
    with f_col2:
        cat_filter = st.selectbox(
            "Filter by Category",
            options=["All Categories"] + all_categories,
            index=0,
            key="filter_clause_cat",
        )

    # 3. Risk Label Filter
    all_labels = set()
    for d in report.clause_details:
        for f in d.flags:
            all_labels.add(f"{f.rule_id} ({f.label})")
    sorted_labels = sorted(list(all_labels))
    with f_col3:
        label_filter = st.selectbox(
            "Filter by Risk Label / Rule",
            options=["All Labels"] + sorted_labels,
            index=0,
            key="filter_risk_label",
        )

    # 4. Source Location Filter
    all_pages = set()
    for d in report.clause_details:
        c_obj = clauses[d.clause_index] if d.clause_index < len(clauses) else None
        p_num = c_obj.page_number if c_obj and c_obj.page_number else 1
        all_pages.add(f"Page {p_num}")
    sorted_pages = sorted(
        list(all_pages),
        key=lambda x: int(x.replace("Page ", "")) if x.replace("Page ", "").isdigit() else 999,
    )
    with f_col4:
        location_filter = st.selectbox(
            "Filter by Source Location",
            options=["All Locations"] + sorted_pages,
            index=0,
            key="filter_source_loc",
        )

    # View Mode Toggle
    view_mode = st.radio(
        "View Mode",
        options=["Detailed Card View", "Side-by-Side Wording Comparison"],
        horizontal=True,
        index=0,
        key="clause_view_mode",
    )

    # Apply filters
    filtered_details = []
    for detail in report.clause_details:
        clause_obj = clauses[detail.clause_index] if detail.clause_index < len(clauses) else None
        page_str = (
            f"Page {clause_obj.page_number}"
            if clause_obj and clause_obj.page_number
            else "Page 1"
        )

        # Level check
        if risk_filter == "High Risk" and detail.risk_level != "high":
            continue
        if risk_filter == "Medium Risk" and detail.risk_level != "medium":
            continue
        if risk_filter == "Low Risk" and detail.risk_level != "low":
            continue
        if risk_filter == "Clean" and detail.has_risk:
            continue

        # Category check
        if cat_filter != "All Categories" and detail.category != cat_filter:
            continue

        # Label check
        if label_filter != "All Labels":
            flag_match = any(
                f"{f.rule_id} ({f.label})" == label_filter for f in detail.flags
            )
            if not flag_match:
                continue

        # Location check
        if location_filter != "All Locations" and page_str != location_filter:
            continue

        filtered_details.append(detail)

    st.caption(
        f"Showing **{len(filtered_details)}** of **{len(report.clause_details)}** clauses matching filters."
    )

    if not filtered_details:
        st.info("No clauses match the current filter selection.")
        return

    for detail in filtered_details:
        level = detail.risk_level
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(level, "⚪")
        clause_obj = clauses[detail.clause_index] if detail.clause_index < len(clauses) else None

        page_str = (
            f"Page {clause_obj.page_number}"
            if clause_obj and clause_obj.page_number
            else "Page 1"
        )
        num_str = f"[{clause_obj.clause_number}] " if clause_obj and clause_obj.clause_number else ""
        expl = expl_map.get(detail.clause_index)

        # Determine revision text if available
        rev_text = ""
        if expl and expl.structured_output and expl.structured_output.suggested_revision:
            rev_text = expl.structured_output.suggested_revision
        elif (
            expl
            and expl.recommendations
            and expl.recommendations != ["No action items at this time."]
        ):
            rev_text = " / ".join(expl.recommendations)

        if view_mode == "Side-by-Side Wording Comparison":
            # Side-by-side comparison view
            with st.container():
                st.markdown(
                    f"**{icon} § {detail.clause_index + 1}: {num_str}{detail.heading}** &nbsp;·&nbsp; "
                    f"*{detail.category}* &nbsp;·&nbsp; <span class='risk-badge risk-{level}'>{level.upper()} ({detail.clause_score:.0f}/100)</span> &nbsp;·&nbsp; "
                    f"<span class='meta-chip'>{page_str}</span>",
                    unsafe_allow_html=True,
                )
                c_orig, c_rev = st.columns(2)
                with c_orig:
                    st.markdown("**Original Clause Wording:**")
                    st.markdown(
                        f"""
                        <div class="diff-box diff-orig">
                            {clause_obj.text if clause_obj else "(Text unavailable)"}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c_rev:
                    st.markdown("**Proposed Revision / Recommendations:**")
                    if rev_text:
                        st.markdown(
                            f"""
                            <div class="diff-box diff-rev">
                                {rev_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div class="diff-box" style="background:#F8F9FA; color:#6C757D; border:1px solid #DEE2E6;">
                                <i>No revisions required. Clause evaluated as low risk.</i>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                st.divider()
        else:
            # Detailed card view
            with st.expander(
                f"{icon}  {num_str}{detail.heading}  —  {detail.category}  "
                f"({detail.clause_score}/100  {level.upper()})",
                expanded=(level == "high"),
            ):
                if clause_obj:
                    cr = (
                        classifications[detail.clause_index]
                        if classifications and detail.clause_index < len(classifications)
                        else None
                    )
                    conf_str = f"Confidence: {cr.confidence * 100:.1f}%" if cr else ""
                    model_str = f"Model: {cr.model_identifier}" if cr else ""
                    status_str = f"Status: {cr.classification_status}" if cr else ""
                    trunc_html = (
                        '<span class="meta-chip" style="background:#FFF3CD; color:#856404;">Truncated (>512 tokens)</span>'
                        if (cr and cr.is_truncated)
                        else ""
                    )

                    st.markdown(
                        f'<span class="meta-chip">ID: {clause_obj.clause_id}</span>'
                        f'<span class="meta-chip">{page_str}</span>'
                        f'<span class="meta-chip">Category: {detail.category}</span>'
                        f'<span class="meta-chip">Suspiciousness: {detail.suspiciousness_score:.0f}/100</span>'
                        f'<span class="meta-chip">Weighted Score: {detail.clause_score:.1f}/100</span>'
                        f'{f"<span class=\'meta-chip\'>{conf_str}</span>" if conf_str else ""}'
                        f'{f"<span class=\'meta-chip\'>{status_str}</span>" if status_str else ""}'
                        f'{f"<span class=\'meta-chip\'>{model_str}</span>" if model_str else ""}'
                        f'{trunc_html}',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "ℹ️ *Note: Category classification identifies the clause topic and is evaluated separately from risk assessment.*"
                    )
                    st.text_area(
                        "Original Clause Text",
                        value=clause_obj.text,
                        height=110,
                        disabled=True,
                        key=f"risk_txt_{detail.clause_index}",
                    )

                # Flags
                if detail.flags:
                    st.markdown("**Risk Flags & Evidence:**")
                    for flag in detail.flags:
                        css_class = flag.severity
                        st.markdown(
                            f'<span class="flag-chip {css_class}"><b>{flag.rule_id}</b> · {flag.label} ({flag.severity.upper()})</span>'
                            f'<span class="meta-chip">Suspiciousness: {flag.suspiciousness_score:.0f}/100</span>'
                            f'<span class="meta-chip">Weight: {flag.weight:.2f}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Issue Description:** {flag.description}")
                        if (
                            flag.matched_snippet
                            and flag.matched_snippet != "(not found in contract text)"
                        ):
                            st.markdown(f"> **Evidence Snippet:** *\"{flag.matched_snippet}\"*")
                        if getattr(flag, "rationale", None):
                            st.markdown(f"💡 **Why it matters:** {flag.rationale}")
                        if getattr(flag, "score_breakdown", None):
                            st.caption(f"📐 *Formula calculation:* {flag.score_breakdown}")
                        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

                # Explanation & Suggested Revision
                if expl:
                    st.markdown("---")
                    if expl.status == "completed_llm" and expl.structured_output:
                        st.markdown(
                            '<span class="meta-chip" style="background:#EBF5FB; color:#1B4F72; border-color:#AED6F1;">'
                            '🤖 AI Generative Review (Grounding Verified)</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Legal Risk & Context Analysis:**")
                        st.markdown(expl.structured_output.explanation)

                        st.markdown("**💼 Potential Practical Business & Operational Consequences:**")
                        st.markdown(expl.structured_output.potential_impact)

                        st.markdown("**📝 Draft Alternative Wording (For Human Review):**")
                        st.info(expl.structured_output.suggested_revision)

                        # Side-by-side comparison box
                        st.markdown(
                            f"""
                            <div class="diff-container">
                                <div class="diff-box diff-orig">
                                    <b>Original Clause:</b><br/>{clause_obj.text if clause_obj else ""}
                                </div>
                                <div class="diff-box diff-rev">
                                    <b>Draft Proposed Revision:</b><br/>{expl.structured_output.suggested_revision}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if expl.structured_output.evidence_quotes:
                            quotes_str = ", ".join(f'"{q}"' for q in expl.structured_output.evidence_quotes)
                            grounding_status = (
                                "🟢 Verbatim Verified"
                                if expl.structured_output.grounding_verified
                                else "🟡 Grounding Warning"
                            )
                            st.caption(f"**Verbatim Evidence Snippet:** {quotes_str} · `{grounding_status}`")

                        if (
                            expl.structured_output.assumptions
                            or expl.structured_output.uncertainty_notes
                        ):
                            with st.expander("Context Assumptions & Uncertainty Notes"):
                                if expl.structured_output.assumptions:
                                    st.markdown("**Assumptions:**")
                                    for a in expl.structured_output.assumptions:
                                        st.markdown(f"- {a}")
                                if expl.structured_output.uncertainty_notes:
                                    st.markdown(
                                        f"**Uncertainty / Context Notes:** {expl.structured_output.uncertainty_notes}"
                                    )

                        st.caption(f"⚖️ *{expl.structured_output.disclaimer}*")
                    else:
                        st.markdown(
                            '<span class="meta-chip" style="background:#F2F4F4; color:#5D6D7E; border-color:#D5D8DC;">'
                            '📋 Template Baseline</span>',
                            unsafe_allow_html=True,
                        )
                        if expl.summary and expl.summary != "No specific risks identified for this clause.":
                            st.markdown("**Legal Risk Analysis:**")
                            st.markdown(expl.summary)
                        if expl.recommendations and expl.recommendations != [
                            "No action items at this time."
                        ]:
                            st.markdown("**Draft Suggested Actions & Recommendations:**")
                            for rec in expl.recommendations:
                                st.markdown(f"- {rec}")
                        st.caption(
                            "⚖️ *Draft revision suggested solely for human legal review; does not constitute legal advice.*"
                        )


def _render_sample_explanations_section() -> None:
    """Display Task 12 official sample outputs for vague payment deadline and unilateral termination."""
    with st.expander("✨ Phase 5 Generative AI Sample Outputs (Task 12 Showcase)"):
        st.markdown(
            "Explore pre-validated structured outputs demonstrating exact verbatim evidence verification, "
            "practical consequence analysis, and balanced revisions for two high-frequency contract risks."
        )
        sample_tab1, sample_tab2 = st.tabs(
            [
                "⏳ Vague Payment Deadline (RULE-DEADLINE-001)",
                "🚪 Potentially One-Sided Termination (RULE-TERM-001)",
            ]
        )

        samples = get_sample_explanations()

        with sample_tab1:
            s1 = samples["vague_payment_deadline"]
            st.markdown(
                '<span class="flag-chip medium">RULE-DEADLINE-001 · Indefinite Performance Timeline</span>'
                '<span class="meta-chip">🟢 Grounding Verified</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "**Original Clause Text:**\n"
                "> *\"Client shall remit payment to Contractor promptly in due course following receipt of each approved invoice.\"*"
            )
            st.markdown(f"**Detailed Risk Analysis:**\n{s1.explanation}")
            st.markdown(f"**💼 Practical Business Impact:**\n{s1.potential_impact}")
            st.markdown("**📝 Draft Proposed Revision (For Human Review):**")
            st.info(s1.suggested_revision)
            st.caption(f"**Verified Evidence Quote:** `\"{s1.evidence_quotes[0]}\"`")
            st.caption(f"**Assumptions & Uncertainty:** {'; '.join(s1.assumptions)} · {s1.uncertainty_notes}")
            st.caption(f"⚖️ *{s1.disclaimer}*")

        with sample_tab2:
            s2 = samples["unilateral_termination"]
            st.markdown(
                '<span class="flag-chip high">RULE-TERM-001 · Potentially One-Sided Termination</span>'
                '<span class="meta-chip">🟢 Grounding Verified</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "**Original Clause Text:**\n"
                "> *\"Company may terminate this Agreement at any time for convenience upon written notice to Vendor, without liability or penalty.\"*"
            )
            st.markdown(f"**Detailed Risk Analysis:**\n{s2.explanation}")
            st.markdown(f"**💼 Practical Business Impact:**\n{s2.potential_impact}")
            st.markdown("**📝 Draft Proposed Revision (For Human Review):**")
            st.info(s2.suggested_revision)
            st.caption(f"**Verified Evidence Quote:** `\"{s2.evidence_quotes[0]}\"`")
            st.caption(f"**Assumptions & Uncertainty:** {'; '.join(s2.assumptions)} · {s2.uncertainty_notes}")
            st.caption(f"⚖️ *{s2.disclaimer}*")


# ══════════════════════════════════════════════════════════════════════
#  MAIN INGESTION & PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════
extracted_doc: Optional[ExtractedDocument] = None
extracted_clauses: Optional[List[Clause]] = None

upload_col, info_col = st.columns([2, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Upload a contract (PDF or DOCX)",
        type=["pdf", "docx"],
        help=f"Maximum file size: {settings.max_upload_size_mb} MB",
    )
with info_col:
    st.info(
        "**Local & Private Processing:**\n\n"
        "• PyMuPDF & python-docx extraction\n"
        "• Automatic scanned PDF / OCR detection\n"
        "• Documents are processed in-memory only\n"
        "• No contract text is stored or sent externally"
    )

# ── File Upload Processing ────────────────────────────────────────────
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    try:
        with st.spinner(f"Ingesting and parsing **{uploaded_file.name}**…"):
            extracted_doc = process_document(uploaded_file.name, file_bytes)
            extracted_clauses = segment_document(extracted_doc)
        st.success(f"✅ Extracted {extracted_doc.metadata.char_count:,} characters from **{uploaded_file.name}**.")
    except ScannedDocumentError as exc:
        st.error(
            "⚠️ **Scanned Document Detected**\n\n"
            f"{exc}\n\n"
            "**Action Required:** Please run this PDF through an OCR scanner (e.g. Adobe Acrobat OCR, "
            "Tesseract, or ABBYY) or provide a digital document with selectable text."
        )
    except EmptyDocumentError as exc:
        st.error(f"⚠️ **Empty Document:** {exc}")
    except CorruptDocumentError as exc:
        st.error(f"❌ **Corrupt File:** {exc}")
    except UnsupportedDocumentError as exc:
        st.error(f"❌ **Unsupported File:** {exc}")
    except DocumentProcessingError as exc:
        st.error(f"❌ **Processing Error:** {exc}")

# ── Sample Contract Loading ───────────────────────────────────────────
if load_sample:
    metadata = DocumentMetadata(
        document_id="doc_sample_msa_01",
        filename="Sample_Master_Services_Agreement.docx",
        file_type="docx",
        file_size_bytes=len(SAMPLE_CONTRACT_TEXT.encode()),
        page_count=2,
        char_count=len(SAMPLE_CONTRACT_TEXT),
        word_count=len(SAMPLE_CONTRACT_TEXT.split()),
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    pages = [
        PageContent(page_number=1, text=SAMPLE_CONTRACT_TEXT[: len(SAMPLE_CONTRACT_TEXT) // 2], char_count=len(SAMPLE_CONTRACT_TEXT) // 2),
        PageContent(page_number=2, text=SAMPLE_CONTRACT_TEXT[len(SAMPLE_CONTRACT_TEXT) // 2 :], char_count=len(SAMPLE_CONTRACT_TEXT) // 2),
    ]
    extracted_doc = ExtractedDocument(
        metadata=metadata,
        raw_text=SAMPLE_CONTRACT_TEXT,
        normalized_text=SAMPLE_CONTRACT_TEXT,
        pages=pages,
    )
    extracted_clauses = segment_document(extracted_doc)
    st.success("✅ Loaded synthetic Master Services Agreement for development & testing.")

# ── Display Clause Preview & Analysis ─────────────────────────────────
if extracted_doc and extracted_clauses:
    # 1. Document summary card
    _render_document_metadata_card(extracted_doc, len(extracted_clauses))

    # 2. Extracted text & clause preview before/with analysis
    _render_clause_preview_section(extracted_clauses, extracted_doc)

    st.markdown("---")
    st.markdown("### ⚡ Legal Risk Analysis")

    # 3. Run analysis
    results = _run_pipeline(
        extracted_doc.normalized_text,
        extracted_clauses,
        backend=selected_backend,
        use_llm_explain=use_llm_explain,
        profile=selected_profile,
        llm_provider=selected_llm_provider,
        llm_api_key=llm_api_key_input,
        llm_endpoint_url=llm_endpoint_input,
    )

    if results:
        report: OverallRiskReport = results["report"]
        explanations: List[Explanation] = results["explanations"]
        clauses: List[Clause] = results["clauses"]
        classifications: List[ClassificationResult] = results["classifications"]

        # Risk dashboard KPI cards
        _render_dashboard(report)
        st.markdown("")

        # Task 12 Sample Explanations showcase
        _render_sample_explanations_section()
        st.markdown("")

        # Detail tabs
        tab_summary, tab_breakdown, tab_missing, tab_clauses = st.tabs(
            [
                "📊 Risk Summary & Label Breakdown",
                "📐 Transparent Score Breakdown",
                "🛡️ Missing Expected Protections",
                "📑 Clause Review & Wording Comparison",
            ]
        )

        with tab_summary:
            _render_category_distribution(report)
            st.markdown("")
            _render_risk_labels_breakdown(report)

        with tab_breakdown:
            _render_score_breakdown(report)

        with tab_missing:
            _render_missing_protections_card(report)

        with tab_clauses:
            _render_clause_details(report, explanations, clauses, classifications)

        # Download PDF report
        st.divider()
        st.markdown("#### 📥 Export Risk Assessment Report")
        try:
            pdf_bytes = generate_pdf_report(
                report,
                explanations,
                clauses=clauses,
                doc_metadata=extracted_doc.metadata,
            )
            st.download_button(
                label="⬇️  Download Executive PDF Report",
                data=pdf_bytes,
                file_name=f"risk_report_{extracted_doc.metadata.document_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Failed to generate PDF report: {exc}")

elif not uploaded_file:
    # Empty state placeholder
    st.markdown("")
    st.markdown(
        """
        <div style="text-align:center; padding:60px 20px; color:#aaa;">
            <div style="font-size:3rem; margin-bottom:12px;">📑</div>
            <div style="font-size:1.15rem; font-weight:500; color:#555;">Upload a PDF or Word contract to begin analysis.</div>
            <div style="font-size:0.9rem; margin-top:6px; color:#888;">Or click "Load Sample Contract" in the sidebar to test immediately.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
