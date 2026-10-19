"""
test_report_generator.py – Comprehensive Tests for report_generator.py (Phase 6).
"""

import io
from clause_splitter import Clause
from document_processor import DocumentMetadata
from llm_explainer import ClauseExplanationOutput, Explanation
from report_generator import generate_pdf_report
from risk_rules import RiskFlag
from risk_scorer import ClauseRiskDetail, FactorScoreBreakdown, MissingProtection, OverallRiskReport


class TestReportGenerator:
    def test_generate_pdf_report_basic(self):
        """Test minimal report generation with basic structure."""
        report = OverallRiskReport(
            overall_score=75.5,
            risk_level="high",
            high_risk_count=1,
            medium_risk_count=1,
            low_risk_count=1,
            document_flags=[
                RiskFlag(
                    rule_id="RISK-006",
                    label="Missing Governing Law",
                    severity="medium",
                    description="No governing law clause found.",
                    matched_snippet="(not found)",
                )
            ],
            clause_details=[
                ClauseRiskDetail(
                    clause_index=0,
                    heading="Indemnification",
                    category="Indemnification",
                    category_weight=0.9,
                    flags=[
                        RiskFlag(
                            rule_id="RISK-004",
                            label="Broad Indemnification",
                            severity="high",
                            description="Overly broad indemnity clause.",
                            matched_snippet="indemnify against any and all claims",
                        )
                    ],
                    clause_score=95.0,
                    risk_level="high",
                )
            ],
            category_distribution={"Indemnification": 1},
        )

        explanations = [
            Explanation(
                clause_index=0,
                heading="Indemnification",
                summary="High risk of uncapped indemnity liability.",
                recommendations=["Cap liability to 1x contract value."],
            )
        ]

        pdf_bytes = generate_pdf_report(report, explanations)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF")

    def test_generate_pdf_with_metadata_and_clauses(self):
        """Test full executive PDF generation including doc metadata, clauses, and factor breakdowns."""
        metadata = DocumentMetadata(
            document_id="doc_test_12345",
            filename="Vendor_Agreement_Final.pdf",
            file_type="pdf",
            file_size_bytes=45200,
            page_count=3,
            char_count=6400,
            word_count=980,
        )

        clauses = [
            Clause(
                index=0,
                heading="Payment Terms",
                text="Client shall pay Contractor promptly in due course following receipt of invoice.",
                page_number=1,
                clause_number="1.1",
                clause_id="clause_000",
                char_count=80,
                word_count=12,
            ),
            Clause(
                index=1,
                heading="Limitation of Liability",
                text="In no event shall Company aggregate liability exceed total fees paid in past 12 months.",
                page_number=2,
                clause_number="2.1",
                clause_id="clause_001",
                char_count=90,
                word_count=15,
            ),
        ]

        factor_bds = [
            FactorScoreBreakdown(
                factor_name="payment",
                category="payment",
                weight=0.15,
                weight_pct=15.0,
                suspiciousness_score=60.0,
                weighted_contribution=9.0,
                trigger_count=1,
                is_applicable=True,
            ),
            FactorScoreBreakdown(
                factor_name="liability",
                category="liability",
                weight=0.25,
                weight_pct=25.0,
                suspiciousness_score=0.0,
                weighted_contribution=0.0,
                trigger_count=0,
                is_applicable=True,
            ),
        ]

        missing_prots = [
            MissingProtection(
                rule_id="RULE-PROT-001",
                label="Missing Governing Law & Jurisdiction",
                severity="medium",
                description="No express governing law or jurisdiction clause identified.",
                rationale="Uncertain legal venue increases dispute costs and litigation risk.",
                recommendation="Add standard governing law clause (e.g. State of Delaware).",
            )
        ]

        report = OverallRiskReport(
            overall_score=42.0,
            risk_level="medium",
            high_risk_count=0,
            medium_risk_count=1,
            low_risk_count=1,
            scoring_profile="standard_commercial",
            factor_breakdowns=factor_bds,
            missing_protections=missing_prots,
            score_formula_explanation="Weighted Risk Score = sum(Suspiciousness_i * Weight_i) / sum(Applicable_Weights)",
            clause_details=[
                ClauseRiskDetail(
                    clause_index=0,
                    heading="Payment Terms",
                    category="payment",
                    category_weight=0.15,
                    flags=[
                        RiskFlag(
                            rule_id="RULE-DEADLINE-001",
                            label="Indefinite Performance Timeline",
                            severity="medium",
                            description="Payment timeline lacks fixed day count.",
                            matched_snippet="promptly in due course",
                            suspiciousness_score=60.0,
                            weight=0.15,
                            rationale="Leaves payment obligation ambiguous.",
                        )
                    ],
                    clause_score=60.0,
                    risk_level="medium",
                ),
                ClauseRiskDetail(
                    clause_index=1,
                    heading="Limitation of Liability",
                    category="liability",
                    category_weight=0.25,
                    flags=[],
                    clause_score=0.0,
                    risk_level="low",
                ),
            ],
            category_distribution={"payment": 1, "liability": 1},
        )

        structured_out = ClauseExplanationOutput(
            clause_id="clause_000",
            risk_labels=["RULE-DEADLINE-001"],
            explanation="The phrase 'promptly in due course' is legally vague and creates collection uncertainty.",
            evidence_quotes=["promptly in due course"],
            potential_impact="Invoices may be delayed indefinitely without breaching the literal terms.",
            suggested_revision="Client shall pay Contractor within thirty (30) calendar days of invoice receipt.",
            assumptions=["Commercial B2B context"],
            uncertainty_notes="No external terms of trade referenced.",
            grounding_verified=True,
        )

        explanations = [
            Explanation(
                clause_index=0,
                heading="Payment Terms",
                summary="Payment timeline is indefinite.",
                recommendations=["Specify 30 days net payment terms."],
                status="completed_llm",
                structured_output=structured_out,
            ),
            Explanation(
                clause_index=1,
                heading="Limitation of Liability",
                summary="No specific risks identified for this clause.",
                recommendations=["No action items at this time."],
                status="completed_offline",
            ),
        ]

        pdf_bytes = generate_pdf_report(
            report,
            explanations,
            clauses=clauses,
            doc_metadata=metadata,
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 2000
        assert pdf_bytes.startswith(b"%PDF")

    def test_generate_pdf_clean_contract(self):
        """Test PDF generation when contract has zero risk flags."""
        report = OverallRiskReport(
            overall_score=0.0,
            risk_level="low",
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=2,
            scoring_profile="standard_commercial",
            clause_details=[
                ClauseRiskDetail(clause_index=0, heading="Recitals", category="other", category_weight=1.0, clause_score=0.0, risk_level="low"),
                ClauseRiskDetail(clause_index=1, heading="Definitions", category="other", category_weight=1.0, clause_score=0.0, risk_level="low"),
            ],
            category_distribution={"other": 2},
        )
        explanations = [
            Explanation(clause_index=0, heading="Recitals", summary="No specific risks identified.", recommendations=[]),
            Explanation(clause_index=1, heading="Definitions", summary="No specific risks identified.", recommendations=[]),
        ]

        pdf_bytes = generate_pdf_report(report, explanations)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")
