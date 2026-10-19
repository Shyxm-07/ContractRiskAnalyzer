"""
test_risk_scorer.py – Tests for the risk scoring engine.
"""

from classifier import ClassificationResult
from risk_rules import RiskFlag
from risk_scorer import OverallRiskReport, score_contract


def _make_cr(index: int, category: str, confidence: float = 0.8) -> ClassificationResult:
    return ClassificationResult(
        clause_index=index,
        heading=f"Clause {index}",
        text="placeholder",
        predicted_category=category,
        confidence=confidence,
    )


class TestScoreContract:
    def test_returns_report(self):
        classifications = [_make_cr(0, "Payment Terms"), _make_cr(1, "Termination")]
        report = score_contract(classifications, {}, [])
        assert isinstance(report, OverallRiskReport)
        assert 0 <= report.overall_score <= 100

    def test_flags_increase_score(self):
        """Clauses with risk flags should score higher than those without."""
        classifications = [_make_cr(0, "Indemnification")]
        report_no_flags = score_contract(classifications, {}, [])

        flag = RiskFlag(
            rule_id="RISK-001",
            label="Unlimited Liability",
            severity="high",
            description="test",
            matched_snippet="test",
        )
        report_with_flags = score_contract(classifications, {0: [flag]}, [])
        assert report_with_flags.overall_score >= report_no_flags.overall_score

    def test_clause_counts(self):
        classifications = [
            _make_cr(0, "Indemnification"),
            _make_cr(1, "Governing Law"),
            _make_cr(2, "Amendment"),
        ]
        flags = {
            0: [
                RiskFlag("RISK-001", "Test", "high", "desc", "snip"),
            ]
        }
        report = score_contract(classifications, flags, [])
        total = report.high_risk_count + report.medium_risk_count + report.low_risk_count
        assert total == 3

    def test_category_distribution(self):
        classifications = [
            _make_cr(0, "Termination"),
            _make_cr(1, "Termination"),
            _make_cr(2, "Payment Terms"),
        ]
        report = score_contract(classifications, {}, [])
        assert report.category_distribution.get("Termination") == 2
        assert report.category_distribution.get("Payment Terms") == 1
