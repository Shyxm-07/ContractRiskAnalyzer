"""
test_llm_explainer.py – Tests for the template-based explainer.
"""

from llm_explainer import Explanation, explain_flags
from risk_rules import RiskFlag


class TestExplainFlags:
    def test_known_rule_id(self):
        flag = RiskFlag(
            rule_id="RISK-001",
            label="Unlimited Liability",
            severity="high",
            description="No cap on liability.",
            matched_snippet="unlimited liability",
        )
        expl = explain_flags(0, "Liability Clause", [flag])
        assert isinstance(expl, Explanation)
        assert "unlimited liability" in expl.summary.lower()
        assert len(expl.recommendations) > 0

    def test_unknown_rule_id(self):
        flag = RiskFlag(
            rule_id="RISK-999",
            label="Custom Flag",
            severity="medium",
            description="A custom risk.",
            matched_snippet="snippet",
        )
        expl = explain_flags(1, "Custom", [flag])
        assert "Custom Flag" in expl.summary
        assert any("counsel" in r.lower() for r in expl.recommendations)

    def test_no_flags(self):
        expl = explain_flags(2, "Benign Clause", [])
        assert "no specific risks" in expl.summary.lower()

    def test_multiple_flags(self):
        flags = [
            RiskFlag("RISK-001", "A", "high", "desc1", "snip1"),
            RiskFlag("RISK-003", "B", "medium", "desc2", "snip2"),
        ]
        expl = explain_flags(3, "Multi", flags)
        assert len(expl.recommendations) >= 2
