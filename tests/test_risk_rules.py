"""
test_risk_rules.py – Tests for the rule-based risk flag engine.
"""

from risk_rules import evaluate_clause, evaluate_document, RiskFlag


class TestClauseRules:
    def test_unlimited_liability(self):
        text = "There shall be no cap or limitation on liability for either party."
        flags = evaluate_clause(text)
        ids = [f.rule_id for f in flags]
        assert "RISK-001" in ids

    def test_unilateral_termination(self):
        text = "Client may terminate this agreement at any time without cause."
        flags = evaluate_clause(text)
        ids = [f.rule_id for f in flags]
        assert "RISK-002" in ids

    def test_auto_renewal(self):
        text = "This agreement shall be automatically renewed for successive one-year terms."
        flags = evaluate_clause(text)
        ids = [f.rule_id for f in flags]
        assert "RISK-003" in ids

    def test_broad_indemnification(self):
        text = "Provider shall indemnify Client against any and all claims, damages and losses."
        flags = evaluate_clause(text)
        ids = [f.rule_id for f in flags]
        assert "RISK-004" in ids

    def test_excessive_non_compete(self):
        text = "The non-compete restriction shall apply for 5 years after termination."
        flags = evaluate_clause(text)
        ids = [f.rule_id for f in flags]
        assert "RISK-005" in ids

    def test_ip_assignment(self):
        text = "Provider assigns all intellectual property rights to Client."
        flags = evaluate_clause(text)
        ids = [f.rule_id for f in flags]
        assert "RISK-007" in ids

    def test_no_flags_for_benign_clause(self):
        text = "The parties agree to cooperate in good faith to resolve any issues."
        flags = evaluate_clause(text)
        assert len(flags) == 0


class TestDocumentRules:
    def test_missing_governing_law(self):
        text = "This contract is between Acme and Beta for consulting services with net 30 payment terms."
        flags = evaluate_document(text)
        ids = [f.rule_id for f in flags]
        assert "RISK-006" in ids

    def test_governing_law_present(self):
        text = "This Agreement shall be governed by the laws of England."
        flags = evaluate_document(text)
        # Should NOT fire RISK-006
        ids = [f.rule_id for f in flags]
        assert "RISK-006" not in ids


class TestRiskFlagFields:
    def test_flag_has_all_fields(self):
        text = "There is unlimited liability under this clause."
        flags = evaluate_clause(text)
        assert len(flags) >= 1
        flag = flags[0]
        assert flag.rule_id
        assert flag.label
        assert flag.severity in ("low", "medium", "high")
        assert flag.description
        assert flag.matched_snippet
