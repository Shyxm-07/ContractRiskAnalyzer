"""
test_integration.py – End-to-end integration test using the sample contract.
"""

from clause_splitter import split_into_clauses
from classifier import classify_clauses
from llm_explainer import explain_flags
from risk_rules import evaluate_clause, evaluate_document
from risk_scorer import score_contract
from sample_contract import SAMPLE_CONTRACT_TEXT


class TestEndToEnd:
    def test_full_pipeline_with_sample(self):
        """Run the entire analysis pipeline on the sample contract."""
        # 1. Split
        clauses = split_into_clauses(SAMPLE_CONTRACT_TEXT)
        assert len(clauses) >= 10

        # 2. Classify
        classifications = classify_clauses(clauses, use_transformer=False)
        assert len(classifications) == len(clauses)

        # 3. Risk rules (clause-level)
        clause_flags = {}
        for clause in clauses:
            flags = evaluate_clause(clause.text)
            if flags:
                clause_flags[clause.index] = flags

        # At least some flags should fire on the sample contract
        assert len(clause_flags) > 0, "Expected at least one clause with risk flags"

        # 4. Document-level rules
        document_flags = evaluate_document(SAMPLE_CONTRACT_TEXT)
        # The sample contract DOES have a governing law clause (Dispute Resolution
        # mentions London), so RISK-006 may or may not fire – just ensure no crash.
        assert isinstance(document_flags, list)

        # 5. Score
        report = score_contract(classifications, clause_flags, document_flags)
        assert 0 <= report.overall_score <= 100
        assert report.risk_level in ("low", "medium", "high")
        assert report.high_risk_count + report.medium_risk_count + report.low_risk_count == len(clauses)

        # 6. Explanations
        explanations = []
        for detail in report.clause_details:
            expl = explain_flags(
                detail.clause_index,
                detail.heading,
                detail.flags,
                use_llm=False,
            )
            explanations.append(expl)
        assert len(explanations) == len(clauses)
