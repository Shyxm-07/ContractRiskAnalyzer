"""
test_classifier.py – Comprehensive tests for contract clause classification.

Covers Phase 3 requirements:
  • Valid predictions across relevant categories (Payment, Termination, Indemnity, etc.)
  • TF-IDF + Logistic Regression baseline ML pipeline
  • Unknown labels and label mapping
  • Low confidence predictions handling and status
  • Empty / whitespace clauses handling
  • Long clauses and sequence truncation handling
  • Unavailable model graceful fallback without crashes
  • Evaluation suite calculating real precision, recall, F1, and confusion matrix
"""

import pytest

import classifier
from classifier import (
    ClassificationResult,
    ClassificationStatus,
    KeywordClauseClassifier,
    LegalBertClauseClassifier,
    TfidfClauseClassifier,
    classify_clause,
    classify_clauses,
    evaluate_classifier,
    get_tfidf_classifier,
    normalize_category_name,
)
from clause_splitter import Clause


class TestTfidfClassifierBaseline:
    def test_payment_classification(self):
        clf = get_tfidf_classifier()
        cat, conf, scores, is_truncated = clf.predict_one(
            "Client shall pay all invoiced amounts within thirty days. Late fees shall apply."
        )
        assert cat == "Payment"
        assert conf > 0.25
        assert "Payment" in scores
        assert not is_truncated

    def test_termination_classification(self):
        clf = get_tfidf_classifier()
        cat, conf, scores, _ = clf.predict_one(
            "Either party may terminate this agreement upon thirty days written notice."
        )
        assert cat == "Termination"
        assert conf > 0.25

    def test_indemnity_classification(self):
        clf = get_tfidf_classifier()
        cat, conf, scores, _ = clf.predict_one(
            "Vendor shall defend, indemnify, and hold harmless Customer from and against any third-party claims."
        )
        assert cat == "Indemnity"
        assert conf > 0.25

    def test_confidentiality_classification(self):
        clf = get_tfidf_classifier()
        cat, conf, scores, _ = clf.predict_one(
            "Recipient agrees to hold all Proprietary and Confidential Information in strict confidence."
        )
        assert cat == "Confidentiality"
        assert conf > 0.25

    def test_liability_classification(self):
        clf = get_tfidf_classifier()
        cat, conf, scores, _ = clf.predict_one(
            "In no event shall either party's aggregate cumulative liability exceed the total fees paid."
        )
        assert cat == "Liability"
        assert conf > 0.25

    def test_non_compete_classification(self):
        clf = get_tfidf_classifier()
        cat, conf, scores, _ = clf.predict_one(
            "During the Term and for a period of eighteen months thereafter, Employee shall not engage in competitive business."
        )
        assert cat == "Non-Compete"
        assert conf > 0.25

    def test_dispute_resolution_classification(self):
        clf = get_tfidf_classifier()
        cat, conf, scores, _ = clf.predict_one(
            "Any dispute arising under this Agreement shall be submitted to binding arbitration in New York."
        )
        assert cat == "Dispute Resolution"
        assert conf > 0.25

    def test_intellectual_property_classification(self):
        clf = get_tfidf_classifier()
        cat, conf, scores, _ = clf.predict_one(
            "Contractor hereby assigns to Client all right, title, and ownership in all inventions, code, patents, and Deliverables."
        )
        assert cat == "Intellectual Property"
        assert conf > 0.25


class TestClassifyClauseInterface:
    def test_valid_clause_prediction(self):
        clause = Clause(
            index=0,
            heading="5. Indemnification",
            text="Provider shall defend, indemnify, and hold harmless Customer against any third-party lawsuits and claims.",
        )
        result = classify_clause(clause, backend="tfidf")
        assert isinstance(result, ClassificationResult)
        assert result.predicted_category in ("Indemnity", "Indemnification")
        assert result.confidence > 0.2
        assert result.clause_index == 0
        assert result.heading == "5. Indemnification"
        assert result.model_identifier == "scikit-learn-tfidf-logistic-regression"
        assert result.classification_status in (
            ClassificationStatus.SUCCESS.value,
            ClassificationStatus.FALLBACK.value,
        )

    def test_empty_clause_handling(self):
        clause = Clause(index=1, heading="Empty Section", text="   ")
        result = classify_clause(clause)
        assert result.predicted_category == "Other / Unknown"
        assert result.confidence == 0.0
        assert result.classification_status == ClassificationStatus.EMPTY_INPUT.value
        assert not result.is_truncated

    def test_low_confidence_clause_handling(self):
        # Text with no legal terms
        clause = Clause(
            index=2,
            heading="Nonsense",
            text="xyz abc 123 random words completely unrelated to law.",
        )
        result = classify_clause(clause, backend="tfidf")
        assert (
            result.classification_status == ClassificationStatus.LOW_CONFIDENCE.value
            or result.predicted_category == "Other / Unknown"
            or result.confidence < 0.35
        )

    def test_long_clause_truncation_flag(self):
        long_body = "The Customer shall pay all monthly fees and invoices within thirty days. " * 60
        clause = Clause(index=3, heading="Long Payment", text=long_body)
        result = classify_clause(clause, backend="tfidf")
        assert result.is_truncated is True
        assert result.predicted_category == "Payment"

    def test_batch_classification_sequence(self):
        clauses = [
            Clause(index=0, heading="P", text="Client shall pay all fees and invoices within 30 days."),
            Clause(index=1, heading="T", text="Either party may terminate this agreement on written notice."),
            Clause(index=2, heading="C", text="Each party agrees to hold all Confidential Information in strict confidence and not disclose it."),
        ]
        results = classify_clauses(clauses, backend="tfidf")
        assert len(results) == 3
        assert results[0].clause_index == 0
        assert results[1].clause_index == 1
        assert results[2].clause_index == 2
        assert results[0].predicted_category == "Payment"
        assert results[1].predicted_category == "Termination"
        assert results[2].predicted_category == "Confidentiality"


class TestUnavailableModelFallback:
    def test_unavailable_transformer_falls_back_gracefully(self, monkeypatch):
        # Simulate LegalBERT download or loading failure
        def mock_load_fail():
            raise RuntimeError("Model download failed (offline test)")

        monkeypatch.setattr(classifier, "get_legalbert_classifier", mock_load_fail)

        clause = Clause(
            index=0,
            heading="Payment",
            text="The buyer shall pay all invoices and remuneration within thirty days.",
        )
        result = classify_clause(clause, use_transformer=True)
        assert isinstance(result, ClassificationResult)
        assert result.predicted_category == "Payment"
        assert result.classification_status in (
            ClassificationStatus.FALLBACK.value,
            ClassificationStatus.SUCCESS.value,
        )


class TestLabelMappingAndNormalization:
    def test_category_synonym_normalization(self):
        assert normalize_category_name("indemnification") == "Indemnity"
        assert normalize_category_name("limitation of liability") == "Liability"
        assert normalize_category_name("payment terms") == "Payment"
        assert normalize_category_name("governing law") == "Dispute Resolution"
        assert normalize_category_name("auto-renewal") == "Renewal"
        assert normalize_category_name("uncategorised") == "Other / Unknown"


class TestEvaluationSuite:
    def test_evaluate_classifier_produces_real_metrics(self):
        clf = get_tfidf_classifier()
        report = evaluate_classifier(clf)

        assert "accuracy" in report
        assert "macro_avg" in report
        assert "weighted_avg" in report
        assert "confusion_matrix" in report
        assert "classification_report_str" in report

        # Accuracy must be a real float between 0.0 and 1.0
        assert 0.0 <= report["accuracy"] <= 1.0
        assert report["total_samples"] > 0
        assert len(report["confusion_matrix"]) > 0

        # Validate macro F1 score existence
        assert "f1-score" in report["macro_avg"]
