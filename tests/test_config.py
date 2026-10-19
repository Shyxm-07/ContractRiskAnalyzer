"""
test_config.py – Tests for config.py
"""

import pytest
from config import AppSettings, risk_level_for_score, settings


class TestAppSettings:
    def test_defaults_load(self):
        """Settings object should load with sensible defaults."""
        s = AppSettings()
        assert s.app_title
        assert s.max_upload_size_mb > 0
        assert s.classifier_model_name
        assert s.explainer_model_name
        assert len(s.clause_categories) > 0
        assert len(s.risk_weights) > 0

    def test_allowed_extensions(self):
        assert ".pdf" in settings.allowed_extensions
        assert ".docx" in settings.allowed_extensions

    def test_log_level_validation(self):
        with pytest.raises(Exception):
            AppSettings(log_level="INVALID")


class TestRiskLevel:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.0, "low"),
            (0.2, "low"),
            (0.39, "low"),
            (0.4, "medium"),
            (0.55, "medium"),
            (0.69, "medium"),
            (0.7, "high"),
            (0.85, "high"),
            (1.0, "high"),
        ],
    )
    def test_risk_level_for_score(self, score, expected):
        assert risk_level_for_score(score) == expected
