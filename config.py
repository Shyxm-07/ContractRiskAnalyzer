"""
config.py – Centralised application configuration.

Loads settings from environment variables (.env) with sensible defaults.
Uses Pydantic for validation so bad values fail fast at startup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# ── Load .env if present ─────────────────────────────────────────────
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


# ── Pydantic settings model ──────────────────────────────────────────
class AppSettings(BaseModel):
    """Validated application settings."""

    app_title: str = Field(
        default_factory=lambda: os.getenv("APP_TITLE", "AI Contract Risk Analyser")
    )
    max_upload_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # ── Model identifiers ────────────────────────────────────────────
    classifier_backend: str = Field(
        default_factory=lambda: os.getenv("CLASSIFIER_BACKEND", "auto")
    )
    classifier_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "CLASSIFIER_MODEL_NAME", "AmirLGass/clauseguard-legalbert"
        )
    )
    legalbert_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "LEGALBERT_MODEL_NAME", "AmirLGass/clauseguard-legalbert"
        )
    )
    confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.35"))
    )
    classifier_max_seq_length: int = Field(
        default_factory=lambda: int(os.getenv("CLASSIFIER_MAX_SEQ_LENGTH", "512"))
    )
    explainer_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "EXPLAINER_MODEL_NAME", "google/flan-t5-base"
        )
    )
    hf_token: str = Field(
        default_factory=lambda: os.getenv("HF_TOKEN", "")
    )

    # ── Phase 5 Generative AI Explanation Engine ─────────────────────
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "template")
    )
    llm_model_name: str = Field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL_NAME", "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        )
    )
    llm_endpoint_url: str = Field(
        default_factory=lambda: os.getenv("LLM_ENDPOINT_URL", "")
    )
    llm_api_key: str = Field(
        default_factory=lambda: os.getenv("LLM_API_KEY", os.getenv("HF_TOKEN", ""))
    )
    llm_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0"))
    )
    llm_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2"))
    )
    llm_temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    llm_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024"))
    )
    enable_external_llm: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_EXTERNAL_LLM", "false").lower()
        in ("true", "1", "yes")
    )

    # ── Supported clause categories ──────────────────────────────────
    clause_categories: List[str] = Field(
        default=[
            "Payment",
            "Payment Terms",
            "Termination",
            "Liability",
            "Limitation of Liability",
            "Indemnity",
            "Indemnification",
            "Confidentiality",
            "Obligations",
            "Renewal",
            "Non-Compete",
            "Governing Law",
            "Dispute Resolution",
            "Intellectual Property",
            "Warranty",
            "Force Majeure",
            "Data Protection",
            "Assignment",
            "Amendment",
            "Other / Unknown",
        ]
    )

    # ── Risk-scoring weights (category → weight 0-1) ─────────────────
    risk_weights: Dict[str, float] = Field(
        default={
            "Indemnity": 0.9,
            "Indemnification": 0.9,
            "Liability": 0.85,
            "Limitation of Liability": 0.85,
            "Termination": 0.7,
            "Confidentiality": 0.6,
            "Obligations": 0.55,
            "Renewal": 0.5,
            "Non-Compete": 0.75,
            "Governing Law": 0.4,
            "Dispute Resolution": 0.5,
            "Intellectual Property": 0.8,
            "Payment": 0.65,
            "Payment Terms": 0.65,
            "Warranty": 0.7,
            "Force Majeure": 0.5,
            "Data Protection": 0.8,
            "Assignment": 0.45,
            "Amendment": 0.4,
            "Other / Unknown": 0.3,
            "Uncategorised": 0.3,
        }
    )

    # ── Phase 4 Scoring Profiles & Factor Weights (Sum = 1.00 / 100%) ──
    # Weights express the relative governance impact of each category within the applicable profile.
    scoring_profile: str = Field(
        default_factory=lambda: os.getenv("SCORING_PROFILE", "standard_commercial")
    )
    scoring_profiles: Dict[str, Dict[str, float]] = Field(
        default={
            "standard_commercial": {
                "liability": 0.20,
                "indemnity": 0.15,
                "termination": 0.15,
                "payment": 0.15,
                "confidentiality": 0.10,
                "missing_protections": 0.10,
                "ambiguity_deadlines": 0.10,
                "renewal": 0.05,
            },
            "vendor_protective": {
                "payment": 0.25,
                "liability": 0.25,
                "indemnity": 0.15,
                "termination": 0.15,
                "ambiguity_deadlines": 0.10,
                "confidentiality": 0.05,
                "missing_protections": 0.05,
            },
            "buyer_protective": {
                "liability": 0.20,
                "termination": 0.20,
                "indemnity": 0.20,
                "missing_protections": 0.15,
                "payment": 0.10,
                "confidentiality": 0.10,
                "ambiguity_deadlines": 0.05,
            },
        }
    )

    # ── Provisional Risk Bands (0-100 scale) ──────────────────────────
    # Note: These thresholds (Low: 0-39.9, Medium: 40-69.9, High: 70-100) are
    # provisional indicators for review prioritization and should be calibrated
    # to organizational risk tolerance and contract portfolio benchmarks.
    risk_threshold_low: float = Field(default=40.0)
    risk_threshold_medium: float = Field(default=70.0)

    # ── Allowed upload extensions ─────────────────────────────────────
    allowed_extensions: List[str] = Field(default=[".pdf", ".docx"])

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v


# Singleton instance – import this everywhere.
settings = AppSettings()


# ── Risk-level thresholds ─────────────────────────────────────────────
# Note: Thresholds are provisional indicators and should be evaluated against
# specific institutional guidelines.
RISK_THRESHOLDS = {
    "low": (0.0, 0.4),
    "medium": (0.4, 0.7),
    "high": (0.7, 1.0),
}


def risk_level_for_score(score: float) -> str:
    """
    Return 'low', 'medium', or 'high' for a score.
    Accepts either normalised 0-1 score or 0-100 score.
    """
    # Normalize if score is on 0-100 scale
    val = score / 100.0 if score > 1.0 else score
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= val < hi:
            return level
    return "high"

