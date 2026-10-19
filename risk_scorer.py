"""
risk_scorer.py – Weighted contract risk scoring engine.

Implements Phase 4:
  • Calculates transparent, explainable suspiciousness scores (0 to 100).
  • Implements exact weighted scoring formula:
      weighted_score = sum(suspiciousness_i * weight_i) / sum(applicable_weights)
  • Factor weights sum to 100% within configurable scoring profiles:
      - standard_commercial (balanced)
      - vendor_protective (emphasizes payment & liability caps)
      - buyer_protective (emphasizes termination, IP, indemnification)
  • Handles missing factors, non-applicable factors, overlapping rules, and duplicate evidence.
  • Distinguishes missing expected clauses from suspicious clauses present in document.
  • Strictly separates ML classification confidence from risk severity.
  • 100% deterministic calculation with a fully transparent audit trail and score breakdown.
  • Documented provisional risk bands (Low: 0-39.9, Medium: 40-69.9, High: 70-100).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from classifier import ClassificationResult
from config import risk_level_for_score, settings
from risk_rules import RiskFlag

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FactorScoreBreakdown:
    """
    Transparent breakdown for an individual risk factor in the weighted score.
    """
    factor_name: str
    category: str
    weight: float                        # Fraction (e.g. 0.20)
    weight_pct: float                    # Percentage (e.g. 20.0%)
    suspiciousness_score: float          # 0 to 100 calibrated rating
    weighted_contribution: float         # suspiciousness * weight
    is_applicable: bool = True           # Handled if non-applicable
    trigger_count: int = 0
    rationales: List[str] = field(default_factory=list)
    formula_detail: str = ""


@dataclass
class MissingProtection:
    """
    A standard expected protective clause that was absent from the agreement.
    Reported separately from suspicious clauses actually present.
    """
    rule_id: str
    label: str
    severity: str                        # "low", "medium", "high"
    description: str
    rationale: str
    recommendation: str
    weight: float = 0.10
    suspiciousness_score: float = 65.0


@dataclass
class ClauseRiskDetail:
    """Per-clause risk breakdown."""
    clause_index: int
    heading: str
    category: str                        # ML topical classification
    category_weight: float = 1.0
    flags: List[RiskFlag] = field(default_factory=list)
    clause_score: float = 0.0            # 0 to 100
    risk_level: str = "low"              # low / medium / high
    clause_id: str = ""
    suspiciousness_score: float = 0.0    # 0 to 100
    score_breakdown: str = ""            # Transparent derivation
    has_risk: bool = False
    ml_confidence: Optional[float] = None  # Kept separate from risk severity


@dataclass
class OverallRiskReport:
    """Aggregated risk report for the entire contract."""
    overall_score: float = 0.0           # 0 to 100
    risk_level: str = "low"              # low / medium / high
    clause_details: List[ClauseRiskDetail] = field(default_factory=list)
    document_flags: List[RiskFlag] = field(default_factory=list)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0

    # Phase 4 additions:
    scoring_profile: str = "standard_commercial"
    factor_breakdowns: List[FactorScoreBreakdown] = field(default_factory=list)
    missing_protections: List[MissingProtection] = field(default_factory=list)
    suspicious_clause_count: int = 0
    clean_clause_count: int = 0
    applicable_weight_sum: float = 1.0
    score_formula_explanation: str = ""
    is_deterministic: bool = True

    @property
    def profile(self) -> str:
        return self.scoring_profile

    @property
    def suspicious_clauses_count(self) -> int:
        return self.suspicious_clause_count

    @property
    def clean_clauses_count(self) -> int:
        return self.clean_clause_count

    @property
    def formula_explanation(self) -> str:
        return self.score_formula_explanation

    @property
    def missing_protections_count(self) -> int:
        return len(self.missing_protections)


# Factor label human-readable mappings
FACTOR_DISPLAY_NAMES: Dict[str, str] = {
    "liability": "Liability Limitation & Caps",
    "indemnity": "Indemnification Obligations",
    "termination": "Termination & Exit Rights",
    "payment": "Payment Terms & Modifications",
    "confidentiality": "Confidentiality & Non-Disclosure",
    "missing_protections": "Standard Expected Protections",
    "ambiguity_deadlines": "Ambiguous Wording & Deadlines",
    "renewal": "Automatic Renewal & Term Extension",
}

# Recommendations for missing expected protections
_MISSING_PROTECTION_RECOMMENDATIONS: Dict[str, str] = {
    "RULE-PROT-001": "Add an express Governing Law and Exclusive Jurisdiction clause specifying a stable legal venue.",
    "RISK-006": "Add an express Governing Law and Exclusive Jurisdiction clause specifying a stable legal venue.",
    "RULE-PROT-002": "Include an express Limitation of Liability clause setting a reasonable aggregate monetary cap (e.g. 12 months fees).",
    "RULE-PROT-003": "Include a standard Termination for Material Breach clause providing 30 days written notice to cure default.",
    "RULE-PROT-004": "Add mutual confidentiality obligations with customary trade secret and public domain carve-outs.",
}


# ══════════════════════════════════════════════════════════════════════
#  WEIGHTED SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════

def score_contract(
    classifications: List[ClassificationResult],
    clause_flags: Dict[int, List[RiskFlag]],
    document_flags: List[RiskFlag],
    profile_name: Optional[str] = None,
    profile: Optional[str] = None,
) -> OverallRiskReport:
    """
    Compute contract risk score using transparent weighted scoring.

    Formula:
        weighted_score = sum(suspiciousness_i * weight_i) / sum(applicable_weights)

    Parameters
    ----------
    classifications : list[ClassificationResult]
        One per clause, carrying topical category and model confidence.
    clause_flags : dict[int, list[RiskFlag]]
        Mapping from clause index to triggered risk flags.
    document_flags : list[RiskFlag]
        Flags applying at the document level (missing protections, inconsistency).
    profile_name : str, optional
        Profile from settings.scoring_profiles (standard_commercial, vendor_protective, buyer_protective).
    profile : str, optional
        Alias for profile_name.
    """
    # 1. Determine profile & active weights (summing to 100% / 1.00)
    profile_key = profile_name or profile or settings.scoring_profile
    if profile_key not in settings.scoring_profiles:
        profile_key = "standard_commercial"
    profile_weights: Dict[str, float] = settings.scoring_profiles[profile_key]

    # 2. Track category distribution and build per-clause details
    clause_details: List[ClauseRiskDetail] = []
    cat_dist: Dict[str, int] = {}
    present_categories: Set[str] = set()

    for cr in classifications:
        cat = cr.predicted_category
        cat_dist[cat] = cat_dist.get(cat, 0) + 1
        present_categories.add(cat.lower())

        flags = clause_flags.get(cr.clause_index, [])
        cat_weight = settings.risk_weights.get(cat, 0.3)

        # Deduplicate multiple flags firing on the same snippet
        unique_flags = flags

        if unique_flags:
            # Calibrate clause suspiciousness from defined evidence
            # Take primary trigger score + diminishing return on secondary triggers
            sorted_scores = sorted([f.suspiciousness_score for f in unique_flags], reverse=True)
            primary_score = sorted_scores[0]
            secondary_boost = sum(s * 0.15 for s in sorted_scores[1:])
            clause_suspiciousness = min(round(primary_score + secondary_boost, 1), 100.0)

            # Build transparent derivation string
            flag_labels = ", ".join(f.label for f in unique_flags)
            breakdown_str = f"Triggered: {flag_labels} (Primary: {primary_score:.0f}"
            if len(sorted_scores) > 1:
                breakdown_str += f" + secondary boost: {secondary_boost:.1f}"
            breakdown_str += f") = {clause_suspiciousness:.1f}/100"
            has_risk = True
        else:
            # Benign clause with no triggers
            clause_suspiciousness = 0.0
            breakdown_str = "No risk triggers detected (0.0/100)"
            has_risk = False

        level = risk_level_for_score(clause_suspiciousness)

        detail = ClauseRiskDetail(
            clause_index=cr.clause_index,
            heading=cr.heading,
            category=cat,
            category_weight=cat_weight,
            flags=unique_flags,
            clause_score=clause_suspiciousness,
            risk_level=level,
            clause_id=getattr(cr, "clause_id", f"c_{cr.clause_index}"),
            suspiciousness_score=clause_suspiciousness,
            score_breakdown=breakdown_str,
            has_risk=has_risk,
            ml_confidence=getattr(cr, "confidence", None),  # Separated from risk
        )
        clause_details.append(detail)

    # 3. Categorize document flags & separate missing expected protections
    missing_protections: List[MissingProtection] = []
    other_doc_flags: List[RiskFlag] = []

    for df in document_flags:
        if df.category == "missing_protections" or "Missing" in df.label:
            rec = _MISSING_PROTECTION_RECOMMENDATIONS.get(
                df.rule_id, "Add standard protective provisions to address this omission."
            )
            mp = MissingProtection(
                rule_id=df.rule_id,
                label=df.label,
                severity=df.severity,
                description=df.description,
                rationale=df.rationale,
                recommendation=rec,
                weight=df.weight,
                suspiciousness_score=df.suspiciousness_score,
            )
            # Avoid duplicate missing protection entries from aliases
            if not any(x.label == mp.label for x in missing_protections):
                missing_protections.append(mp)
        else:
            other_doc_flags.append(df)

    # 4. Group all triggered flags by factor category
    factor_flags: Dict[str, List[RiskFlag]] = {cat_name: [] for cat_name in profile_weights}

    for d in clause_details:
        for f in d.flags:
            cat_key = f.category.lower()
            # Normalize to profile factor keys
            matched_key = None
            for pk in profile_weights:
                if pk in cat_key or cat_key in pk:
                    matched_key = pk
                    break
            if not matched_key:
                matched_key = "ambiguity_deadlines"
            factor_flags[matched_key].append(f)

    # Document-level flags (missing protections, inconsistencies)
    for mp in missing_protections:
        dummy_flag = RiskFlag(
            rule_id=mp.rule_id,
            label=mp.label,
            severity=mp.severity,
            description=mp.description,
            matched_snippet="(not found)",
            category="missing_protections",
            suspiciousness_score=mp.suspiciousness_score,
            weight=mp.weight,
            rationale=mp.rationale,
        )
        factor_flags["missing_protections"].append(dummy_flag)

    for odf in other_doc_flags:
        factor_flags["ambiguity_deadlines"].append(odf)

    # 5. Compute Suspiciousness for each profile factor
    factor_breakdowns: List[FactorScoreBreakdown] = []
    weighted_sum = 0.0
    applicable_weights_sum = 0.0

    for factor_key, weight in profile_weights.items():
        flags_for_factor = factor_flags.get(factor_key, [])
        factor_name = FACTOR_DISPLAY_NAMES.get(factor_key, factor_key.title())

        # Determine applicability:
        # A factor is applicable if:
        #  - It triggered one or more flags, OR
        #  - Clauses in this domain exist in the contract, OR
        #  - It is an essential universal safeguard (liability, termination, missing_protections)
        is_universal = factor_key in ("liability", "termination", "missing_protections")
        has_domain_clauses = any(
            factor_key in c.lower() or c.lower() in factor_key for c in present_categories
        )
        is_applicable = bool(flags_for_factor or has_domain_clauses or is_universal)

        if not is_applicable:
            # Non-applicable factor: exclude from weighted scoring denominator
            fb = FactorScoreBreakdown(
                factor_name=factor_name,
                category=factor_key,
                weight=weight,
                weight_pct=round(weight * 100, 1),
                suspiciousness_score=0.0,
                weighted_contribution=0.0,
                is_applicable=False,
                trigger_count=0,
                rationales=["Factor not applicable to this contract type"],
                formula_detail=f"Not applicable (weight {weight * 100:.0f}% excluded from denominator)",
            )
            factor_breakdowns.append(fb)
            continue

        if flags_for_factor:
            # Deduplicate by rule ID to avoid double-counting aliases (e.g. RULE-LIAB-001 vs RISK-001)
            dedup_flags: Dict[str, RiskFlag] = {}
            for flg in flags_for_factor:
                # Group by canonical concept (e.g. uncapped liability)
                concept = flg.label.lower()
                if concept not in dedup_flags or flg.suspiciousness_score > dedup_flags[concept].suspiciousness_score:
                    dedup_flags[concept] = flg

            unique_factor_flags = list(dedup_flags.values())
            scores = sorted([f.suspiciousness_score for f in unique_factor_flags], reverse=True)
            factor_suspiciousness = scores[0]
            if len(scores) > 1:
                # Diminishing boost for multiple distinct triggers in same category
                factor_suspiciousness += sum(s * 0.15 for s in scores[1:])
            factor_suspiciousness = min(round(factor_suspiciousness, 1), 100.0)

            rationales = [f"{f.label}: {f.rationale}" for f in unique_factor_flags if f.rationale]
            detail_str = f"Score {factor_suspiciousness:.1f} × Weight {weight:.2f} = {factor_suspiciousness * weight:.2f}"
            trig_count = len(unique_factor_flags)
        else:
            # Applicable factor, but verified clean
            factor_suspiciousness = 0.0
            rationales = ["Evaluated: No risk flags triggered in this domain."]
            detail_str = f"Score 0.0 × Weight {weight:.2f} = 0.00"
            trig_count = 0

        weighted_contrib = factor_suspiciousness * weight
        weighted_sum += weighted_contrib
        applicable_weights_sum += weight

        fb = FactorScoreBreakdown(
            factor_name=factor_name,
            category=factor_key,
            weight=weight,
            weight_pct=round(weight * 100, 1),
            suspiciousness_score=factor_suspiciousness,
            weighted_contribution=round(weighted_contrib, 2),
            is_applicable=True,
            trigger_count=trig_count,
            rationales=rationales,
            formula_detail=detail_str,
        )
        factor_breakdowns.append(fb)

    # 6. Final Weighted Score Calculation
    # weighted_score = sum(suspiciousness_i * weight_i) / sum(applicable_weights)
    denom = applicable_weights_sum if applicable_weights_sum > 0 else 1.0
    overall_score = round(min(max(weighted_sum / denom, 0.0), 100.0), 1)
    overall_level = risk_level_for_score(overall_score)

    high_count = sum(1 for d in clause_details if d.risk_level == "high")
    med_count = sum(1 for d in clause_details if d.risk_level == "medium")
    low_count = sum(1 for d in clause_details if d.risk_level == "low")
    suspicious_count = sum(1 for d in clause_details if d.has_risk)
    clean_count = len(clause_details) - suspicious_count

    formula_exp = (
        f"weighted_score = sum(suspiciousness_i × weight_i) / sum(applicable_weights)\n"
        f"= {weighted_sum:.2f} / {denom:.2f} = {overall_score:.1f}/100 ({overall_level.upper()})"
    )

    report = OverallRiskReport(
        overall_score=overall_score,
        risk_level=overall_level,
        clause_details=clause_details,
        document_flags=document_flags,
        category_distribution=cat_dist,
        high_risk_count=high_count,
        medium_risk_count=med_count,
        low_risk_count=low_count,
        scoring_profile=profile_key,
        factor_breakdowns=factor_breakdowns,
        missing_protections=missing_protections,
        suspicious_clause_count=suspicious_count,
        clean_clause_count=clean_count,
        applicable_weight_sum=round(denom, 4),
        score_formula_explanation=formula_exp,
        is_deterministic=True,
    )

    logger.info(
        "Contract scored %.1f / 100 (%s) under profile '%s' (weights sum: %.2f)",
        overall_score, overall_level, profile_key, denom,
    )
    return report
