"""
risk_rules.py – Rule-based contract risk detection engine.

Implements Phase 4:
  • Configurable rules across ambiguous wording, missing deadlines, unclear payment terms,
    one-sided termination, potentially uncapped liability, automatic renewal, confidentiality
    concerns, inconsistent clauses, and missing protections.
  • Rich RiskFlag data structure storing rule ID, label, category, severity, evidence snippet,
    suspiciousness score (0-100), transparent score breakdown, and legal rationale.
  • Context-aware matching with exception checking (avoids keyword-only false positives).
  • All findings treated as review indicators rather than declaring clauses illegal or invalid.
  • Preserves backward compatibility with RISK-001 through RISK-007.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RiskFlag:
    """
    A single risk finding raised against a clause or document.
    Treated as a review indicator for legal scrutiny.
    """
    rule_id: str
    label: str
    severity: str                           # "low", "medium", "high"
    description: str
    matched_snippet: str                    # Evidence: the portion of text triggering the rule
    category: str = "general"               # liability, termination, payment, ambiguity, etc.
    suspiciousness_score: float = 50.0      # 0 to 100 based on validated logic
    weight: float = 0.10                    # Configurable factor weight (fraction 0-1)
    rationale: str = ""                     # Why this issue may matter for review
    score_breakdown: str = ""               # Transparent derivation of suspiciousness score
    evidence_text: str = ""                 # Expanded context around matched snippet
    clause_id: Optional[str] = None
    clause_index: Optional[int] = None


@dataclass
class RuleDefinition:
    """Metadata specification for a contract risk rule."""
    rule_id: str
    name: str
    category: str
    severity: str                           # "low", "medium", "high"
    default_weight: float
    description: str
    evidence_requirement: str
    rationale: str
    test_cases_summary: str = ""


# ══════════════════════════════════════════════════════════════════════
#  RULE REGISTRY METADATA
# ══════════════════════════════════════════════════════════════════════

RULE_DEFINITIONS: Dict[str, RuleDefinition] = {
    # ── Potentially Uncapped Liability ────────────────────────────────
    "RULE-LIAB-001": RuleDefinition(
        rule_id="RULE-LIAB-001",
        name="Potentially Uncapped Liability",
        category="liability",
        severity="high",
        default_weight=0.20,
        description="Identifies language that disclaims, waives, or removes limitation of liability caps.",
        evidence_requirement="Explicit statements waiving liability caps or accepting unlimited liability without standard monetary limits.",
        rationale="Exposes a party to uncapped recovery for direct, incidental, or catastrophic damages.",
        test_cases_summary="Fires on 'no cap on liability', 'unlimited liability'. Does NOT fire on 'liability capped at fees paid'.",
    ),
    "RULE-LIAB-002": RuleDefinition(
        rule_id="RULE-LIAB-002",
        name="Broad Liability Cap Carve-outs",
        category="liability",
        severity="medium",
        default_weight=0.15,
        description="Identifies liability limitation clauses where exceptions swallow ordinary operational breaches.",
        evidence_requirement="Carve-outs from liability caps that include broad contract breach or performance failure.",
        rationale="Extensive carve-outs effectively nullify the protection provided by monetary liability limits.",
        test_cases_summary="Fires when carve-outs include 'any breach of agreement' without fault thresholds.",
    ),

    # ── Potentially One-Sided Termination Rights ───────────────────────
    "RULE-TERM-001": RuleDefinition(
        rule_id="RULE-TERM-001",
        name="Potentially One-Sided Termination for Convenience",
        category="termination",
        severity="high",
        default_weight=0.15,
        description="Identifies termination for convenience rights granted exclusively to one party.",
        evidence_requirement="Right to terminate at will or without cause given solely to Client or Provider rather than mutually.",
        rationale="One party may terminate after substantial resource commitment without reciprocal protection.",
        test_cases_summary="Fires on 'Client may terminate at any time without cause'. Does NOT fire on mutual termination.",
    ),
    "RULE-TERM-002": RuleDefinition(
        rule_id="RULE-TERM-002",
        name="Immediate Termination Without Cure Period",
        category="termination",
        severity="high",
        default_weight=0.15,
        description="Permits immediate contract termination for breach without customary notice or opportunity to cure.",
        evidence_requirement="Clause allowing immediate cancellation on breach without a 15-30 day cure period.",
        rationale="Denies the defaulting party an opportunity to rectify administrative or minor operational defaults.",
        test_cases_summary="Fires on 'terminate immediately upon breach'. Does NOT fire when a 30-day cure period is provided.",
    ),

    # ── Broad Indemnification ─────────────────────────────────────────
    "RULE-INDEM-001": RuleDefinition(
        rule_id="RULE-INDEM-001",
        name="Broad / Unhedged Indemnification",
        category="indemnity",
        severity="high",
        default_weight=0.15,
        description="Imposes expansive indemnification obligations covering any and all claims without standard fault standards.",
        evidence_requirement="Phrases requiring indemnification for 'any and all claims whatsoever' without reasonable limits.",
        rationale="May require a party to act as an unhedged commercial insurer for counterparty liabilities.",
        test_cases_summary="Fires on 'indemnify against any and all claims'.",
    ),

    # ── Ambiguous Wording ─────────────────────────────────────────────
    "RULE-AMBIG-001": RuleDefinition(
        rule_id="RULE-AMBIG-001",
        name="Subjective / Unilateral Discretion",
        category="ambiguity_deadlines",
        severity="medium",
        default_weight=0.10,
        description="Grants one party sole and absolute subjective discretion without objective criteria.",
        evidence_requirement="Phrases such as 'sole and absolute discretion' or 'at its unilateral option' unqualified by reasonableness.",
        rationale="Allows one party to make self-serving determinations regarding performance, approval, or pricing.",
        test_cases_summary="Fires on 'in its sole and absolute discretion'. Modulated if 'acting reasonably' is added.",
    ),
    "RULE-AMBIG-002": RuleDefinition(
        rule_id="RULE-AMBIG-002",
        name="Undefined High-Effort Standards",
        category="ambiguity_deadlines",
        severity="low",
        default_weight=0.05,
        description="Imposes vague high-effort commitments ('best efforts', 'utmost efforts') without defined expenditure caps.",
        evidence_requirement="High-effort phrasing without objective criteria or monetary limits.",
        rationale="May be interpreted under some jurisdictions as requiring extensive expenditure of time and resources.",
        test_cases_summary="Fires on 'shall use best efforts to complete'.",
    ),

    # ── Missing Deadlines ─────────────────────────────────────────────
    "RULE-DEADLINE-001": RuleDefinition(
        rule_id="RULE-DEADLINE-001",
        name="Indefinite Performance Timeline",
        category="ambiguity_deadlines",
        severity="medium",
        default_weight=0.10,
        description="Applies vague timing phrases without definite calendar or business day backstops.",
        evidence_requirement="Use of 'as soon as practicable', 'promptly', or 'in due course' without an explicit day limit.",
        rationale="Without definite calendar day deadlines, enforcing prompt performance or remedy delivery is difficult.",
        test_cases_summary="Fires on 'deliver promptly' when no specific number of days is stated.",
    ),

    # ── Unclear Payment Terms ─────────────────────────────────────────
    "RULE-PAY-001": RuleDefinition(
        rule_id="RULE-PAY-001",
        name="Unilateral Fee Modification",
        category="payment",
        severity="high",
        default_weight=0.15,
        description="Allows one party to increase rates or adjust fees during the term without mutual consent.",
        evidence_requirement="Terms permitting fee or rate modifications at Company discretion without prior client approval.",
        rationale="Creates financial unpredictability and exposes the payer to unbudgeted cost escalations.",
        test_cases_summary="Fires on 'fees are subject to change at Company's discretion'.",
    ),
    "RULE-PAY-002": RuleDefinition(
        rule_id="RULE-PAY-002",
        name="Contingent Payment / Pay-If-Paid",
        category="payment",
        severity="medium",
        default_weight=0.10,
        description="Conditions payment upon receipt of funds from an unrelated third party or subjective satisfaction.",
        evidence_requirement="Clauses stating pay-when-paid, pay-if-paid, or payment contingent upon client satisfaction.",
        rationale="Transfers credit and insolvency risk of third parties onto the service provider.",
        test_cases_summary="Fires on 'payment is contingent upon receipt of customer funds'.",
    ),

    # ── Automatic Renewal ─────────────────────────────────────────────
    "RULE-RENEW-001": RuleDefinition(
        rule_id="RULE-RENEW-001",
        name="Automatic Renewal Lock-In",
        category="renewal",
        severity="medium",
        default_weight=0.05,
        description="Automatically extends contract term for successive periods unless timely affirmative notice is given.",
        evidence_requirement="Automatic evergreen renewal provisions with narrow or unstated opt-out windows.",
        rationale="Risk of accidental multi-year lock-in and unexpected recurring payment obligations.",
        test_cases_summary="Fires on 'shall automatically renew for successive one-year terms'.",
    ),

    # ── Confidentiality Concerns ──────────────────────────────────────
    "RULE-CONF-001": RuleDefinition(
        rule_id="RULE-CONF-001",
        name="Perpetual Confidentiality Obligation",
        category="confidentiality",
        severity="medium",
        default_weight=0.10,
        description="Imposes indefinite confidentiality obligations on general commercial information without a sunset period.",
        evidence_requirement="Confidentiality lasting 'in perpetuity' or 'indefinitely' for all non-trade-secret information.",
        rationale="Creates indefinite compliance burdens and lingering exposure years after commercial engagement ends.",
        test_cases_summary="Fires on 'shall remain confidential in perpetuity'. Does NOT fire when sunset (e.g. 3 years) is present.",
    ),
    "RULE-CONF-002": RuleDefinition(
        rule_id="RULE-CONF-002",
        name="Missing Standard Confidentiality Carve-outs",
        category="confidentiality",
        severity="low",
        default_weight=0.05,
        description="Strict non-disclosure clause that lacks standard carve-outs for public domain or independent development.",
        evidence_requirement="Broad confidentiality definitions without express exclusions for publicly known information.",
        rationale="Could technically restrict use of information that is already public or developed independently.",
        test_cases_summary="Fires when confidentiality contains zero exclusions.",
    ),

    # ── Inconsistent Clauses ──────────────────────────────────────────
    "RULE-INCONSIST-001": RuleDefinition(
        rule_id="RULE-INCONSIST-001",
        name="Conflicting Dispute Resolution Forums",
        category="ambiguity_deadlines",
        severity="medium",
        default_weight=0.10,
        description="Contract contains conflicting dispute resolution mechanisms (e.g. both binding arbitration and exclusive court jurisdiction).",
        evidence_requirement="Simultaneous mandatory arbitration clause and exclusive judicial jurisdiction clause in the same document.",
        rationale="Creates expensive threshold procedural battles over which forum has jurisdiction over claims.",
        test_cases_summary="Fires on full text referencing both binding arbitration and exclusive court jurisdiction without carve-outs.",
    ),

    # ── Missing Expected Protections (Document Level) ─────────────────
    "RULE-PROT-001": RuleDefinition(
        rule_id="RULE-PROT-001",
        name="Missing Governing Law / Jurisdiction",
        category="missing_protections",
        severity="medium",
        default_weight=0.15,
        description="No governing law or choice of jurisdiction clause was detected in the agreement.",
        evidence_requirement="Absence of standard governing law or jurisdiction phrasing across the entire document.",
        rationale="Ambiguity over applicable law increases litigation costs and unpredictable conflict-of-laws rulings.",
        test_cases_summary="Fires when no governing law clause is found in full contract text.",
    ),
    "RULE-PROT-002": RuleDefinition(
        rule_id="RULE-PROT-002",
        name="Missing Limitation of Liability Cap",
        category="missing_protections",
        severity="high",
        default_weight=0.25,
        description="No express monetary cap or limitation of liability clause was found in the agreement.",
        evidence_requirement="Absence of limitation of liability terms in commercial agreements.",
        rationale="Without an express cap, liability is governed by common law which may include full foreseeable damages.",
        test_cases_summary="Fires when no limitation of liability clause exists in full document.",
    ),
    "RULE-PROT-003": RuleDefinition(
        rule_id="RULE-PROT-003",
        name="Missing Termination for Default / Breach Mechanism",
        category="missing_protections",
        severity="medium",
        default_weight=0.15,
        description="No clear termination for breach or cure mechanism was detected in the contract.",
        evidence_requirement="Absence of breach termination provisions.",
        rationale="Leaves parties without an agreed contractual exit roadmap when non-performance occurs.",
        test_cases_summary="Fires when no breach termination provision is identified.",
    ),
    "RULE-PROT-004": RuleDefinition(
        rule_id="RULE-PROT-004",
        name="Missing Confidentiality Protections",
        category="missing_protections",
        severity="low",
        default_weight=0.10,
        description="No standard non-disclosure or confidentiality obligations were detected in the agreement.",
        evidence_requirement="Absence of confidentiality terms in commercial agreement.",
        rationale="Leaves exchanged proprietary information and trade secrets without express contractual non-disclosure protection.",
        test_cases_summary="Fires when confidentiality is absent from full contract.",
    ),

    # ── Preserved Legacy Rules (RISK-005 & RISK-007) ──────────────────
    "RISK-005": RuleDefinition(
        rule_id="RISK-005",
        name="Excessive Non-Compete Duration",
        category="ambiguity_deadlines",
        severity="high",
        default_weight=0.10,
        description="Non-compete restrictive covenant duration exceeds customary 1–2 year commercial norms.",
        evidence_requirement="Non-compete duration specified as greater than 24 months / 2 years.",
        rationale="Restrictive covenants longer than 24 months are frequently held unreasonable and unenforceable by courts.",
        test_cases_summary="Fires on 'non-compete for 5 years'.",
    ),
    "RISK-007": RuleDefinition(
        rule_id="RISK-007",
        name="Blanket IP Assignment",
        category="liability",
        severity="high",
        default_weight=0.15,
        description="Assigns all intellectual property rights without excluding pre-existing or background IP.",
        evidence_requirement="Language assigning 'all intellectual property' or 'any and all IP' without carve-outs.",
        rationale="May inadvertently assign background tools, pre-existing IP, or non-project proprietary assets.",
        test_cases_summary="Fires on 'assigns all intellectual property rights'.",
    ),
}

# Backward compatibility mapping for legacy rule IDs
_LEGACY_RULE_MAP: Dict[str, str] = {
    "RISK-001": "RULE-LIAB-001",
    "RISK-002": "RULE-TERM-001",
    "RISK-003": "RULE-RENEW-001",
    "RISK-004": "RULE-INDEM-001",
    "RISK-005": "RISK-005",
    "RISK-006": "RULE-PROT-001",
    "RISK-007": "RISK-007",
}


# ══════════════════════════════════════════════════════════════════════
#  INDIVIDUAL RULE LOGIC WITH CONTEXT & EXCEPTION CHECKING
# ══════════════════════════════════════════════════════════════════════

def _check_uncapped_liability(text: str) -> List[RiskFlag]:
    """
    Check for uncapped liability.
    EXCEPTION: If the text expressly provides a numeric, aggregate, or fee-based cap
    (e.g., 'liability shall not exceed the fees paid'), do NOT flag as uncapped liability!
    """
    # 1. First check if text is actually setting a cap:
    # "shall not exceed the total amount paid", "limited to $1,000,000", "capped at the fees"
    has_protective_cap = bool(
        re.search(
            r"(?:shall\s+(?:not\s+exceed|be\s+limited\s+to)|aggregate\s+liability\s+(?:is|shall\s+be)\s+limited|"
            r"liability\s+cap\s+of|capped\s+at)\s+.*?\b(?:fees?|amount|payments?|\$|\d+)",
            text,
            re.IGNORECASE,
        )
    )

    patterns = [
        (r"\bunlimited\s+liabilit\w*", "Explicit uncapped liability language detected"),
        (
            r"\bno\s+(?:cap|limit(?:ation)?)(?:\s+or\s+(?:cap|limit(?:ation)?))*\s+on\s+(?:liabilit|damage)",
            "Disclaimed liability cap",
        ),
        (
            r"(?:without|no)\s+(?:any\s+)?(?:cap|limit(?:ation)?)\b.*?\b(?:liabilit|damage)",
            "Absence of damage limitation",
        ),
        (
            r"shall\s+not\s+be\s+limited\s+(?:in\s+any\s+(?:way|manner)|by\s+any\s+cap)",
            "Disclaimer of liability limitation",
        ),
    ]

    flags: List[RiskFlag] = []
    for pat, label_desc in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            # If the snippet is part of an exception to a cap (e.g., "except liability for fraud shall not be limited"),
            # that is normal carve-out, but if it states the whole party's liability has no cap, flag it.
            matched = m.group(0)
            if has_protective_cap and "except" in text.lower():
                # Cap exists with standard carve-outs; only flag if broad uncapped liability applies
                continue

            suspiciousness = 85.0
            score_breakdown = "Base severity HIGH (85)"
            if re.search(r"\bany\s+and\s+all\s+(?:damages|losses|claims)\b", text, re.IGNORECASE):
                suspiciousness += 10.0
                score_breakdown += " + applies to all damages without carve-outs (+10) = 95/100"
            else:
                score_breakdown += " = 85/100"

            flags.append(
                RiskFlag(
                    rule_id="RULE-LIAB-001",  # Also matched by legacy RISK-001
                    label="Potentially Uncapped Liability",
                    severity="high",
                    category="liability",
                    suspiciousness_score=min(suspiciousness, 100.0),
                    weight=0.20,
                    description="Clause appears to remove, disclaim, or waive limitation of liability caps.",
                    matched_snippet=matched,
                    rationale="Uncapped liability exposes a party to enterprise-threatening recovery for unforeseen losses.",
                    score_breakdown=score_breakdown,
                    evidence_text=text[:250],
                )
            )
            break
    return flags


def _check_broad_liability_carveouts(text: str) -> List[RiskFlag]:
    """Check for liability cap carve-outs that swallow ordinary breach of contract."""
    # Look for "The limitations of liability shall not apply to: ... any breach of this agreement"
    pat = r"(?:limitations?\s+of\s+liability\s+(?:shall\s+not\s+apply|do\s+not\s+apply)|carve[\s-]?outs?\s+include).*?\b(any\s+breach\s+of\s+this\s+agreement|any\s+default|any\s+performance\s+failure)\b"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        return [
            RiskFlag(
                rule_id="RULE-LIAB-002",
                label="Broad Liability Cap Carve-outs",
                severity="medium",
                category="liability",
                suspiciousness_score=65.0,
                weight=0.15,
                description="Liability cap exceptions include ordinary breach of contract, which may nullify the cap.",
                matched_snippet=m.group(0)[:120],
                rationale="Carve-outs that include general contract breach undermine the predictability of agreed liability caps.",
                score_breakdown="Base severity MEDIUM (65) = 65/100",
                evidence_text=text[:250],
            )
        ]
    return []


def _check_unilateral_termination(text: str) -> List[RiskFlag]:
    """
    Check for termination for convenience rights.
    EXCEPTION: If mutual ('either party may terminate on 30 days notice'), do NOT flag as high-risk unilateral!
    """
    convenience_pat = r"(?:may|shall|can)\s+terminate\s+(?:this\s+)?(?:agreement|contract)\s+(?:at\s+any\s+time|without\s+cause|for\s+(?:any|no)\s+reason|for\s+convenience)"
    m = re.search(convenience_pat, text, re.IGNORECASE)
    if not m:
        return []

    # Check context: is it mutual?
    is_mutual = bool(
        re.search(
            r"\b(?:either\s+party|both\s+parties|each\s+party|neither\s+party)\b",
            text,
            re.IGNORECASE,
        )
    )

    if is_mutual:
        # Mutual termination for convenience with reasonable notice is common; evaluate notice
        has_notice = bool(re.search(r"\b(?:\d+|thirty|sixty|ninety)\s+days?\b", text, re.IGNORECASE))
        if has_notice:
            # Benign mutual convenience termination with notice
            return []
        else:
            # Mutual but lacking specified notice
            return [
                RiskFlag(
                    rule_id="RULE-TERM-001",
                    label="Mutual Termination for Convenience Without Notice",
                    severity="low",
                    category="termination",
                    suspiciousness_score=35.0,
                    weight=0.10,
                    description="Mutual termination for convenience is permitted but notice period is unspecified.",
                    matched_snippet=m.group(0),
                    rationale="Immediate mutual termination creates business disruption; a standard 30-day notice is advisable.",
                    score_breakdown="Base severity LOW (35) [mutual with unstated notice] = 35/100",
                    evidence_text=text[:250],
                )
            ]

    # Truly unilateral termination
    matched = m.group(0)
    has_notice = bool(re.search(r"\b(?:\d+|thirty|sixty)\s+days?\b", text, re.IGNORECASE))
    suspiciousness = 75.0
    score_breakdown = "Base severity HIGH (75) [unilateral termination right]"
    if not has_notice:
        suspiciousness += 15.0
        score_breakdown += " + no advance notice required (+15) = 90/100"
    else:
        score_breakdown += " = 75/100"

    return [
        RiskFlag(
            rule_id="RULE-TERM-001",  # Maps to legacy RISK-002
            label="Potentially One-Sided Termination for Convenience",
            severity="high",
            category="termination",
            suspiciousness_score=min(suspiciousness, 100.0),
            weight=0.15,
            description="One party appears to have the unilateral right to terminate the contract at will.",
            matched_snippet=matched,
            rationale="One-sided termination allows the counterparty to cancel the engagement after investments have been made.",
            score_breakdown=score_breakdown,
            evidence_text=text[:250],
        )
    ]


def _check_immediate_termination_no_cure(text: str) -> List[RiskFlag]:
    """Check for immediate termination for breach without standard 15-30 day cure period."""
    pat = r"(?:terminate\s+immediately|effective\s+immediately)\s+(?:upon\s+(?:written\s+)?notice\s+of\s+breach|without\s+(?:any\s+)?(?:opportunity\s+to\s+cure|cure\s+period))"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        # Check if cure period exception is explicitly stated
        if not re.search(r"\b(?:cure\s+period|remedy\s+such\s+breach\s+within)\b", text, re.IGNORECASE):
            return [
                RiskFlag(
                    rule_id="RULE-TERM-002",
                    label="Immediate Breach Termination Without Cure Period",
                    severity="high",
                    category="termination",
                    suspiciousness_score=80.0,
                    weight=0.15,
                    description="Allows immediate termination for breach without a mandatory 15-30 day cure period.",
                    matched_snippet=m.group(0),
                    rationale="Denies the non-breaching party an opportunity to cure administrative or inadvertent defaults.",
                    score_breakdown="Base severity HIGH (80) = 80/100",
                    evidence_text=text[:250],
                )
            ]
    return []


def _check_broad_indemnification(text: str) -> List[RiskFlag]:
    """Check for broad, unhedged indemnification obligations."""
    patterns = [
        r"indemnif[y|ies|ication].*?\b(?:any\s+and\s+all|all\s+claims|whatsoever|without\s+limitation)\b",
        r"hold\s+harmless.*?\b(?:any\s+and\s+all|all\s+claims|whatsoever)\b",
    ]
    flags: List[RiskFlag] = []
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            matched = m.group(0)[:120]
            suspiciousness = 75.0
            score_breakdown = "Base severity HIGH (75)"
            if re.search(r"\bdefend\b", text, re.IGNORECASE):
                suspiciousness += 10.0
                score_breakdown += " + defense obligation included (+10) = 85/100"
            else:
                score_breakdown += " = 75/100"

            flags.append(
                RiskFlag(
                    rule_id="RULE-INDEM-001",  # Maps to legacy RISK-004
                    label="Broad / Unhedged Indemnification",
                    severity="high",
                    category="indemnity",
                    suspiciousness_score=min(suspiciousness, 100.0),
                    weight=0.15,
                    description="Indemnification language is expansive and may require defense against all third-party claims.",
                    matched_snippet=matched,
                    rationale="May convert the indemnifying party into an unhedged insurer for counterparty liabilities.",
                    score_breakdown=score_breakdown,
                    evidence_text=text[:250],
                )
            )
            break
    return flags


def _check_ambiguous_discretion(text: str) -> List[RiskFlag]:
    """Check for sole and absolute unilateral discretion."""
    pat = r"\b(?:sole\s+and\s+absolute\s+discretion|unilateral\s+discretion|at\s+its\s+sole\s+discretion)\b"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        # Check if qualified by reasonableness standard
        has_reasonableness = bool(
            re.search(r"\b(?:acting\s+reasonably|commercially\s+reasonable|good\s+faith)\b", text, re.IGNORECASE)
        )
        if has_reasonableness:
            suspiciousness = 40.0
            breakdown = "Base MEDIUM (60) - tempered by reasonableness/good faith (-20) = 40/100"
            severity = "low"
        else:
            suspiciousness = 70.0
            breakdown = "Base severity MEDIUM (70) [unbounded subjective discretion] = 70/100"
            severity = "medium"

        return [
            RiskFlag(
                rule_id="RULE-AMBIG-001",
                label="Unilateral Subjective Discretion",
                severity=severity,
                category="ambiguity_deadlines",
                suspiciousness_score=suspiciousness,
                weight=0.10,
                description="Grants one party subjective, unilateral discretion without objective benchmarks.",
                matched_snippet=m.group(0),
                rationale="Allows unilateral determination of deliverables, standards, or performance criteria without objective benchmarks.",
                score_breakdown=breakdown,
                evidence_text=text[:250],
            )
        ]
    return []


def _check_high_effort_standards(text: str) -> List[RiskFlag]:
    """Check for open-ended 'best efforts' without defined limits."""
    pat = r"\b(?:shall|will|agrees\s+to)\s+(?:use\s+)?(?:best\s+efforts|utmost\s+efforts)\b"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        return [
            RiskFlag(
                rule_id="RULE-AMBIG-002",
                label="Undefined High-Effort Standards",
                severity="low",
                category="ambiguity_deadlines",
                suspiciousness_score=40.0,
                weight=0.05,
                description="Uses 'best efforts' without defining financial, resource, or milestone limits.",
                matched_snippet=m.group(0),
                rationale="Under certain jurisdictions, 'best efforts' may require extreme expenditure of diligence and resources.",
                score_breakdown="Base severity LOW (40) = 40/100",
                evidence_text=text[:250],
            )
        ]
    return []


def _check_missing_deadlines(text: str) -> List[RiskFlag]:
    """Check for vague temporal obligations lacking calendar day limits."""
    pat = r"\b(?:as\s+soon\s+as\s+practicable|in\s+a\s+timely\s+manner|promptly\s+thereafter|in\s+due\s+course)\b"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        # Check if accompanied by explicit day count
        has_days = bool(re.search(r"\b(?:within\s+\d+|in\s+no\s+event\s+later\s+than)\s+(?:business\s+)?days?\b", text, re.IGNORECASE))
        if not has_days:
            return [
                RiskFlag(
                    rule_id="RULE-DEADLINE-001",
                    label="Indefinite Performance Timeline",
                    severity="medium",
                    category="ambiguity_deadlines",
                    suspiciousness_score=50.0,
                    weight=0.10,
                    description="Key timing obligation governed by vague phrasing without calendar day limits.",
                    matched_snippet=m.group(0),
                    rationale="Without calendar backstops, non-performing parties can delay delivery without technical breach.",
                    score_breakdown="Base severity MEDIUM (50) = 50/100",
                    evidence_text=text[:250],
                )
            ]
    return []


def _check_unilateral_fee_modification(text: str) -> List[RiskFlag]:
    """Check for unilateral fee increases without mutual written consent."""
    pat = r"(?:fees?|rates?|prices?|pricing)\s+(?:may\s+be\s+increased|subject\s+to\s+change|may\s+be\s+modified)\s+(?:at\s+(?:any\s+time|Company's\s+discretion)|without\s+(?:prior\s+)?(?:written\s+)?consent)"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        return [
            RiskFlag(
                rule_id="RULE-PAY-001",
                label="Unilateral Fee Modification",
                severity="high",
                category="payment",
                suspiciousness_score=80.0,
                weight=0.15,
                description="Allows one party to increase fees or change pricing without mutual written agreement.",
                matched_snippet=m.group(0)[:120],
                rationale="Unilateral pricing modifications expose the paying party to unbudgeted cost escalations during the term.",
                score_breakdown="Base severity HIGH (80) = 80/100",
                evidence_text=text[:250],
            )
        ]
    return []


def _check_contingent_payment(text: str) -> List[RiskFlag]:
    """Check for pay-if-paid or contingent payment clauses."""
    pat = (
        r"\b(?:"
        r"pay[\s-]when[\s-]paid|"
        r"pay[\s-]if[\s-]paid|"
        r"payment\s+(?:is\s+|shall\s+be\s+)?contingent\s+(?:upon|on)\s+(?:receipt|customer)|"
        r"contingent\s+(?:upon|on)\s+(?:receipt\s+of\s+customer|client\s+satisfaction)|"
        r"conditioned\s+(?:upon|on)\s+receipt\s+of\s+(?:payment|funds)"
        r")\b"
    )
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        return [
            RiskFlag(
                rule_id="RULE-PAY-002",
                label="Contingent Payment / Pay-If-Paid",
                severity="medium",
                category="payment",
                suspiciousness_score=65.0,
                weight=0.10,
                description="Conditions payment upon receipt of funds from third parties or subjective customer satisfaction.",
                matched_snippet=m.group(0),
                rationale="Shifts credit risk and non-payment risk of unrelated parties onto the vendor or subcontractor.",
                score_breakdown="Base severity MEDIUM (65) = 65/100",
                evidence_text=text[:250],
            )
        ]
    return []


def _check_auto_renewal(text: str) -> List[RiskFlag]:
    """Check for automatic renewal with evaluation of opt-out notice windows."""
    pat = r"\b(?:auto(?:matic(?:ally)?)?[\s-]+renew|shall\s+(?:be\s+)?automatically\s+renewed)\b"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        # Check notice window
        notice_m = re.search(r"(\d+)\s*days?", text, re.IGNORECASE)
        days = int(notice_m.group(1)) if notice_m else None

        suspiciousness = 50.0
        score_breakdown = "Base severity MEDIUM (50)"
        if days is None:
            suspiciousness += 15.0
            score_breakdown += " + no opt-out notice window stated (+15) = 65/100"
        elif days < 15 or days > 90:
            suspiciousness += 20.0
            score_breakdown += f" + non-standard opt-out window ({days} days) (+20) = 70/100"
        else:
            score_breakdown += f" [standard {days}-day notice window] = 50/100"

        return [
            RiskFlag(
                rule_id="RULE-RENEW-001",  # Maps to legacy RISK-003
                label="Automatic Renewal Lock-In",
                severity="medium",
                category="renewal",
                suspiciousness_score=suspiciousness,
                weight=0.05,
                description="Contract automatically extends for successive terms unless affirmative cancellation is given.",
                matched_snippet=m.group(0),
                rationale="May result in accidental long-term contract extension and ongoing billing obligations.",
                score_breakdown=score_breakdown,
                evidence_text=text[:250],
            )
        ]
    return []


def _check_perpetual_confidentiality(text: str) -> List[RiskFlag]:
    """
    Check for perpetual confidentiality obligations on general information.
    EXCEPTION: Trade secrets are legitimately perpetual; if limited to trade secrets, do NOT flag.
    """
    pat = r"\b(?:in\s+perpetuity|shall\s+survive\s+indefinitely|confidential\s+forever|shall\s+never\s+expire)\b"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        # Check if expressly limited only to trade secrets
        is_only_trade_secrets = bool(
            re.search(
                r"\b(?:trade\s+secrets?\s+(?:shall|only)|except\s+for\s+trade\s+secrets)\b",
                text,
                re.IGNORECASE,
            )
        )
        if is_only_trade_secrets:
            # Legitimate trade secret exception
            return []

        return [
            RiskFlag(
                rule_id="RULE-CONF-001",
                label="Perpetual Confidentiality Obligation",
                severity="medium",
                category="confidentiality",
                suspiciousness_score=60.0,
                weight=0.10,
                description="Imposes indefinite confidentiality obligations on general commercial information without a sunset.",
                matched_snippet=m.group(0),
                rationale="Indefinite confidentiality creates compliance tracking overhead years after the commercial relationship concludes.",
                score_breakdown="Base severity MEDIUM (60) = 60/100",
                evidence_text=text[:250],
            )
        ]
    return []


def _check_non_compete_duration(text: str) -> List[RiskFlag]:
    """Check for excessive non-compete duration (>24 months)."""
    m = re.search(
        r"(?:non[\s-]?compete|restrictive\s+covenant).*?(\d+)\s*(?:year|month)",
        text,
        re.IGNORECASE,
    )
    if m:
        value = int(m.group(1))
        unit_is_year = "year" in m.group(0).lower()
        months = value * 12 if unit_is_year else value
        if months > 24:
            return [
                RiskFlag(
                    rule_id="RISK-005",
                    label="Excessive Non-Compete Duration",
                    severity="high",
                    category="ambiguity_deadlines",
                    suspiciousness_score=85.0,
                    weight=0.10,
                    description=f"Non-compete duration of {value} {'years' if unit_is_year else 'months'} exceeds standard commercial limits.",
                    matched_snippet=m.group(0)[:120],
                    rationale="Restrictive covenants longer than 24 months are heavily scrutinized and often held unenforceable by courts.",
                    score_breakdown=f"Base severity HIGH (85) [{months} months > 24 months limit] = 85/100",
                    evidence_text=text[:250],
                )
            ]
    return []


def _check_ip_assignment(text: str) -> List[RiskFlag]:
    """Check for blanket pre-existing IP assignment."""
    patterns = [
        r"assign[s]?\s+(?:all|any\s+and\s+all)\s+(?:intellectual\s+property|ip|rights)",
        r"work[\s-]+(?:made[\s-]+)?for[\s-]+hire",
    ]
    flags: List[RiskFlag] = []
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            flags.append(
                RiskFlag(
                    rule_id="RISK-007",
                    label="Blanket IP Assignment",
                    severity="high",
                    category="liability",
                    suspiciousness_score=75.0,
                    weight=0.15,
                    description="Clause assigns all intellectual property rights, potentially capturing pre-existing tools or background IP.",
                    matched_snippet=m.group(0),
                    rationale="Review scope to ensure pre-existing background IP and reusable tools are expressly carved out.",
                    score_breakdown="Base severity HIGH (75) = 75/100",
                    evidence_text=text[:250],
                )
            )
            break
    return flags


# ══════════════════════════════════════════════════════════════════════
#  DOCUMENT-LEVEL CHECKS (MISSING EXPECTED CLAUSES & INCONSISTENCIES)
# ══════════════════════════════════════════════════════════════════════

def _check_missing_governing_law(full_text: str) -> List[RiskFlag]:
    """Flag when no governing law or jurisdiction clause is present in the full contract."""
    if not re.search(
        r"governing\s+law|governed\s+by(?:\s+the)?\s+laws?|jurisdiction|applicable\s+law",
        full_text,
        re.IGNORECASE,
    ):
        return [
            RiskFlag(
                rule_id="RULE-PROT-001",  # Maps to legacy RISK-006
                label="Missing Governing Law",
                severity="medium",
                category="missing_protections",
                suspiciousness_score=70.0,
                weight=0.15,
                description="No governing law or choice of jurisdiction clause was detected in the agreement.",
                matched_snippet="(not found in contract text)",
                rationale="Absence of governing law leaves legal interpretation to unpredictable conflict-of-laws principles.",
                score_breakdown="Base severity MEDIUM (70) = 70/100",
                evidence_text="Entire document scanned: no governing law or jurisdiction clause found.",
            )
        ]
    return []


def _check_missing_liability_cap(full_text: str) -> List[RiskFlag]:
    """Flag when a commercial contract lacks an express limitation of liability clause."""
    has_cap = bool(
        re.search(
            r"limitation\s+of\s+liability|aggregate\s+liability|liability\s+(?:cap|shall\s+not\s+exceed)|maximum\s+liability",
            full_text,
            re.IGNORECASE,
        )
    )
    if not has_cap:
        return [
            RiskFlag(
                rule_id="RULE-PROT-002",
                label="Missing Limitation of Liability Cap",
                severity="high",
                category="missing_protections",
                suspiciousness_score=85.0,
                weight=0.25,
                description="No limitation of liability or monetary cap clause was detected in the document.",
                matched_snippet="(not found in contract text)",
                rationale="In the absence of an express cap, liability is governed by common law which permits full foreseeable damages.",
                score_breakdown="Base severity HIGH (85) = 85/100",
                evidence_text="Entire document scanned: no limitation of liability or cap clause found.",
            )
        ]
    return []


def _check_missing_breach_termination(full_text: str) -> List[RiskFlag]:
    """Flag when no termination for breach / default provision is found."""
    has_breach_term = bool(
        re.search(
            r"terminate.*?\b(?:material\s+breach|default|failure\s+to\s+cure|remedy\s+such\s+breach)\b",
            full_text,
            re.IGNORECASE,
        )
    )
    if not has_breach_term:
        return [
            RiskFlag(
                rule_id="RULE-PROT-003",
                label="Missing Termination for Breach Mechanism",
                severity="medium",
                category="missing_protections",
                suspiciousness_score=65.0,
                weight=0.15,
                description="No clear termination for breach or cure period mechanism was identified.",
                matched_snippet="(not found in contract text)",
                rationale="Leaves parties without an agreed contractual roadmap for orderly termination upon non-performance.",
                score_breakdown="Base severity MEDIUM (65) = 65/100",
                evidence_text="Entire document scanned: no breach termination clause found.",
            )
        ]
    return []


def _check_conflicting_dispute_forums(full_text: str) -> List[RiskFlag]:
    """Check for contradictory dispute resolution forums in the same document."""
    has_arbitration = bool(
        re.search(r"\b(?:binding\s+arbitration|arbitrated\s+under|rules\s+of\s+arbitration)\b", full_text, re.IGNORECASE)
    )
    has_exclusive_court = bool(
        re.search(
            r"\b(?:exclusive\s+jurisdiction\s+of\s+the\s+courts|sole\s+jurisdiction\s+of\s+(?:the\s+)?courts)\b",
            full_text,
            re.IGNORECASE,
        )
    )
    if has_arbitration and has_exclusive_court:
        # Check if one is expressly carved out for injunctive relief
        has_carveout = bool(re.search(r"\b(?:injunctive\s+relief|equitable\s+relief)\b", full_text, re.IGNORECASE))
        if not has_carveout:
            return [
                RiskFlag(
                    rule_id="RULE-INCONSIST-001",
                    label="Conflicting Dispute Resolution Forums",
                    severity="medium",
                    category="ambiguity_deadlines",
                    suspiciousness_score=65.0,
                    weight=0.10,
                    description="Agreement references both mandatory binding arbitration and exclusive judicial court jurisdiction.",
                    matched_snippet="Mandatory arbitration and exclusive court jurisdiction both present",
                    rationale="Conflicting forum clauses create expensive jurisdictional disputes before claims can be heard on the merits.",
                    score_breakdown="Base severity MEDIUM (65) = 65/100",
                    evidence_text="Full text analysis: both binding arbitration and exclusive court jurisdiction declared without clear demarcation.",
                )
            ]
    return []


# ══════════════════════════════════════════════════════════════════════
#  MASTER REGISTRY & EVALUATION DISPATCHERS
# ══════════════════════════════════════════════════════════════════════

_ALL_CLAUSE_RULES = [
    _check_uncapped_liability,
    _check_broad_liability_carveouts,
    _check_unilateral_termination,
    _check_immediate_termination_no_cure,
    _check_broad_indemnification,
    _check_ambiguous_discretion,
    _check_high_effort_standards,
    _check_missing_deadlines,
    _check_unilateral_fee_modification,
    _check_contingent_payment,
    _check_auto_renewal,
    _check_perpetual_confidentiality,
    _check_non_compete_duration,
    _check_ip_assignment,
]

_ALL_DOCUMENT_RULES = [
    _check_missing_governing_law,
    _check_missing_liability_cap,
    _check_missing_breach_termination,
    _check_conflicting_dispute_forums,
]


def deduplicate_flags(flags: List[RiskFlag]) -> List[RiskFlag]:
    """
    Deduplicate flags that trigger on overlapping snippets within the same clause.
    Retains the flag with the higher suspiciousness score.
    """
    seen: Dict[Tuple[str, str], RiskFlag] = {}
    for flag in flags:
        key = (flag.rule_id, flag.matched_snippet.strip().lower()[:50])
        if key not in seen or flag.suspiciousness_score > seen[key].suspiciousness_score:
            seen[key] = flag
    return list(seen.values())


def evaluate_clause(
    text: str,
    clause_id: Optional[str] = None,
    clause_index: Optional[int] = None,
) -> List[RiskFlag]:
    """
    Run all clause-level rules against *text* with context and exception checking.
    Injects clause metadata (clause_id, clause_index) when provided.
    """
    if not text or not text.strip():
        return []

    flags: List[RiskFlag] = []
    for rule_fn in _ALL_CLAUSE_RULES:
        rule_flags = rule_fn(text)
        for rf in rule_flags:
            rf.clause_id = clause_id
            rf.clause_index = clause_index
            flags.append(rf)

    # Also support legacy alias IDs in results for backward compatibility
    # e.g., if RULE-LIAB-001 fired, tests expecting RISK-001 should find it
    enhanced_flags: List[RiskFlag] = []
    for f in deduplicate_flags(flags):
        enhanced_flags.append(f)
        # Create legacy alias flag with same fields if mapping exists
        for legacy_id, new_id in _LEGACY_RULE_MAP.items():
            if f.rule_id == new_id and legacy_id != new_id:
                alias_flag = RiskFlag(
                    rule_id=legacy_id,
                    label=f.label,
                    severity=f.severity,
                    description=f.description,
                    matched_snippet=f.matched_snippet,
                    category=f.category,
                    suspiciousness_score=f.suspiciousness_score,
                    weight=f.weight,
                    rationale=f.rationale,
                    score_breakdown=f.score_breakdown,
                    evidence_text=f.evidence_text,
                    clause_id=clause_id,
                    clause_index=clause_index,
                )
                enhanced_flags.append(alias_flag)

    return enhanced_flags


def evaluate_document(full_text: str) -> List[RiskFlag]:
    """
    Run document-level checks across the entire contract text.
    Evaluates missing expected clauses (e.g. governing law, liability caps)
    and cross-clause contradictions.
    """
    if not full_text or not full_text.strip():
        return []

    flags: List[RiskFlag] = []
    for rule_fn in _ALL_DOCUMENT_RULES:
        flags.extend(rule_fn(full_text))

    # Add legacy alias for document checks (RISK-006 -> RULE-PROT-001)
    enhanced_flags: List[RiskFlag] = []
    for f in deduplicate_flags(flags):
        enhanced_flags.append(f)
        if f.rule_id == "RULE-PROT-001":
            alias = RiskFlag(
                rule_id="RISK-006",
                label=f.label,
                severity=f.severity,
                description=f.description,
                matched_snippet=f.matched_snippet,
                category=f.category,
                suspiciousness_score=f.suspiciousness_score,
                weight=f.weight,
                rationale=f.rationale,
                score_breakdown=f.score_breakdown,
                evidence_text=f.evidence_text,
            )
            enhanced_flags.append(alias)

    return enhanced_flags


def get_rule_definitions() -> Dict[str, RuleDefinition]:
    """Return dictionary of all registered rule definitions and metadata."""
    return RULE_DEFINITIONS
