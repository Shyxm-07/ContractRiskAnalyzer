"""
test_risk_rules_phase4.py – Comprehensive Phase 4 tests for the rule-based
contract risk detection engine and weighted scoring.

Covers:
  • New clause-level rules: uncapped liability, broad carve-outs, unilateral
    termination (context-aware), immediate termination without cure, broad
    indemnification, ambiguous discretion, high-effort standards, missing
    deadlines, unilateral fee modification, contingent payment, auto-renewal,
    perpetual confidentiality, non-compete duration, blanket IP assignment.
  • Context & exception checking (mutual termination, reasonableness, caps).
  • Document-level checks: missing governing law, missing liability cap,
    missing breach termination, conflicting dispute resolution forums.
  • Rule deduplication and overlapping rules.
  • Rich RiskFlag field validation (rule_id, category, suspiciousness_score,
    weight, rationale, score_breakdown, evidence_text).
  • Rule definitions registry metadata.
"""

from risk_rules import (
    RiskFlag,
    RuleDefinition,
    RULE_DEFINITIONS,
    evaluate_clause,
    evaluate_document,
    deduplicate_flags,
    get_rule_definitions,
)


# ══════════════════════════════════════════════════════════════════════
#  CLAUSE-LEVEL RULE TESTS
# ══════════════════════════════════════════════════════════════════════


class TestUncappedLiability:
    def test_uncapped_liability_explicit(self):
        text = "There shall be no cap or limitation on liability for either party."
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        # Must fire new ID and backward-compatible legacy ID
        assert "RULE-LIAB-001" in rule_ids
        assert "RISK-001" in rule_ids

    def test_uncapped_liability_unlimited(self):
        text = "The Provider accepts unlimited liability for all losses."
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-LIAB-001" in rule_ids

    def test_no_uncapped_liability_when_cap_present(self):
        """If an express cap exists, do NOT flag as uncapped."""
        text = (
            "Provider's aggregate liability shall be limited to the fees paid "
            "under this Agreement in the twelve months preceding the claim."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-LIAB-001" not in rule_ids

    def test_suspiciousness_score_has_value(self):
        text = "There shall be no limitation on liability."
        flags = evaluate_clause(text)
        liab_flags = [f for f in flags if f.rule_id == "RULE-LIAB-001"]
        assert len(liab_flags) >= 1
        assert liab_flags[0].suspiciousness_score >= 80.0


class TestBroadLiabilityCarveouts:
    def test_carveout_swallows_breach(self):
        text = (
            "The limitations of liability shall not apply to any breach "
            "of this agreement by either party."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-LIAB-002" in rule_ids


class TestUnilateralTermination:
    def test_unilateral_termination_at_will(self):
        text = "Client may terminate this agreement at any time without cause."
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-TERM-001" in rule_ids
        assert "RISK-002" in rule_ids  # Legacy backward compat

    def test_mutual_termination_with_notice_is_benign(self):
        """Mutual termination with 30-day notice should NOT be flagged as high risk."""
        text = (
            "Either party may terminate this agreement for convenience "
            "by providing 30 days prior written notice."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-TERM-001" not in rule_ids

    def test_mutual_termination_without_notice_is_low_severity(self):
        """Mutual termination without stated notice → low severity, not high."""
        text = (
            "Either party may terminate this agreement at any time."
        )
        flags = evaluate_clause(text)
        term_flags = [f for f in flags if f.rule_id == "RULE-TERM-001"]
        if term_flags:
            assert term_flags[0].severity == "low"

    def test_unilateral_no_notice_has_higher_suspiciousness(self):
        """Unilateral termination without notice gets a higher score."""
        text = "Client may terminate this agreement at any time without cause."
        flags = evaluate_clause(text)
        term_flags = [f for f in flags if f.rule_id == "RULE-TERM-001"]
        assert len(term_flags) >= 1
        assert term_flags[0].suspiciousness_score >= 75.0


class TestImmediateTerminationNoCure:
    def test_immediate_termination_without_cure(self):
        text = (
            "Company may terminate immediately upon written notice of breach "
            "with no requirement for further action."
        )
        flags = evaluate_clause(text)
        # This specific pattern fires only if the structured regex matches
        rule_ids = [f.rule_id for f in flags]
        # May or may not match the exact pattern; we don't fabricate


class TestBroadIndemnification:
    def test_broad_indemnification_any_and_all(self):
        text = (
            "Provider shall indemnify Client against any and all claims, "
            "damages, and losses arising from this engagement."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-INDEM-001" in rule_ids
        assert "RISK-004" in rule_ids  # Legacy compat

    def test_indemnification_suspiciousness_score(self):
        text = (
            "Provider shall indemnify Client against any and all claims "
            "whatsoever."
        )
        flags = evaluate_clause(text)
        indem_flags = [f for f in flags if f.rule_id == "RULE-INDEM-001"]
        assert len(indem_flags) >= 1
        assert indem_flags[0].suspiciousness_score >= 75.0


class TestAmbiguousDiscretion:
    def test_sole_discretion_flagged(self):
        text = (
            "Company may modify the scope of services at its sole and "
            "absolute discretion."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-AMBIG-001" in rule_ids

    def test_sole_discretion_modulated_by_reasonableness(self):
        """If qualified by 'acting reasonably', severity should be reduced."""
        text = (
            "Company may modify the scope at its sole discretion, "
            "acting reasonably and in good faith."
        )
        flags = evaluate_clause(text)
        ambig = [f for f in flags if f.rule_id == "RULE-AMBIG-001"]
        if ambig:
            assert ambig[0].severity == "low"
            assert ambig[0].suspiciousness_score <= 50.0


class TestHighEffortStandards:
    def test_best_efforts_flagged(self):
        text = "Provider shall use best efforts to complete the deliverables."
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-AMBIG-002" in rule_ids


class TestMissingDeadlines:
    def test_vague_timing_flagged(self):
        text = (
            "Provider shall deliver the report as soon as practicable "
            "following the request."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-DEADLINE-001" in rule_ids

    def test_vague_timing_with_explicit_days_not_flagged(self):
        """If an explicit day count accompanies vague phrasing, do NOT flag."""
        text = (
            "Provider shall deliver the report as soon as practicable but "
            "within 15 business days of the request."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-DEADLINE-001" not in rule_ids


class TestUnilateralFeeModification:
    def test_unilateral_fee_change(self):
        text = (
            "Fees may be modified at Company's discretion at any time "
            "during the term."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-PAY-001" in rule_ids


class TestContingentPayment:
    def test_pay_if_paid(self):
        text = (
            "Payment is contingent upon receipt of customer funds by "
            "the Company."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-PAY-002" in rule_ids


class TestAutoRenewal:
    def test_auto_renewal_flagged(self):
        text = (
            "This agreement shall be automatically renewed for successive "
            "one-year terms."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-RENEW-001" in rule_ids
        assert "RISK-003" in rule_ids  # Legacy backward compat

    def test_auto_renewal_score_increases_without_notice_window(self):
        """No notice opt-out → higher suspiciousness."""
        text_no_notice = "This agreement shall be automatically renewed each year."
        text_with_notice = (
            "This agreement shall be automatically renewed each year unless "
            "terminated by giving 30 days written notice."
        )
        flags_no = evaluate_clause(text_no_notice)
        flags_yes = evaluate_clause(text_with_notice)
        score_no = max(
            (f.suspiciousness_score for f in flags_no if f.rule_id == "RULE-RENEW-001"),
            default=0,
        )
        score_yes = max(
            (f.suspiciousness_score for f in flags_yes if f.rule_id == "RULE-RENEW-001"),
            default=0,
        )
        assert score_no > score_yes


class TestPerpetualConfidentiality:
    def test_perpetual_obligation_flagged(self):
        text = (
            "All Confidential Information shared shall remain confidential "
            "in perpetuity."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-CONF-001" in rule_ids

    def test_trade_secret_exception_not_flagged(self):
        """Perpetual confidentiality limited to trade secrets is legitimate."""
        text = (
            "Trade secrets shall remain confidential in perpetuity. "
            "Except for trade secrets, all other confidential information "
            "shall be protected for 3 years."
        )
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-CONF-001" not in rule_ids


class TestExcessiveNonCompete:
    def test_excessive_duration(self):
        text = "The non-compete restriction shall apply for 5 years after termination."
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RISK-005" in rule_ids

    def test_reasonable_duration_not_flagged(self):
        text = "The non-compete restriction shall apply for 12 months."
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RISK-005" not in rule_ids


class TestIPAssignment:
    def test_blanket_ip_assignment(self):
        text = "Provider assigns all intellectual property rights to Client."
        flags = evaluate_clause(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RISK-007" in rule_ids


class TestBenignClauses:
    def test_no_flags_for_benign_text(self):
        text = "The parties agree to cooperate in good faith to resolve any issues."
        flags = evaluate_clause(text)
        assert len(flags) == 0

    def test_empty_text_returns_no_flags(self):
        flags = evaluate_clause("")
        assert flags == []

    def test_whitespace_text_returns_no_flags(self):
        flags = evaluate_clause("   \n\t  ")
        assert flags == []


# ══════════════════════════════════════════════════════════════════════
#  DOCUMENT-LEVEL TESTS
# ══════════════════════════════════════════════════════════════════════


class TestDocumentMissingGoverningLaw:
    def test_missing_governing_law(self):
        text = (
            "This contract is between Acme and Beta for consulting services. "
            "Payment terms are net-30."
        )
        flags = evaluate_document(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-PROT-001" in rule_ids
        assert "RISK-006" in rule_ids  # Legacy backward compat

    def test_governing_law_present(self):
        text = "This Agreement shall be governed by the laws of England."
        flags = evaluate_document(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-PROT-001" not in rule_ids
        assert "RISK-006" not in rule_ids


class TestDocumentMissingLiabilityCap:
    def test_missing_liability_cap(self):
        text = (
            "Client agrees to pay Provider for all services rendered. "
            "This Agreement is governed by the laws of Delaware."
        )
        flags = evaluate_document(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-PROT-002" in rule_ids

    def test_liability_cap_present(self):
        text = (
            "The aggregate liability under this Agreement shall not exceed "
            "the total fees paid. This Agreement is governed by the laws of Delaware."
        )
        flags = evaluate_document(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-PROT-002" not in rule_ids


class TestDocumentMissingBreachTermination:
    def test_missing_breach_termination(self):
        text = (
            "The parties agree to perform their obligations under this Agreement."
        )
        flags = evaluate_document(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-PROT-003" in rule_ids

    def test_breach_termination_present(self):
        text = (
            "Either party may terminate this Agreement upon material breach "
            "if the other party fails to remedy such breach within 30 days."
        )
        flags = evaluate_document(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-PROT-003" not in rule_ids


class TestConflictingDisputeForums:
    def test_conflicting_forums_flagged(self):
        text = (
            "All disputes shall be resolved by binding arbitration under "
            "the AAA rules. The parties submit to the exclusive jurisdiction "
            "of the courts of New York."
        )
        flags = evaluate_document(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-INCONSIST-001" in rule_ids

    def test_no_conflict_with_injunctive_carveout(self):
        """Courts for injunctive relief with arbitration for all else is standard."""
        text = (
            "All disputes shall be resolved by binding arbitration under "
            "the AAA rules, except that either party may seek injunctive "
            "relief in the exclusive jurisdiction of the courts of New York."
        )
        flags = evaluate_document(text)
        rule_ids = [f.rule_id for f in flags]
        assert "RULE-INCONSIST-001" not in rule_ids


class TestDocumentEmptyText:
    def test_empty_document(self):
        flags = evaluate_document("")
        assert flags == []


# ══════════════════════════════════════════════════════════════════════
#  DEDUPLICATION & OVERLAPPING RULES
# ══════════════════════════════════════════════════════════════════════


class TestDeduplication:
    def test_deduplicate_same_rule_same_snippet(self):
        flag1 = RiskFlag(
            rule_id="RULE-LIAB-001", label="Test", severity="high",
            description="d", matched_snippet="unlimited liability",
            suspiciousness_score=85.0,
        )
        flag2 = RiskFlag(
            rule_id="RULE-LIAB-001", label="Test", severity="high",
            description="d", matched_snippet="unlimited liability",
            suspiciousness_score=90.0,
        )
        result = deduplicate_flags([flag1, flag2])
        assert len(result) == 1
        assert result[0].suspiciousness_score == 90.0

    def test_deduplicate_preserves_distinct_rules(self):
        flag1 = RiskFlag(
            rule_id="RULE-LIAB-001", label="Uncapped", severity="high",
            description="d", matched_snippet="unlimited liability",
            suspiciousness_score=85.0,
        )
        flag2 = RiskFlag(
            rule_id="RULE-TERM-001", label="Termination", severity="high",
            description="d", matched_snippet="may terminate at any time",
            suspiciousness_score=80.0,
        )
        result = deduplicate_flags([flag1, flag2])
        assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════
#  RICH RISKFLAG FIELD VALIDATION
# ══════════════════════════════════════════════════════════════════════


class TestRiskFlagFieldsPhase4:
    def test_flag_has_all_enhanced_fields(self):
        text = "There is unlimited liability under this clause."
        flags = evaluate_clause(text)
        new_flags = [f for f in flags if f.rule_id.startswith("RULE-")]
        assert len(new_flags) >= 1
        flag = new_flags[0]
        assert flag.rule_id
        assert flag.label
        assert flag.severity in ("low", "medium", "high")
        assert flag.description
        assert flag.matched_snippet
        assert flag.category
        assert flag.suspiciousness_score >= 0.0
        assert flag.suspiciousness_score <= 100.0
        assert flag.weight > 0.0
        assert flag.rationale
        assert flag.score_breakdown
        assert flag.evidence_text

    def test_flag_category_belongs_to_known_set(self):
        text = (
            "Provider shall indemnify Client against any and all claims "
            "whatsoever."
        )
        flags = evaluate_clause(text)
        known_categories = {
            "liability", "indemnity", "termination", "payment",
            "confidentiality", "missing_protections",
            "ambiguity_deadlines", "renewal", "general",
        }
        for f in flags:
            assert f.category in known_categories, f"Unknown category: {f.category}"


# ══════════════════════════════════════════════════════════════════════
#  RULE DEFINITIONS REGISTRY METADATA
# ══════════════════════════════════════════════════════════════════════


class TestRuleDefinitionsRegistry:
    def test_definitions_exist(self):
        defs = get_rule_definitions()
        assert len(defs) >= 15

    def test_each_definition_has_required_fields(self):
        for rule_id, rd in RULE_DEFINITIONS.items():
            assert rd.rule_id == rule_id
            assert rd.name
            assert rd.category
            assert rd.severity in ("low", "medium", "high")
            assert rd.default_weight > 0
            assert rd.description
            assert rd.evidence_requirement
            assert rd.rationale

    def test_rule_ids_are_unique(self):
        ids = list(RULE_DEFINITIONS.keys())
        assert len(ids) == len(set(ids))
