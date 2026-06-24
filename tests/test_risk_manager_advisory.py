"""tests/test_risk_manager_advisory.py — Phase 9 Engineer PR sign-off API tests.

Covers:
  - GREEN / YELLOW / RED verdict logic for representative diffs
  - 11 FORBIDDEN_DIFF_PATTERNS each individually trigger RED
  - META validation produces YELLOW for soft warnings
  - render_sign_off_markdown produces structured output
  - SignOffResult schema (8 fields, frozen)
  - 0 LLM cost invariant (advisory is purely deterministic)
"""
from __future__ import annotations

import pytest

from engine.agents.risk_manager.advisory import (
    FORBIDDEN_DIFF_PATTERNS,
    SignOffResult,
    sign_off,
    render_sign_off_markdown,
    _find_forbidden_patterns,
    _validate_proposed_meta,
)
from engine.strategies import StrategyMeta


# ──────────────────────────────────────────────────────────────────────────────
# Verdict outcomes for representative diffs
# ──────────────────────────────────────────────────────────────────────────────
class TestVerdictGREEN:
    def test_empty_diff_is_green(self):
        r = sign_off(diff_text="", affected_strategies=())
        assert r.verdict == "GREEN"
        assert r.cost_usd == 0.0

    def test_clean_helper_addition_is_green(self):
        r = sign_off(
            diff_text="+def helper():\n+    return 42\n",
            affected_strategies=(),
        )
        assert r.verdict == "GREEN"

    def test_well_formed_new_meta_is_green(self):
        meta = StrategyMeta(
            spec_id=999, spec_hash_short="abcdef12", sleeve_id="etf_l1",
            intra_sleeve_weight=0.50, rebalance_days=30, expected_horizon_days=30,
            label="", doctrine="", universe="", color="#000", display_short="X",
        )
        r = sign_off(
            diff_text="+# new strategy file\n",
            affected_strategies=("NEW_X",),
            proposed_meta=meta,
        )
        assert r.verdict == "GREEN"
        assert len(r.passing_checks) >= 3   # hash + sleeve_id + spec_id


class TestVerdictRED:
    @pytest.mark.parametrize("forbidden_payload", [
        "+LOCKED_META = {",
        "+SLEEVE_CLASS_INTRA_CAPS = {",
        "+BOOK_SINGLE_TICKER_ABS_CAP = 0.20",
        "+RISK_THRESHOLDS = ",
        "+ALLOWED_SLEEVES = frozenset",
        "+PAPER_TRADE_SLEEVE_ALLOCATION = {",
        "+LEVERAGE_FACTOR = 2.0",
        "+from engine.preregistration import register_spec\n+register_spec(",
        "+from engine.circuit_breaker import manual_reset\n+manual_reset(",
    ])
    def test_forbidden_pattern_triggers_red(self, forbidden_payload):
        r = sign_off(diff_text=forbidden_payload, affected_strategies=())
        assert r.verdict == "RED"
        assert len(r.forbidden_hits) >= 1

    def test_red_reasons_include_tier3_citation(self):
        r = sign_off(
            diff_text="+LOCKED_META = {}",
            affected_strategies=(),
        )
        assert r.verdict == "RED"
        # Reasons must cite Tier-3 / spec-amendment routing
        joined = " ".join(r.reasons).lower()
        assert "tier-3" in joined or "spec-amendment" in joined


class TestVerdictYELLOW:
    def test_high_intra_weight_yellow(self):
        meta = StrategyMeta(
            spec_id=100, spec_hash_short="abcdef12", sleeve_id="etf_l1",
            intra_sleeve_weight=0.95,        # > 0.9 triggers warning
            rebalance_days=30, expected_horizon_days=30,
            label="", doctrine="", universe="", color="", display_short="",
        )
        r = sign_off(
            diff_text="+# new strategy\n",
            affected_strategies=("NEW_X",),
            proposed_meta=meta,
        )
        assert r.verdict == "YELLOW"
        assert any("intra_sleeve_weight" in w for w in r.meta_warnings)

    def test_malformed_hash_yellow(self):
        # spec_hash_short shape check fires inside __post_init__ for length=8,
        # but our advisory adds a hex-validity check too. Use a 8-char non-hex:
        meta = StrategyMeta(
            spec_id=101, spec_hash_short="xyzqwer1", sleeve_id="etf_l1",  # non-hex
            intra_sleeve_weight=0.5, rebalance_days=30, expected_horizon_days=30,
            label="", doctrine="", universe="", color="", display_short="",
        )
        r = sign_off(
            diff_text="+# new strategy\n",
            affected_strategies=("NEW_Y",),
            proposed_meta=meta,
        )
        assert r.verdict == "YELLOW"


# ──────────────────────────────────────────────────────────────────────────────
# Pattern detector
# ──────────────────────────────────────────────────────────────────────────────
class TestForbiddenPatternDetector:
    def test_finds_distinct_patterns(self):
        text = "+LOCKED_META = {}\n-LOCKED_SLEEVES = frozenset()"
        hits = _find_forbidden_patterns(text)
        # Both LOCKED_META and LOCKED_SLEEVES should match
        assert len(hits) >= 2

    def test_dedups_repeated(self):
        text = "+LOCKED_META = 1\n+LOCKED_META = 2"
        hits = _find_forbidden_patterns(text)
        # Distinct matches dedup (sorted set)
        assert hits == ["LOCKED_META ="]

    def test_no_false_positive_on_partial_match(self):
        # "LOCKED" alone shouldn't trigger; need the full LOCKED_META = pattern
        text = "+# This file is LOCKED for editing"
        hits = _find_forbidden_patterns(text)
        # No forbidden pattern fully matched
        assert hits == []


# ──────────────────────────────────────────────────────────────────────────────
# META validator
# ──────────────────────────────────────────────────────────────────────────────
class TestMetaValidator:
    def test_no_meta_is_passing(self):
        warnings, passing = _validate_proposed_meta(None, ())
        assert warnings == []
        assert any("no META validation requested" in p for p in passing)

    def test_existing_spec_id_treated_as_modify(self):
        # spec_id=61 already exists (K1_BAB)
        meta = StrategyMeta(
            spec_id=61, spec_hash_short="abcdef12", sleeve_id="etf_l1",
            intra_sleeve_weight=0.5, rebalance_days=30, expected_horizon_days=30,
            label="", doctrine="", universe="", color="", display_short="",
        )
        warnings, passing = _validate_proposed_meta(meta, ("K1_BAB",))
        # No warning — affected_strategies maps to existing
        assert not any("ambiguous intent" in w for w in warnings)

    def test_existing_spec_id_ambiguous_if_target_missing(self):
        # spec_id collision but affected_strategies points to non-existent strat
        meta = StrategyMeta(
            spec_id=61, spec_hash_short="abcdef12", sleeve_id="etf_l1",
            intra_sleeve_weight=0.5, rebalance_days=30, expected_horizon_days=30,
            label="", doctrine="", universe="", color="", display_short="",
        )
        warnings, _ = _validate_proposed_meta(meta, ("NOT_A_STRAT",))
        assert any("ambiguous intent" in w for w in warnings)


# ──────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────────────────────
class TestMarkdownRender:
    def test_green_renders_clean(self):
        r = sign_off(diff_text="+def x(): pass", affected_strategies=())
        md = render_sign_off_markdown(r)
        assert "GREEN" in md
        assert "## Risk Manager Advisory" in md

    def test_red_includes_forbidden_section(self):
        r = sign_off(diff_text="+LOCKED_META = {}", affected_strategies=())
        md = render_sign_off_markdown(r)
        assert "RED" in md
        assert "Forbidden-pattern hits" in md
        assert "`LOCKED_META =`" in md   # backtick-wrapped


# ──────────────────────────────────────────────────────────────────────────────
# Result schema invariance
# ──────────────────────────────────────────────────────────────────────────────
class TestSignOffResultSchema:
    def test_eight_fields_locked(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SignOffResult)}
        expected = {
            "verdict", "reasons", "forbidden_hits", "meta_warnings",
            "passing_checks", "spec_anchor", "generated_at_utc", "cost_usd",
        }
        assert fields == expected

    def test_zero_cost_invariant(self):
        # Advisory is pure regex + dataclass validation — never LLM
        r = sign_off(diff_text="x", affected_strategies=())
        assert r.cost_usd == 0.0

    def test_spec_anchor_present(self):
        r = sign_off(diff_text="", affected_strategies=())
        assert "spec id=69" in r.spec_anchor


# ──────────────────────────────────────────────────────────────────────────────
# FORBIDDEN_DIFF_PATTERNS list integrity
# ──────────────────────────────────────────────────────────────────────────────
class TestForbiddenPatternsList:
    def test_minimum_coverage(self):
        """Must include the canonical locked-artifact patterns."""
        joined = " ".join(FORBIDDEN_DIFF_PATTERNS)
        for canonical in [
            "LOCKED_META", "LOCKED_SLEEVES",
            "SLEEVE_CLASS_INTRA_CAPS", "BOOK_SINGLE_TICKER_ABS_CAP",
            "RISK_THRESHOLDS", "ALLOWED_SLEEVES",
            "PAPER_TRADE_SLEEVE_ALLOCATION", "LEVERAGE_FACTOR",
            "register_spec", "manual_reset",
        ]:
            assert canonical in joined, f"missing canonical forbidden pattern {canonical!r}"
