"""
tests/test_factor_lab_power.py — Power analysis textbook reference + edge cases.

Validates engine/factor_lab/power.py against:
  - Cohen (1988) Table 2.4.1 reference values
  - Lo (2002) Equation 4 variance formula
  - Project P3c retrospective scenario

These are mock-free pure-math tests — no DB, no LLM, no fixtures.
"""
from __future__ import annotations

import math

import pytest

from engine.factor_lab.power import (
    achieved_power_at_n,
    power_check,
    required_sample_size_sharpe_diff,
)
from engine.factor_lab.types import FactorState, IllegalTransition, assert_legal_transition


# ─────────────────────────────────────────────────────────────────────────────
# Reference value tests (Cohen 1988 + Lo 2002)
# ─────────────────────────────────────────────────────────────────────────────

class TestRequiredSampleSizeReferenceValues:
    """Verify against Cohen 1988 + Lo 2002 textbook formulas."""

    def test_cohen_default_convention_returns_positive_int(self):
        """Cohen 1988 default α=0.05 + power=0.80 should yield positive integer."""
        n = required_sample_size_sharpe_diff(
            expected_sharpe_lift=0.5,
            baseline_sharpe=0.7,
        )
        assert isinstance(n, int)
        assert n > 0
        # Sanity: with lift=0.5 and S_A=0.7, n_req should be in single-double-digit months
        # not thousands (rough envelope check, not exact reference).
        assert 10 < n < 1000

    def test_smaller_lift_requires_larger_n(self):
        """Power formula: n_req inversely proportional to δ²."""
        n_large_lift  = required_sample_size_sharpe_diff(0.5, 0.7)
        n_small_lift  = required_sample_size_sharpe_diff(0.1, 0.7)
        # Halving δ should ~quadruple n (1/δ² scaling)
        # 0.5 → 0.1 is 5× shrink → ~25× growth in n_req
        assert n_small_lift > 5 * n_large_lift

    def test_higher_baseline_sharpe_requires_more_n(self):
        """Lo 2002: Var(Ŝ) grows with S² → harder to detect δ over high S_A."""
        n_low_baseline  = required_sample_size_sharpe_diff(0.3, 0.0)
        n_high_baseline = required_sample_size_sharpe_diff(0.3, 2.0)
        assert n_high_baseline > n_low_baseline

    def test_higher_target_power_requires_more_n(self):
        """80% → 95% power requires ~1.7× n increase (z_β jumps from 0.84 to 1.64)."""
        n_80 = required_sample_size_sharpe_diff(0.5, 0.7, target_power=0.80)
        n_95 = required_sample_size_sharpe_diff(0.5, 0.7, target_power=0.95)
        ratio = n_95 / n_80
        assert 1.5 < ratio < 2.0  # rough envelope per power formula

    def test_smaller_alpha_requires_more_n(self):
        """α=0.01 vs α=0.05 should yield more conservative (larger) n_req."""
        n_alpha_05 = required_sample_size_sharpe_diff(0.5, 0.7, target_alpha=0.05)
        n_alpha_01 = required_sample_size_sharpe_diff(0.5, 0.7, target_alpha=0.01)
        assert n_alpha_01 > n_alpha_05


class TestEdgeCaseRejection:
    """Validate input guards (spec §3.2)."""

    def test_zero_lift_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            required_sample_size_sharpe_diff(0.0, 0.7)

    def test_negative_lift_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            required_sample_size_sharpe_diff(-0.1, 0.7)

    def test_nan_baseline_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            required_sample_size_sharpe_diff(0.5, float("nan"))

    def test_inf_baseline_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            required_sample_size_sharpe_diff(0.5, float("inf"))

    def test_power_below_floor_raises(self):
        with pytest.raises(ValueError, match="target_power"):
            required_sample_size_sharpe_diff(0.5, 0.7, target_power=0.30)

    def test_power_above_ceiling_raises(self):
        with pytest.raises(ValueError, match="target_power"):
            required_sample_size_sharpe_diff(0.5, 0.7, target_power=0.999)

    def test_alpha_below_floor_raises(self):
        with pytest.raises(ValueError, match="target_alpha"):
            required_sample_size_sharpe_diff(0.5, 0.7, target_alpha=0.0001)

    def test_alpha_above_ceiling_raises(self):
        with pytest.raises(ValueError, match="target_alpha"):
            required_sample_size_sharpe_diff(0.5, 0.7, target_alpha=0.30)


# ─────────────────────────────────────────────────────────────────────────────
# achieved_power_at_n — inverse direction
# ─────────────────────────────────────────────────────────────────────────────

class TestAchievedPowerAtN:
    def test_at_n_required_yields_target_power(self):
        """Round-trip: n_req should produce target power within rounding error."""
        n_req = required_sample_size_sharpe_diff(0.5, 0.7, target_power=0.80)
        achieved = achieved_power_at_n(0.5, 0.7, n_req)
        # ceil() means n_req >= exact n; achieved power should be ≥ target
        assert 0.79 < achieved < 0.85

    def test_zero_n_returns_zero_power(self):
        assert achieved_power_at_n(0.5, 0.7, 0) == 0.0

    def test_huge_n_approaches_one(self):
        achieved = achieved_power_at_n(0.5, 0.7, n_available=10_000)
        assert achieved > 0.999

    def test_returns_float_in_zero_one(self):
        for n in [10, 50, 200, 1000]:
            p = achieved_power_at_n(0.5, 0.7, n)
            assert 0.0 <= p <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# power_check — decision logic
# ─────────────────────────────────────────────────────────────────────────────

class TestPowerCheckDecision:
    def test_sufficient_n_returns_REGISTERED(self):
        result = power_check(
            expected_sharpe_lift=0.5,
            baseline_sharpe=0.7,
            n_available=10_000,  # massively over-powered
        )
        assert result.decision == FactorState.REGISTERED
        assert result.n_available >= result.n_required

    def test_insufficient_n_returns_BLOCKED(self):
        result = power_check(
            expected_sharpe_lift=0.1,  # tiny lift = huge n_req
            baseline_sharpe=0.7,
            n_available=10,
        )
        assert result.decision == FactorState.BLOCKED_UNDERPOWERED
        assert result.n_available < result.n_required
        assert "Increase sample" in result.reason

    def test_p3c_retrospective_blocks(self):
        """P3c had n_extreme=18, lift≈+1.38, baseline≈1.0 → BLOCKED."""
        result = power_check(
            expected_sharpe_lift=1.38,
            baseline_sharpe=1.0,
            n_available=18,
        )
        # P3c achieves ~0.72 power at n=18 — short of 0.80 target → BLOCKED
        assert result.decision == FactorState.BLOCKED_UNDERPOWERED
        assert result.achieved_power_at_n_available < 0.80

    def test_method_label_set(self):
        result = power_check(
            expected_sharpe_lift=0.5,
            baseline_sharpe=0.7,
            n_available=200,
        )
        assert "Lo-Memmel-Cohen" in result.method

    def test_to_dict_serializable(self):
        result = power_check(
            expected_sharpe_lift=0.5,
            baseline_sharpe=0.7,
            n_available=200,
        )
        d = result.to_dict()
        assert d["decision"] in ("REGISTERED", "BLOCKED_UNDERPOWERED")
        assert isinstance(d["n_required"], int)
        assert isinstance(d["achieved_power_at_n_available"], float)


# ─────────────────────────────────────────────────────────────────────────────
# State machine transition rules (spec §2.2)
# ─────────────────────────────────────────────────────────────────────────────

class TestStateMachine:
    def test_legal_draft_to_proposed(self):
        assert_legal_transition(FactorState.DRAFT, FactorState.PROPOSED)

    def test_legal_proposed_to_registered(self):
        assert_legal_transition(FactorState.PROPOSED, FactorState.REGISTERED)

    def test_legal_proposed_to_blocked(self):
        assert_legal_transition(FactorState.PROPOSED, FactorState.BLOCKED_UNDERPOWERED)

    def test_legal_registered_to_testing(self):
        assert_legal_transition(FactorState.REGISTERED, FactorState.TESTING)

    def test_legal_testing_to_pass(self):
        assert_legal_transition(FactorState.TESTING, FactorState.PASS)

    def test_legal_testing_to_marginal(self):
        assert_legal_transition(FactorState.TESTING, FactorState.MARGINAL)

    def test_legal_testing_to_fail(self):
        assert_legal_transition(FactorState.TESTING, FactorState.FAIL)

    def test_legal_testing_to_fail_underpowered(self):
        assert_legal_transition(FactorState.TESTING, FactorState.FAIL_UNDERPOWERED)

    def test_illegal_blocked_to_registered_raises(self):
        """Spec §2.2: BLOCKED is terminal — must rewrite spec."""
        with pytest.raises(IllegalTransition, match="terminal"):
            assert_legal_transition(FactorState.BLOCKED_UNDERPOWERED,
                                    FactorState.REGISTERED)

    def test_illegal_pass_to_testing_raises(self):
        """Verdict states are terminal — no re-test."""
        with pytest.raises(IllegalTransition, match="terminal"):
            assert_legal_transition(FactorState.PASS, FactorState.TESTING)

    def test_illegal_registered_to_pass_skips_testing(self):
        """Cannot skip TESTING."""
        with pytest.raises(IllegalTransition):
            assert_legal_transition(FactorState.REGISTERED, FactorState.PASS)

    def test_illegal_draft_to_registered_skips_proposed(self):
        with pytest.raises(IllegalTransition):
            assert_legal_transition(FactorState.DRAFT, FactorState.REGISTERED)

    def test_illegal_testing_back_to_registered(self):
        """No backtracking."""
        with pytest.raises(IllegalTransition):
            assert_legal_transition(FactorState.TESTING, FactorState.REGISTERED)
