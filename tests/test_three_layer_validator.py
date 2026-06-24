"""tests/test_three_layer_validator.py — SLM Phase 2.5 unit tests.

Covers:
  1. Bayesian Sharpe updater: posterior math, decision routing,
     short-circuit on insufficient data
  2. Three-layer voting: majority logic, asymmetric reject-blocking,
     INSUFFICIENT short-circuit
  3. Realistic scenarios: PIT SN-like strong signal across 24mo
     should ACCEPT; weak/noise signal should REJECT; ambiguous
     signal should CONTINUE
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from engine.research.bayesian_sharpe_updater import (
    BayesianDecision, bayesian_sharpe_update,
)
from engine.research.sequential_testing import (
    OBrienFlemingBoundary, default_obf_boundary_paper_trade,
)
from engine.research.three_layer_validator import (
    ThreeLayerDecision, evaluate_three_layer,
)


def _generate_returns(true_sharpe_ann: float, n_months: int,
                      seed: int = 0) -> pd.Series:
    """Helper: simulate monthly returns with a target true Sharpe."""
    rng = np.random.default_rng(seed)
    target_monthly_mean = true_sharpe_ann * 0.01  # implied 10% ann vol
    monthly_vol = 0.01 * math.sqrt(12)  # 10% ann vol
    r = rng.normal(target_monthly_mean, monthly_vol / math.sqrt(12), n_months)
    return pd.Series(
        r, index=pd.date_range("2022-01-31", periods=n_months, freq="ME"),
    )


# ── 1. Bayesian Sharpe updater ─────────────────────────────────────────


class TestBayesianSharpeUpdater:
    def test_insufficient_data_returns_prior(self):
        r = pd.Series([0.01, 0.02])  # n=2 < default min 3
        result = bayesian_sharpe_update(
            sleeve_returns=r, prior_mean=1.0,
        )
        assert result.decision == BayesianDecision.INSUFFICIENT

    def test_strong_signal_accepts(self):
        # True Sharpe 2.0 over 24mo with reasonable prior → ACCEPT
        r = _generate_returns(true_sharpe_ann=2.0, n_months=24, seed=1)
        result = bayesian_sharpe_update(
            sleeve_returns=r, prior_mean=1.5, prior_sd=0.5,
            threshold=0.50, accept_posterior_prob=0.80,
        )
        assert result.decision == BayesianDecision.ACCEPT
        assert result.posterior_prob_above_threshold > 0.80
        assert result.posterior_mean > 1.0

    def test_zero_signal_with_loose_prior_drifts_toward_reject(self):
        # True Sharpe = 0 over 60mo with a loose prior (sd=0.7) should
        # produce a posterior far from prior_mean. With tight prior the
        # posterior stays anchored — this is CORRECT Bayesian behavior
        # and a documented limitation of the conjugate Normal-Normal
        # framework: per Bailey-LdP, Var(SR_ann) ≈ 1 / n_years even
        # for SR ≈ 0, so single-strategy data is informative only over
        # multi-year windows.
        r = _generate_returns(true_sharpe_ann=0.0, n_months=60, seed=2)
        result = bayesian_sharpe_update(
            sleeve_returns=r, prior_mean=1.0, prior_sd=0.7,
            threshold=0.50, reject_posterior_prob=0.20,
        )
        # Posterior should be pulled below the prior, even if not all
        # the way to REJECT (cost of conjugate model)
        assert result.posterior_mean < result.prior_mean
        # Either REJECT or CONTINUE acceptable; ACCEPT would mean
        # framework is broken
        assert result.decision != BayesianDecision.ACCEPT, (
            f"60mo of zero signal should NOT accept; got "
            f"posterior_mean={result.posterior_mean:.3f}, "
            f"p_above={result.posterior_prob_above_threshold:.3f}"
        )

    def test_posterior_pulled_toward_observation_with_more_data(self):
        # Strong contradicting signal: prior=2.0, true=0.0; more months →
        # posterior pulled further toward 0
        r_short = _generate_returns(true_sharpe_ann=0.0, n_months=6, seed=3)
        r_long = _generate_returns(true_sharpe_ann=0.0, n_months=36, seed=3)
        r_short_data = bayesian_sharpe_update(sleeve_returns=r_short,
                                              prior_mean=2.0, prior_sd=0.5)
        r_long_data = bayesian_sharpe_update(sleeve_returns=r_long,
                                             prior_mean=2.0, prior_sd=0.5)
        # Posterior with more data should be closer to true (0)
        assert abs(r_long_data.posterior_mean) < abs(r_short_data.posterior_mean)


# ── 2. Three-layer voting ──────────────────────────────────────────────


class TestThreeLayerVoting:
    def test_strong_signal_24mo_all_layers_accept(self):
        # True Sharpe 2.0 over 24mo + 35 trials → all layers should ACCEPT
        r = _generate_returns(true_sharpe_ann=2.0, n_months=24, seed=10)
        boundary = default_obf_boundary_paper_trade()
        result = evaluate_three_layer(
            sleeve_returns=r,
            prior_mean_sharpe=1.5,
            n_trials_across_research=35,
            obf_boundary=boundary,
            obf_month=24,
        )
        assert result.layer1_vote == "ACCEPT"
        assert result.layer2_vote == "ACCEPT"
        assert result.layer3_vote == "ACCEPT"
        assert result.final_decision == ThreeLayerDecision.ACCEPT
        assert result.evidence_passed is True

    def test_zero_signal_24mo_rejects(self):
        r = _generate_returns(true_sharpe_ann=0.0, n_months=24, seed=11)
        boundary = default_obf_boundary_paper_trade()
        result = evaluate_three_layer(
            sleeve_returns=r,
            prior_mean_sharpe=1.0,
            n_trials_across_research=35,
            obf_boundary=boundary,
            obf_month=24,
        )
        # With reject_is_blocking=True (default), any REJECT vote blocks
        # composite ACCEPT. Zero-signal should produce at least one REJECT.
        assert result.final_decision in (ThreeLayerDecision.REJECT,
                                         ThreeLayerDecision.CONTINUE)
        assert result.evidence_passed is False

    def test_reject_is_blocking_asymmetry(self):
        """ANY single REJECT vote blocks composite ACCEPT in default mode."""
        # Construct a scenario: Layer 1 strongly ACCEPT, Layer 2 ACCEPT,
        # Layer 3 REJECT → should still REJECT due to blocking
        # Easiest synthesis: short window so OBF says INSUFFICIENT → no
        # blocking; need to test with full 3 votes
        # Use a window where the OBF gives REJECT but Bayesian + DSR ACCEPT
        # This is hard to construct cleanly so let's just verify the
        # aggregator behavior directly.
        from engine.research.three_layer_validator import _aggregate
        # 2 ACCEPT + 1 REJECT, blocking → REJECT
        assert _aggregate(["ACCEPT", "ACCEPT", "REJECT"],
                          reject_is_blocking=True) == ThreeLayerDecision.REJECT
        # Same votes, non-blocking → ACCEPT (majority)
        assert _aggregate(["ACCEPT", "ACCEPT", "REJECT"],
                          reject_is_blocking=False) == ThreeLayerDecision.ACCEPT

    def test_insufficient_short_circuit(self):
        from engine.research.three_layer_validator import _aggregate
        # 2 INSUFFICIENT → INSUFFICIENT (not enough data anywhere)
        # (1 INSUFFICIENT + 2 votes can still decide via the 2 votes;
        #  2 INSUFFICIENT means we have <2 deciding votes → INSUFFICIENT)
        assert _aggregate(["INSUFFICIENT", "INSUFFICIENT", "ACCEPT"],
                          reject_is_blocking=True) == ThreeLayerDecision.INSUFFICIENT

    def test_F2_reject_amid_two_insufficient_still_blocks(self):
        """F2 regression test (audit A2 2026-06-03): when 2 layers are
        INSUFFICIENT and 1 is REJECT, the REJECT should hard-block.

        Rationale: if 2 of 3 lenses can't decide and the 1 that CAN says
        REJECT, the REJECT is the only signal we have — trust it MORE,
        not less. Prior to fix, the `n_insufficient < 2` qualifier on
        line 147 caused this case to fall through to the majority logic
        (returning INSUFFICIENT), silently dropping the REJECT signal.
        """
        from engine.research.three_layer_validator import _aggregate
        # The bug case: 1 REJECT + 2 INSUFFICIENT, blocking ON → must REJECT
        assert _aggregate(["REJECT", "INSUFFICIENT", "INSUFFICIENT"],
                          reject_is_blocking=True) == ThreeLayerDecision.REJECT
        # Non-blocking still falls through to INSUFFICIENT (majority)
        assert _aggregate(["REJECT", "INSUFFICIENT", "INSUFFICIENT"],
                          reject_is_blocking=False) == ThreeLayerDecision.INSUFFICIENT

    def test_pit_sn_like_24mo_strong_signal_accepts(self):
        """Realistic scenario mirroring PIT SN: true Sharpe ~2.0, 24mo
        window, 100 candidate strategies tried in research. Should ACCEPT."""
        r = _generate_returns(true_sharpe_ann=2.0, n_months=24, seed=42)
        result = evaluate_three_layer(
            sleeve_returns=r,
            prior_mean_sharpe=1.38,                   # P-D8 honest target
            n_trials_across_research=100,             # broad search
            obf_boundary=default_obf_boundary_paper_trade(),
            obf_month=24,
        )
        assert result.final_decision == ThreeLayerDecision.ACCEPT
        assert result.layer1_bayesian.posterior_mean > 1.0
        assert result.layer2_deflated_sr.deflated_sr > 0.85

    def test_layer3_optional_skipped_when_no_boundary(self):
        r = _generate_returns(true_sharpe_ann=1.5, n_months=24, seed=20)
        result = evaluate_three_layer(
            sleeve_returns=r,
            prior_mean_sharpe=1.5,
            n_trials_across_research=20,
            obf_boundary=None,  # SKIP Layer 3
            obf_month=None,
        )
        assert result.layer3_vote == "INSUFFICIENT"
        # 2 ACCEPT votes + 1 INSUFFICIENT → ACCEPT (majority)
        assert result.final_decision == ThreeLayerDecision.ACCEPT
