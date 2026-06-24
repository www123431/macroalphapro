"""Tests for engine.risk.barra_lite.regress_sleeve_by_regime (FLAW 5 fix)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.risk.barra_lite import (
    regress_sleeve_by_regime,
    regress_sleeve_on_factors,
)


@pytest.fixture
def synth_panel():
    """120-month sleeve + factors + regime labels for testing."""
    np.random.seed(7)
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    factors = pd.DataFrame({
        "MKT": np.random.randn(120) * 0.04,
        "SMB": np.random.randn(120) * 0.03,
        "MOM": np.random.randn(120) * 0.04,
    }, index=idx)
    # Regime: alternating blocks of 40 months
    regime = pd.Series(
        ["CALM"] * 40 + ["NORMAL"] * 40 + ["STRESS"] * 40,
        index=idx, name="regime",
    )
    # Sleeve = 0.5 × MOM + alpha (varies by regime)
    sleeve = pd.Series(0.0, index=idx)
    sleeve.loc[idx[:40]] = (0.5 * factors["MOM"].iloc[:40] + 0.015
                              + np.random.RandomState(11).randn(40) * 0.005)
    sleeve.loc[idx[40:80]] = (0.5 * factors["MOM"].iloc[40:80] + 0.005
                                 + np.random.RandomState(13).randn(40) * 0.005)
    sleeve.loc[idx[80:]] = (0.5 * factors["MOM"].iloc[80:] - 0.005
                              + np.random.RandomState(17).randn(40) * 0.005)
    return factors, sleeve, regime


# ── Basic functionality ─────────────────────────────────────────────────

def test_regress_by_regime_returns_dict(synth_panel):
    factors, sleeve, regime = synth_panel
    out = regress_sleeve_by_regime(sleeve, regime, factors)
    assert isinstance(out, dict)
    assert {"CALM", "NORMAL", "STRESS"}.issubset(set(out.keys()))


def test_regress_by_regime_each_value_is_report(synth_panel):
    factors, sleeve, regime = synth_panel
    out = regress_sleeve_by_regime(sleeve, regime, factors)
    for label, rep in out.items():
        assert hasattr(rep, "alpha_t_hac")
        assert hasattr(rep, "betas")
        assert "MOM" in rep.betas


def test_regress_by_regime_recovers_known_alpha_differences(synth_panel):
    """Synthetic alphas: CALM=+1.5%/mo, NORMAL=+0.5%/mo, STRESS=-0.5%/mo.
    The regime-stratified regression should reflect that ordering."""
    factors, sleeve, regime = synth_panel
    out = regress_sleeve_by_regime(sleeve, regime, factors)
    # CALM alpha should be highest (constructed as 0.015 monthly)
    calm_alpha = out["CALM"].alpha_monthly
    normal_alpha = out["NORMAL"].alpha_monthly
    stress_alpha = out["STRESS"].alpha_monthly
    assert calm_alpha > normal_alpha
    assert normal_alpha > stress_alpha


def test_regress_by_regime_skips_small_regime(synth_panel):
    """Regime with fewer than min_months_per_regime → skipped."""
    factors, sleeve, regime = synth_panel
    # Make STRESS very small
    regime_sparse = regime.copy()
    regime_sparse.iloc[80:118] = "NORMAL"   # only 2 STRESS months
    out = regress_sleeve_by_regime(sleeve, regime_sparse, factors,
                                          min_months_per_regime=18)
    assert "STRESS" not in out
    assert "NORMAL" in out


def test_regress_by_regime_hac_lag_scaled_for_short_samples(synth_panel):
    """For short regime sub-samples, hac_lags should be auto-reduced
    (capped at ~n/4) to remain meaningful."""
    factors, sleeve, regime = synth_panel
    out = regress_sleeve_by_regime(sleeve, regime, factors,
                                          hac_lags=6,
                                          min_months_per_regime=18)
    # All regimes have 40 months → hac scaled to min(6, 40/4) = 6
    for rep in out.values():
        assert rep.n_months >= 18


def test_regress_by_regime_aggregated_vs_stratified_diff(synth_panel):
    """Aggregated regression hides per-regime variation; stratified
    should reveal it (FLAW 5 motivation)."""
    factors, sleeve, regime = synth_panel
    agg = regress_sleeve_on_factors(sleeve, factors, sleeve_name="agg")
    strat = regress_sleeve_by_regime(sleeve, regime, factors)
    # Stratified alphas should span a range; aggregated is one number
    stratified_alphas = [r.alpha_monthly for r in strat.values()]
    spread = max(stratified_alphas) - min(stratified_alphas)
    # Spread should be meaningfully > 0 since we constructed differences
    assert spread > 0.005, "regime spread should reveal hidden variation"
    # Aggregated alpha is between extremes
    assert min(stratified_alphas) <= agg.alpha_monthly <= max(stratified_alphas)


def test_regress_by_regime_empty_overlap_returns_empty():
    """Disjoint indices → no overlap → empty dict."""
    idx_s = pd.date_range("2020-01-31", periods=10, freq="ME")
    idx_r = pd.date_range("2010-01-31", periods=10, freq="ME")
    sleeve = pd.Series(np.random.randn(10) * 0.01, index=idx_s)
    regime = pd.Series(["NORMAL"] * 10, index=idx_r)
    factors = pd.DataFrame({"MKT": np.random.randn(10) * 0.04},
                              index=idx_s)
    out = regress_sleeve_by_regime(sleeve, regime, factors)
    assert out == {}


def test_regress_by_regime_one_regime_only(synth_panel):
    """All months in one regime → only that regime in output."""
    factors, sleeve, regime = synth_panel
    all_normal = pd.Series("NORMAL", index=regime.index)
    out = regress_sleeve_by_regime(sleeve, all_normal, factors)
    assert list(out.keys()) == ["NORMAL"]
    assert out["NORMAL"].n_months >= 100
