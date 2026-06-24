"""tests/test_regime_multivariate_v2.py — v2 D1+D3+D4 fix correctness.

Spec: docs/spec_multivariate_msm_v2.md (registered post-2026-05-08; supersedes v1
which was withdrawn for architectural defect — NOT falsification).
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# D1 fix: _identify_regimes_by_vix (VIX-anchored regime ID)
# ─────────────────────────────────────────────────────────────────────────────

def test_identify_regimes_by_vix_high_vix_means_off():
    """Regime with HIGHER VIX mean → risk-OFF (stress); LOWER → risk-ON (calm)."""
    from engine.regime import _identify_regimes_by_vix
    # K=2, 3 features (yield_spread, vix, ig_hy_credit_spread)
    means = np.array([
        [+1.5, 14.0, 3.0],   # regime 0: high yield_spread, LOW VIX (calm)
        [-0.3, 32.0, 6.5],   # regime 1: low yield_spread, HIGH VIX (stress)
    ])
    on_idx, off_idx = _identify_regimes_by_vix(means)
    assert on_idx == 0, f"expected risk-on=0 (low VIX); got {on_idx}"
    assert off_idx == 1, f"expected risk-off=1 (high VIX); got {off_idx}"


def test_identify_regimes_by_vix_inverse_assignment():
    """Swap rows; verify VIX anchor still works regardless of fitted index order."""
    from engine.regime import _identify_regimes_by_vix
    means = np.array([
        [-0.3, 32.0, 6.5],   # regime 0: HIGH VIX (stress)
        [+1.5, 14.0, 3.0],   # regime 1: LOW VIX (calm)
    ])
    on_idx, off_idx = _identify_regimes_by_vix(means)
    assert on_idx == 1, f"low-VIX regime should be on regardless of array order; got {on_idx}"
    assert off_idx == 0


def test_identify_regimes_by_vix_tie_falls_back_to_yield_spread():
    """If VIX means tie exactly, fallback to yield_spread anchor."""
    from engine.regime import _identify_regimes_by_vix
    means = np.array([
        [+2.0, 20.0, 4.0],
        [-1.0, 20.0, 4.0],   # exact VIX tie
    ])
    on_idx, off_idx = _identify_regimes_by_vix(means)
    # Tiebreaker: argmax(yield_spread) for risk-on, argmin for risk-off
    assert on_idx == 0
    assert off_idx == 1


def test_identify_regimes_by_vix_requires_vix_column():
    """If feature_names doesn't contain 'vix', should raise."""
    from engine.regime import _identify_regimes_by_vix
    means = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="vix"):
        _identify_regimes_by_vix(means, feature_names=("yield_spread", "credit_spread"))


# ─────────────────────────────────────────────────────────────────────────────
# D3 fix: compute_overlay_returns_ternary (hysteresis band)
# ─────────────────────────────────────────────────────────────────────────────

def test_ternary_overlay_above_upper_long():
    """p > 0.55 → position +1 → overlay = base."""
    from engine.multivariate_msm_verdict import compute_overlay_returns_ternary
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    p = pd.Series([0.6, 0.7, 0.8, 0.9, 1.0], index=idx)
    base = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02], index=idx)
    overlay = compute_overlay_returns_ternary(p, base)
    assert np.allclose(overlay.values, base.values)


def test_ternary_overlay_below_lower_short():
    """p < 0.45 → position -1 → overlay = -base."""
    from engine.multivariate_msm_verdict import compute_overlay_returns_ternary
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    p = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4], index=idx)
    base = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02], index=idx)
    overlay = compute_overlay_returns_ternary(p, base)
    assert np.allclose(overlay.values, -base.values)


def test_ternary_overlay_within_band_zero():
    """0.45 ≤ p ≤ 0.55 → position 0 → overlay = 0."""
    from engine.multivariate_msm_verdict import compute_overlay_returns_ternary
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    p = pd.Series([0.45, 0.48, 0.50, 0.52, 0.55], index=idx)
    base = pd.Series([0.05, -0.03, 0.04, -0.02, 0.01], index=idx)
    overlay = compute_overlay_returns_ternary(p, base)
    assert np.allclose(overlay.values, 0.0)


def test_ternary_overlay_invalid_band_raises():
    """upper ≤ lower or band outside (0,1) → raises."""
    from engine.multivariate_msm_verdict import compute_overlay_returns_ternary
    p = pd.Series([0.5])
    b = pd.Series([0.01])
    with pytest.raises(ValueError, match="hysteresis band"):
        compute_overlay_returns_ternary(p, b, upper=0.4, lower=0.6)
    with pytest.raises(ValueError, match="hysteresis band"):
        compute_overlay_returns_ternary(p, b, upper=1.5, lower=0.4)


# ─────────────────────────────────────────────────────────────────────────────
# v2 verdict: compute_verdict_v2 (descriptive-only labels)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_paired_returns(
    n_months: int = 72,
    multi_mean: float = 0.005,
    uni_mean:   float = 0.004,
    vol:        float = 0.04,
    seed:       int = 53,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    base = rng.normal(0, vol, n_months)
    a = pd.Series(base + rng.normal(multi_mean, 0.005, n_months), index=idx)
    b = pd.Series(base + rng.normal(uni_mean, 0.005, n_months), index=idx)
    return a, b


def test_verdict_v2_descriptive_negative_label():
    """ΔŜ < 0 → DESCRIPTIVE_NEGATIVE."""
    from engine.multivariate_msm_verdict import compute_verdict_v2
    a, b = _synthetic_paired_returns(multi_mean=0.000, uni_mean=0.012, seed=61)
    v = compute_verdict_v2(a, b, fallback_rate=0.05, n_resamples=300, random_state=71)
    assert v.decision_label == "DESCRIPTIVE_NEGATIVE", (
        f"strong negative ΔŜ should give NEGATIVE; got {v.decision_label} (ΔŜ={v.delta_sharpe})"
    )


def test_verdict_v2_uninterpretable_at_high_fallback():
    """fallback rate ≥ 50% → UNINTERPRETABLE regardless of ΔŜ."""
    from engine.multivariate_msm_verdict import compute_verdict_v2
    a, b = _synthetic_paired_returns(multi_mean=0.020, uni_mean=0.000, seed=79)
    v = compute_verdict_v2(a, b, fallback_rate=0.70, n_resamples=200, random_state=83)
    assert v.decision_label == "UNINTERPRETABLE"


def test_verdict_v2_no_pass_label_in_taxonomy():
    """v2 spec §3.2 explicitly drops PASS — verify no PASS label can be produced."""
    from engine.multivariate_msm_verdict import compute_verdict_v2
    # Generate strong positive case
    a, b = _synthetic_paired_returns(n_months=120, multi_mean=0.020, uni_mean=0.000, seed=89)
    v = compute_verdict_v2(a, b, fallback_rate=0.0, n_resamples=300, random_state=97)
    assert v.decision_label != "PASS", "v2 spec §3.2 drops PASS gate; label must never be 'PASS'"
    assert v.decision_label in {"DESCRIPTIVE_POSITIVE", "DESCRIPTIVE_INSUFFICIENT",
                                 "DESCRIPTIVE_NEGATIVE", "UNINTERPRETABLE"}


def test_verdict_v2_descriptive_positive_requires_threshold_and_ci():
    """DESCRIPTIVE_POSITIVE requires ΔŜ ≥ +0.05 AND CI lower > 0."""
    from engine.multivariate_msm_verdict import compute_verdict_v2
    # Strong positive case
    a, b = _synthetic_paired_returns(n_months=120, multi_mean=0.025, uni_mean=0.000, seed=101)
    v = compute_verdict_v2(a, b, fallback_rate=0.0, n_resamples=500, random_state=103)
    if v.delta_sharpe >= 0.05 and v.bootstrap_ci_lower > 0:
        assert v.decision_label == "DESCRIPTIVE_POSITIVE"
    # Otherwise label = INSUFFICIENT (don't assert; depends on bootstrap noise)


def test_verdict_v2_carries_all_diagnostic_fields():
    """Snapshot must include all fields reviewer / supervisor needs."""
    from engine.multivariate_msm_verdict import compute_verdict_v2, MultivariateVerdictV2
    a, b = _synthetic_paired_returns()
    v = compute_verdict_v2(a, b, fallback_rate=0.10, n_resamples=200, random_state=109)
    assert isinstance(v, MultivariateVerdictV2)
    for attr in ["delta_sharpe", "sharpe_multivariate", "sharpe_univariate",
                 "bootstrap_ci_lower", "bootstrap_ci_upper", "memmel_z",
                 "paired_correlation", "fallback_rate", "n_oos_months",
                 "achieved_power_descriptive", "ci_lower_above_zero",
                 "ci_lower_above_threshold", "decision_label"]:
        assert hasattr(v, attr), f"MultivariateVerdictV2 missing {attr}"


# ─────────────────────────────────────────────────────────────────────────────
# v2 spec lock: thresholds / constants
# ─────────────────────────────────────────────────────────────────────────────

def test_v2_overlay_band_locked():
    """Spec_v2 §2.5 hysteresis band locked at 0.45 / 0.55."""
    from engine import regime, multivariate_msm_verdict
    assert regime._V2_OVERLAY_UPPER_THRESHOLD == 0.55
    assert regime._V2_OVERLAY_LOWER_THRESHOLD == 0.45
    assert multivariate_msm_verdict._V2_OVERLAY_UPPER == 0.55
    assert multivariate_msm_verdict._V2_OVERLAY_LOWER == 0.45


def test_v2_proxy_validation_thresholds_locked():
    """Spec_v2 §3.6 r ≥ 0.7 = validated, 0.5 ≤ r < 0.7 = weakly_validated, r < 0.5 = INVALID."""
    from engine import regime
    assert regime._V2_PROXY_R_INVALID_BELOW == 0.50
    assert regime._V2_PROXY_R_VALIDATED_ABOVE == 0.70
