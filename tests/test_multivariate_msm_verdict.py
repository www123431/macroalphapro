"""tests/test_multivariate_msm_verdict.py — verdict computation correctness.

Spec: docs/spec_multivariate_msm_v1.md §3.1 / §3.2 / §3.4

Synthetic-data tests for pure compute functions (Memmel Z / bootstrap CI /
decision rule / overlay returns). The actual OOS run requires FRED + yfinance
network access (script: scripts/run_multivariate_msm_d6.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_returns(
    n: int     = 72,
    mean: float = 0.005,
    vol: float  = 0.04,
    seed: int   = 13,
) -> pd.Series:
    """6-year (72 monthly obs) synthetic return series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-31", periods=n, freq="ME")
    return pd.Series(rng.normal(mean, vol, n), index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# compute_overlay_returns
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_overlay_returns_full_long_at_p_one():
    """p_risk_on=1.0 → position=+1 → overlay return == base return."""
    from engine.multivariate_msm_verdict import compute_overlay_returns
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    p = pd.Series([1.0] * 12, index=idx)
    base = pd.Series(np.linspace(-0.02, 0.03, 12), index=idx)
    overlay = compute_overlay_returns(p, base)
    assert np.allclose(overlay.values, base.values)


def test_compute_overlay_returns_full_short_at_p_zero():
    """p_risk_on=0 → position=-1 → overlay = -base."""
    from engine.multivariate_msm_verdict import compute_overlay_returns
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    p = pd.Series([0.0] * 12, index=idx)
    base = pd.Series(np.linspace(-0.02, 0.03, 12), index=idx)
    overlay = compute_overlay_returns(p, base)
    assert np.allclose(overlay.values, -base.values)


def test_compute_overlay_returns_zero_at_p_half():
    """p_risk_on=0.5 → position=0 → overlay = 0."""
    from engine.multivariate_msm_verdict import compute_overlay_returns
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    p = pd.Series([0.5] * 12, index=idx)
    base = pd.Series(np.linspace(-0.02, 0.03, 12), index=idx)
    overlay = compute_overlay_returns(p, base)
    assert np.allclose(overlay.values, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# annualized_sharpe / Memmel Z
# ─────────────────────────────────────────────────────────────────────────────

def test_annualized_sharpe_matches_lo_2002_convention():
    """For monthly returns with mean μ and std σ, annualized Sharpe = (μ/σ)·√12."""
    from engine.multivariate_msm_verdict import annualized_sharpe
    rets = _synthetic_returns(n=120, mean=0.008, vol=0.04, seed=11)
    s = annualized_sharpe(rets)
    # Hand-check
    expected = (rets.mean() / rets.std(ddof=1)) * np.sqrt(12)
    assert abs(s - expected) < 1e-10


def test_annualized_sharpe_returns_nan_on_zero_vol():
    """Constant returns → std=0 → Sharpe undefined → NaN."""
    from engine.multivariate_msm_verdict import annualized_sharpe
    rets = pd.Series([0.005] * 24, index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    assert np.isnan(annualized_sharpe(rets))


def test_memmel_z_returns_finite_on_correlated_pair():
    """Two correlated synthetic series → Memmel Z + ρ̂ + V̂ all finite."""
    from engine.multivariate_msm_verdict import memmel_z_paired_sharpe_diff
    rng = np.random.default_rng(19)
    n = 72
    idx = pd.date_range("2019-01-31", periods=n, freq="ME")
    base = rng.normal(0.005, 0.04, n)
    a = pd.Series(base + rng.normal(0, 0.005, n), index=idx)
    b = pd.Series(base + rng.normal(0, 0.005, n), index=idx)
    z, rho, V = memmel_z_paired_sharpe_diff(a, b)
    assert np.isfinite(z) and np.isfinite(rho) and np.isfinite(V)
    assert 0.5 < rho < 1.0, f"expected high paired corr; got {rho}"
    assert V > 0


def test_memmel_z_returns_nan_on_short_series():
    """< 12 paired observations → NaN (cannot estimate stable Sharpe)."""
    from engine.multivariate_msm_verdict import memmel_z_paired_sharpe_diff
    a = pd.Series([0.01, 0.02, 0.03, -0.01], index=pd.date_range("2020-01-31", periods=4, freq="ME"))
    b = pd.Series([0.005, 0.015, 0.025, 0.0], index=a.index)
    z, rho, V = memmel_z_paired_sharpe_diff(a, b)
    assert np.isnan(z)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def test_bootstrap_ci_returns_finite_on_realistic_input():
    """72-month paired returns → bootstrap returns finite (lower, upper, block_size)."""
    from engine.multivariate_msm_verdict import bootstrap_sharpe_diff_ci
    a = _synthetic_returns(n=72, mean=0.005, vol=0.04, seed=21)
    b = _synthetic_returns(n=72, mean=0.004, vol=0.04, seed=23)
    lo, up, block = bootstrap_sharpe_diff_ci(a, b, n_resamples=500, random_state=99)
    assert np.isfinite(lo) and np.isfinite(up)
    assert lo <= up
    assert block >= 1


def test_bootstrap_ci_lower_above_zero_when_strong_positive_diff():
    """Construct a series A clearly stronger than B → bootstrap CI lower bound > 0."""
    from engine.multivariate_msm_verdict import bootstrap_sharpe_diff_ci
    rng = np.random.default_rng(31)
    n = 120  # 10y for tighter CI
    idx = pd.date_range("2014-01-31", periods=n, freq="ME")
    base = rng.normal(0, 0.04, n)
    a = pd.Series(base + rng.normal(+0.012, 0.005, n), index=idx)  # +14%/yr drift
    b = pd.Series(base + rng.normal(0.000, 0.005, n), index=idx)   # 0 drift
    lo, up, _ = bootstrap_sharpe_diff_ci(a, b, n_resamples=500, random_state=41)
    assert lo > 0, f"strong positive ΔŜ should have CI lower > 0; got [{lo}, {up}]"


# ─────────────────────────────────────────────────────────────────────────────
# Decision rule (spec §3.2 locked)
# ─────────────────────────────────────────────────────────────────────────────

def test_decision_rule_pass_requires_threshold_and_ci_above_zero():
    from engine.multivariate_msm_verdict import apply_decision_rule
    assert apply_decision_rule(0.15, 0.05, 0.25, 0.0) == "PASS"
    assert apply_decision_rule(0.10, 0.001, 0.20, 0.0) == "PASS"


def test_decision_rule_marginal_insufficient_precision_when_ci_crosses_zero():
    from engine.multivariate_msm_verdict import apply_decision_rule
    assert apply_decision_rule(0.15, -0.05, 0.30, 0.0) == "MARGINAL_INSUFFICIENT_PRECISION"


def test_decision_rule_marginal_when_below_threshold():
    from engine.multivariate_msm_verdict import apply_decision_rule
    assert apply_decision_rule(0.05, -0.10, 0.20, 0.0) == "MARGINAL"


def test_decision_rule_fail_on_negative_delta():
    from engine.multivariate_msm_verdict import apply_decision_rule
    assert apply_decision_rule(-0.05, -0.20, 0.10, 0.0) == "FAIL"


def test_decision_rule_uninterpretable_at_50pct_fallback():
    """Per spec §3.4: ≥ 50% fallback → UNINTERPRETABLE regardless of ΔŜ."""
    from engine.multivariate_msm_verdict import apply_decision_rule
    # Even with a "PASS-shaped" ΔŜ + CI, fallback ≥ 50% wins
    assert apply_decision_rule(0.15, 0.05, 0.25, 0.50) == "UNINTERPRETABLE"
    assert apply_decision_rule(0.15, 0.05, 0.25, 0.75) == "UNINTERPRETABLE"


def test_decision_rule_below_uninterpretable_threshold_proceeds():
    """49% fallback → still proceeds with normal decision logic."""
    from engine.multivariate_msm_verdict import apply_decision_rule
    assert apply_decision_rule(0.15, 0.05, 0.25, 0.49) == "PASS"


# ─────────────────────────────────────────────────────────────────────────────
# Top-level compute_verdict
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_verdict_returns_complete_snapshot():
    """End-to-end verdict on synthetic 72-month overlay returns."""
    from engine.multivariate_msm_verdict import compute_verdict, MultivariateVerdict
    a = _synthetic_returns(n=72, mean=0.006, vol=0.04, seed=51)
    b = _synthetic_returns(n=72, mean=0.005, vol=0.04, seed=53)
    v = compute_verdict(a, b, fallback_rate=0.05, n_resamples=500, random_state=61)
    assert isinstance(v, MultivariateVerdict)
    assert v.n_oos_months == 72
    assert v.decision in {"PASS", "MARGINAL_INSUFFICIENT_PRECISION",
                          "MARGINAL", "FAIL", "UNINTERPRETABLE"}
    assert np.isfinite(v.delta_sharpe)
    assert np.isfinite(v.bootstrap_ci_lower) and np.isfinite(v.bootstrap_ci_upper)
    assert v.bootstrap_block_size >= 1
    assert 0 <= v.achieved_power_descriptive <= 1


def test_compute_verdict_uninterpretable_at_high_fallback():
    """High fallback rate overrides any ΔŜ verdict."""
    from engine.multivariate_msm_verdict import compute_verdict
    a = _synthetic_returns(n=72, mean=0.020, vol=0.04, seed=71)
    b = _synthetic_returns(n=72, mean=0.000, vol=0.04, seed=73)
    v = compute_verdict(a, b, fallback_rate=0.60, n_resamples=200, random_state=81)
    assert v.decision == "UNINTERPRETABLE"
