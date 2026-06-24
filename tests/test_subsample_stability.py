"""tests/test_subsample_stability.py — Tier C L2-5 Commit 1.

Pure-function tests for compute_subsample_stability. Synthetic
factor PnL series with known stability properties (stable / decaying
/ growing / crisis-killed) — assert the function recovers the right
diagnostic flags.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def idx_120():
    return pd.date_range("2014-01-31", periods=120, freq="ME")


@pytest.fixture
def idx_240():
    return pd.date_range("2005-01-31", periods=240, freq="ME")


# ────────────────────────────────────────────────────────────────────
# STABLE factor — constant Sharpe across windows
# ────────────────────────────────────────────────────────────────────
def test_stable_factor_passes_institutional_bar(idx_240):
    """High-SNR stable factor (mean 1%/mo, vol 0.5%/mo) — window
    sample noise should NOT dominate. Per
    [[feedback-random-data-test-tolerances-from-theory-2026-06-09]]:
    test tolerance band must accommodate sampling SE. Here SE(Sharpe)
    per 60-mo window ≈ sqrt((1+0.5·SR²)/5yr) ≈ 0.7 on a true SR ≈ 6.9,
    so realized SR ranges roughly ±2.1 around 6.9 → worst/best ratio
    consistently > 0.5 across seeds."""
    from engine.research.subsample_stability import (
        compute_subsample_stability,
    )
    rng = np.random.default_rng(7)
    n = 240
    factor = pd.Series(
        0.010 + rng.normal(0, 0.005, n), index=idx_240,
    )
    out = compute_subsample_stability(factor, n_splits=4)
    assert out is not None
    assert out["n_splits"] == 4
    assert out["n_total_months"] == 240
    assert len(out["windows"]) == 4
    # High-SNR stable factor — ratio should be high
    assert out["worst_best_sharpe_ratio"] > 0.50
    assert out["institutional_stable"] is True


def test_stable_factor_no_monotone_trend_on_white_noise(idx_240):
    """A truly stable factor (no time-varying mean) should NOT trigger
    monotone_decay or monotone_growth flags. Different seed than the
    institutional-bar test so we cover multiple sampling realizations."""
    from engine.research.subsample_stability import (
        compute_subsample_stability,
    )
    rng = np.random.default_rng(101)
    n = 240
    factor = pd.Series(
        0.010 + rng.normal(0, 0.005, n), index=idx_240,
    )
    out = compute_subsample_stability(factor, n_splits=4)
    assert out is not None
    assert out["monotone_decay"]  is False
    assert out["monotone_growth"] is False


# ────────────────────────────────────────────────────────────────────
# DECAYING factor — McLean-Pontiff post-pub style
# ────────────────────────────────────────────────────────────────────
def test_decaying_factor_flagged(idx_240):
    """Each sub-window has progressively lower mean → monotone decay
    flag should fire; decay_slope_per_year significantly negative."""
    from engine.research.subsample_stability import (
        compute_subsample_stability,
    )
    rng = np.random.default_rng(11)
    n = 240
    # Linear decay: starts at 1.5%/mo, ends at -0.5%/mo
    time_drift = np.linspace(0.015, -0.005, n)
    noise = rng.normal(0, 0.015, n)
    factor = pd.Series(time_drift + noise, index=idx_240)
    out = compute_subsample_stability(factor, n_splits=4)
    assert out is not None
    # Sharpe pattern should be strictly decreasing across all 4 splits
    sharpes = [w["sharpe_ann"] for w in out["windows"]]
    assert out["monotone_decay"] is True, f"sharpes={sharpes}"
    # Decay slope should be significantly negative (t < -1.96)
    assert out["decay_slope_per_year"] < 0
    assert out["decay_slope_t"] is not None
    assert out["decay_slope_t"] < -1.96, (
        f"decay slope t={out['decay_slope_t']} should be < -1.96"
    )


# ────────────────────────────────────────────────────────────────────
# GROWING factor — suspicious (could be non-stationary trend)
# ────────────────────────────────────────────────────────────────────
def test_growing_factor_flagged(idx_240):
    from engine.research.subsample_stability import (
        compute_subsample_stability,
    )
    rng = np.random.default_rng(23)
    n = 240
    time_drift = np.linspace(-0.005, 0.015, n)
    noise = rng.normal(0, 0.015, n)
    factor = pd.Series(time_drift + noise, index=idx_240)
    out = compute_subsample_stability(factor, n_splits=4)
    assert out is not None
    assert out["monotone_growth"] is True
    assert out["decay_slope_per_year"] > 0
    assert out["decay_slope_t"] > 1.96


# ────────────────────────────────────────────────────────────────────
# CRISIS-KILLED factor — passes 3 windows but window 4 craters
# ────────────────────────────────────────────────────────────────────
def test_crisis_killed_factor_fails_institutional_bar(idx_240):
    """Three positive windows then a strongly negative one → fails
    institutional bar (worst window < 0)."""
    from engine.research.subsample_stability import (
        compute_subsample_stability,
    )
    rng = np.random.default_rng(31)
    n = 240
    # First 180 months: +5bp/mo. Last 60 months: -2%/mo (crisis)
    mean = np.concatenate([np.full(180, 0.005), np.full(60, -0.02)])
    noise = rng.normal(0, 0.015, n)
    factor = pd.Series(mean + noise, index=idx_240)
    out = compute_subsample_stability(factor, n_splits=4)
    assert out is not None
    sharpes = [w["sharpe_ann"] for w in out["windows"]]
    # Last window cratered
    assert sharpes[-1] < 0
    # First 3 positive
    assert all(s > 0 for s in sharpes[:3])
    # Institutional bar fails (min Sharpe is negative)
    assert out["institutional_stable"] is False


# ────────────────────────────────────────────────────────────────────
# Refusal cases
# ────────────────────────────────────────────────────────────────────
def test_returns_none_when_too_few_total_months():
    """4 splits × 24 min = 96 floor. Below → None."""
    from engine.research.subsample_stability import (
        compute_subsample_stability,
    )
    idx = pd.date_range("2023-01-31", periods=60, freq="ME")
    factor = pd.Series(np.zeros(60), index=idx)
    assert compute_subsample_stability(
        factor, n_splits=4, min_per_sub=24,
    ) is None


def test_returns_none_on_empty_series():
    from engine.research.subsample_stability import (
        compute_subsample_stability,
    )
    assert compute_subsample_stability(pd.Series(dtype=float)) is None


def test_returns_none_when_index_not_datetime():
    from engine.research.subsample_stability import (
        compute_subsample_stability,
    )
    factor = pd.Series(np.zeros(120), index=range(120))
    assert compute_subsample_stability(factor) is None


def test_two_split_works_on_smaller_sample(idx_120):
    """With n_splits=2 and min_per_sub=24, 120 months is plenty."""
    from engine.research.subsample_stability import (
        compute_subsample_stability,
    )
    rng = np.random.default_rng(43)
    factor = pd.Series(rng.normal(0.005, 0.02, 120), index=idx_120)
    out = compute_subsample_stability(factor, n_splits=2)
    assert out is not None
    assert len(out["windows"]) == 2


# ────────────────────────────────────────────────────────────────────
# compute_for_tier_c_pnl_series helper
# ────────────────────────────────────────────────────────────────────
def test_tier_c_helper_reads_pnl_net_13bp(idx_240):
    from engine.research.subsample_stability import (
        compute_for_tier_c_pnl_series,
    )
    rng = np.random.default_rng(53)
    df = pd.DataFrame({
        "pnl_gross":    rng.normal(0.006, 0.04, 240),
        "pnl_net_13bp": rng.normal(0.005, 0.04, 240),
        "pnl_net_80bp": rng.normal(0.003, 0.04, 240),
        "turnover":    rng.uniform(0.3, 0.6, 240),
    }, index=idx_240)
    out = compute_for_tier_c_pnl_series(df, n_splits=4)
    assert out is not None
    assert "windows" in out


def test_tier_c_helper_returns_none_when_column_missing(idx_240):
    from engine.research.subsample_stability import (
        compute_for_tier_c_pnl_series,
    )
    df = pd.DataFrame({"only_gross": np.zeros(240)}, index=idx_240)
    assert compute_for_tier_c_pnl_series(df) is None


# ────────────────────────────────────────────────────────────────────
# Real GP/A parquet integration (skipped if missing)
# ────────────────────────────────────────────────────────────────────
def test_gpa_real_parquet_when_present():
    """Run on the actual seeded GP/A parquet — should produce sensible
    results that future regression tests can pin."""
    p = (Path(__file__).resolve().parents[1] / "data" / "research_store"
         / "tier_c_pnl" / "dc4cf6beaa247880_GREEN.parquet")
    if not p.exists():
        pytest.skip("GP/A parquet not present — run seed_tier_c_idle.py")
    from engine.research.subsample_stability import (
        compute_for_tier_c_pnl_series,
    )
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    out = compute_for_tier_c_pnl_series(df, n_splits=4)
    assert out is not None
    assert out["n_total_months"] == 395
    assert len(out["windows"]) == 4
    # GP/A 1992-2024 = ~33 years. 4 splits = ~99 months each
    for w in out["windows"]:
        assert 95 < w["n_months"] < 105
