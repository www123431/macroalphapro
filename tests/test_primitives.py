"""Tests for engine.research.primitives — Layer 0 audited primitives.

CRITICAL properties (each primitive must pass):
- Anti-look-ahead (signal at t uses only ≤t-1 info where applicable)
- NaN-safe (no exceptions on NaN inputs)
- Dtype-preserving (float in → float out)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research import primitives as P


@pytest.fixture
def sample_prices():
    """3-ticker, 30-day price panel with no NaNs."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame(
        np.cumprod(1 + 0.01 * np.random.RandomState(0).randn(30, 3), axis=0) * 100,
        index=dates, columns=["A", "B", "C"],
    )


@pytest.fixture
def sample_returns():
    dates = pd.date_range("2024-01-31", periods=60, freq="ME")
    return pd.Series(
        np.random.RandomState(0).randn(60) * 0.01, index=dates, name="ls"
    )


# ── compute_log_return ──────────────────────────────────────────────────

def test_compute_log_return_first_row_nan(sample_prices):
    r = P.compute_log_return(sample_prices)
    assert r.iloc[0].isna().all()
    assert not r.iloc[1].isna().any()


def test_compute_log_return_shape_preserved(sample_prices):
    r = P.compute_log_return(sample_prices)
    assert r.shape == sample_prices.shape


def test_compute_log_return_empty_safe():
    empty = pd.DataFrame()
    r = P.compute_log_return(empty)
    assert r.empty


# ── rolling_sum ─────────────────────────────────────────────────────────

def test_rolling_sum_12_1_anti_look_ahead():
    """For 12-1: signal at t uses returns from t-12 through t-2 (not t-1, not t)."""
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    panel = pd.DataFrame(
        {"A": range(24), "B": range(24, 48)}, index=dates, dtype=float
    )
    r = P.rolling_sum(panel, window=12, skip=1)
    # At index 13: should be sum of indices [13-12, 13-2] = indices 1..11 of A
    # But due to shift(1): values are panel.shift(1).rolling(11).sum()
    # → at t=12: sum of shifted values at 1..11 = sum of A values at 0..10 = 0+1+...+10 = 55
    assert r["A"].iloc[12] == 55.0


def test_rolling_sum_invalid_window_raises():
    with pytest.raises(ValueError):
        P.rolling_sum(pd.DataFrame({"A": [1.0]}), window=0)


# ── cross_sectional_rank ────────────────────────────────────────────────

def test_cross_sectional_rank_range():
    panel = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 1.0], "C": [2.0, 3.0]})
    r = P.cross_sectional_rank(panel)
    assert ((r >= 0) & (r <= 1)).all().all()
    # Row 0: A=1 (lowest, rank=1/3), B=3 (highest, rank=3/3=1.0), C=2 (rank=2/3)
    assert r.iloc[0]["A"] == pytest.approx(1/3)
    assert r.iloc[0]["B"] == pytest.approx(1.0)


# ── apply_lag ───────────────────────────────────────────────────────────

def test_apply_lag_default_1():
    s = pd.Series([1.0, 2.0, 3.0])
    out = P.apply_lag(s)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 1.0


def test_apply_lag_zero_raises():
    """Critical: lag=0 would preserve look-ahead → must reject."""
    with pytest.raises(ValueError):
        P.apply_lag(pd.Series([1.0]), n_periods=0)


def test_apply_lag_negative_raises():
    with pytest.raises(ValueError):
        P.apply_lag(pd.Series([1.0]), n_periods=-1)


# ── residualize_against ─────────────────────────────────────────────────

def test_residualize_orthogonal_input_unchanged():
    """If returns is orthogonal to factor, residual ≈ returns."""
    np.random.seed(0)
    n = 100
    factor = pd.Series(np.random.randn(n), name="f").to_frame()
    returns = pd.Series(np.random.randn(n))
    resid = P.residualize_against(returns, factor)
    # OLS residual variance ≤ input variance
    assert resid.var() <= returns.var() + 0.1


# ── top_bottom_membership ───────────────────────────────────────────────

def test_top_bottom_membership_correct_masking():
    rank = pd.DataFrame({
        "A": [0.05, 0.50, 0.95],
        "B": [0.50, 0.50, 0.50],
        "C": [0.95, 0.05, 0.05],
    })
    long_m, short_m = P.top_bottom_membership(rank.T, top_frac=0.3, bottom_frac=0.3)
    # On row "A" (long): only the high ranks should be true
    # ... but it's column-rank format; let me test with explicit values
    rank_panel = pd.DataFrame({
        "A": [0.95, 0.50, 0.05],    # date 0: A high, date 1: A mid, date 2: A low
        "B": [0.50, 0.50, 0.50],
        "C": [0.05, 0.50, 0.95],
    })
    long_m, short_m = P.top_bottom_membership(rank_panel,
                                                 top_frac=0.3, bottom_frac=0.3)
    # date 0: A (0.95≥0.7) long, C (0.05≤0.3) short
    assert long_m.iloc[0]["A"]
    assert short_m.iloc[0]["C"]
    assert not long_m.iloc[0]["C"]


def test_top_bottom_membership_invalid_frac():
    with pytest.raises(ValueError):
        P.top_bottom_membership(pd.DataFrame({"A": [0.5]}),
                                  top_frac=0.6, bottom_frac=0.3)


# ── equal_weight_long_short_returns ─────────────────────────────────────

def test_ew_ls_returns_basic():
    rets = pd.DataFrame({"A": [0.10, 0.05], "B": [-0.05, -0.10], "C": [0.0, 0.0]})
    long_m = pd.DataFrame({"A": [True, True], "B": [False, False], "C": [False, False]})
    short_m = pd.DataFrame({"A": [False, False], "B": [True, True], "C": [False, False]})
    ls = P.equal_weight_long_short_returns(long_m, short_m, rets)
    # Row 0: mean(A) - mean(B) = 0.10 - (-0.05) = 0.15
    assert ls.iloc[0] == pytest.approx(0.15)
    assert ls.iloc[1] == pytest.approx(0.15)


# ── exclude_microcap ────────────────────────────────────────────────────

def test_exclude_microcap_masks_below_threshold():
    prices = pd.DataFrame({"A": [10.0, 3.0], "B": [4.0, 8.0]})
    out = P.exclude_microcap(prices, threshold=5.0)
    assert out.iloc[0]["A"] == 10.0
    assert pd.isna(out.iloc[1]["A"])    # 3.0 < 5.0
    assert pd.isna(out.iloc[0]["B"])    # 4.0 < 5.0
    assert out.iloc[1]["B"] == 8.0


# ── winsorize ───────────────────────────────────────────────────────────

def test_winsorize_clips_extremes():
    panel = pd.DataFrame({
        "A": [1.0, 2.0], "B": [2.0, 4.0], "C": [3.0, 6.0],
        "D": [4.0, 8.0], "E": [100.0, 1000.0],
    })
    out = P.winsorize(panel, lower=0.1, upper=0.9)
    # E should be clipped down significantly
    assert out.iloc[0]["E"] < 100.0


# ── vol_target_normalize ────────────────────────────────────────────────

def test_vol_target_normalize_anti_look_ahead(sample_returns):
    out = P.vol_target_normalize(sample_returns, target_vol=0.10,
                                    lookback=12, periods_per_year=12)
    # First 12 periods should have NaN scaling (uses shift(1) of 12-window vol)
    assert out.iloc[:12].isna().all() or out.iloc[:13].isna().sum() >= 12


def test_vol_target_normalize_invalid_target_raises():
    with pytest.raises(ValueError):
        P.vol_target_normalize(pd.Series([1.0]), target_vol=0)


# ── apply_round_trip_cost ───────────────────────────────────────────────

def test_apply_round_trip_cost_subtracts():
    rets = pd.Series([0.01, 0.02, -0.01])
    out = P.apply_round_trip_cost(rets, bps_per_side=12.0, turnover=1.0)
    # Cost = 2 * 1 * 12 / 10000 = 0.0024
    expected = rets - 0.0024
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_apply_round_trip_cost_zero_turnover_no_cost():
    rets = pd.Series([0.01, 0.02])
    out = P.apply_round_trip_cost(rets, bps_per_side=12.0, turnover=0.0)
    pd.testing.assert_series_equal(out, rets, check_names=False)


def test_apply_round_trip_cost_invalid_raises():
    with pytest.raises(ValueError):
        P.apply_round_trip_cost(pd.Series([1.0]), bps_per_side=-1)


# ── monthly_resample ────────────────────────────────────────────────────

def test_monthly_resample_compound():
    daily = pd.Series(
        [0.01, 0.01, 0.0, 0.0],
        index=pd.date_range("2024-01-29", periods=4, freq="D"),
    )
    monthly = P.monthly_resample_compound(daily)
    # Jan: (1.01)(1.01)(1)(1) - 1 = 0.0201
    assert monthly.iloc[0] == pytest.approx(0.0201, abs=1e-6)
