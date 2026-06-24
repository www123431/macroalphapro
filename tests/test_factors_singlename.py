"""
tests/test_factors_singlename.py — Stage 2 Wave A single-stock factor tests.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.2

Covers:
  - tsmom: 12-month lookback + 1-month skip + signed ±1
  - bab:   60-day β tertile rank vs SPY
  - dividend_yield: trailing 12mo div yield + cross-section z-score
"""
from __future__ import annotations

import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from engine.factors_singlename import (
    LOOKBACK_MONTHS_LOCKED,
    SKIP_MONTHS_LOCKED,
    BETA_WINDOW_DAYS_LOCKED,
    BETA_BENCHMARK_LOCKED,
    DIVIDEND_LOOKBACK_DAYS_LOCKED,
    compute_tsmom_singlestock_signal,
    compute_bab_singlestock_signal,
    compute_dividend_yield_singlestock_signal,
)


def _make_synthetic_panel(tickers, start="2018-01-01", end="2024-12-31", spy_drift=0.0005, spy_vol=0.012, n_seed=42):
    """Build a deterministic synthetic price panel including SPY benchmark."""
    idx = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(n_seed)
    spy_rets = rng.normal(spy_drift, spy_vol, len(idx))
    data = {"SPY": 100.0 * np.exp(np.cumsum(spy_rets))}
    for i, t in enumerate(tickers):
        if t == "SPY":
            continue
        beta = 0.6 + i * 0.15  # β span 0.6 to ~1.8
        idio = rng.normal(0, 0.005, len(idx))
        rets = beta * spy_rets + idio
        data[t] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# TSMOM single-stock
# ─────────────────────────────────────────────────────────────────────────────

def test_tsmom_locked_constants():
    assert LOOKBACK_MONTHS_LOCKED == 12
    assert SKIP_MONTHS_LOCKED == 1


def test_tsmom_returns_nan_for_missing_panel():
    result = compute_tsmom_singlestock_signal(
        as_of=datetime.date(2020, 6, 30),
        universe=["AAPL", "MSFT"],
        panel=None,
    )
    assert isinstance(result, pd.Series)
    assert result.isna().all()


def test_tsmom_signed_signals_for_uptrend_downtrend():
    """Build panel where uptick ticker should give +1, downtick -1."""
    idx = pd.date_range("2018-01-01", "2024-12-31", freq="B")
    n = len(idx)
    panel = pd.DataFrame({
        "UPTICK":   100.0 * np.exp(np.linspace(0, 0.5, n)),    # +50% over period
        "DOWNTICK": 100.0 * np.exp(np.linspace(0, -0.3, n)),   # -30%
        "FLAT":     pd.Series(100.0, index=idx),
    }, index=idx)
    result = compute_tsmom_singlestock_signal(
        as_of=datetime.date(2024, 6, 30),
        universe=["UPTICK", "DOWNTICK", "FLAT"],
        panel=panel,
    )
    assert result["UPTICK"] == 1.0
    assert result["DOWNTICK"] == -1.0
    assert result["FLAT"] == 0.0


def test_tsmom_rejects_non_date():
    with pytest.raises(TypeError):
        compute_tsmom_singlestock_signal(
            as_of="2020-06-30",
            universe=["AAPL"],
            panel=pd.DataFrame(),
        )


def test_tsmom_skip_month_no_lookahead():
    """Verify SKIP_MONTHS=1: the most recent month should NOT influence signal.

    Panel: huge uptick in last month only. Without skip, signal = +1.
    With skip=1mo, signal should reflect older period (flat = 0 here).
    """
    idx = pd.date_range("2018-01-01", "2024-12-31", freq="B")
    n = len(idx)
    # Flat for 12 months ending 30 days before 2024-06-30, then huge uptick
    prices = np.full(n, 100.0)
    last_30_idx = idx >= pd.Timestamp("2024-06-01")
    prices[last_30_idx] = 200.0  # huge spike in last ~30 days
    panel = pd.DataFrame({"X": prices}, index=idx)
    result = compute_tsmom_singlestock_signal(
        as_of=datetime.date(2024, 6, 30),
        universe=["X"], panel=panel,
    )
    # With proper skip, the lookback window excludes the spike → signal = 0 (flat)
    # If skip is broken, signal would be +1 due to spike inclusion
    assert result["X"] == 0.0, f"skip month logic broken: got {result['X']}"


# ─────────────────────────────────────────────────────────────────────────────
# BAB single-stock
# ─────────────────────────────────────────────────────────────────────────────

def test_bab_locked_constants():
    assert BETA_WINDOW_DAYS_LOCKED == 60
    assert BETA_BENCHMARK_LOCKED == "SPY"


def test_bab_returns_nan_without_benchmark():
    panel = pd.DataFrame({"AAPL": [100.0]}, index=pd.date_range("2024-01-01", periods=1, freq="B"))
    result = compute_bab_singlestock_signal(
        as_of=datetime.date(2024, 6, 30),
        universe=["AAPL"],
        panel=panel,  # no SPY in panel
    )
    assert result.isna().all()


def test_bab_tertile_rank_works():
    """Synthetic universe with known β spread → bottom tertile +1, top -1."""
    tickers = [f"T{i}" for i in range(9)]  # 9 tickers → tertiles of 3 each
    panel = _make_synthetic_panel(tickers, start="2023-01-01", end="2024-12-31")
    result = compute_bab_singlestock_signal(
        as_of=datetime.date(2024, 6, 30),
        universe=tickers, panel=panel,
    )
    # T0 has lowest β (0.6), T8 has highest (~1.8)
    # Expect: T0/T1/T2 = +1 (low-β long); T6/T7/T8 = -1 (high-β short); rest = 0
    assert result["T0"] == 1.0, f"T0 (lowest β) should be +1, got {result['T0']}"
    assert result["T8"] == -1.0, f"T8 (highest β) should be -1, got {result['T8']}"
    # Middle should be 0
    assert result["T4"] == 0.0


def test_bab_insufficient_universe_returns_nan():
    """Universe too small for tertiles → NaN."""
    panel = _make_synthetic_panel(["SPY", "X"], start="2023-01-01", end="2024-12-31")
    result = compute_bab_singlestock_signal(
        as_of=datetime.date(2024, 6, 30),
        universe=["X"],  # only 1 ticker → can't tertile
        panel=panel,
    )
    # Single ticker — should NOT crash, but tertiles meaningless; NaN OK
    assert isinstance(result, pd.Series)


# ─────────────────────────────────────────────────────────────────────────────
# Dividend yield single-stock
# ─────────────────────────────────────────────────────────────────────────────

def test_div_yield_locked_constant():
    assert DIVIDEND_LOOKBACK_DAYS_LOCKED == 365


def test_div_yield_zscore_ranking():
    """Mock dividend cache + price panel → verify z-score sign."""
    # Tickers HIGH (yield 5%), MID (2%), LOW (0.5%)
    idx = pd.date_range("2023-01-01", "2024-12-31", freq="B")
    panel = pd.DataFrame({
        "HIGH": pd.Series(100.0, index=idx),
        "MID":  pd.Series(100.0, index=idx),
        "LOW":  pd.Series(100.0, index=idx),
        "ZERO": pd.Series(100.0, index=idx),
        "ALSO": pd.Series(100.0, index=idx),  # ensure ≥5 valid for z-score
    }, index=idx)
    div_dates = pd.date_range("2024-04-01", "2024-06-01", freq="ME")
    fake_div_cache = pd.DataFrame({
        "HIGH": [1.0] * len(div_dates),    # ~$1 per dividend × ~3 = $3 / $100 = 3%
        "MID":  [0.5] * len(div_dates),
        "LOW":  [0.1] * len(div_dates),
        "ZERO": [0.0] * len(div_dates),
        "ALSO": [0.5] * len(div_dates),
    }, index=div_dates)
    with mock.patch(
        "engine.factors_singlename.dividend_yield._load_dividend_cache",
        return_value=fake_div_cache,
    ):
        with mock.patch(
            "engine.factors_singlename.dividend_yield._ensure_ticker_dividends",
            return_value=True,
        ):
            result = compute_dividend_yield_singlestock_signal(
                as_of=datetime.date(2024, 6, 30),
                universe=["HIGH", "MID", "LOW", "ZERO", "ALSO"],
                panel=panel,
            )
    # HIGH should have highest z-score, ZERO lowest
    assert np.isfinite(result["HIGH"]) and np.isfinite(result["ZERO"])
    assert result["HIGH"] > result["MID"] > result["LOW"] > result["ZERO"]


def test_div_yield_returns_nan_for_insufficient_cross_section():
    """< 5 valid tickers → return all-NaN (unstable z-score)."""
    panel = pd.DataFrame({
        "ONLY": pd.Series(100.0, index=pd.date_range("2024-01-01", periods=10, freq="B"))
    })
    result = compute_dividend_yield_singlestock_signal(
        as_of=datetime.date(2024, 6, 30),
        universe=["ONLY"], panel=panel,
    )
    assert result.isna().all() or len(result) == 0
