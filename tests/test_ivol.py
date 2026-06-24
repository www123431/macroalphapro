"""
tests/test_ivol.py — unit tests for engine.factors_singlename.ivol (F-LAB-E2).

Coverage:
  - Locked constants match design lock
  - Math correctness (synthetic panel with known β + known residual vol)
  - Cross-section z-score properties (mean=0, std=1, n=universe)
  - NaN propagation (missing ticker, missing benchmark, insufficient data)
  - API parity with Wave A factors (asset_classes optional, panel optional)
  - register_factor() called at module load → factor in registry
"""
from __future__ import annotations

import datetime
import inspect

import numpy as np
import pandas as pd
import pytest

from engine.factors_singlename import ivol
from engine.factors_singlename.ivol import (
    IVOL_BENCHMARK_LOCKED,
    IVOL_MIN_OBS_RATIO_LOCKED,
    IVOL_WINDOW_DAYS_LOCKED,
    MIN_UNIVERSE_FOR_ZSCORE_LOCKED,
    TRADING_DAYS_PER_YEAR,
    compute_ivol_singlestock_signal,
)


# ── Locked constants ────────────────────────────────────────────────────────
def test_locked_constants_match_design() -> None:
    assert IVOL_WINDOW_DAYS_LOCKED == 60
    assert IVOL_BENCHMARK_LOCKED == "SPY"
    assert IVOL_MIN_OBS_RATIO_LOCKED == 0.5
    assert TRADING_DAYS_PER_YEAR == 252
    assert MIN_UNIVERSE_FOR_ZSCORE_LOCKED == 5


# ── Helper: build a synthetic price panel with known structure ──────────────
def _make_synthetic_panel(
    tickers: list[str],
    as_of:   datetime.date,
    n_days:  int = 120,
    seed:    int = 42,
) -> pd.DataFrame:
    """Geometric Brownian motion panel with SPY benchmark."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp(as_of), periods=n_days, freq="B")
    panel = pd.DataFrame(index=dates, columns=sorted(set(tickers + ["SPY"])), dtype=float)
    daily_drift, daily_vol = 0.08 / 252.0, 0.20 / np.sqrt(252.0)
    for t in panel.columns:
        rets = rng.normal(daily_drift, daily_vol, n_days)
        panel[t] = 100.0 * np.exp(np.cumsum(rets))
    return panel


# ── Happy path — basic shape ────────────────────────────────────────────────
def test_ivol_returns_zscore_series_correct_index() -> None:
    universe = [f"T{i:02d}" for i in range(8)]
    panel = _make_synthetic_panel(universe, datetime.date(2024, 6, 28))
    sig = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe,
        panel=panel,
    )
    assert isinstance(sig, pd.Series)
    assert set(sig.index) == set(universe)
    assert sig.notna().all()


def test_ivol_zscore_sample_properties_n10() -> None:
    """Cross-section z-score across 10 tickers should have mean ≈ 0, std ≈ 1."""
    universe = [f"T{i:02d}" for i in range(10)]
    panel = _make_synthetic_panel(universe, datetime.date(2024, 6, 28))
    sig = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel,
    )
    valid = sig.dropna()
    assert len(valid) == 10
    assert abs(valid.mean()) < 1e-9
    assert abs(valid.std(ddof=1) - 1.0) < 1e-9


def test_ivol_deterministic_for_same_panel() -> None:
    universe = [f"T{i:02d}" for i in range(8)]
    panel = _make_synthetic_panel(universe, datetime.date(2024, 6, 28), seed=7)
    s1 = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    s2 = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    pd.testing.assert_series_equal(s1, s2)


# ── Math correctness — controlled synthetic panel ───────────────────────────
def test_ivol_math_zero_residual_means_low_zscore() -> None:
    """A ticker that perfectly tracks SPY (β=1, residual=0) should have IVOL≈0,
    so its cross-section z-score should be the lowest (most negative) in the panel.
    """
    rng = np.random.default_rng(123)
    n_days = 120
    dates = pd.bdate_range(end="2024-06-28", periods=n_days, freq="B")
    spy_rets = rng.normal(0.0, 0.01, n_days)
    spy_prices = 100.0 * np.exp(np.cumsum(spy_rets))

    universe = ["NOISY1", "NOISY2", "NOISY3", "NOISY4", "PERFECT", "NOISY5"]
    panel = pd.DataFrame(index=dates, dtype=float)
    panel["SPY"] = spy_prices
    # PERFECT tracks SPY exactly (returns identical → residual = 0)
    panel["PERFECT"] = spy_prices.copy()
    # 5 NOISY tickers with extra idiosyncratic noise
    for i, t in enumerate([n for n in universe if n != "PERFECT"]):
        noise = rng.normal(0.0, 0.02, n_days)
        rets_t = spy_rets + noise
        panel[t] = 100.0 * np.exp(np.cumsum(rets_t))

    sig = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel,
    )
    # PERFECT should have the most negative z-score (lowest IVOL)
    assert sig["PERFECT"] < 0
    assert sig["PERFECT"] == sig.min()


def test_ivol_math_high_residual_means_high_zscore() -> None:
    """Ticker with extra noise but β=1 should have higher IVOL → higher z-score."""
    rng = np.random.default_rng(7)
    n_days = 120
    dates = pd.bdate_range(end="2024-06-28", periods=n_days, freq="B")
    spy_rets = rng.normal(0.0, 0.01, n_days)
    spy_prices = 100.0 * np.exp(np.cumsum(spy_rets))

    universe = ["LOW1", "LOW2", "LOW3", "LOW4", "LOW5", "HIGH"]
    panel = pd.DataFrame(index=dates, dtype=float)
    panel["SPY"] = spy_prices
    # LOW: small noise; HIGH: large noise
    for t in universe:
        noise_scale = 0.04 if t == "HIGH" else 0.005
        noise = rng.normal(0.0, noise_scale, n_days)
        panel[t] = 100.0 * np.exp(np.cumsum(spy_rets + noise))

    sig = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28),
        universe=universe, panel=panel,
    )
    # HIGH should have the highest z-score (highest IVOL)
    assert sig["HIGH"] == sig.max()
    assert sig["HIGH"] > 0


# ── NaN propagation ─────────────────────────────────────────────────────────
def test_missing_ticker_returns_nan() -> None:
    universe = [f"T{i:02d}" for i in range(8)] + ["MISSING"]
    panel = _make_synthetic_panel([f"T{i:02d}" for i in range(8)], datetime.date(2024, 6, 28))
    sig = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    assert pd.isna(sig["MISSING"])
    assert sig.dropna().shape[0] == 8


def test_missing_benchmark_returns_all_nan() -> None:
    universe = [f"T{i:02d}" for i in range(8)]
    panel = pd.DataFrame(index=pd.bdate_range(end="2024-06-28", periods=120, freq="B"))
    for t in universe:
        panel[t] = 100.0 + np.arange(120) * 0.1  # no SPY
    sig = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    assert sig.isna().all()


def test_min_universe_gate_below_5() -> None:
    universe = ["A", "B", "C"]   # only 3 < 5
    panel = _make_synthetic_panel(universe, datetime.date(2024, 6, 28))
    sig = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=panel,
    )
    assert sig.isna().all()


def test_empty_universe_returns_empty() -> None:
    panel = _make_synthetic_panel(["X"], datetime.date(2024, 6, 28))
    sig = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=[], panel=panel,
    )
    assert sig.empty


def test_empty_panel_returns_all_nan() -> None:
    universe = [f"T{i:02d}" for i in range(8)]
    sig = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=pd.DataFrame(),
    )
    assert sig.isna().all()


def test_invalid_as_of_type() -> None:
    with pytest.raises(TypeError, match="as_of must be datetime.date"):
        compute_ivol_singlestock_signal(
            as_of="2024-06-28",  # type: ignore
            universe=["A"], panel=pd.DataFrame(),
        )


def test_insufficient_lookback_returns_all_nan() -> None:
    """Panel with < min_obs trading days → all NaN."""
    universe = [f"T{i:02d}" for i in range(8)]
    short_panel = _make_synthetic_panel(universe, datetime.date(2024, 6, 28), n_days=10)
    sig = compute_ivol_singlestock_signal(
        as_of=datetime.date(2024, 6, 28), universe=universe, panel=short_panel,
    )
    assert sig.isna().all()


# ── API parity with Wave A factors ──────────────────────────────────────────
def test_api_parity_with_dividend_yield_and_bab() -> None:
    """IVOL should be drop-in replaceable in Wave A walk-forward / mining_runner."""
    from engine.factors_singlename.dividend_yield import (
        compute_dividend_yield_singlestock_signal,
    )
    from engine.factors_singlename.bab import compute_bab_singlestock_signal

    common = {"as_of", "universe", "asset_classes", "panel"}
    for name, fn in [
        ("dividend_yield", compute_dividend_yield_singlestock_signal),
        ("bab",            compute_bab_singlestock_signal),
        ("ivol",           compute_ivol_singlestock_signal),
    ]:
        sig = inspect.signature(fn)
        assert common.issubset(set(sig.parameters)), f"{name}: {sig}"


# ── Registration into Tier 1 mining content layer ──────────────────────────
def test_ivol_registered_in_singlename_factor_library() -> None:
    """Module-load-time register_factor() call should populate the registry."""
    from engine.factor_library_singlename import get_factor

    # Force populate by calling get_factor (lazy import path)
    spec = get_factor("ivol_singlestock")
    assert spec.factor_id == "ivol_singlestock"
    assert spec.asset_class == "equity_singlename"
    assert spec.expected_sign == -1   # AHX-Z 2006 prior
    assert spec.signal_fn is compute_ivol_singlestock_signal
    assert "Ang" in spec.citation and "Hodrick" in spec.citation


def test_ivol_factor_id_in_list_factors() -> None:
    from engine.factor_library_singlename import list_factors
    assert "ivol_singlestock" in list_factors()
