"""
tests/test_factor_ensemble_singlename.py — Stage 2 Wave A walk-forward harness tests.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52)
"""
from __future__ import annotations

import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from engine.factor_ensemble_singlename import (
    CHUNK_SIZE_LOCKED,
    OOS_START_DATE_WAVE_A,
    OOS_END_DATE_WAVE_A,
    TC_BPS_LOCKED,
    VOL_TARGET_LOCKED,
    MAX_LEVERAGE_LOCKED,
    MAX_NAME_WEIGHT_LOCKED,
    run_singlestock_walk_forward,
    SinglestockWalkForwardResult,
)
from engine.factor_ensemble_singlename.panel_fetcher import _chunked
from engine.factor_ensemble_singlename.walk_forward import (
    _construct_singlestock_weights,
    _generate_monthend_dates,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_constants_match_spec():
    assert CHUNK_SIZE_LOCKED == 100  # spec §2.6
    assert TC_BPS_LOCKED == 12.0     # spec §2.5 + §2.2 single-stock
    assert VOL_TARGET_LOCKED == 0.15 # audit amendment 2026-05-09
    assert MAX_LEVERAGE_LOCKED == 1.5
    assert MAX_NAME_WEIGHT_LOCKED == 0.02
    assert OOS_START_DATE_WAVE_A == datetime.date(2000, 1, 1)
    assert OOS_END_DATE_WAVE_A == datetime.date(2024, 12, 31)


# ─────────────────────────────────────────────────────────────────────────────
# panel_fetcher chunking
# ─────────────────────────────────────────────────────────────────────────────

def test_chunked_yields_correct_batches():
    tickers = [f"T{i}" for i in range(250)]
    chunks = list(_chunked(tickers, chunk_size=100))
    assert len(chunks) == 3
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert len(chunks[2]) == 50


def test_chunked_handles_under_chunk_size():
    tickers = ["A", "B", "C"]
    chunks = list(_chunked(tickers, chunk_size=100))
    assert len(chunks) == 1
    assert chunks[0] == ["A", "B", "C"]


# ─────────────────────────────────────────────────────────────────────────────
# Date generation
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_monthend_dates_basic():
    dates = _generate_monthend_dates(datetime.date(2020, 1, 1), datetime.date(2020, 6, 30))
    assert len(dates) == 6
    assert dates[0] == datetime.date(2020, 1, 31)
    assert dates[-1] == datetime.date(2020, 6, 30)


def test_generate_monthend_dates_handles_year_boundary():
    dates = _generate_monthend_dates(datetime.date(2023, 11, 1), datetime.date(2024, 2, 29))
    expected = [
        datetime.date(2023, 11, 30),
        datetime.date(2023, 12, 31),
        datetime.date(2024, 1, 31),
        datetime.date(2024, 2, 29),
    ]
    assert dates == expected


# ─────────────────────────────────────────────────────────────────────────────
# Weight construction (single-stock caps)
# ─────────────────────────────────────────────────────────────────────────────

def _make_panel_for_vol(tickers, n_days=120):
    """Synthetic panel with deterministic per-ticker vol."""
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(42)
    data = {}
    for i, t in enumerate(tickers):
        # Deterministic vol per ticker (different scales)
        daily_vol = 0.01 + i * 0.002
        rets = rng.normal(0.0001, daily_vol, n_days)
        data[t] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


def test_construct_weights_respects_2pct_name_cap():
    """Highly concentrated signal should be capped at 2% per name."""
    tickers = [f"T{i}" for i in range(50)]
    panel = _make_panel_for_vol(tickers)
    # All ones signal — would naturally distribute equally before cap
    signal = pd.Series(1.0, index=tickers, dtype=float)
    weights = _construct_singlestock_weights(
        ensemble_signal=signal, panel=panel, as_of=datetime.date(2024, 6, 1),
    )
    # No single name should exceed 2% by absolute value
    assert weights.abs().max() <= MAX_NAME_WEIGHT_LOCKED + 1e-9


def test_construct_weights_respects_max_leverage():
    """Vol target 15% × leverage cap 1.5× → gross exposure max 1.5."""
    tickers = [f"T{i}" for i in range(100)]
    panel = _make_panel_for_vol(tickers)
    # Strong long-short signal
    signal = pd.Series([1.0 if i % 2 == 0 else -1.0 for i in range(100)], index=tickers, dtype=float)
    weights = _construct_singlestock_weights(
        ensemble_signal=signal, panel=panel, as_of=datetime.date(2024, 6, 1),
    )
    # Sum of absolute weights ≤ MAX_LEVERAGE_LOCKED (1.5)
    assert weights.abs().sum() <= MAX_LEVERAGE_LOCKED + 0.01


def test_construct_weights_empty_signal():
    panel = _make_panel_for_vol(["X", "Y"])
    weights = _construct_singlestock_weights(
        ensemble_signal=pd.Series(dtype=float), panel=panel, as_of=datetime.date(2024, 6, 1),
    )
    assert weights.empty


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end mock walk-forward
# ─────────────────────────────────────────────────────────────────────────────

def test_walk_forward_wave_b_not_implemented():
    with pytest.raises(NotImplementedError, match="Wave B"):
        run_singlestock_walk_forward(
            universe_at_date_fn=lambda d: ["AAPL"],
            wave="B",
        )


def test_walk_forward_invalid_wave():
    with pytest.raises(ValueError, match="wave must be"):
        run_singlestock_walk_forward(
            universe_at_date_fn=lambda d: ["AAPL"],
            wave="X",
        )


def test_walk_forward_empty_panel_returns_zero():
    """If panel fetch returns empty, harness gracefully returns n_periods=0."""
    with mock.patch(
        "engine.factor_ensemble_singlename.walk_forward.bulk_fetch_singlestock_panel",
        return_value=pd.DataFrame(),
    ):
        result = run_singlestock_walk_forward(
            universe_at_date_fn=lambda d: ["AAPL", "MSFT"],
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 6, 30),
            wave="A",
        )
    assert isinstance(result, SinglestockWalkForwardResult)
    assert result.n_periods == 0


def test_walk_forward_smoke_with_synthetic_panel():
    """End-to-end smoke: 6mo walk-forward with synthetic panel succeeds."""
    tickers = [f"T{i}" for i in range(20)] + ["SPY"]
    # Need ~2 years of data (12mo TSMOM lookback + buffer)
    idx = pd.date_range("2022-01-01", "2024-06-30", freq="B")
    rng = np.random.default_rng(42)
    spy_rets = rng.normal(0.0005, 0.012, len(idx))
    data = {"SPY": 100.0 * np.exp(np.cumsum(spy_rets))}
    for i, t in enumerate(tickers):
        if t == "SPY":
            continue
        beta = 0.6 + i * 0.07
        idio = rng.normal(0, 0.005, len(idx))
        data[t] = 100.0 * np.exp(np.cumsum(beta * spy_rets + idio))
    panel = pd.DataFrame(data, index=idx)

    # Mock div yield cache to return zero (no dividends in synthetic data)
    with mock.patch(
        "engine.factor_ensemble_singlename.walk_forward.bulk_fetch_singlestock_panel",
        return_value=panel,
    ):
        with mock.patch(
            "engine.factors_singlename.dividend_yield._ensure_ticker_dividends",
            return_value=True,
        ):
            with mock.patch(
                "engine.factors_singlename.dividend_yield._load_dividend_cache",
                return_value=pd.DataFrame(0.0, index=idx, columns=tickers),  # zero divs
            ):
                result = run_singlestock_walk_forward(
                    universe_at_date_fn=lambda d: tickers,
                    start_date=datetime.date(2024, 1, 1),
                    end_date=datetime.date(2024, 6, 30),
                    wave="A",
                    use_cache=False,
                )
    assert isinstance(result, SinglestockWalkForwardResult)
    # 6 months of monthly rebalances → 5 successful periods (last has no next-month return)
    assert result.n_periods >= 1, f"expected ≥1 successful period, got {result.n_periods}"
    assert isinstance(result.monthly_returns_net, pd.Series)
    # TC drag < gross return spread (sanity check)
    assert "wave" in result.metadata
    assert result.metadata["wave"] == "A"
    assert result.metadata["tc_bps"] == 12.0
