"""
tests/test_walk_forward_bulk_fetch.py — Bulk-fetch + disk cache patch tests.

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50) §四 amendment 2026-05-09
(post-Gate-0 infrastructure speedup; clarification +0 trials).

Verifies:
  • _bulk_prefetch_panel: cache hit path (no yfinance call), cache miss path (1 bulk call)
  • _panel_slice: empty-panel safety, ticker-subset filtering, date-range filtering
  • _fetch_realized_vols(panel=...): 0 yfinance calls when panel covers tickers + range
  • _fetch_realized_vols(panel=None): falls back to per-ticker fetch (existing slow path)
  • _compute_realized_return(panel=...): same fast/slow path bifurcation
  • Backwards compat: panel=None default preserves existing test suite behavior
"""
from __future__ import annotations

import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from engine.factor_ensemble_walk_forward import (
    _bulk_prefetch_panel,
    _panel_slice,
    _fetch_inv_vols,
    _fetch_realized_vols,
    _compute_realized_return,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_panel(tickers, start, end):
    """Build a deterministic synthetic price panel covering full [start, end] range."""
    idx = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(42)
    data = {}
    for t in tickers:
        # Geometric random walk starting at 100
        rets = rng.normal(0.0005, 0.012, len(idx))
        prices = 100.0 * np.exp(np.cumsum(rets))
        data[t] = prices
    return pd.DataFrame(data, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# _panel_slice
# ─────────────────────────────────────────────────────────────────────────────

def test_panel_slice_returns_empty_for_none_panel():
    out = _panel_slice(None, ["XLF"], datetime.date(2020, 1, 1), datetime.date(2020, 6, 30))
    assert out.empty


def test_panel_slice_returns_empty_for_empty_dataframe():
    out = _panel_slice(pd.DataFrame(), ["XLF"], datetime.date(2020, 1, 1), datetime.date(2020, 6, 30))
    assert out.empty


def test_panel_slice_filters_tickers_and_dates():
    panel = _make_panel(["XLF", "XLE", "QQQ"], "2020-01-01", "2020-12-31")
    out = _panel_slice(panel, ["XLF", "QQQ"],
                      datetime.date(2020, 3, 1), datetime.date(2020, 4, 30))
    assert set(out.columns) == {"XLF", "QQQ"}
    assert out.index.min() >= pd.Timestamp("2020-03-01")
    assert out.index.max() <= pd.Timestamp("2020-04-30")
    assert "XLE" not in out.columns


def test_panel_slice_skips_missing_tickers():
    panel = _make_panel(["XLF"], "2020-01-01", "2020-06-30")
    # NONEXIST not in panel → only XLF returned
    out = _panel_slice(panel, ["XLF", "NONEXIST"],
                      datetime.date(2020, 1, 1), datetime.date(2020, 6, 30))
    assert list(out.columns) == ["XLF"]


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_realized_vols fast path (panel given) — must NOT call yfinance
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_realized_vols_fast_path_no_yfinance_call():
    panel = _make_panel(["XLF", "XLE"], "2019-01-01", "2020-12-31")

    with mock.patch("engine.signal._fetch_closes") as m_yf:
        result = _fetch_realized_vols(
            ["XLF", "XLE"],
            datetime.date(2020, 6, 30),
            panel=panel,
        )
        assert not m_yf.called, "fast path must not call _fetch_closes"

    assert isinstance(result, pd.Series)
    assert "XLF" in result.index and "XLE" in result.index
    # vol should be roughly the synthetic 0.012 daily * sqrt(252) ≈ 0.19
    assert 0.05 < result["XLF"] < 0.40, f"unrealistic vol: {result['XLF']}"


def test_fetch_realized_vols_fast_path_skips_missing_ticker():
    panel = _make_panel(["XLF"], "2019-01-01", "2020-12-31")
    result = _fetch_realized_vols(
        ["XLF", "NOTINPANEL"],
        datetime.date(2020, 6, 30),
        panel=panel,
    )
    assert "XLF" in result.index
    assert "NOTINPANEL" not in result.index


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_realized_vols slow path (panel=None) — backwards compat
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_realized_vols_slow_path_when_panel_none():
    """Slow path must still call _fetch_closes when panel not provided."""
    fake_closes = pd.DataFrame(
        {"XLF": np.linspace(100, 110, 100)},
        index=pd.date_range("2020-01-01", periods=100, freq="B"),
    )
    with mock.patch("engine.signal._fetch_closes", return_value=fake_closes) as m_yf:
        result = _fetch_realized_vols(["XLF"], datetime.date(2020, 6, 30), panel=None)
        assert m_yf.called, "slow path should call _fetch_closes"


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_inv_vols fast path (delegates to _fetch_realized_vols)
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_inv_vols_fast_path_no_yfinance_call():
    panel = _make_panel(["XLF"], "2019-01-01", "2020-12-31")
    with mock.patch("engine.signal._fetch_closes") as m_yf:
        result = _fetch_inv_vols(["XLF"], datetime.date(2020, 6, 30), panel=panel)
        assert not m_yf.called

    assert "XLF" in result.index
    assert result["XLF"] > 0  # 1/vol must be positive


# ─────────────────────────────────────────────────────────────────────────────
# _compute_realized_return fast path (panel given)
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_realized_return_fast_path_no_yfinance_call():
    panel = _make_panel(["XLF", "XLE"], "2019-01-01", "2020-12-31")
    weights = pd.Series({"XLF": 0.6, "XLE": -0.4})

    with mock.patch("engine.signal._fetch_closes") as m_yf:
        ret = _compute_realized_return(
            weights=weights,
            period_start=datetime.date(2020, 6, 30),
            period_end=datetime.date(2020, 7, 31),
            panel=panel,
        )
        assert not m_yf.called

    # Return must be finite and reasonable
    assert isinstance(ret, float)
    assert -0.50 < ret < 0.50, f"unrealistic monthly return: {ret}"


def test_compute_realized_return_zero_weight_skipped():
    panel = _make_panel(["XLF"], "2019-01-01", "2020-12-31")
    weights = pd.Series({"XLF": 0.0})  # zero weight
    ret = _compute_realized_return(
        weights=weights,
        period_start=datetime.date(2020, 6, 30),
        period_end=datetime.date(2020, 7, 31),
        panel=panel,
    )
    assert ret == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _bulk_prefetch_panel cache round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_bulk_prefetch_panel_cache_hit_no_yfinance_call(tmp_path, monkeypatch):
    """Cache hit (panel covers requested range × tickers) → no yf.download call."""
    # Redirect cache to tmp_path
    cache_path = tmp_path / "_yf_cache.parquet"
    monkeypatch.setattr("engine.factor_ensemble_walk_forward._PRICE_PANEL_CACHE", cache_path)

    # Pre-populate cache that covers range + tickers
    panel = _make_panel(["XLF", "XLE"], "2018-01-01", "2025-12-31")
    panel.to_parquet(cache_path)

    with mock.patch("yfinance.download") as m_yf:
        result = _bulk_prefetch_panel(
            tickers=["XLF", "XLE"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 12, 31),
        )
        assert not m_yf.called, "cache hit must not call yf.download"
    assert "XLF" in result.columns and "XLE" in result.columns


def test_bulk_prefetch_panel_cache_miss_calls_yfinance(tmp_path, monkeypatch):
    """Cache miss (no cache file) → 1 bulk yf.download call."""
    cache_path = tmp_path / "_yf_cache.parquet"
    monkeypatch.setattr("engine.factor_ensemble_walk_forward._PRICE_PANEL_CACHE", cache_path)

    fake_panel = _make_panel(["XLF"], "2019-01-01", "2025-12-31")
    fake_yf_response = pd.concat({"Close": fake_panel}, axis=1)

    with mock.patch("yfinance.download", return_value=fake_yf_response) as m_yf:
        result = _bulk_prefetch_panel(
            tickers=["XLF"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 12, 31),
        )
        assert m_yf.call_count == 1, "cache miss should call yf.download exactly once"
    # Cache file persisted
    assert cache_path.exists()
    # Re-read produces same columns
    reloaded = pd.read_parquet(cache_path)
    assert "XLF" in reloaded.columns


def test_bulk_prefetch_panel_returns_empty_when_yfinance_fails(tmp_path, monkeypatch):
    """yf.download raises → returns existing cache if present, else empty DataFrame."""
    cache_path = tmp_path / "_yf_cache.parquet"
    monkeypatch.setattr("engine.factor_ensemble_walk_forward._PRICE_PANEL_CACHE", cache_path)

    with mock.patch("yfinance.download", side_effect=RuntimeError("network failure")):
        result = _bulk_prefetch_panel(
            tickers=["XLF"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 12, 31),
        )
        assert result.empty
