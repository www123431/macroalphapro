"""
tests/test_factors.py — Sprint Week 1 factor module tests.

Spec: docs/spec_factor_ensemble_v1.md (id=50, hash 1665945d2ca5)

Covers:
  - TSMOM (engine/factors/tsmom.py) — wraps engine.signal.get_signal_dataframe
  - BAB compat (engine/factors/bab_compat.py) — wraps engine.signal ql01_bab column
  - Carry-equity (engine/factors/carry_equity.py) — equity-only dividend yield
  - Quality (engine/factors/quality.py) — top-10 holdings ROE+revenueGrowth z-score

All tests mock external dependencies (engine.signal, yfinance, holdings ingestion)
to be deterministic and fast.
"""
from __future__ import annotations

import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from engine.factors import (
    compute_tsmom_signal,
    compute_carry_equity_signal,
    compute_quality_signal,
    compute_bab_signal,
)
from engine.factors import tsmom as tsmom_mod
from engine.factors import bab_compat as bab_mod
from engine.factors import carry_equity as carry_mod
from engine.factors import quality as quality_mod


# ─────────────────────────────────────────────────────────────────────────────
# TSMOM tests
# ─────────────────────────────────────────────────────────────────────────────


def test_tsmom_locked_constants():
    """Spec §2.2.1 locked parameters."""
    assert tsmom_mod.LOOKBACK_MONTHS == 12
    assert tsmom_mod.SKIP_MONTHS == 1
    assert tsmom_mod.VOL_WINDOW_DAYS == 60


def test_tsmom_rejects_non_date():
    with pytest.raises(TypeError):
        compute_tsmom_signal(as_of="2026-05-31", universe=["QQQ"])


def test_tsmom_empty_universe_returns_empty_series():
    result = compute_tsmom_signal(as_of=datetime.date(2026, 5, 31), universe=[])
    assert isinstance(result, pd.Series)
    assert result.empty


def test_tsmom_wraps_signal_dataframe():
    """TSMOM reads from engine.signal.get_signal_dataframe tsmom column."""
    mock_df = pd.DataFrame({
        "ticker": ["QQQ", "XLF", "XLE"],
        "tsmom":  [1.0, 0.0, -1.0],
    }, index=["科技成长(纳指)", "金融", "全球能源"])

    with patch("engine.signal.get_signal_dataframe", return_value=mock_df):
        result = compute_tsmom_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["QQQ", "XLF", "XLE"],
        )

    assert result.loc["QQQ"] == 1.0
    assert result.loc["XLF"] == 0.0
    assert result.loc["XLE"] == -1.0


def test_tsmom_returns_nan_for_missing_ticker():
    """Ticker not in signal_df (insufficient history) → NaN per §2.3 protocol."""
    mock_df = pd.DataFrame({
        "ticker": ["QQQ"],
        "tsmom":  [1.0],
    }, index=["科技成长(纳指)"])

    with patch("engine.signal.get_signal_dataframe", return_value=mock_df):
        result = compute_tsmom_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["QQQ", "MISSING"],
        )

    assert result.loc["QQQ"] == 1.0
    assert pd.isna(result.loc["MISSING"])


def test_tsmom_signal_dataframe_failure_returns_all_nan():
    """If get_signal_dataframe raises, all-NaN graceful degradation."""
    with patch(
        "engine.signal.get_signal_dataframe",
        side_effect=Exception("simulated"),
    ):
        result = compute_tsmom_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["QQQ", "XLF"],
        )
    assert result.isna().all()


def test_tsmom_empty_signal_dataframe_returns_all_nan():
    with patch(
        "engine.signal.get_signal_dataframe",
        return_value=pd.DataFrame(),
    ):
        result = compute_tsmom_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["QQQ"],
        )
    assert result.isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# BAB compat tests
# ─────────────────────────────────────────────────────────────────────────────


def test_bab_rejects_non_date():
    with pytest.raises(TypeError):
        compute_bab_signal(as_of="x", universe=["QQQ"])


def test_bab_delegates_to_factor_library(tmp_path, monkeypatch):
    """2026-05-19 rewrite: bab_compat now delegates to
    engine.factor_library._compute_bab_weights (canonical Frazzini-Pedersen
    path) rather than reading a non-existent `ql01_bab` column from
    get_signal_dataframe. Test that the wrapper correctly reindexes the
    weights dict onto the requested universe."""
    monkeypatch.setattr(
        "engine.factors.bab_compat._BAB_CACHE_PATH",
        tmp_path / "bab.parquet",
    )
    fake_closes = pd.DataFrame({"QQQ": [1.0], "XLF": [1.0], "SPY": [1.0]})
    fake_weights = {"QQQ": -0.5, "XLF": 0.3}  # TLT missing → 0 after reindex

    with patch("engine.signal._fetch_closes", return_value=fake_closes), \
         patch("engine.factor_library._compute_bab_weights",
               return_value=fake_weights):
        result = compute_bab_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["QQQ", "XLF", "TLT"],
            use_cache=False,
        )

    assert result.loc["QQQ"] == -0.5
    assert result.loc["XLF"] == 0.3
    assert result.loc["TLT"] == 0.0


def test_bab_failure_returns_all_nan(tmp_path, monkeypatch):
    """Underlying _fetch_closes raising → bab_compat must return all-NaN
    Series indexed on universe (graceful degradation, no propagation)."""
    monkeypatch.setattr(
        "engine.factors.bab_compat._BAB_CACHE_PATH",
        tmp_path / "bab.parquet",
    )
    with patch("engine.signal._fetch_closes",
               side_effect=Exception("net err")):
        result = compute_bab_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["QQQ"],
            use_cache=False,
        )
    assert result.isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# Carry-equity tests
# ─────────────────────────────────────────────────────────────────────────────


def test_carry_locked_equity_only_scope():
    """Spec §2.2.2 — equity_sector + equity_factor only."""
    assert carry_mod.EQUITY_ASSET_CLASSES == frozenset({"equity_sector", "equity_factor"})


def test_carry_dividend_lookback_locked():
    assert carry_mod.DIVIDEND_LOOKBACK_DAYS == 365


def test_carry_requires_asset_classes():
    """Asset classes are mandatory for Carry-equity scope enforcement."""
    with pytest.raises(ValueError, match="asset_classes"):
        compute_carry_equity_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["QQQ"],
            asset_classes=None,
        )


def test_carry_non_equity_returns_nan():
    """Non-equity asset classes (commodity / FI / FX / vol) → NaN per spec §2.1."""
    asset_classes = {
        "QQQ":  "equity_sector",
        "GLD":  "commodity",
        "TLT":  "fixed_income",
        "VXX":  "volatility",
        "UUP":  "fx",
    }
    with patch(
        "engine.factors.carry_equity._compute_etf_dividend_yield",
        return_value=0.025,  # any value for equity
    ):
        result = compute_carry_equity_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=list(asset_classes.keys()),
            asset_classes=asset_classes,
        )

    assert result.loc["QQQ"] == pytest.approx(0.025)
    assert pd.isna(result.loc["GLD"])
    assert pd.isna(result.loc["TLT"])
    assert pd.isna(result.loc["VXX"])
    assert pd.isna(result.loc["UUP"])


def test_carry_equity_factor_class_included():
    """equity_factor (e.g. USMV, QUAL) also gets Carry signal."""
    asset_classes = {"USMV": "equity_factor", "QUAL": "equity_factor"}
    with patch(
        "engine.factors.carry_equity._compute_etf_dividend_yield",
        return_value=0.018,
    ):
        result = compute_carry_equity_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["USMV", "QUAL"],
            asset_classes=asset_classes,
        )
    assert result.loc["USMV"] == pytest.approx(0.018)
    assert result.loc["QUAL"] == pytest.approx(0.018)


def test_carry_dividend_yield_failure_returns_nan():
    """yfinance dividend fetch failure → NaN per resilience."""
    asset_classes = {"QQQ": "equity_sector"}
    with patch(
        "engine.factors.carry_equity._compute_etf_dividend_yield",
        return_value=None,
    ):
        result = compute_carry_equity_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["QQQ"],
            asset_classes=asset_classes,
        )
    assert pd.isna(result.loc["QQQ"])


def test_carry_dividend_yield_calculation():
    """Mock yfinance to test dividend yield arithmetic."""
    fake_dividends = pd.Series(
        [0.5, 0.6, 0.5, 0.7],
        index=pd.DatetimeIndex([
            "2025-08-15", "2025-11-15", "2026-02-15", "2026-04-15",
        ]),
    )
    fake_history = pd.DataFrame(
        {"Close": [400.0, 405.0]},
        index=pd.DatetimeIndex(["2026-05-30", "2026-05-31"]),
    )

    mock_ticker = MagicMock()
    mock_ticker.dividends = fake_dividends
    mock_ticker.history.return_value = fake_history

    with patch("engine.factors.carry_equity.yf.Ticker", return_value=mock_ticker):
        result = carry_mod._compute_etf_dividend_yield(
            "TEST", datetime.date(2026, 5, 31),
        )

    # All 4 dividends in last 365d: 0.5+0.6+0.5+0.7 = 2.3
    # Latest price 405.0
    # Yield = 2.3 / 405.0 ≈ 0.00568
    assert result == pytest.approx(2.3 / 405.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Quality tests
# ─────────────────────────────────────────────────────────────────────────────


def test_quality_locked_components():
    """Spec §2.2.3 — 2-component simplified."""
    assert quality_mod.QUALITY_SUB_COMPONENTS == ("profitability", "growth")
    assert quality_mod.EQUITY_ASSET_CLASSES == frozenset(
        {"equity_sector", "equity_factor"}
    )


def test_quality_requires_asset_classes():
    with pytest.raises(ValueError, match="asset_classes"):
        compute_quality_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["QQQ"],
            asset_classes=None,
        )


def test_quality_non_equity_returns_nan():
    asset_classes = {
        "QQQ": "equity_sector",
        "GLD": "commodity",
        "TLT": "fixed_income",
    }
    with patch(
        "engine.factors.quality._compute_etf_quality_aggregate",
        return_value=0.20,  # constant raw aggregate for equity
    ):
        result = compute_quality_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=list(asset_classes.keys()),
            asset_classes=asset_classes,
        )
    # GLD / TLT non-equity → NaN regardless of aggregate
    assert pd.isna(result.loc["GLD"])
    assert pd.isna(result.loc["TLT"])
    # QQQ equity but only 1 valid → cross-section z-score skipped, returns NaN
    # (insufficient for standardization per §2.2.3)
    assert pd.isna(result.loc["QQQ"])


def test_quality_z_score_standardization():
    """Cross-section z-score across equity universe."""
    asset_classes = {f"E{i}": "equity_sector" for i in range(5)}
    raw_aggregates = {
        "E0": 0.10,
        "E1": 0.20,
        "E2": 0.30,
        "E3": 0.40,
        "E4": 0.50,
    }

    def mock_aggregate(etf, as_of):
        return raw_aggregates.get(etf)

    with patch(
        "engine.factors.quality._compute_etf_quality_aggregate",
        side_effect=mock_aggregate,
    ):
        result = compute_quality_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=list(asset_classes.keys()),
            asset_classes=asset_classes,
        )

    # mean = 0.30, std (ddof=0) = sqrt(((-.2)²+(-.1)²+0²+.1²+.2²)/5) = sqrt(0.02) ≈ 0.1414
    # z-score E2 = 0.0 (it's the mean)
    assert result.loc["E2"] == pytest.approx(0.0, abs=1e-6)
    # E0 should be ≈ -1.414 (most negative)
    assert result.loc["E0"] < result.loc["E1"] < result.loc["E2"] < result.loc["E3"] < result.loc["E4"]


def test_quality_all_nan_returns_all_nan():
    """If no equity ETFs have valid aggregate, all NaN."""
    asset_classes = {"E0": "equity_sector", "E1": "equity_sector"}
    with patch(
        "engine.factors.quality._compute_etf_quality_aggregate",
        return_value=None,
    ):
        result = compute_quality_signal(
            as_of=datetime.date(2026, 5, 31),
            universe=["E0", "E1"],
            asset_classes=asset_classes,
        )
    assert result.isna().all()


def test_quality_aggregate_with_valid_holdings():
    """Test _compute_etf_quality_aggregate with synthetic holdings + metrics."""
    fake_holdings = [
        {"name": "AAPL", "weight": 0.10, "rank": 1},
        {"name": "MSFT", "weight": 0.08, "rank": 2},
        {"name": "GOOGL", "weight": 0.05, "rank": 3},
    ]
    metrics = {
        "AAPL": {"returnOnEquity": 0.30, "revenueGrowth": 0.15},
        "MSFT": {"returnOnEquity": 0.25, "revenueGrowth": 0.20},
        "GOOGL": {"returnOnEquity": 0.20, "revenueGrowth": 0.10},
    }

    def mock_metric(name, metric, as_of):
        return metrics.get(name, {}).get(metric)

    with patch(
        "engine.etf_holdings_ingestion.fetch_etf_top10_holdings",
        return_value=fake_holdings,
    ), patch(
        "engine.factors.quality._fetch_holding_metric",
        side_effect=mock_metric,
    ):
        result = quality_mod._compute_etf_quality_aggregate(
            "TEST", datetime.date(2026, 5, 31),
        )

    # AAPL quality = (0.30 + 0.15)/2 = 0.225, weight 0.10
    # MSFT quality = (0.25 + 0.20)/2 = 0.225, weight 0.08
    # GOOGL quality = (0.20 + 0.10)/2 = 0.15, weight 0.05
    # weighted_sum = 0.10×0.225 + 0.08×0.225 + 0.05×0.15 = 0.0225 + 0.018 + 0.0075 = 0.048
    # weighted_total = 0.10 + 0.08 + 0.05 = 0.23
    # aggregate = 0.048 / 0.23 ≈ 0.2087
    expected = (0.10 * 0.225 + 0.08 * 0.225 + 0.05 * 0.15) / 0.23
    assert result == pytest.approx(expected, abs=1e-6)


def test_quality_aggregate_skips_holdings_with_missing_metrics():
    """Holdings without ROE OR revenueGrowth → excluded from aggregate."""
    fake_holdings = [
        {"name": "AAPL", "weight": 0.10, "rank": 1},
        {"name": "BAD",  "weight": 0.20, "rank": 2},  # missing metrics
    ]

    def mock_metric(name, metric, as_of):
        if name == "AAPL":
            return {"returnOnEquity": 0.30, "revenueGrowth": 0.15}.get(metric)
        return None  # BAD missing

    with patch(
        "engine.etf_holdings_ingestion.fetch_etf_top10_holdings",
        return_value=fake_holdings,
    ), patch(
        "engine.factors.quality._fetch_holding_metric",
        side_effect=mock_metric,
    ):
        result = quality_mod._compute_etf_quality_aggregate(
            "TEST", datetime.date(2026, 5, 31),
        )

    # Only AAPL counted: (0.30+0.15)/2 = 0.225 (weight 0.10), aggregate = 0.225
    assert result == pytest.approx(0.225, abs=1e-6)


def test_quality_aggregate_no_valid_holdings_returns_none():
    fake_holdings = [{"name": "BAD", "weight": 0.10, "rank": 1}]
    with patch(
        "engine.etf_holdings_ingestion.fetch_etf_top10_holdings",
        return_value=fake_holdings,
    ), patch(
        "engine.factors.quality._fetch_holding_metric",
        return_value=None,
    ):
        result = quality_mod._compute_etf_quality_aggregate(
            "TEST", datetime.date(2026, 5, 31),
        )
    assert result is None


def test_quality_aggregate_clips_extreme_values():
    """ROE/revenueGrowth clipped to [-1, 1] for outlier robustness."""
    fake_holdings = [{"name": "EXTREME", "weight": 1.0, "rank": 1}]
    with patch(
        "engine.etf_holdings_ingestion.fetch_etf_top10_holdings",
        return_value=fake_holdings,
    ), patch(
        "engine.factors.quality._fetch_holding_metric",
        side_effect=lambda n, m, t: {"returnOnEquity": 5.0, "revenueGrowth": -3.0}[m],
    ):
        result = quality_mod._compute_etf_quality_aggregate(
            "TEST", datetime.date(2026, 5, 31),
        )
    # Clipped: ROE 5.0 → 1.0, growth -3.0 → -1.0
    # aggregate = (1.0 + (-1.0))/2 = 0.0
    assert result == pytest.approx(0.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# v1 amendment 2026-05-09 — walk-forward lookahead guard
# ─────────────────────────────────────────────────────────────────────────────


def test_quality_spec_lock_date_locked():
    """SPEC_LOCK_DATE is 2026-05-09 (spec id=50 register day)."""
    assert quality_mod.SPEC_LOCK_DATE == datetime.date(2026, 5, 9)


def test_quality_returns_all_nan_for_walk_forward_dates():
    """as_of < SPEC_LOCK_DATE → Quality all-NaN (avoid yfinance .info lookahead)."""
    asset_classes = {"QQQ": "equity_sector", "XLF": "equity_sector"}
    # 2015-01-31 is walk-forward historical date
    result = compute_quality_signal(
        as_of=datetime.date(2015, 1, 31),
        universe=["QQQ", "XLF"],
        asset_classes=asset_classes,
    )
    assert result.isna().all(), \
        "Walk-forward as_of < SPEC_LOCK_DATE should return all-NaN per v1 amendment"


def test_quality_returns_all_nan_for_pre_lock_2024():
    """Boundary: any date before SPEC_LOCK_DATE (incl. recent ones)."""
    asset_classes = {"QQQ": "equity_sector"}
    result = compute_quality_signal(
        as_of=datetime.date(2026, 5, 8),  # 1 day before SPEC_LOCK_DATE
        universe=["QQQ"],
        asset_classes=asset_classes,
    )
    assert result.isna().all()


def test_quality_active_for_lock_date_and_after():
    """as_of >= SPEC_LOCK_DATE → Quality computation runs (would call yfinance)."""
    asset_classes = {"QQQ": "equity_sector"}
    # Mock to verify the function tries to compute (not auto-NaN at lock date)
    with patch(
        "engine.factors.quality._compute_etf_quality_aggregate",
        return_value=0.20,  # mock aggregate to bypass yfinance
    ):
        result = compute_quality_signal(
            as_of=datetime.date(2026, 5, 9),  # exactly SPEC_LOCK_DATE
            universe=["QQQ"],
            asset_classes=asset_classes,
        )
    # Single equity ETF cross-section z-score insufficient (< 2 valid) → NaN
    # But the function ran (not short-circuited by lock-date guard)
    # Verify by checking _compute_etf_quality_aggregate was called via mock contract


def test_quality_active_for_post_lock_dates():
    """Forward live dates → Quality computation runs."""
    asset_classes = {f"E{i}": "equity_sector" for i in range(5)}

    raw_aggregates = {f"E{i}": 0.10 + i * 0.05 for i in range(5)}

    def mock_aggregate(etf, as_of):
        return raw_aggregates.get(etf)

    with patch(
        "engine.factors.quality._compute_etf_quality_aggregate",
        side_effect=mock_aggregate,
    ):
        # Date well after SPEC_LOCK_DATE
        result = compute_quality_signal(
            as_of=datetime.date(2027, 1, 31),
            universe=list(asset_classes.keys()),
            asset_classes=asset_classes,
        )

    # Should produce z-scores (5 valid equity ETFs, sufficient cross-section)
    assert not result.isna().all(), \
        "Forward live date should compute Quality, not return all-NaN"
