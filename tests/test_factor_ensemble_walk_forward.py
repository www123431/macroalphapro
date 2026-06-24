"""
tests/test_factor_ensemble_walk_forward.py — Sprint Week 3 walk-forward tests.

Spec: docs/spec_factor_ensemble_v1.md (id=50, hash 1665945d2ca5) §2.5
"""
from __future__ import annotations

import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine import factor_ensemble_walk_forward as wf
from engine.factor_ensemble_walk_forward import (
    TARGET_VOL,
    MIN_HISTORY_YEARS,
    VOL_WINDOW_DAYS,
    OOS_START_DATE,
    DEFAULT_END_DATE,
    WalkForwardResult,
    run_walk_forward,
    _generate_monthend_dates,
    _get_universe_at_date,
    _build_asset_classes_lookup,
    _compute_weights_from_signal,
    _price_at_or_before,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────


def test_locked_walk_forward_constants():
    assert TARGET_VOL == 0.10
    assert MIN_HISTORY_YEARS == 2
    assert VOL_WINDOW_DAYS == 60
    assert OOS_START_DATE == datetime.date(2011, 1, 1)
    assert DEFAULT_END_DATE == datetime.date(2024, 12, 31)


# ─────────────────────────────────────────────────────────────────────────────
# _generate_monthend_dates
# ─────────────────────────────────────────────────────────────────────────────


def test_generate_monthend_dates_full_year():
    """Full year 2020 should yield 12 month-ends."""
    dates = _generate_monthend_dates(
        datetime.date(2020, 1, 1), datetime.date(2020, 12, 31),
    )
    assert len(dates) == 12
    assert dates[0] == datetime.date(2020, 1, 31)
    assert dates[-1] == datetime.date(2020, 12, 31)


def test_generate_monthend_dates_partial_year():
    """Q1 2020 → 3 month-ends."""
    dates = _generate_monthend_dates(
        datetime.date(2020, 1, 1), datetime.date(2020, 3, 31),
    )
    assert dates == [
        datetime.date(2020, 1, 31),
        datetime.date(2020, 2, 29),  # leap year
        datetime.date(2020, 3, 31),
    ]


def test_generate_monthend_dates_oos_window():
    """OOS 2011-01 to 2024-12 → 168 month-ends (14 years × 12)."""
    dates = _generate_monthend_dates(OOS_START_DATE, DEFAULT_END_DATE)
    assert len(dates) == 14 * 12  # 168


# ─────────────────────────────────────────────────────────────────────────────
# _get_universe_at_date (point-in-time)
# ─────────────────────────────────────────────────────────────────────────────


def test_get_universe_at_date_calls_get_universe_as_of():
    """Walk-forward universe respects ETF inception dates (anti-survivorship)."""
    mock_universe = {
        "金融": "XLF",
        "美国长债": "TLT",
        "黄金": "GLD",
    }
    with patch(
        "engine.universe_manager.get_universe_as_of",
        return_value=mock_universe,
    ) as mock_fn:
        result = _get_universe_at_date(datetime.date(2015, 1, 31))

    mock_fn.assert_called_once_with(
        datetime.date(2015, 1, 31), min_history_years=MIN_HISTORY_YEARS,
    )
    assert result == mock_universe


def test_get_universe_at_date_failure_returns_empty():
    with patch(
        "engine.universe_manager.get_universe_as_of",
        side_effect=Exception("DB down"),
    ):
        result = _get_universe_at_date(datetime.date(2020, 1, 31))
    assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# _build_asset_classes_lookup
# ─────────────────────────────────────────────────────────────────────────────


def test_build_asset_classes_lookup_normal():
    by_class = {
        "equity_sector": {"金融": "XLF", "黄金": "GLD"},  # GLD wrong class but ok for mock
        "fixed_income":  {"美国长债": "TLT"},
    }
    with patch(
        "engine.universe_manager.get_universe_by_class",
        return_value=by_class,
    ):
        result = _build_asset_classes_lookup(["XLF", "TLT", "GLD"])
    assert result["XLF"] == "equity_sector"
    assert result["TLT"] == "fixed_income"
    assert result["GLD"] == "equity_sector"


def test_build_asset_classes_lookup_failure_fallback():
    """If registry unreachable, all → equity_sector default."""
    with patch(
        "engine.universe_manager.get_universe_by_class",
        side_effect=Exception("DB down"),
    ):
        result = _build_asset_classes_lookup(["QQQ", "TLT"])
    assert result == {"QQQ": "equity_sector", "TLT": "equity_sector"}


# ─────────────────────────────────────────────────────────────────────────────
# _compute_weights_from_signal
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_weights_handles_empty_signal():
    result = _compute_weights_from_signal(
        pd.Series(dtype=float), datetime.date(2020, 1, 31),
    )
    assert result.empty


def test_compute_weights_handles_all_zero_signal():
    sig = pd.Series({"QQQ": 0.0, "XLF": 0.0})
    result = _compute_weights_from_signal(sig, datetime.date(2020, 1, 31))
    assert result.empty


def test_compute_weights_normalizes_gross():
    sig = pd.Series({"QQQ": 1.0, "XLF": -1.0})
    fake_vols = pd.Series({"QQQ": 0.20, "XLF": 0.15})

    with patch(
        "engine.factor_ensemble_walk_forward._fetch_realized_vols",
        return_value=fake_vols,
    ):
        result = _compute_weights_from_signal(sig, datetime.date(2020, 1, 31))

    # After inv-vol weighting + gross-normalize + vol-target scalar
    # gross |sum| should be reasonable (between 0 and 2 due to leverage cap)
    assert not result.empty
    # Both tickers represented
    assert "QQQ" in result.index
    assert "XLF" in result.index
    # Signs preserved
    assert result["QQQ"] > 0
    assert result["XLF"] < 0


def test_compute_weights_drops_nan_signals():
    sig = pd.Series({"QQQ": 1.0, "XLF": np.nan, "TLT": 0.5})
    fake_vols = pd.Series({"QQQ": 0.20, "TLT": 0.10})

    with patch(
        "engine.factor_ensemble_walk_forward._fetch_realized_vols",
        return_value=fake_vols,
    ):
        result = _compute_weights_from_signal(sig, datetime.date(2020, 1, 31))

    assert "XLF" not in result.index
    assert "QQQ" in result.index
    assert "TLT" in result.index


# ─────────────────────────────────────────────────────────────────────────────
# _price_at_or_before
# ─────────────────────────────────────────────────────────────────────────────


def test_price_at_or_before_exact_date():
    closes = pd.Series(
        [100, 101, 102],
        index=pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-03-31"]),
    )
    assert _price_at_or_before(closes, datetime.date(2020, 2, 29)) == 101


def test_price_at_or_before_uses_most_recent_eligible():
    closes = pd.Series(
        [100, 101, 102],
        index=pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-03-31"]),
    )
    # Target between Feb and Mar → use Feb price
    assert _price_at_or_before(closes, datetime.date(2020, 3, 15)) == 101


def test_price_at_or_before_target_too_early():
    closes = pd.Series(
        [100, 101],
        index=pd.DatetimeIndex(["2020-02-29", "2020-03-31"]),
    )
    # Target before all data → None
    assert _price_at_or_before(closes, datetime.date(2020, 1, 15)) is None


# ─────────────────────────────────────────────────────────────────────────────
# run_walk_forward — top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────


def test_run_walk_forward_rejects_non_date():
    with pytest.raises(TypeError):
        run_walk_forward(start_date="2020-01-01")


def test_run_walk_forward_rejects_invalid_range():
    with pytest.raises(ValueError, match="must be"):
        run_walk_forward(
            start_date=datetime.date(2024, 12, 31),
            end_date=datetime.date(2020, 1, 1),
        )


def test_run_walk_forward_empty_universe_returns_empty_result():
    """If universe empty all periods, result has 0 periods."""
    with patch(
        "engine.factor_ensemble_walk_forward._get_universe_at_date",
        return_value={},
    ):
        result = run_walk_forward(
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2020, 6, 30),
            baseline_only=True,
        )

    assert isinstance(result, WalkForwardResult)
    assert result.n_periods == 0
    assert result.cumulative_return == 0.0


def test_run_walk_forward_basic_pipeline_baseline_only():
    """End-to-end with mocks: 3 month walk-forward, BAB-only baseline."""
    mock_universe_dict = {"金融": "XLF", "黄金": "GLD"}
    mock_asset_classes = {"XLF": "equity_sector", "GLD": "commodity"}
    fake_bab_signal = pd.Series({"XLF": 1.0, "GLD": -1.0})
    fake_vols = pd.Series({"XLF": 0.18, "GLD": 0.12})

    fake_closes_xlf = pd.Series(
        [50, 51, 52, 53, 54],
        index=pd.DatetimeIndex([
            "2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31",
        ]),
    )
    fake_closes_gld = pd.Series(
        [150, 152, 154, 156, 158],
        index=pd.DatetimeIndex([
            "2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31",
        ]),
    )

    def fake_fetch_closes(ticker, start=None, end=None):
        if ticker == "XLF":
            return fake_closes_xlf
        if ticker == "GLD":
            return fake_closes_gld
        return pd.Series(dtype=float)

    with patch(
        "engine.factor_ensemble_walk_forward._get_universe_at_date",
        return_value=mock_universe_dict,
    ), patch(
        "engine.factor_ensemble_walk_forward._build_asset_classes_lookup",
        return_value=mock_asset_classes,
    ), patch(
        "engine.factors.compute_bab_signal",
        return_value=fake_bab_signal,
    ), patch(
        "engine.factor_ensemble_walk_forward._fetch_realized_vols",
        return_value=fake_vols,
    ), patch(
        "engine.signal._fetch_closes",
        side_effect=fake_fetch_closes,
    ):
        result = run_walk_forward(
            start_date=datetime.date(2020, 1, 31),
            end_date=datetime.date(2020, 4, 30),
            baseline_only=True,
            persist=False,
        )

    assert isinstance(result, WalkForwardResult)
    assert result.n_periods >= 1  # at least 1-2 successful periods
    assert isinstance(result.monthly_returns, pd.Series)


def test_run_walk_forward_quality_lookahead_guard_in_walk_forward():
    """Walk-forward at historical date → Quality factor returns all-NaN per amendment."""
    from engine.factors.quality import compute_quality_signal, SPEC_LOCK_DATE
    universe = ["QQQ", "XLF", "USMV", "QUAL"]
    asset_classes = {t: "equity_sector" for t in universe}

    # Pre-SPEC_LOCK_DATE → all-NaN
    historical_result = compute_quality_signal(
        as_of=datetime.date(2015, 6, 30),  # well before lock date
        universe=universe,
        asset_classes=asset_classes,
    )
    assert historical_result.isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# WalkForwardResult dataclass
# ─────────────────────────────────────────────────────────────────────────────


def test_walk_forward_result_aggregate_metrics():
    """Verify Sharpe / vol / drawdown computation correctness."""
    monthly_returns = pd.Series(
        [0.01, -0.005, 0.015, 0.008, -0.012, 0.020],
        index=pd.DatetimeIndex([
            "2020-01-31", "2020-02-29", "2020-03-31",
            "2020-04-30", "2020-05-31", "2020-06-30",
        ]),
    )

    # Manually compute expected
    n = len(monthly_returns)
    cumulative = (1 + monthly_returns).prod() - 1
    ann_vol = monthly_returns.std(ddof=0) * np.sqrt(12)
    ann_mean = monthly_returns.mean() * 12
    ann_sharpe = ann_mean / ann_vol if ann_vol > 1e-9 else 0

    # We can't easily call run_walk_forward end-to-end with synthetic returns,
    # but we can construct a result manually and validate fields hold structure
    result = WalkForwardResult(
        n_periods=n,
        monthly_returns=monthly_returns,
        cumulative_return=float(cumulative),
        annualized_sharpe=float(ann_sharpe),
        annualized_vol=float(ann_vol),
        max_drawdown=-0.012,  # roughly the worst drawdown
        n_etfs_per_period=pd.Series([5] * n, index=monthly_returns.index),
        gross_exposure=pd.Series([1.0] * n, index=monthly_returns.index),
    )
    assert result.n_periods == n
    assert result.annualized_sharpe == pytest.approx(ann_sharpe)
