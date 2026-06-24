"""
tests/test_path_c_pead_ts_signal_panel.py — Sprint D-2 PEAD time-series SUE panel tests.

Pre-registration: docs/spec_path_d_dhs_behavioral_2factor_v1.md (id=62)
"""
from __future__ import annotations

import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from engine.path_c.pead_ts_signal_panel import (
    SEASONAL_LAG_QUARTERS,
    SIGMA_WINDOW_QUARTERS,
    SIGMA_MIN_PERIODS,
    SIGMA_MIN_VALUE,
    SUE_WINSORIZE_LOW,
    SUE_WINSORIZE_HIGH,
    PeadTsSignalPanelResult,
    bulk_fetch_pead_ts_signal_panel,
    is_wrds_available,
    _mock_pead_ts_panel,
    _COMP_FUNDQ_PEAD_TS_SQL,
    _CRSP_MSE_TICKER_SQL,
    _CRSP_COMP_LINK_SQL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_constants_match_spec():
    assert SEASONAL_LAG_QUARTERS == 4         # spec §2.3 step 3 (seasonal Δ vs q-4)
    assert SIGMA_WINDOW_QUARTERS == 8         # spec §2.3 step 4 (8Q rolling)
    assert SIGMA_MIN_PERIODS == 4             # spec §六 (≥ 4 prior obs)
    assert SIGMA_MIN_VALUE == 0.01            # spec §2.3 step 6 (σ floor)
    assert SUE_WINSORIZE_LOW == -10.0         # spec §六 (winsorize bounds)
    assert SUE_WINSORIZE_HIGH == +10.0


# ─────────────────────────────────────────────────────────────────────────────
# Mock panel schema + determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_panel_schema():
    df = _mock_pead_ts_panel(
        ["AAPL", "MSFT"],
        datetime.date(2014, 1, 1),
        datetime.date(2014, 12, 31),
    )
    expected = {
        "permno", "ticker", "gvkey", "fiscal_yearq", "rdq",
        "eps_adj", "eps_adj_lag4", "delta_eps", "sigma_8q",
        "sue_raw", "sue", "market_cap_at_q",
    }
    assert expected.issubset(set(df.columns))


def test_mock_panel_deterministic():
    d1 = _mock_pead_ts_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    d2 = _mock_pead_ts_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    pd.testing.assert_frame_equal(d1.reset_index(drop=True), d2.reset_index(drop=True))


def test_mock_panel_empty_tickers():
    assert _mock_pead_ts_panel([], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31)).empty


def test_mock_panel_rdq_within_window():
    start = datetime.date(2014, 1, 1)
    end = datetime.date(2014, 12, 31)
    df = _mock_pead_ts_panel(["AAPL", "MSFT"], start, end)
    assert (df["rdq"] >= start).all() and (df["rdq"] <= end).all()


def test_mock_panel_eps_adj_positive():
    """Mock EPS is constructed > 0 by max(0.01, ...) clamp."""
    df = _mock_pead_ts_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    assert (df["eps_adj"] > 0).all()
    assert (df["eps_adj_lag4"] > 0).all()


def test_mock_panel_sigma_above_floor():
    df = _mock_pead_ts_panel(["A", "B", "C", "D"], datetime.date(2014, 1, 1), datetime.date(2016, 12, 31))
    if not df.empty:
        assert (df["sigma_8q"] >= SIGMA_MIN_VALUE).all()


def test_mock_panel_sue_winsorized():
    df = _mock_pead_ts_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2016, 12, 31))
    if not df.empty:
        assert (df["sue"] >= SUE_WINSORIZE_LOW).all()
        assert (df["sue"] <= SUE_WINSORIZE_HIGH).all()


def test_mock_panel_sue_formula_consistency():
    """sue should equal clipped sue_raw, and sue_raw should match delta_eps / sigma_8q."""
    df = _mock_pead_ts_panel(["AAPL", "MSFT"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    if df.empty:
        pytest.skip("no rows in mock window")
    # sue_raw formula
    np.testing.assert_allclose(
        df["sue_raw"].values,
        (df["delta_eps"] / df["sigma_8q"]).values,
        rtol=1e-9,
    )
    # winsorize
    expected_sue = df["sue_raw"].clip(SUE_WINSORIZE_LOW, SUE_WINSORIZE_HIGH).values
    np.testing.assert_allclose(df["sue"].values, expected_sue, rtol=1e-9)


def test_mock_panel_delta_eps_matches_seasonal_diff():
    df = _mock_pead_ts_panel(["AAPL", "MSFT"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    if df.empty:
        pytest.skip("no rows")
    np.testing.assert_allclose(
        df["delta_eps"].values,
        (df["eps_adj"] - df["eps_adj_lag4"]).values,
        rtol=1e-9,
    )


def test_mock_panel_mcap_positive():
    df = _mock_pead_ts_panel(["A", "B"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    assert (df["market_cap_at_q"] > 0).all()


def test_mock_panel_10y_size():
    df = _mock_pead_ts_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2023, 12, 31))
    # 3 firms × ~40 quarters minus seasonal+σ warm-up exclusions
    assert 80 <= len(df) <= 130


# ─────────────────────────────────────────────────────────────────────────────
# Public API: bulk_fetch_pead_ts_signal_panel
# ─────────────────────────────────────────────────────────────────────────────

def test_bulk_fetch_mock(tmp_path):
    cache_path = tmp_path / "pead_ts_panel.parquet"
    result = bulk_fetch_pead_ts_signal_panel(
        tickers=["AAPL", "MSFT"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert isinstance(result, PeadTsSignalPanelResult)
    assert result.mode == "mock"
    assert result.n_firm_quarters > 0


def test_bulk_fetch_persists_cache(tmp_path):
    cache_path = tmp_path / "pead_ts_panel.parquet"
    bulk_fetch_pead_ts_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert cache_path.exists()
    assert cache_path.with_suffix(cache_path.suffix + ".meta.json").exists()


def test_bulk_fetch_cache_hit(tmp_path):
    cache_path = tmp_path / "pead_ts_panel.parquet"
    bulk_fetch_pead_ts_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    r2 = bulk_fetch_pead_ts_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert r2.exclusion_stats.get("from_cache") is True


def test_bulk_fetch_cache_miss_wider_window(tmp_path):
    cache_path = tmp_path / "pead_ts_panel.parquet"
    bulk_fetch_pead_ts_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 6, 30),
        mock_mode=True, cache_path=cache_path,
    )
    r2 = bulk_fetch_pead_ts_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2020, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert r2.exclusion_stats.get("from_cache") is not True


def test_bulk_fetch_cache_corrupt_falls_back(tmp_path):
    cache_path = tmp_path / "pead_ts_panel.parquet"
    cache_path.write_text("NOT VALID PARQUET", encoding="utf-8")
    cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text("{}", encoding="utf-8")
    result = bulk_fetch_pead_ts_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert not result.panel.empty


def test_bulk_fetch_auto_detect_mock(tmp_path):
    cache_path = tmp_path / "pead_ts_panel.parquet"
    with mock.patch("engine.path_c.pead_ts_signal_panel.is_wrds_available", return_value=False):
        result = bulk_fetch_pead_ts_signal_panel(
            ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
            mock_mode=None, cache_path=cache_path,
        )
    assert result.mode == "mock"


# ─────────────────────────────────────────────────────────────────────────────
# SQL template integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_sql_fundq_has_eps_and_split_adjust():
    """spec §2.2: must pull epspxq + ajexq + cshoq + prccq + atq + rdq."""
    for col in ("epspxq", "ajexq", "cshoq", "prccq", "atq", "rdq"):
        assert col in _COMP_FUNDQ_PEAD_TS_SQL, f"missing {col}"


def test_sql_fundq_standard_filters():
    assert "indfmt = 'INDL'" in _COMP_FUNDQ_PEAD_TS_SQL
    assert "datafmt = 'STD'" in _COMP_FUNDQ_PEAD_TS_SQL
    assert "popsrc = 'D'" in _COMP_FUNDQ_PEAD_TS_SQL
    assert "consol = 'C'" in _COMP_FUNDQ_PEAD_TS_SQL


def test_sql_msenames_placeholders():
    for ph in ("%(tickers)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _CRSP_MSE_TICKER_SQL


def test_sql_link_placeholders():
    for ph in ("%(permnos)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _CRSP_COMP_LINK_SQL


def test_is_wrds_available_returns_bool():
    assert isinstance(is_wrds_available(), bool)
