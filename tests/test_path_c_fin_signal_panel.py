"""
tests/test_path_c_fin_signal_panel.py — Sprint D-3 FIN raw panel tests.

Pre-registration: docs/spec_path_d_dhs_behavioral_2factor_v1.md (id=62)
"""
from __future__ import annotations

import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from engine.path_c.fin_signal_panel import (
    NSI_LAG_QUARTERS,
    NSI_WINSORIZE_LOW,
    NSI_WINSORIZE_HIGH,
    ACC_WINSORIZE_LOW,
    ACC_WINSORIZE_HIGH,
    FinSignalPanelResult,
    bulk_fetch_fin_signal_panel,
    is_wrds_available,
    _mock_fin_panel,
    _COMP_FUNDQ_FIN_SQL,
    _CRSP_MSE_TICKER_SQL,
    _CRSP_COMP_LINK_SQL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_constants_match_spec():
    assert NSI_LAG_QUARTERS == 4
    assert NSI_WINSORIZE_LOW == -0.5
    assert NSI_WINSORIZE_HIGH == +1.0
    assert ACC_WINSORIZE_LOW == -0.3
    assert ACC_WINSORIZE_HIGH == +0.3


# ─────────────────────────────────────────────────────────────────────────────
# Mock panel schema + determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_panel_schema():
    df = _mock_fin_panel(
        ["AAPL", "MSFT"],
        datetime.date(2014, 1, 1),
        datetime.date(2014, 12, 31),
    )
    expected = {
        "permno", "ticker", "gvkey", "fiscal_yearq", "rdq",
        "shares_adj", "shares_adj_lag4", "nsi_raw", "nsi",
        "acc_raw", "atq_lag1", "acc_scaled_raw", "acc_scaled",
        "market_cap_at_q",
    }
    assert expected.issubset(set(df.columns))


def test_mock_panel_deterministic():
    d1 = _mock_fin_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    d2 = _mock_fin_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    pd.testing.assert_frame_equal(d1.reset_index(drop=True), d2.reset_index(drop=True))


def test_mock_panel_empty_tickers():
    assert _mock_fin_panel([], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31)).empty


def test_mock_panel_rdq_within_window():
    start = datetime.date(2014, 1, 1)
    end = datetime.date(2014, 12, 31)
    df = _mock_fin_panel(["AAPL", "MSFT"], start, end)
    assert (df["rdq"] >= start).all() and (df["rdq"] <= end).all()


def test_mock_panel_shares_positive():
    df = _mock_fin_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    assert (df["shares_adj"] > 0).all()
    assert (df["shares_adj_lag4"] > 0).all()


def test_mock_panel_atq_lag1_positive():
    df = _mock_fin_panel(["A", "B"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    assert (df["atq_lag1"] > 0).all()


def test_mock_panel_nsi_within_winsorize():
    df = _mock_fin_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    if not df.empty:
        assert (df["nsi"] >= NSI_WINSORIZE_LOW).all()
        assert (df["nsi"] <= NSI_WINSORIZE_HIGH).all()


def test_mock_panel_acc_scaled_within_winsorize():
    df = _mock_fin_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    if not df.empty:
        assert (df["acc_scaled"] >= ACC_WINSORIZE_LOW).all()
        assert (df["acc_scaled"] <= ACC_WINSORIZE_HIGH).all()


def test_mock_panel_nsi_formula_consistency():
    """nsi_raw should equal log(shares_adj / shares_adj_lag4); nsi = clipped nsi_raw."""
    df = _mock_fin_panel(["AAPL", "MSFT"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    if df.empty:
        pytest.skip("no rows")
    expected_nsi_raw = np.log(df["shares_adj"] / df["shares_adj_lag4"])
    np.testing.assert_allclose(df["nsi_raw"].values, expected_nsi_raw.values, rtol=1e-9)
    expected_nsi = expected_nsi_raw.clip(NSI_WINSORIZE_LOW, NSI_WINSORIZE_HIGH).values
    np.testing.assert_allclose(df["nsi"].values, expected_nsi, rtol=1e-9)


def test_mock_panel_acc_scaled_formula_consistency():
    df = _mock_fin_panel(["AAPL", "MSFT"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    if df.empty:
        pytest.skip("no rows")
    np.testing.assert_allclose(
        df["acc_scaled_raw"].values,
        (df["acc_raw"] / df["atq_lag1"]).values,
        rtol=1e-9,
    )


def test_mock_panel_mcap_positive():
    df = _mock_fin_panel(["A", "B"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    assert (df["market_cap_at_q"] > 0).all()


def test_mock_panel_10y_size():
    df = _mock_fin_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2023, 12, 31))
    # 3 firms × ~40 quarters minus 4Q NSI lookback exclusions
    assert 100 <= len(df) <= 130


# ─────────────────────────────────────────────────────────────────────────────
# Public API: bulk_fetch_fin_signal_panel
# ─────────────────────────────────────────────────────────────────────────────

def test_bulk_fetch_mock(tmp_path):
    cache_path = tmp_path / "fin_panel.parquet"
    result = bulk_fetch_fin_signal_panel(
        tickers=["AAPL", "MSFT"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert isinstance(result, FinSignalPanelResult)
    assert result.mode == "mock"
    assert result.n_firm_quarters > 0


def test_bulk_fetch_persists_cache(tmp_path):
    cache_path = tmp_path / "fin_panel.parquet"
    bulk_fetch_fin_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert cache_path.exists()
    assert cache_path.with_suffix(cache_path.suffix + ".meta.json").exists()


def test_bulk_fetch_cache_hit(tmp_path):
    cache_path = tmp_path / "fin_panel.parquet"
    bulk_fetch_fin_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    r2 = bulk_fetch_fin_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert r2.exclusion_stats.get("from_cache") is True


def test_bulk_fetch_cache_miss_wider_window(tmp_path):
    cache_path = tmp_path / "fin_panel.parquet"
    bulk_fetch_fin_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 6, 30),
        mock_mode=True, cache_path=cache_path,
    )
    r2 = bulk_fetch_fin_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2020, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert r2.exclusion_stats.get("from_cache") is not True


def test_bulk_fetch_cache_corrupt_falls_back(tmp_path):
    cache_path = tmp_path / "fin_panel.parquet"
    cache_path.write_text("NOT VALID PARQUET", encoding="utf-8")
    cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text("{}", encoding="utf-8")
    result = bulk_fetch_fin_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert not result.panel.empty


def test_bulk_fetch_auto_detect_mock(tmp_path):
    cache_path = tmp_path / "fin_panel.parquet"
    with mock.patch("engine.path_c.fin_signal_panel.is_wrds_available", return_value=False):
        result = bulk_fetch_fin_signal_panel(
            ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
            mock_mode=None, cache_path=cache_path,
        )
    assert result.mode == "mock"


# ─────────────────────────────────────────────────────────────────────────────
# SQL template integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_sql_fundq_has_balance_sheet_fields():
    """spec §2.4: must pull cshoq + ajexq + atq + actq + lctq + cheq + dlcq + txpq + dpq."""
    for col in ("cshoq", "ajexq", "atq", "actq", "lctq", "cheq",
                "dlcq", "txpq", "dpq", "niq", "prccq", "rdq"):
        assert col in _COMP_FUNDQ_FIN_SQL, f"missing {col}"


def test_sql_fundq_standard_filters():
    assert "indfmt = 'INDL'" in _COMP_FUNDQ_FIN_SQL
    assert "datafmt = 'STD'" in _COMP_FUNDQ_FIN_SQL
    assert "popsrc = 'D'" in _COMP_FUNDQ_FIN_SQL
    assert "consol = 'C'" in _COMP_FUNDQ_FIN_SQL


def test_sql_msenames_placeholders():
    for ph in ("%(tickers)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _CRSP_MSE_TICKER_SQL


def test_sql_link_placeholders():
    for ph in ("%(permnos)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _CRSP_COMP_LINK_SQL


def test_is_wrds_available_returns_bool():
    assert isinstance(is_wrds_available(), bool)
