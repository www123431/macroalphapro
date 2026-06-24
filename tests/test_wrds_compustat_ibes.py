"""Tests for engine.data.fetchers.wrds_compustat + wrds_ibes (mocked)."""
from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from engine.data.fetchers import wrds_compustat as comp
from engine.data.fetchers import wrds_ibes as ibes


# -- Compustat -----------------------------------------------------------

def test_compustat_PIT_FILTER_present_in_query():
    """Every Compustat query must include the standard PIT filter."""
    captured = {}
    def fake(sql, account="${WRDS_USER_2}"):
        captured["sql"] = sql
        return pd.DataFrame({c: [] for c in comp.DEFAULT_FUNDA_COLS})

    with mock.patch.object(comp, "_get_connector",
                                return_value=type("W", (), {"raw_sql": staticmethod(fake)})):
        comp.fetch_funda("2020-01-01", "2020-12-31")
    assert comp.PIT_FILTER in captured["sql"]
    assert "indfmt='INDL'" in captured["sql"]
    assert "consol='C'" in captured["sql"]


def test_compustat_gvkey_filter_quotes_strings():
    """gvkey is a string in Compustat; filter must quote it."""
    captured = {}
    def fake(sql, account="${WRDS_USER_2}"):
        captured["sql"] = sql
        return pd.DataFrame({c: [] for c in comp.DEFAULT_FUNDA_COLS})

    with mock.patch.object(comp, "_get_connector",
                                return_value=type("W", (), {"raw_sql": staticmethod(fake)})):
        comp.fetch_funda("2020-01-01", "2020-12-31", gvkeys=["001690", "001045"])
    assert "gvkey IN ('001690','001045')" in captured["sql"]


def test_compustat_fetch_handles_connector_unavailable():
    """No wrds_direct in env → returns empty df, doesn't crash."""
    with mock.patch.object(comp, "_get_connector", return_value=None):
        df = comp.fetch_funda("2020-01-01", "2020-12-31")
    assert df.empty
    assert "datadate" in df.columns


def test_compustat_book_to_market_panel():
    funda = pd.DataFrame({
        "gvkey": ["A", "A", "B"],
        "datadate": pd.to_datetime(["2020-12-31", "2021-12-31", "2020-12-31"]),
        "ceq": [100.0, 110.0, 50.0],
    })
    mcap = pd.DataFrame({
        "gvkey": ["A", "A", "B"],
        "date": pd.to_datetime(["2020-12-31", "2021-12-31", "2020-12-31"]),
        "market_cap_m": [200.0, 220.0, 25.0],
    })
    out = comp.book_to_market_panel(funda, mcap)
    assert len(out) == 3
    A_2020 = out[(out["gvkey"] == "A") & (out["datadate"] == pd.Timestamp("2020-12-31"))]
    assert abs(A_2020["b_to_m"].iloc[0] - 0.5) < 1e-9
    B_row = out[out["gvkey"] == "B"]
    assert abs(B_row["b_to_m"].iloc[0] - 2.0) < 1e-9


def test_compustat_roe_panel():
    funda = pd.DataFrame({
        "gvkey": ["A", "A"],
        "datadate": pd.to_datetime(["2020-12-31", "2021-12-31"]),
        "ni": [10.0, 15.0],
        "ceq": [100.0, 110.0],
    })
    out = comp.roe_panel(funda)
    assert len(out) == 2
    assert abs(out["roe"].iloc[0] - 0.10) < 1e-9
    assert abs(out["roe"].iloc[1] - 15.0/110.0) < 1e-9


def test_compustat_roe_handles_zero_equity():
    funda = pd.DataFrame({
        "gvkey": ["A"],
        "datadate": pd.to_datetime(["2020-12-31"]),
        "ni": [10.0],
        "ceq": [0.0],
    })
    out = comp.roe_panel(funda)
    assert pd.isna(out["roe"].iloc[0])


# -- IBES ----------------------------------------------------------------

def test_ibes_default_measure_is_eps():
    captured = {}
    def fake(sql, account="${WRDS_USER_2}"):
        captured["sql"] = sql
        return pd.DataFrame({c: [] for c in ibes.DEFAULT_STATSUM_COLS})

    with mock.patch.object(ibes, "_get_connector",
                                return_value=type("W", (), {"raw_sql": staticmethod(fake)})):
        ibes.fetch_statsum_eps("2020-01-01", "2020-03-31")
    assert "measure='EPS'" in captured["sql"]


def test_ibes_pit_uses_statpers_not_fpedats():
    """For statsum, PIT pin is statpers (not fpedats which is fiscal end).
    The filter must use statpers."""
    captured = {}
    def fake(sql, account="${WRDS_USER_2}"):
        captured["sql"] = sql
        return pd.DataFrame({c: [] for c in ibes.DEFAULT_STATSUM_COLS})

    with mock.patch.object(ibes, "_get_connector",
                                return_value=type("W", (), {"raw_sql": staticmethod(fake)})):
        ibes.fetch_statsum_eps("2020-01-01", "2020-12-31")
    assert "statpers BETWEEN" in captured["sql"]


def test_ibes_det_pit_uses_revdats():
    """For detail-level, PIT pin is revdats (analyst publication date)."""
    captured = {}
    def fake(sql, account="${WRDS_USER_2}"):
        captured["sql"] = sql
        return pd.DataFrame({c: [] for c in ibes.DEFAULT_DET_COLS})

    with mock.patch.object(ibes, "_get_connector",
                                return_value=type("W", (), {"raw_sql": staticmethod(fake)})):
        ibes.fetch_det_eps("2020-01-01", "2020-12-31")
    assert "revdats BETWEEN" in captured["sql"]


def test_ibes_fpi_filter_quoted_strings():
    """fpi is a string column; filter quotes it."""
    captured = {}
    def fake(sql, account="${WRDS_USER_2}"):
        captured["sql"] = sql
        return pd.DataFrame({c: [] for c in ibes.DEFAULT_STATSUM_COLS})

    with mock.patch.object(ibes, "_get_connector",
                                return_value=type("W", (), {"raw_sql": staticmethod(fake)})):
        ibes.fetch_statsum_eps("2020-01-01", "2020-03-31", fpi=["1", "2"])
    assert "fpi IN ('1','2')" in captured["sql"]


def test_ibes_revision_count_panel():
    det = pd.DataFrame({
        "ticker": ["AAPL", "AAPL", "AAPL", "MSFT"],
        "analys": [101, 101, 102, 201],
        "value": [3.00, 3.10, 3.05, 2.50],
        "revdats": pd.to_datetime(["2020-01-15", "2020-02-15", "2020-02-20", "2020-01-15"]),
    })
    out = ibes.revision_count_panel(det)
    # 1 revision for analyst 101 AAPL (Jan -> Feb), 0 for 102 first obs, 0 for 201 first obs
    aapl_101 = out[(out["ticker"] == "AAPL")]
    assert len(aapl_101) == 1
    assert abs(aapl_101["delta"].iloc[0] - 0.10) < 1e-9


def test_ibes_fetch_handles_connector_unavailable():
    with mock.patch.object(ibes, "_get_connector", return_value=None):
        df = ibes.fetch_statsum_eps("2020-01-01", "2020-03-31")
    assert df.empty


# -- Module-level constants -----------------------------------------------

def test_compustat_default_cols_includes_key_columns():
    """Required for BARRA Phase 2 HML/QMJ:"""
    must_have = {"gvkey", "datadate", "ceq", "ni", "at", "sale"}
    assert must_have.issubset(set(comp.DEFAULT_FUNDA_COLS))


def test_ibes_default_cols_includes_key_columns():
    """Required for analyst-revision sleeve:"""
    must_have = {"ticker", "statpers", "meanest", "actual", "anndats_act"}
    assert must_have.issubset(set(ibes.DEFAULT_STATSUM_COLS))
