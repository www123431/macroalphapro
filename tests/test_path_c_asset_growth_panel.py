"""
tests/test_path_c_asset_growth_panel.py — Sprint J-2 asset growth panel tests.

Pre-registration: docs/spec_path_j_asset_growth_drift_v1.md (id=60)
"""
from __future__ import annotations

import datetime
from unittest import mock

import pandas as pd
import pytest

from engine.path_c.asset_growth_signal_panel import (
    ATQ_LOOKBACK_QUARTERS,
    MIN_ATQ_DOLLAR_M,
    MAX_ABSOLUTE_GROWTH,
    AssetGrowthSignalPanelResult,
    bulk_fetch_asset_growth_signal_panel,
    is_wrds_available,
    _mock_asset_growth_panel,
    _COMP_FUNDQ_AG_SQL,
    _CRSP_MSE_TICKER_SQL,
    _CRSP_COMP_LINK_BY_PERMNO_SQL,
)


def test_locked_constants_match_spec():
    assert ATQ_LOOKBACK_QUARTERS == 4
    assert MIN_ATQ_DOLLAR_M == 100.0
    assert MAX_ABSOLUTE_GROWTH == 5.0


def test_mock_panel_schema():
    df = _mock_asset_growth_panel(
        ["AAPL", "MSFT"],
        datetime.date(2014, 1, 1),
        datetime.date(2014, 12, 31),
    )
    expected = {"permno", "ticker", "gvkey", "fiscal_yearq", "rdq",
                "atq_recent", "atq_prior", "market_cap_at_q"}
    assert expected.issubset(set(df.columns))


def test_mock_panel_deterministic():
    d1 = _mock_asset_growth_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    d2 = _mock_asset_growth_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    pd.testing.assert_frame_equal(d1.reset_index(drop=True), d2.reset_index(drop=True))


def test_mock_panel_rdq_within_window():
    start = datetime.date(2014, 1, 1)
    end = datetime.date(2014, 12, 31)
    df = _mock_asset_growth_panel(["AAPL"], start, end)
    assert (df["rdq"] >= start).all() and (df["rdq"] <= end).all()


def test_mock_panel_atq_recent_positive():
    df = _mock_asset_growth_panel(["A", "B"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    assert (df["atq_recent"] > 0).all()


def test_mock_panel_atq_prior_nan_first_year():
    """In first year of mock data, atq_prior should be NaN for quarter_idx < 4."""
    df = _mock_asset_growth_panel(["A"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    # All firm-quarters in first year have quarter_idx < 4 → atq_prior is NaN
    # (mock generator returns NaN for these per the conditional)
    # In our 1y window, all rows should have NaN atq_prior
    assert df["atq_prior"].isna().all()


def test_mock_panel_atq_prior_present_after_1y():
    """After year 1, atq_prior should be populated."""
    df = _mock_asset_growth_panel(
        ["A"], datetime.date(2014, 1, 1), datetime.date(2017, 12, 31)
    )
    later_rows = df[df["fiscal_yearq"].str[:4].astype(int) >= 2015]
    assert later_rows["atq_prior"].notna().any()


def test_mock_panel_empty():
    df = _mock_asset_growth_panel([], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    assert df.empty


def test_mock_panel_market_cap_positive():
    df = _mock_asset_growth_panel(["A", "B"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    assert (df["market_cap_at_q"] > 0).all()


def test_mock_panel_10y_size():
    df = _mock_asset_growth_panel(["A", "B", "C"],
                                   datetime.date(2014, 1, 1),
                                   datetime.date(2023, 12, 31))
    assert 100 <= len(df) <= 130


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def test_bulk_fetch_mock(tmp_path):
    cache_path = tmp_path / "ag.parquet"
    r = bulk_fetch_asset_growth_signal_panel(
        ["AAPL", "MSFT"],
        datetime.date(2014, 1, 1),
        datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert isinstance(r, AssetGrowthSignalPanelResult)
    assert r.mode == "mock"
    assert r.n_firm_quarters > 0


def test_bulk_fetch_cache_persists(tmp_path):
    cache_path = tmp_path / "ag.parquet"
    bulk_fetch_asset_growth_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert cache_path.exists()
    assert cache_path.with_suffix(cache_path.suffix + ".meta.json").exists()


def test_bulk_fetch_cache_hit(tmp_path):
    cache_path = tmp_path / "ag.parquet"
    bulk_fetch_asset_growth_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    r2 = bulk_fetch_asset_growth_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert r2.exclusion_stats.get("from_cache") is True


def test_bulk_fetch_cache_miss_wider(tmp_path):
    cache_path = tmp_path / "ag.parquet"
    bulk_fetch_asset_growth_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 6, 30),
        mock_mode=True, cache_path=cache_path,
    )
    r2 = bulk_fetch_asset_growth_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2020, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert r2.exclusion_stats.get("from_cache") is not True


def test_bulk_fetch_cache_corrupt_falls_back(tmp_path):
    cache_path = tmp_path / "ag.parquet"
    cache_path.write_text("BAD", encoding="utf-8")
    cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text("{}", encoding="utf-8")
    r = bulk_fetch_asset_growth_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert not r.panel.empty


def test_bulk_fetch_auto_mock(tmp_path):
    cache_path = tmp_path / "ag.parquet"
    with mock.patch("engine.path_c.asset_growth_signal_panel.is_wrds_available", return_value=False):
        r = bulk_fetch_asset_growth_signal_panel(
            ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
            mock_mode=None, cache_path=cache_path,
        )
    assert r.mode == "mock"


# ─────────────────────────────────────────────────────────────────────────────
# SQL templates
# ─────────────────────────────────────────────────────────────────────────────

def test_sql_fundq_pulls_atq():
    """spec §2.2 requires atq + market cap inputs."""
    for col in ("atq", "cshoq", "prccq", "rdq"):
        assert col in _COMP_FUNDQ_AG_SQL, f"missing {col}"


def test_sql_fundq_standard_filters():
    assert "indfmt = 'INDL'" in _COMP_FUNDQ_AG_SQL
    assert "datafmt = 'STD'" in _COMP_FUNDQ_AG_SQL
    assert "popsrc = 'D'" in _COMP_FUNDQ_AG_SQL
    assert "consol = 'C'" in _COMP_FUNDQ_AG_SQL


def test_sql_msenames_placeholders():
    for ph in ("%(tickers)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _CRSP_MSE_TICKER_SQL


def test_sql_link_by_permno_placeholders():
    for ph in ("%(permnos)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _CRSP_COMP_LINK_BY_PERMNO_SQL


def test_is_wrds_available_returns_bool():
    assert isinstance(is_wrds_available(), bool)
