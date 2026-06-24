"""
tests/test_path_c_rd_signal_panel.py — Sprint I-2 R&D signal panel tests.

Pre-registration: docs/spec_path_i_rd_premium_drift_v1.md (id=59)
"""
from __future__ import annotations

import datetime
from unittest import mock

import pandas as pd
import pytest

from engine.path_c.rd_signal_panel import (
    R_AND_D_RECENT_QUARTERS,
    R_AND_D_PRIOR_QUARTERS,
    R_AND_D_MIN_DISCLOSED_QUARTERS,
    R_AND_D_MIN_DOLLAR_M,
    RdSignalPanelResult,
    bulk_fetch_rd_signal_panel,
    is_wrds_available,
    _mock_rd_panel,
    _COMP_FUNDQ_RD_SQL,
    _CRSP_MSE_TICKER_SQL,
    _CRSP_COMP_LINK_SQL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_constants_match_spec():
    assert R_AND_D_RECENT_QUARTERS == 4         # spec §2.3 (trailing 4Q)
    assert R_AND_D_PRIOR_QUARTERS == 4          # spec §2.3 (prior 4Q baseline)
    assert R_AND_D_MIN_DISCLOSED_QUARTERS == 2  # spec §2.3 (≥2 disclosed)
    assert R_AND_D_MIN_DOLLAR_M == 1.0          # spec §2.3 ($1M threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Mock panel schema + determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_panel_schema():
    df = _mock_rd_panel(
        ["AAPL", "MSFT"],
        datetime.date(2014, 1, 1),
        datetime.date(2014, 12, 31),
    )
    expected = {
        "permno", "ticker", "gvkey", "fiscal_yearq", "rdq",
        "r_and_d_4q_recent", "r_and_d_4q_prior",
        "n_quarters_recent", "n_quarters_prior",
        "atq", "market_cap_at_q",
    }
    assert expected.issubset(set(df.columns))


def test_mock_panel_deterministic():
    d1 = _mock_rd_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    d2 = _mock_rd_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    pd.testing.assert_frame_equal(d1.reset_index(drop=True), d2.reset_index(drop=True))


def test_mock_panel_empty_tickers():
    assert _mock_rd_panel([], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31)).empty


def test_mock_panel_rdq_within_window():
    start = datetime.date(2014, 1, 1)
    end = datetime.date(2014, 12, 31)
    df = _mock_rd_panel(["AAPL", "MSFT"], start, end)
    assert (df["rdq"] >= start).all() and (df["rdq"] <= end).all()


def test_mock_panel_rd_sums_positive():
    df = _mock_rd_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    assert (df["r_and_d_4q_recent"] > 0).all()
    assert (df["r_and_d_4q_prior"] > 0).all()


def test_mock_panel_n_quarters_in_range():
    df = _mock_rd_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    assert (df["n_quarters_recent"] >= 2).all()
    assert (df["n_quarters_recent"] <= 4).all()
    assert (df["n_quarters_prior"] >= 2).all()
    assert (df["n_quarters_prior"] <= 4).all()


def test_mock_panel_atq_positive():
    df = _mock_rd_panel(["A", "B"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    assert (df["atq"] > 0).all()


def test_mock_panel_10y_size():
    df = _mock_rd_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2023, 12, 31))
    assert 100 <= len(df) <= 130


# ─────────────────────────────────────────────────────────────────────────────
# Public API: bulk_fetch_rd_signal_panel
# ─────────────────────────────────────────────────────────────────────────────

def test_bulk_fetch_mock(tmp_path):
    cache_path = tmp_path / "rd_panel.parquet"
    result = bulk_fetch_rd_signal_panel(
        tickers=["AAPL", "MSFT"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert isinstance(result, RdSignalPanelResult)
    assert result.mode == "mock"
    assert result.n_firm_quarters > 0


def test_bulk_fetch_persists_cache(tmp_path):
    cache_path = tmp_path / "rd_panel.parquet"
    bulk_fetch_rd_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert cache_path.exists()
    assert cache_path.with_suffix(cache_path.suffix + ".meta.json").exists()


def test_bulk_fetch_cache_hit(tmp_path):
    cache_path = tmp_path / "rd_panel.parquet"
    bulk_fetch_rd_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    r2 = bulk_fetch_rd_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert r2.exclusion_stats.get("from_cache") is True


def test_bulk_fetch_cache_miss_wider_window(tmp_path):
    cache_path = tmp_path / "rd_panel.parquet"
    bulk_fetch_rd_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 6, 30),
        mock_mode=True, cache_path=cache_path,
    )
    r2 = bulk_fetch_rd_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2020, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert r2.exclusion_stats.get("from_cache") is not True


def test_bulk_fetch_cache_corrupt_falls_back(tmp_path):
    cache_path = tmp_path / "rd_panel.parquet"
    cache_path.write_text("NOT VALID PARQUET", encoding="utf-8")
    cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text("{}", encoding="utf-8")
    result = bulk_fetch_rd_signal_panel(
        ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
        mock_mode=True, cache_path=cache_path,
    )
    assert not result.panel.empty


def test_bulk_fetch_auto_detect_mock(tmp_path):
    cache_path = tmp_path / "rd_panel.parquet"
    with mock.patch("engine.path_c.rd_signal_panel.is_wrds_available", return_value=False):
        result = bulk_fetch_rd_signal_panel(
            ["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31),
            mock_mode=None, cache_path=cache_path,
        )
    assert result.mode == "mock"


# ─────────────────────────────────────────────────────────────────────────────
# SQL template integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_sql_fundq_has_xrdq_and_atq():
    """spec §2.2: must pull xrdq + atq + cshoq + prccq."""
    for col in ("xrdq", "atq", "cshoq", "prccq", "rdq"):
        assert col in _COMP_FUNDQ_RD_SQL, f"missing {col}"


def test_sql_fundq_standard_filters():
    assert "indfmt = 'INDL'" in _COMP_FUNDQ_RD_SQL
    assert "datafmt = 'STD'" in _COMP_FUNDQ_RD_SQL
    assert "popsrc = 'D'" in _COMP_FUNDQ_RD_SQL
    assert "consol = 'C'" in _COMP_FUNDQ_RD_SQL


def test_sql_msenames_placeholders():
    for ph in ("%(tickers)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _CRSP_MSE_TICKER_SQL


def test_sql_link_placeholders():
    for ph in ("%(permnos)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _CRSP_COMP_LINK_SQL


def test_is_wrds_available_returns_bool():
    assert isinstance(is_wrds_available(), bool)
