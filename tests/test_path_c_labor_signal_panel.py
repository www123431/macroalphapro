"""
tests/test_path_c_labor_signal_panel.py — Sprint G2 labor signal panel tests.

Pre-registration: docs/spec_path_c_labor_signal_drift_v1.md (id=58)

Surface (mirrors test_path_c_earnings_panel.py structure):
  - Locked constants match spec
  - Mock panel schema + determinism + window filter
  - Cache HIT-MISS-corruption via sidecar metadata
  - SQL templates syntactic check (placeholder integrity)
  - is_wrds_available smoke
  - L6 / B12 / layoff_flag semantics in mock
"""
from __future__ import annotations

import datetime
from unittest import mock

import pandas as pd
import pytest

from engine.path_c.labor_signal_panel import (
    L6_WINDOW_MONTHS,
    B12_WINDOW_MONTHS,
    LAYOFF_WINDOW_DAYS,
    MIN_L6_POSTINGS_REQUIRED,
    MIN_B12_POSTINGS_REQUIRED,
    LaborSignalPanelResult,
    bulk_fetch_labor_signal_panel,
    is_wrds_available,
    _mock_labor_panel,
    _REVELIO_COMPANY_MAPPING_SQL,
    _COMP_FUNDQ_RDQ_SQL,
    _REVELIO_POSTINGS_AGG_SQL,
    _REVELIO_LAYOFFS_SQL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants per spec §2.3 + §六
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_constants_match_spec():
    assert L6_WINDOW_MONTHS == 6          # spec §2.3 (rolling 6mo)
    assert B12_WINDOW_MONTHS == 12        # spec §2.3 (12mo baseline)
    assert LAYOFF_WINDOW_DAYS == 90       # spec §2.3 (layoff lookback)
    assert MIN_L6_POSTINGS_REQUIRED == 5  # spec §2.3 fallback
    assert MIN_B12_POSTINGS_REQUIRED == 10  # spec §2.3 fallback


# ─────────────────────────────────────────────────────────────────────────────
# Mock panel generation
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_panel_schema_columns():
    """Mock panel has the 10 columns required by LaborSignalPanelResult docstring."""
    df = _mock_labor_panel(
        tickers=["AAPL", "MSFT"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
    )
    expected = {
        "permno", "ticker", "gvkey", "rcid", "fiscal_yearq", "rdq",
        "l6_postings_count", "b12_postings_count", "layoff_flag",
        "market_cap_at_q",
    }
    assert expected.issubset(set(df.columns))


def test_mock_panel_deterministic():
    """Same tickers + window → identical panel."""
    d1 = _mock_labor_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    d2 = _mock_labor_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    pd.testing.assert_frame_equal(d1.reset_index(drop=True), d2.reset_index(drop=True))


def test_mock_panel_empty_tickers():
    df = _mock_labor_panel([], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    assert df.empty


def test_mock_panel_rdq_within_window():
    start = datetime.date(2014, 1, 1)
    end = datetime.date(2014, 12, 31)
    df = _mock_labor_panel(["AAPL", "MSFT"], start, end)
    assert (df["rdq"] >= start).all()
    assert (df["rdq"] <= end).all()


def test_mock_panel_postings_counts_positive():
    """L6 and B12 should be non-negative integers."""
    df = _mock_labor_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    assert (df["l6_postings_count"] >= 0).all()
    assert (df["b12_postings_count"] >= 0).all()


def test_mock_panel_layoff_flag_binary():
    """layoff_flag is 0 or 1."""
    df = _mock_labor_panel(["A", "B", "C"], datetime.date(2014, 1, 1), datetime.date(2015, 12, 31))
    assert set(df["layoff_flag"].unique()).issubset({0, 1})


def test_mock_panel_10y_sample_size():
    """10y × 3 tickers → ~120 firm-quarters."""
    df = _mock_labor_panel(
        ["A", "B", "C"],
        datetime.date(2014, 1, 1),
        datetime.date(2023, 12, 31),
    )
    assert 100 <= len(df) <= 130


# ─────────────────────────────────────────────────────────────────────────────
# Public API: bulk_fetch_labor_signal_panel
# ─────────────────────────────────────────────────────────────────────────────

def test_bulk_fetch_mock_mode(tmp_path):
    cache_path = tmp_path / "labor_panel.parquet"
    result = bulk_fetch_labor_signal_panel(
        tickers=["AAPL", "MSFT"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert isinstance(result, LaborSignalPanelResult)
    assert result.mode == "mock"
    assert result.n_firm_quarters > 0


def test_bulk_fetch_persists_cache(tmp_path):
    cache_path = tmp_path / "labor_panel.parquet"
    bulk_fetch_labor_signal_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert cache_path.exists()
    meta = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    assert meta.exists()


def test_bulk_fetch_cache_hit(tmp_path):
    cache_path = tmp_path / "labor_panel.parquet"
    bulk_fetch_labor_signal_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    r2 = bulk_fetch_labor_signal_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert r2.exclusion_stats.get("from_cache") is True


def test_bulk_fetch_cache_miss_wider_window(tmp_path):
    """Wider requested window → cache miss."""
    cache_path = tmp_path / "labor_panel.parquet"
    bulk_fetch_labor_signal_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 6, 30),
        mock_mode=True,
        cache_path=cache_path,
    )
    r2 = bulk_fetch_labor_signal_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2020, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert r2.exclusion_stats.get("from_cache") is not True


def test_bulk_fetch_cache_corrupt_falls_back(tmp_path):
    cache_path = tmp_path / "labor_panel.parquet"
    cache_path.write_text("NOT VALID PARQUET", encoding="utf-8")
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    meta_path.write_text("{}", encoding="utf-8")
    result = bulk_fetch_labor_signal_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert not result.panel.empty


def test_bulk_fetch_no_cache(tmp_path):
    cache_path = tmp_path / "labor_panel.parquet"
    bulk_fetch_labor_signal_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    r2 = bulk_fetch_labor_signal_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
        use_cache=False,
    )
    assert r2.exclusion_stats.get("from_cache") is not True


def test_bulk_fetch_window_filters_rdq(tmp_path):
    cache_path = tmp_path / "labor_panel.parquet"
    start = datetime.date(2014, 6, 1)
    end = datetime.date(2014, 9, 30)
    result = bulk_fetch_labor_signal_panel(
        tickers=["A", "B", "C"],
        start_date=start,
        end_date=end,
        mock_mode=True,
        cache_path=cache_path,
    )
    if not result.panel.empty:
        assert (result.panel["rdq"] >= start).all()
        assert (result.panel["rdq"] <= end).all()


def test_bulk_fetch_auto_detect_mock(tmp_path):
    cache_path = tmp_path / "labor_panel.parquet"
    with mock.patch("engine.path_c.labor_signal_panel.is_wrds_available", return_value=False):
        result = bulk_fetch_labor_signal_panel(
            tickers=["AAPL"],
            start_date=datetime.date(2014, 1, 1),
            end_date=datetime.date(2014, 12, 31),
            mock_mode=None,
            cache_path=cache_path,
        )
    assert result.mode == "mock"


def test_bulk_fetch_stores_window(tmp_path):
    cache_path = tmp_path / "labor_panel.parquet"
    start = datetime.date(2014, 1, 1)
    end = datetime.date(2014, 12, 31)
    result = bulk_fetch_labor_signal_panel(
        tickers=["AAPL"], start_date=start, end_date=end,
        mock_mode=True, cache_path=cache_path,
    )
    assert result.window_start == start
    assert result.window_end == end


# ─────────────────────────────────────────────────────────────────────────────
# SQL template integrity (compile-only)
# ─────────────────────────────────────────────────────────────────────────────

def test_sql_company_mapping_placeholders():
    assert "%(tickers)s" in _REVELIO_COMPANY_MAPPING_SQL
    assert "revelio.company_mapping" in _REVELIO_COMPANY_MAPPING_SQL


def test_sql_postings_agg_placeholders():
    for ph in ("%(rcids)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _REVELIO_POSTINGS_AGG_SQL
    assert "revelio.postings_cosmos" in _REVELIO_POSTINGS_AGG_SQL
    assert "DATE_TRUNC('month', post_date)" in _REVELIO_POSTINGS_AGG_SQL


def test_sql_layoffs_placeholders():
    for ph in ("%(rcids)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _REVELIO_LAYOFFS_SQL
    assert "revelio.layoffs" in _REVELIO_LAYOFFS_SQL


def test_sql_company_mapping_filters_us_exchange():
    """spec §2.2 US-exchange filter; probe 2026-05-12 confirmed actual values
    are 'NASDAQ' and 'New York Stock Exchange' (not 'NYSE')."""
    assert "'NASDAQ'" in _REVELIO_COMPANY_MAPPING_SQL
    assert "'New York Stock Exchange'" in _REVELIO_COMPANY_MAPPING_SQL


def test_sql_company_mapping_deterministic_dedupe():
    """SQL uses DISTINCT ON (ticker) + ORDER BY ticker, rcid ASC to
    deterministically pick lowest rcid per ticker (parent / earliest entity).
    Audit fix 2026-05-12 — without this, multi-rcid tickers (~65% of SP500
    sample) returned arbitrary rcid across runs."""
    assert "DISTINCT ON (ticker)" in _REVELIO_COMPANY_MAPPING_SQL
    assert "ORDER BY ticker, rcid ASC" in _REVELIO_COMPANY_MAPPING_SQL


def test_sql_comp_fundq_standard_filters():
    """Compustat standard filters per spec §2.2."""
    assert "indfmt = 'INDL'" in _COMP_FUNDQ_RDQ_SQL
    assert "datafmt = 'STD'" in _COMP_FUNDQ_RDQ_SQL
    assert "popsrc = 'D'" in _COMP_FUNDQ_RDQ_SQL
    assert "consol = 'C'" in _COMP_FUNDQ_RDQ_SQL


# ─────────────────────────────────────────────────────────────────────────────
# is_wrds_available smoke
# ─────────────────────────────────────────────────────────────────────────────

def test_is_wrds_available_returns_bool():
    assert isinstance(is_wrds_available(), bool)
