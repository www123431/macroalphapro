"""
tests/test_path_c_earnings_panel.py — Path C #1 PEAD Sprint 2 panel builder tests.

Pre-registration: docs/spec_path_c_earnings_pead_v1.md (id=57)

Per feedback_test_isolation_no_disk_pollution.md: all disk-writing tests use
tmp_path + cache_path override so no real disk state is touched.

Test surface:
  - Locked constants match spec hash-locked values
  - Mock mode generates correct schema + deterministic
  - Cache round-trip (write → read same panel)
  - Cache miss + corruption resilience
  - Window filter
  - SQL templates syntactically valid (compile-only smoke)
  - Public API auto-detects mock_mode
"""
from __future__ import annotations

import datetime
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from engine.path_c import (
    CONSENSUS_LOCK_WINDOW_DAYS,
    MIN_ANALYSTS_REQUIRED,
    WINDOW_START_LOCKED,
    WINDOW_END_LOCKED,
    UNIVERSE_TOP_N_LOCKED,
    HOLD_TRADING_DAYS_LOCKED,
    DECILE_LONG_THRESHOLD,
    DECILE_SHORT_THRESHOLD,
    NW_LAG_TRADING_DAYS_LOCKED,
    TC_BPS_ROUNDTRIP_LOCKED,
    bulk_fetch_earnings_panel,
    is_wrds_available,
    EarningsPanelResult,
)
from engine.path_c.earnings_panel import (
    _mock_earnings_panel,
    _COMP_FUNDQ_RDQ_SQL,
    _IBES_DET_EPSUS_SQL,
    _IBES_ACT_EPSUS_SQL,
    _IBES_PERMNO_LINK_SQL,
    _CRSP_COMP_LINK_SQL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants per spec §2.3, §2.4, §2.5, §六
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_constants_match_spec():
    """Verify __init__.py locked constants exactly match spec §六 hash-locked values."""
    assert CONSENSUS_LOCK_WINDOW_DAYS == 90        # spec §2.3 + §六
    assert MIN_ANALYSTS_REQUIRED == 2              # spec §2.3 fallback
    assert WINDOW_START_LOCKED == "2014-01-01"     # spec §六 universe row
    assert WINDOW_END_LOCKED == "2023-12-31"       # spec §六 universe row
    assert UNIVERSE_TOP_N_LOCKED == 200            # spec §六 + kickoff brief §12
    assert HOLD_TRADING_DAYS_LOCKED == 60          # spec §六
    assert DECILE_LONG_THRESHOLD == 0.90           # spec §2.4 + §六
    assert DECILE_SHORT_THRESHOLD == 0.10          # spec §2.4 + §六
    assert NW_LAG_TRADING_DAYS_LOCKED == 60        # spec §2.5 (typo resolution)
    assert TC_BPS_ROUNDTRIP_LOCKED == 30.0         # spec §六 + kickoff brief §8


# ─────────────────────────────────────────────────────────────────────────────
# Mock mode panel generation
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_panel_schema_columns():
    """Mock panel has the 10 columns required by EarningsPanelResult docstring."""
    df = _mock_earnings_panel(
        tickers=["AAPL", "MSFT", "GOOG"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
    )
    expected_cols = {
        "permno", "ticker_ibes", "gvkey", "fiscal_yearq", "rdq",
        "actual_eps", "consensus_median", "consensus_dispersion",
        "n_analysts", "market_cap_at_q",
    }
    assert expected_cols.issubset(set(df.columns))


def test_mock_panel_deterministic_across_runs():
    """Same ticker + window → identical panel (hash-based seeding)."""
    df1 = _mock_earnings_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    df2 = _mock_earnings_panel(["AAPL"], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_mock_panel_handles_empty_tickers():
    df = _mock_earnings_panel([], datetime.date(2014, 1, 1), datetime.date(2014, 12, 31))
    assert df.empty


def test_mock_panel_rdq_within_window():
    """All rdq values must fall within requested window."""
    start = datetime.date(2014, 1, 1)
    end = datetime.date(2014, 12, 31)
    df = _mock_earnings_panel(["AAPL", "MSFT"], start, end)
    assert (df["rdq"] >= start).all()
    assert (df["rdq"] <= end).all()


def test_mock_panel_n_analysts_meets_minimum():
    """All firm-quarter rows must have n_analysts ≥ MIN_ANALYSTS_REQUIRED."""
    df = _mock_earnings_panel(
        tickers=["A", "B", "C", "D", "E"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2015, 12, 31),
    )
    assert (df["n_analysts"] >= MIN_ANALYSTS_REQUIRED).all()


def test_mock_panel_dispersion_positive():
    """Dispersion must be strictly positive (spec §2.3: zero-dispersion excluded)."""
    df = _mock_earnings_panel(
        tickers=["A", "B", "C", "D", "E"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2015, 12, 31),
    )
    # Mock generator uses rng so dispersion = std with random floats → > 0
    assert (df["consensus_dispersion"] > 0).all()


def test_mock_panel_10y_full_window_size():
    """10y × 5 tickers should produce ~40 quarters × 5 = ~200 firm-quarters."""
    df = _mock_earnings_panel(
        tickers=["A", "B", "C", "D", "E"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2023, 12, 31),
    )
    # Allow some slack — final quarter may overshoot window
    assert 150 <= len(df) <= 220


def test_mock_panel_quarter_labels_sequential():
    """fiscal_yearq labels should span the expected range."""
    df = _mock_earnings_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
    )
    labels = set(df["fiscal_yearq"].unique())
    # 2013Q4 (rdq Jan) through 2014Q3 (rdq Oct-Dec) all possible
    assert any(lbl.startswith("2014") for lbl in labels)


# ─────────────────────────────────────────────────────────────────────────────
# Public API: bulk_fetch_earnings_panel
# ─────────────────────────────────────────────────────────────────────────────

def test_bulk_fetch_mock_mode_explicit(tmp_path):
    """Explicit mock_mode=True produces non-empty panel."""
    cache_path = tmp_path / "earnings_panel.parquet"
    result = bulk_fetch_earnings_panel(
        tickers=["AAPL", "MSFT"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert isinstance(result, EarningsPanelResult)
    assert result.mode == "mock"
    assert result.n_firm_quarters > 0
    assert not result.panel.empty


def test_bulk_fetch_persists_cache(tmp_path):
    """First call writes cache; cache file exists after."""
    cache_path = tmp_path / "earnings_panel.parquet"
    bulk_fetch_earnings_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert cache_path.exists()


def test_bulk_fetch_cache_hit(tmp_path):
    """Second call (same tickers/window/cache) loads from cache."""
    cache_path = tmp_path / "earnings_panel.parquet"
    r1 = bulk_fetch_earnings_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    r2 = bulk_fetch_earnings_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    # Cache HIT: exclusion_stats should reflect cache load
    assert r2.exclusion_stats.get("from_cache") is True
    # Same n_firm_quarters
    assert r1.n_firm_quarters == r2.n_firm_quarters


def test_bulk_fetch_cache_miss_when_window_extended(tmp_path):
    """Requesting wider window than cached forces refetch."""
    cache_path = tmp_path / "earnings_panel.parquet"
    bulk_fetch_earnings_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 6, 30),
        mock_mode=True,
        cache_path=cache_path,
    )
    # Now request 10y window — should miss
    r2 = bulk_fetch_earnings_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2023, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    # Cache miss → no `from_cache` flag
    assert r2.exclusion_stats.get("from_cache") is not True


def test_bulk_fetch_cache_corrupt_falls_back_to_fetch(tmp_path):
    """If cache file is corrupted, fall back to refetch instead of crashing."""
    cache_path = tmp_path / "earnings_panel.parquet"
    cache_path.write_text("NOT VALID PARQUET", encoding="utf-8")
    result = bulk_fetch_earnings_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    assert not result.panel.empty


def test_bulk_fetch_no_cache_flag(tmp_path):
    """use_cache=False skips cache load even if present."""
    cache_path = tmp_path / "earnings_panel.parquet"
    bulk_fetch_earnings_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
    )
    r2 = bulk_fetch_earnings_panel(
        tickers=["AAPL"],
        start_date=datetime.date(2014, 1, 1),
        end_date=datetime.date(2014, 12, 31),
        mock_mode=True,
        cache_path=cache_path,
        use_cache=False,
    )
    # use_cache=False → from_cache flag NOT set
    assert r2.exclusion_stats.get("from_cache") is not True


def test_bulk_fetch_window_filters_rdq(tmp_path):
    """Returned panel.rdq strictly within [start, end]."""
    cache_path = tmp_path / "earnings_panel.parquet"
    start = datetime.date(2014, 6, 1)
    end = datetime.date(2014, 9, 30)
    result = bulk_fetch_earnings_panel(
        tickers=["AAPL", "MSFT", "GOOG"],
        start_date=start,
        end_date=end,
        mock_mode=True,
        cache_path=cache_path,
    )
    if not result.panel.empty:
        assert (result.panel["rdq"] >= start).all()
        assert (result.panel["rdq"] <= end).all()


def test_bulk_fetch_auto_detect_mock_when_wrds_unavailable(tmp_path):
    """When WRDS not configured, mock_mode=None should auto-resolve to mock."""
    cache_path = tmp_path / "earnings_panel.parquet"
    with mock.patch("engine.path_c.earnings_panel.is_wrds_available", return_value=False):
        result = bulk_fetch_earnings_panel(
            tickers=["AAPL"],
            start_date=datetime.date(2014, 1, 1),
            end_date=datetime.date(2014, 12, 31),
            mock_mode=None,
            cache_path=cache_path,
        )
    assert result.mode == "mock"


def test_bulk_fetch_window_start_end_stored(tmp_path):
    """Result captures requested window bounds."""
    cache_path = tmp_path / "earnings_panel.parquet"
    start = datetime.date(2014, 1, 1)
    end = datetime.date(2014, 12, 31)
    result = bulk_fetch_earnings_panel(
        tickers=["AAPL"],
        start_date=start,
        end_date=end,
        mock_mode=True,
        cache_path=cache_path,
    )
    assert result.window_start == start
    assert result.window_end == end


# ─────────────────────────────────────────────────────────────────────────────
# SQL template smoke (compile-only — no WRDS execution)
# ─────────────────────────────────────────────────────────────────────────────

def test_sql_templates_have_required_placeholders():
    """SQL templates use named placeholders matching SQL caller params."""
    # comp.fundq SQL needs start_date, end_date, gvkeys
    for ph in ("%(start_date)s", "%(end_date)s", "%(gvkeys)s"):
        assert ph in _COMP_FUNDQ_RDQ_SQL
    # ibes.det_epsus SQL needs start_date, end_date, tickers
    for ph in ("%(start_date)s", "%(end_date)s", "%(tickers)s"):
        assert ph in _IBES_DET_EPSUS_SQL
    # ibes.act_epsus SQL same
    for ph in ("%(start_date)s", "%(end_date)s", "%(tickers)s"):
        assert ph in _IBES_ACT_EPSUS_SQL
    # ibes.id ↔ crsp.stocknames linkage needs tickers + date bounds
    for ph in ("%(tickers)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _IBES_PERMNO_LINK_SQL
    # ccmxpf_lnkhist linkage needs permnos + date bounds
    for ph in ("%(permnos)s", "%(start_date)s", "%(end_date)s"):
        assert ph in _CRSP_COMP_LINK_SQL


def test_sql_templates_reference_locked_tables():
    """Each SQL targets the locked WRDS table per spec §2.2."""
    assert "comp.fundq" in _COMP_FUNDQ_RDQ_SQL
    assert "ibes.det_epsus" in _IBES_DET_EPSUS_SQL
    assert "ibes.act_epsus" in _IBES_ACT_EPSUS_SQL
    assert "ibes.id" in _IBES_PERMNO_LINK_SQL
    assert "crsp.stocknames" in _IBES_PERMNO_LINK_SQL
    assert "crsp.ccmxpf_lnkhist" in _CRSP_COMP_LINK_SQL


def test_det_epsus_filters_to_all_quarterly_horizons():
    """fpi 6..11 are all quarterly horizons (Q1-ahead .. Q6-ahead).

    Rigor audit 2026-05-12: original fpi='6' was too narrow (excluded valid
    forecasts made when target was 2Q+ ahead). Fixed to include all quarterly
    fpi codes — clarification +0 trials, methodology unchanged.
    """
    for fpi_code in ("'6'", "'7'", "'8'", "'9'", "'10'", "'11'"):
        assert fpi_code in _IBES_DET_EPSUS_SQL, f"missing fpi code {fpi_code}"


def test_act_epsus_filters_to_quarterly_pdicity():
    """Spec §2.3 implicit: pdicity='QTR' for quarterly actuals."""
    assert "pdicity = 'QTR'" in _IBES_ACT_EPSUS_SQL


def test_fundq_filters_to_standard_indfmt():
    """Compustat best-practice: indfmt='INDL' + datafmt='STD' + popsrc='D' + consol='C'."""
    assert "indfmt = 'INDL'" in _COMP_FUNDQ_RDQ_SQL
    assert "datafmt = 'STD'" in _COMP_FUNDQ_RDQ_SQL
    assert "popsrc = 'D'" in _COMP_FUNDQ_RDQ_SQL
    assert "consol = 'C'" in _COMP_FUNDQ_RDQ_SQL


# ─────────────────────────────────────────────────────────────────────────────
# is_wrds_available smoke (delegates to crsp_loader)
# ─────────────────────────────────────────────────────────────────────────────

def test_is_wrds_available_returns_bool():
    """Should always return bool (whether configured or not)."""
    result = is_wrds_available()
    assert isinstance(result, bool)
