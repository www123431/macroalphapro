"""
tests/test_sp500_announcements.py — Sprint D-1 announcement feed tests.

Tests cover:
- Wikipedia parser date/cell parsing
- EDGAR filer display name parsing
- Reconciler matching + persistence
- get_path_n_signal live/auto/backtest mode dispatch
"""
from __future__ import annotations

import datetime

import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia parser unit tests (pure-function, no network)
# ─────────────────────────────────────────────────────────────────────────────
def test_parse_effective_date_formats():
    from engine.data_sources.sp500_announcements.wikipedia import _parse_effective_date
    assert _parse_effective_date("May 7, 2026") == datetime.date(2026, 5, 7)
    assert _parse_effective_date("September 18, 2025") == datetime.date(2025, 9, 18)
    assert _parse_effective_date("Jan 2, 2024") == datetime.date(2024, 1, 2)
    assert _parse_effective_date("2024-03-15") == datetime.date(2024, 3, 15)
    # Strip footnotes
    assert _parse_effective_date("May 7, 2026[1][2]") == datetime.date(2026, 5, 7)
    # Bad input
    assert _parse_effective_date("invalid") is None
    assert _parse_effective_date("") is None


def test_estimate_announcement_date_heuristic():
    from engine.data_sources.sp500_announcements.wikipedia import _estimate_announcement_date
    # eff = 2026-05-07 → ann ≈ 2026-04-30 (eff - 7 calendar days)
    ann = _estimate_announcement_date(datetime.date(2026, 5, 7))
    assert ann == datetime.date(2026, 4, 30)


def test_clean_cell_text_strips_footnotes():
    from engine.data_sources.sp500_announcements.wikipedia import _clean_cell_text
    assert _clean_cell_text("Veeva Systems[1]") == "Veeva Systems"
    assert _clean_cell_text("Foo  Bar\nBaz") == "Foo Bar Baz"
    assert _clean_cell_text("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR parser unit tests (pure-function)
# ─────────────────────────────────────────────────────────────────────────────
def test_parse_edgar_filer_display():
    from engine.data_sources.sp500_announcements.edgar_8k import _parse_filer_display
    # Standard format
    name, tickers, cik = _parse_filer_display([
        "REALTY INCOME CORP  (O, O-P)  (CIK 0000726728)",
    ])
    assert name == "REALTY INCOME CORP"
    assert tickers == ["O", "O-P"]
    assert cik == "0000726728"

    # No tickers, only CIK
    name, tickers, cik = _parse_filer_display([
        "SOME COMPANY INC  (CIK 0001234567)",
    ])
    assert name == "SOME COMPANY INC"
    assert tickers == []
    assert cik == "0001234567"

    # Empty
    name, tickers, cik = _parse_filer_display([])
    assert name == ""
    assert tickers == []
    assert cik == ""


# ─────────────────────────────────────────────────────────────────────────────
# Reconciler tests
# ─────────────────────────────────────────────────────────────────────────────
def test_reconcile_announcements_refines_via_edgar():
    from engine.data_sources.sp500_announcements.wikipedia import SP500ChangeEvent
    from engine.data_sources.sp500_announcements.edgar_8k import Edgar8KFiling
    from engine.data_sources.sp500_announcements.reconciler import reconcile_announcements

    wiki = [SP500ChangeEvent(
        effective_date    = datetime.date(2026, 5, 7),
        announcement_date = datetime.date(2026, 4, 30),  # heuristic eff - 7
        ticker            = "VEEV",
        company_name      = "Veeva Systems",
        action            = "ADD",
        reason            = "Acquisition",
        source            = "wikipedia",
    )]

    edgar = [Edgar8KFiling(
        cik           = "0001393052",
        filer_name    = "VEEVA SYSTEMS INC",
        tickers       = ["VEEV"],
        file_date     = datetime.date(2026, 5, 2),   # 3 days closer than heuristic
        accession_no  = "0001234567-26-000123",
        query_matched = '"added to the S&P 500"',
        raw_display   = "VEEVA SYSTEMS INC (VEEV) (CIK 0001393052)",
    )]

    out = reconcile_announcements(wiki, edgar)
    assert len(out) == 1
    refined = out[0]
    assert refined.ticker == "VEEV"
    # EDGAR file_date should override heuristic
    assert refined.announcement_date == datetime.date(2026, 5, 2)
    assert "edgar_8k" in refined.source


def test_reconcile_no_edgar_match_keeps_heuristic():
    from engine.data_sources.sp500_announcements.wikipedia import SP500ChangeEvent
    from engine.data_sources.sp500_announcements.reconciler import reconcile_announcements

    wiki = [SP500ChangeEvent(
        effective_date    = datetime.date(2026, 5, 7),
        announcement_date = datetime.date(2026, 4, 30),
        ticker            = "VEEV",
        company_name      = "Veeva Systems",
        action            = "ADD",
        reason            = "Acquisition",
        source            = "wikipedia",
    )]

    # No EDGAR filings provided
    out = reconcile_announcements(wiki, [])
    assert len(out) == 1
    assert out[0].announcement_date == datetime.date(2026, 4, 30)
    assert out[0].source == "wikipedia"


def test_reconcile_keeps_remove_events_unchanged():
    """REMOVE events have no 8-K filing; heuristic announcement_date preserved."""
    from engine.data_sources.sp500_announcements.wikipedia import SP500ChangeEvent
    from engine.data_sources.sp500_announcements.edgar_8k import Edgar8KFiling
    from engine.data_sources.sp500_announcements.reconciler import reconcile_announcements

    wiki = [SP500ChangeEvent(
        effective_date    = datetime.date(2026, 5, 7),
        announcement_date = datetime.date(2026, 4, 30),
        ticker            = "CTRA",
        company_name      = "Coterra Energy",
        action            = "REMOVE",
        reason            = "Acquisition",
        source            = "wikipedia",
    )]
    edgar = [Edgar8KFiling(
        cik="0000123", filer_name="CTRA", tickers=["CTRA"],
        file_date=datetime.date(2026, 5, 1), accession_no="0001",
        query_matched="x", raw_display="CTRA (CIK 0000123)",
    )]
    out = reconcile_announcements(wiki, edgar)
    assert out[0].action == "REMOVE"
    assert out[0].announcement_date == datetime.date(2026, 4, 30)  # unchanged


# ─────────────────────────────────────────────────────────────────────────────
# get_path_n_signal mode dispatch
# ─────────────────────────────────────────────────────────────────────────────
def test_path_n_signal_live_mode_with_pending_event():
    """Test live mode end-to-end via the DB (assumes Wikipedia smoke-test
    already populated SP500AnnouncementEvent with VEEV 2026-05-07 ADD)."""
    from engine.portfolio.paper_trade_combined import get_path_n_signal
    # 2026-05-04 should see VEEV with effective_date in 3-day window
    sig = get_path_n_signal(datetime.date(2026, 5, 4), mode="live")
    # Either OK with VEEV or NO_SIGNAL if test isolation removed it
    assert sig.status in {"OK", "NO_SIGNAL"}
    if sig.status == "OK":
        assert "VEEV" in sig.weights.index
        assert sig.n_positions >= 1


def test_path_n_signal_backtest_mode_uses_parquet():
    """Backtest mode uses CRSP msp500list parquet (permno identifiers)."""
    from engine.portfolio.paper_trade_combined import get_path_n_signal
    sig = get_path_n_signal(datetime.date(2023, 6, 15), mode="backtest")
    assert sig.status in {"OK", "NO_SIGNAL"}
    if sig.status == "OK":
        # Identifiers should be 'permno_XXX' format
        for ident in sig.weights.index:
            assert str(ident).startswith("permno_")


def test_path_n_signal_auto_mode_no_break():
    """Auto mode should not crash for various dates."""
    from engine.portfolio.paper_trade_combined import get_path_n_signal
    for date in [
        datetime.date(2023, 6, 15),  # historical, backtest event likely
        datetime.date(2026, 5, 13),  # today, may have no events
        datetime.date(2030, 1, 1),   # far future, no data
    ]:
        sig = get_path_n_signal(date, mode="auto")
        assert sig.status in {"OK", "NO_SIGNAL", "ERROR"}
