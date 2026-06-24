"""
engine/data_sources/sp500_announcements/ — Real-time S&P 500 reconstitution feed.

Sprint D-1 (2026-05-13 night): free alternative to paid S&P Global data feed.

Purpose
-------
Path N S&P 500 Reconstitution Drift strategy needs to detect S&P 500 add/delete
events at ANNOUNCEMENT-time (T-5) for forward paper trade and real-money
execution. CRSP msp500list (used in backtest) is historical effective-date
data — too late for live alpha capture (drift window T-5 to T-1).

This module provides free real-time announcement detection from two sources:

Primary: Wikipedia "List of S&P 500 companies" (Selected changes table)
  - Structured table with Effective Date / Added / Removed / Reason
  - Community-maintained, typically updated within hours of S&P announcement
  - Easy parsing via MediaWiki API

Secondary: SEC EDGAR full-text search for 8-K filings
  - Federal API, stable, structured
  - Search "added to the S&P 500" in 8-K body
  - Filing date ≈ announcement date (more precise than Wikipedia effective_date)

Strategy
--------
1. Wikipedia gives us effective_date with high reliability
2. announcement_date ≈ effective_date - 5 trading days (S&P standard practice)
3. EDGAR optionally refines announcement_date when 8-K filed earlier
4. Persist to SP500AnnouncementEvent table for orchestrator consumption
"""
from engine.data_sources.sp500_announcements.wikipedia import (
    WIKIPEDIA_SP500_URL,
    SP500ChangeEvent,
    fetch_wikipedia_sp500_changes,
)
from engine.data_sources.sp500_announcements.edgar_8k import (
    EDGAR_SEARCH_API,
    fetch_edgar_8k_sp500_filings,
)
from engine.data_sources.sp500_announcements.reconciler import (
    reconcile_announcements,
    persist_announcements,
    load_pending_path_n_events,
)

__all__ = [
    "WIKIPEDIA_SP500_URL",
    "SP500ChangeEvent",
    "fetch_wikipedia_sp500_changes",
    "EDGAR_SEARCH_API",
    "fetch_edgar_8k_sp500_filings",
    "reconcile_announcements",
    "persist_announcements",
    "load_pending_path_n_events",
]
