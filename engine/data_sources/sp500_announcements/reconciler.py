"""
engine/data_sources/sp500_announcements/reconciler.py — Wikipedia + EDGAR merge.

Sprint D-1 reconciliation layer. Takes:
  - Wikipedia events (effective_date + ticker + ADD/REMOVE, heuristic ann_date)
  - EDGAR 8-K filings (precise filing_date ≈ announcement_date + ticker)

Output:
  - Reconciled list of SP500ChangeEvent with announcement_date refined
    where EDGAR found a matching filing
  - Persistence to SP500AnnouncementEvent DB table for orchestrator consumption
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

from engine.data_sources.sp500_announcements.wikipedia import SP500ChangeEvent
from engine.data_sources.sp500_announcements.edgar_8k import Edgar8KFiling

logger = logging.getLogger(__name__)


def reconcile_announcements(
    wikipedia_events: list[SP500ChangeEvent],
    edgar_filings:    list[Edgar8KFiling],
    match_window_days: int = 30,
) -> list[SP500ChangeEvent]:
    """Refine Wikipedia events' announcement_date via EDGAR 8-K filing dates.

    Matching: for each Wikipedia ADD event, search EDGAR filings for the same
    ticker filed within ±match_window_days of the Wikipedia announcement_date.
    If found, use the EDGAR filing_date as announcement_date.

    Wikipedia REMOVE events are kept with heuristic announcement_date (no 8-K
    typically filed for index removal).
    """
    if not edgar_filings:
        return wikipedia_events

    # Build ticker → list of EDGAR filings
    edgar_by_ticker: dict[str, list[Edgar8KFiling]] = {}
    for f in edgar_filings:
        for t in f.tickers:
            edgar_by_ticker.setdefault(t.upper(), []).append(f)

    refined: list[SP500ChangeEvent] = []
    n_refined = 0
    for event in wikipedia_events:
        if event.action != "ADD":
            refined.append(event)
            continue

        ticker = event.ticker.upper()
        candidates = edgar_by_ticker.get(ticker, [])
        if not candidates:
            refined.append(event)
            continue

        # Find EDGAR filing closest to (but no later than) effective_date
        # and within ±match_window_days of heuristic announcement_date
        best_filing = None
        best_distance = None
        for f in candidates:
            if f.file_date > event.effective_date:
                continue  # filing after effective is irrelevant
            distance_days = abs((f.file_date - (event.announcement_date or event.effective_date)).days)
            if distance_days > match_window_days:
                continue
            if best_distance is None or distance_days < best_distance:
                best_distance = distance_days
                best_filing = f

        if best_filing is not None:
            refined.append(SP500ChangeEvent(
                effective_date    = event.effective_date,
                announcement_date = best_filing.file_date,
                ticker            = event.ticker,
                company_name      = event.company_name,
                action            = event.action,
                reason            = event.reason,
                source            = f"wikipedia+edgar_8k_{best_filing.accession_no}",
            ))
            n_refined += 1
        else:
            refined.append(event)

    logger.info(
        "reconcile_announcements: refined %d of %d ADD events via EDGAR matching",
        n_refined, sum(1 for e in wikipedia_events if e.action == "ADD"),
    )
    return refined


def persist_announcements(
    events:  list[SP500ChangeEvent],
    session: Optional[object] = None,
) -> dict:
    """Persist reconciled events to SP500AnnouncementEvent DB table.

    Idempotent: existing (ticker, effective_date, action) tuples are updated
    in place; new events are inserted.

    Returns dict with counts of {inserted, updated, errors}.
    """
    from engine.memory import init_db, SessionFactory
    from engine.db_models import SP500AnnouncementEvent

    init_db()
    own_session = session is None
    sess = session if session is not None else SessionFactory()

    inserted = 0
    updated = 0
    errors = 0
    try:
        for event in events:
            existing = (
                sess.query(SP500AnnouncementEvent)
                    .filter_by(
                        ticker=event.ticker,
                        effective_date=event.effective_date,
                        action=event.action,
                    )
                    .first()
            )
            if existing:
                # Update announcement_date / source / company_name / reason
                changed = False
                if existing.announcement_date != event.announcement_date:
                    existing.announcement_date = event.announcement_date
                    changed = True
                if existing.source != event.source:
                    existing.source = event.source
                    changed = True
                if existing.company_name != event.company_name and event.company_name:
                    existing.company_name = event.company_name
                    changed = True
                if existing.reason != event.reason and event.reason:
                    existing.reason = event.reason
                    changed = True
                if changed:
                    existing.updated_at = datetime.datetime.utcnow()
                    updated += 1
            else:
                row = SP500AnnouncementEvent(
                    ticker             = event.ticker,
                    effective_date     = event.effective_date,
                    announcement_date  = event.announcement_date,
                    company_name       = event.company_name,
                    action             = event.action,
                    reason             = event.reason,
                    source             = event.source,
                )
                sess.add(row)
                inserted += 1
        sess.commit()
    except Exception as exc:
        logger.exception("persist_announcements failed: %s", exc)
        sess.rollback()
        errors = 1
    finally:
        if own_session:
            sess.close()

    return {"inserted": inserted, "updated": updated, "errors": errors}


def load_pending_path_n_events(
    as_of:         datetime.date,
    lookahead_days: int = 5,
) -> list[dict]:
    """Load S&P 500 ADD events with effective_date in (as_of, as_of + lookahead].

    Returns list of dicts {ticker, effective_date, announcement_date, ...}
    for orchestrator consumption (Path N forward signal).

    Filters:
      - Only ADD events (Path N is long-only)
      - effective_date strictly future
      - effective_date within lookahead window
    """
    from engine.memory import init_db, SessionFactory
    from engine.db_models import SP500AnnouncementEvent

    init_db()
    sess = SessionFactory()
    try:
        cutoff_end = as_of + datetime.timedelta(days=lookahead_days)
        rows = (
            sess.query(SP500AnnouncementEvent)
                .filter(SP500AnnouncementEvent.action == "ADD")
                .filter(SP500AnnouncementEvent.effective_date > as_of)
                .filter(SP500AnnouncementEvent.effective_date <= cutoff_end)
                .order_by(SP500AnnouncementEvent.effective_date.asc())
                .all()
        )
        return [
            {
                "ticker":            r.ticker,
                "effective_date":    r.effective_date,
                "announcement_date": r.announcement_date,
                "company_name":      r.company_name,
                "reason":            r.reason,
                "source":            r.source,
            }
            for r in rows
        ]
    finally:
        sess.close()
