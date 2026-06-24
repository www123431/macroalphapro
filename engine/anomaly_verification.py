"""
engine/anomaly_verification.py — D4.5 of S6 anomaly_screener (Slimmed, 2026-05-05)

Pre-registration: docs/decisions/s6_anomaly_screener_spec_2026-05-05.md
                  §1.4 + §M1 Reported Metrics

Slimmed scope (vs Full-S6):
  ✅ M1 precision (per detector)
  ✅ M1 recall    (universe sweep)
  ✅ M1 F1        (harmonic mean)
  ❌ Calibration plot   — REMOVED
  ❌ Brier score        — REMOVED
  ❌ ROC AUC            — REMOVED
  ❌ Sensitivity sweep  — REMOVED

Pipeline:
  Daily cron (D4.6) calls verify_due_flags(today) at end of each trading day
  to set event_occurred / event_date / event_return / event_sigma on flags
  whose horizon window has just closed (scan_date + horizon_days < today).

  Independently, populate_universe_events(scan_date) populates
  AnomalyUniverseEvent for the universe (current portfolio ∪ recently
  flagged tickers) so recall can be computed.

  compute_metrics(...) joins flags ↔ universe events and returns precision,
  recall, F1 per detector + composite verdict.
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Iterable

import pandas as pd

from engine.memory import (
    AnomalyFlag,
    AnomalyUniverseEvent,
    SessionFactory,
)

logger = logging.getLogger(__name__)

# ── Pre-registered constants (centralized in engine/config.py 2026-05-06) ────
from engine.config import (
    EVENT_SIGMA_THRESHOLD,
    ROLLING_VOL_WINDOW,
    CUTOFF_DATE,
    COMPOSITE_WIN_PP,
    M2_MIN_ACCEPT_PCT,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fetch_close_history(ticker: str, end_date: datetime.date, days: int = 90) -> pd.DataFrame:
    """Fetch close history for a ticker; same as anomaly_screener._fetch_price_history."""
    from engine.anomaly_screener import _fetch_price_history
    return _fetch_price_history(ticker, end_date, days=days)


def _detect_event_in_window(
    ticker: str,
    window_start: datetime.date,
    window_end: datetime.date,
) -> tuple[datetime.date | None, float | None, float | None]:
    """
    Find the first day in [window_start, window_end] (inclusive) where
    abs daily return > EVENT_SIGMA_THRESHOLD × σ_60d.

    Returns (event_date, event_return, event_sigma) or (None, None, None).
    σ_60d is computed using closes ≤ window_start - 1 (forward-only).
    """
    prices = _fetch_close_history(ticker, window_end + datetime.timedelta(days=2), days=120)
    if prices.empty:
        return None, None, None
    closes = prices["Close"].dropna()
    if len(closes) < ROLLING_VOL_WINDOW + 5:
        return None, None, None
    rets = closes.pct_change().dropna()

    # σ baseline ends at window_start - 1 day to avoid leakage
    pre_window = rets.loc[rets.index < window_start]
    if len(pre_window) < ROLLING_VOL_WINDOW:
        return None, None, None
    sigma_60 = float(pre_window.iloc[-ROLLING_VOL_WINDOW:].std())
    if sigma_60 <= 1e-9:
        return None, None, None

    # Scan window
    in_window = rets.loc[(rets.index >= window_start) & (rets.index <= window_end)]
    for d, r in in_window.items():
        z = abs(r) / sigma_60
        if z >= EVENT_SIGMA_THRESHOLD:
            return d, float(r), float(z)
    return None, None, None


# ── Verify flags whose window has closed ─────────────────────────────────────

def verify_due_flags(as_of: datetime.date | None = None) -> dict:
    """
    For all AnomalyFlag rows where event_occurred IS NULL and
    scan_date + horizon_days <= as_of, run forward verification.

    Sets:
      event_occurred  (Boolean)
      verified_at     (now)
      event_date / event_return / event_sigma  (if occurred)

    Returns counts {n_verified, n_occurred, n_no_event, errors}.
    """
    as_of = as_of or datetime.date.today()
    n_verified = 0
    n_occurred = 0
    n_errors = 0
    with SessionFactory() as session:
        flags = (
            session.query(AnomalyFlag)
            .filter(AnomalyFlag.event_occurred.is_(None))
            .all()
        )
        for f in flags:
            window_end = f.scan_date + datetime.timedelta(days=f.horizon_days)
            if window_end > as_of:
                continue   # window not yet closed
            window_start = f.scan_date + datetime.timedelta(days=1)
            try:
                ev_date, ev_ret, ev_sigma = _detect_event_in_window(
                    f.ticker, window_start, window_end
                )
                f.event_occurred = ev_date is not None
                f.verified_at    = datetime.datetime.utcnow()
                if ev_date is not None:
                    f.event_date   = ev_date
                    f.event_return = ev_ret
                    f.event_sigma  = ev_sigma
                    n_occurred += 1
                n_verified += 1
            except Exception as exc:
                logger.warning("verify_due_flags: error on flag id=%s ticker=%s: %s",
                               f.id, f.ticker, exc)
                n_errors += 1
        session.commit()
    return {
        "as_of":       str(as_of),
        "n_verified":  n_verified,
        "n_occurred":  n_occurred,
        "n_no_event":  n_verified - n_occurred,
        "n_errors":    n_errors,
    }


# ── Universe sweep for recall ────────────────────────────────────────────────

def populate_universe_events(
    scan_date: datetime.date,
    *,
    horizon_days: int = 5,
) -> dict:
    """
    Populate AnomalyUniverseEvent with all 2σ events on date `scan_date` for
    the universe = (current portfolio holdings) ∪ (tickers flagged in last
    horizon_days × 4 days). Skips events with date < CUTOFF_DATE.

    Idempotent — uses unique constraint on (ticker, event_date).

    Returns {tickers_scanned, events_found}.
    """
    if scan_date < CUTOFF_DATE:
        return {"scan_date": str(scan_date), "skipped": "before_cutoff"}

    from engine.anomaly_screener import _get_current_holdings
    holdings = _get_current_holdings(scan_date)
    universe = set(holdings.keys())

    # Add recently-flagged tickers
    cutoff_recent = scan_date - datetime.timedelta(days=horizon_days * 4)
    with SessionFactory() as session:
        recent = (
            session.query(AnomalyFlag.ticker)
            .filter(AnomalyFlag.scan_date >= cutoff_recent)
            .filter(AnomalyFlag.scan_date <= scan_date)
            .distinct()
            .all()
        )
    universe.update(t for (t,) in recent if t)

    events_found = 0
    with SessionFactory() as session:
        for ticker in universe:
            ev_date, ev_ret, ev_sigma = _detect_event_in_window(
                ticker, scan_date, scan_date
            )
            if ev_date is None:
                continue
            sector = holdings.get(ticker, {}).get("sector", "—")
            existing = (
                session.query(AnomalyUniverseEvent)
                .filter(
                    AnomalyUniverseEvent.ticker == ticker,
                    AnomalyUniverseEvent.event_date == ev_date,
                )
                .first()
            )
            if existing:
                continue
            session.add(AnomalyUniverseEvent(
                event_date            = ev_date,
                sector                = sector,
                ticker                = ticker,
                event_return          = ev_ret,
                event_sigma           = ev_sigma,
                detected_by_baseline_a = False,
                detected_by_baseline_b = False,
                detected_by_llm        = False,
            ))
            events_found += 1
        session.commit()

    return {
        "scan_date":       str(scan_date),
        "tickers_scanned": len(universe),
        "events_found":    events_found,
    }


def link_flags_to_universe_events() -> dict:
    """
    Update AnomalyUniverseEvent.detected_by_* booleans + matched_flag_ids
    by joining against AnomalyFlag rows.

    A universe event at (ticker, event_date) is "detected by" detector X
    if there exists an AnomalyFlag row with detector=X, ticker=ticker,
    scan_date <= event_date <= scan_date + horizon_days.
    """
    n_updated = 0
    with SessionFactory() as session:
        events = session.query(AnomalyUniverseEvent).all()
        for ev in events:
            # Find flags whose window covers this event
            flags = (
                session.query(AnomalyFlag)
                .filter(AnomalyFlag.ticker == ev.ticker)
                .filter(AnomalyFlag.scan_date <= ev.event_date)
                .all()
            )
            matching: list[int] = []
            det_a = det_b = det_l = False
            for f in flags:
                window_end = f.scan_date + datetime.timedelta(days=f.horizon_days)
                if ev.event_date > window_end:
                    continue
                matching.append(f.id)
                if f.detector == "rule_baseline_a":
                    det_a = True
                elif f.detector == "rule_baseline_b":
                    det_b = True
                elif f.detector == "llm":
                    det_l = True
            updated = False
            if ev.detected_by_baseline_a != det_a:
                ev.detected_by_baseline_a = det_a; updated = True
            if ev.detected_by_baseline_b != det_b:
                ev.detected_by_baseline_b = det_b; updated = True
            if ev.detected_by_llm != det_l:
                ev.detected_by_llm = det_l; updated = True
            new_match = json.dumps(sorted(matching))
            if (ev.matched_flag_ids or "") != new_match:
                ev.matched_flag_ids = new_match
                updated = True
            if updated:
                n_updated += 1
        session.commit()
    return {"n_updated": n_updated}


# ── M1 metrics ───────────────────────────────────────────────────────────────

def compute_metrics_for_detector(
    detector: str,
    *,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> dict:
    """
    Compute precision, recall, F1 for a single detector over [start_date, end_date].
    end_date defaults to today; start_date defaults to CUTOFF_DATE.
    """
    end_date   = end_date   or datetime.date.today()
    start_date = start_date or CUTOFF_DATE
    if end_date < start_date:
        return {"detector": detector, "precision": None, "recall": None, "f1": None,
                "n_flags": 0, "n_events": 0}

    with SessionFactory() as session:
        flags = (
            session.query(AnomalyFlag)
            .filter(AnomalyFlag.detector == detector)
            .filter(AnomalyFlag.scan_date >= start_date)
            .filter(AnomalyFlag.scan_date <= end_date)
            .filter(AnomalyFlag.event_occurred.isnot(None))   # only verified
            .all()
        )
        events = (
            session.query(AnomalyUniverseEvent)
            .filter(AnomalyUniverseEvent.event_date >= start_date)
            .filter(AnomalyUniverseEvent.event_date <= end_date)
            .all()
        )

    n_flags = len(flags)
    if n_flags == 0:
        return {
            "detector": detector,
            "precision": None, "recall": None, "f1": None,
            "n_flags": 0, "n_events": len(events),
            "n_true_positive": 0, "n_false_positive": 0, "n_false_negative": 0,
            "window": (str(start_date), str(end_date)),
        }

    tp = sum(1 for f in flags if f.event_occurred)
    fp = sum(1 for f in flags if not f.event_occurred)
    precision = tp / max(n_flags, 1)

    # Recall = events covered / total events. detected_by_X is precomputed via link_flags_to_universe_events
    det_attr = {
        "rule_baseline_a": "detected_by_baseline_a",
        "rule_baseline_b": "detected_by_baseline_b",
        "llm":             "detected_by_llm",
    }.get(detector)
    if det_attr is None or len(events) == 0:
        recall = None
    else:
        n_covered = sum(1 for ev in events if getattr(ev, det_attr))
        recall = n_covered / max(len(events), 1)

    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "detector":         detector,
        "precision":        round(precision, 4) if precision is not None else None,
        "recall":           round(recall, 4)    if recall    is not None else None,
        "f1":               round(f1, 4)        if f1        is not None else None,
        "n_flags":          n_flags,
        "n_events":         len(events),
        "n_true_positive":  tp,
        "n_false_positive": fp,
        "n_false_negative": (None if recall is None
                             else int(round((1 - recall) * len(events)))),
        "window":           (str(start_date), str(end_date)),
    }


def compute_m2_for_detector(
    detector: str,
    *,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> dict:
    """Supervisor acceptance rate over the window."""
    end_date   = end_date   or datetime.date.today()
    start_date = start_date or CUTOFF_DATE
    with SessionFactory() as session:
        flags = (
            session.query(AnomalyFlag)
            .filter(AnomalyFlag.detector == detector)
            .filter(AnomalyFlag.scan_date >= start_date)
            .filter(AnomalyFlag.scan_date <= end_date)
            .filter(AnomalyFlag.supervisor_useful.isnot(None))
            .all()
        )
    n = len(flags)
    n_useful = sum(1 for f in flags if f.supervisor_useful)
    return {
        "detector":     detector,
        "n_labeled":    n,
        "n_useful":     n_useful,
        "accept_rate":  (n_useful / n) if n > 0 else None,
        "window":       (str(start_date), str(end_date)),
    }


def compute_composite_verdict(
    *,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> dict:
    """
    Implement spec §1.4 composite verdict:
      CLEAR_WIN     iff M1_diff_A ≥ +5pp AND M1_diff_B ≥ +5pp AND M2_LLM ≥ 30%
      CLEAR_LOSS   iff M1_diff_A ≤ -5pp OR  M1_diff_B ≤ -5pp
      INCONCLUSIVE  otherwise
      CATASTROPHIC iff (M1_LLM precision < 30%) AND (M1_baseline_a precision < 30%)
                       AND (M2_LLM accept_rate < 30%) at 60d hard checkpoint
    """
    m1_a   = compute_metrics_for_detector("rule_baseline_a", start_date=start_date, end_date=end_date)
    m1_b   = compute_metrics_for_detector("rule_baseline_b", start_date=start_date, end_date=end_date)
    m1_llm = compute_metrics_for_detector("llm",             start_date=start_date, end_date=end_date)
    m2_llm = compute_m2_for_detector("llm", start_date=start_date, end_date=end_date)

    pa = (m1_a["precision"] or 0)
    pb = (m1_b["precision"] or 0)
    pl = (m1_llm["precision"] or 0)
    diff_a_pp = (pl - pa) * 100
    diff_b_pp = (pl - pb) * 100
    m2_pct    = (m2_llm["accept_rate"] or 0) * 100

    catastrophic = (
        m1_llm["n_flags"] > 0 and m1_a["n_flags"] > 0 and
        pl < 0.30 and pa < 0.30 and m2_pct < M2_MIN_ACCEPT_PCT
    )
    if catastrophic:
        verdict = "CATASTROPHIC"
    elif diff_a_pp >= COMPOSITE_WIN_PP and diff_b_pp >= COMPOSITE_WIN_PP and m2_pct >= M2_MIN_ACCEPT_PCT:
        verdict = "CLEAR_WIN"
    elif diff_a_pp <= -COMPOSITE_WIN_PP or diff_b_pp <= -COMPOSITE_WIN_PP:
        verdict = "CLEAR_LOSS"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict":         verdict,
        "diff_a_pp":       round(diff_a_pp, 2),
        "diff_b_pp":       round(diff_b_pp, 2),
        "m2_llm_pct":      round(m2_pct, 2),
        "m1_baseline_a":   m1_a,
        "m1_baseline_b":   m1_b,
        "m1_llm":          m1_llm,
        "m2_llm":          m2_llm,
        "thresholds": {
            "composite_win_pp":  COMPOSITE_WIN_PP,
            "m2_min_accept_pct": M2_MIN_ACCEPT_PCT,
        },
        "as_of": str(end_date or datetime.date.today()),
    }


# ── Cron entry point ─────────────────────────────────────────────────────────

def run_daily_verification(scan_date: datetime.date | None = None) -> dict:
    """
    Daily cron — D4.6 will call this. Three-step pipeline:
      1. verify_due_flags(today)
      2. populate_universe_events(today)
      3. link_flags_to_universe_events()

    Returns combined stats for the day.
    """
    today = scan_date or datetime.date.today()
    a = verify_due_flags(today)
    b = populate_universe_events(today)
    c = link_flags_to_universe_events()
    return {"as_of": str(today), "verify": a, "universe_sweep": b, "link": c}
