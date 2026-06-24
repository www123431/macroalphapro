"""
engine/agents/dq_inspector/source_inspectors.py — Phase 4 per-source helpers.

Each function inspects ONE data source and returns SourceCheckResult.
gates.py composes these into Breach objects for the 10 detector modes.

Source inspectors are pure functional + side-effect free (read-only on
DB / file system; no writes). Failures degrade gracefully — exception
inside an inspector returns is_breach=False with extra={"error": "..."}
to avoid masking real data-quality issues with bug noise.

Source coverage:
  - FRED freshness via engine.macro_fetcher._fetch_observations
  - yfinance bab_compat cache via file mtime
  - D-PEAD panel cache via file mtime
  - S&P 500 feed via SP500AnnouncementEvent.detected_at MAX
  - K1 / D-PEAD universe coverage via universe loader + price snapshot
  - Price anomaly via class-aware caps (Q3 resolution)
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd

from engine.agents.dq_inspector.thresholds import (
    DQ_THRESHOLDS,
    FRED_DEFAULT_FALLBACK_BDAYS,
    FRED_MAX_STALENESS_BDAYS,
    MODE_7_CAP_BY_TICKER_CLASS,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SourceCheckResult:
    """One per-source check output.

    is_breach          : True if the source state violates threshold
    observed_value     : measurable quantity (days stale / fraction / count)
    threshold          : the rule's threshold (same units as observed)
    extra              : free-form context for narrator / persist
    """
    source_id:      str
    is_breach:      bool
    observed_value: float
    threshold:      float
    extra:          dict


# ──────────────────────────────────────────────────────────────────────────────
# Business-day arithmetic helper
# ──────────────────────────────────────────────────────────────────────────────
def _bdays_between(start: datetime.date, end: datetime.date) -> int:
    """Count business days (Mon-Fri) between start (exclusive) and end (inclusive).

    NYSE-holiday-naive — DQ Inspector uses calendar weekdays, not NYSE
    trading calendar, because FRED publication cadence is calendar-based
    (US gov holidays delay but rarely synchronise with NYSE).
    """
    if start >= end:
        return 0
    days = 0
    cur = start + datetime.timedelta(days=1)
    while cur <= end:
        if cur.weekday() < 5:   # Mon=0, Fri=4
            days += 1
        cur += datetime.timedelta(days=1)
    return days


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1 — FRED series staleness (per-series threshold)
# ──────────────────────────────────────────────────────────────────────────────
def check_fred_freshness(
    as_of:      datetime.date,
    series_id:  str,
) -> SourceCheckResult:
    """Fetch most-recent FRED observation; compute business-day staleness.

    Returns is_breach=True iff (today - last_obs_date).bdays exceeds
    the per-series threshold from FRED_MAX_STALENESS_BDAYS. Unknown
    series fall back to FRED_DEFAULT_FALLBACK_BDAYS.

    Graceful degradation: FRED API failure → is_breach=False + extra
    flags "api_unreachable"; this avoids HALTing the daily cycle on
    transient network issues while still flagging real staleness.
    """
    threshold = FRED_MAX_STALENESS_BDAYS.get(series_id, FRED_DEFAULT_FALLBACK_BDAYS)
    try:
        from engine.macro_fetcher import _fetch_observations, _get_api_key
        api_key = _get_api_key()
        obs = _fetch_observations(series_id, n=1, api_key=api_key)
        if not obs:
            return SourceCheckResult(
                source_id      = f"fred:{series_id}",
                is_breach      = False,        # don't HALT on API empty / network issue
                observed_value = float("nan"),
                threshold      = float(threshold),
                extra          = {
                    "error":       "no_observations_returned",
                    "series_id":   series_id,
                    "fallback":    "FRED API empty — daily cycle proceeds; investigate",
                },
            )
        last_obs_date = datetime.date.fromisoformat(obs[0]["date"])
        bdays_stale = _bdays_between(last_obs_date, as_of)
        return SourceCheckResult(
            source_id      = f"fred:{series_id}",
            is_breach      = bdays_stale > threshold,
            observed_value = float(bdays_stale),
            threshold      = float(threshold),
            extra          = {
                "series_id":         series_id,
                "last_obs_date":     last_obs_date.isoformat(),
                "last_obs_value":    obs[0]["value"],
                "is_known_series":   series_id in FRED_MAX_STALENESS_BDAYS,
            },
        )
    except Exception as exc:
        logger.warning("check_fred_freshness(%s) failed: %s", series_id, exc)
        return SourceCheckResult(
            source_id      = f"fred:{series_id}",
            is_breach      = False,
            observed_value = float("nan"),
            threshold      = float(threshold),
            extra          = {"error": str(exc), "series_id": series_id},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2 — yfinance bab_compat cache freshness
# ──────────────────────────────────────────────────────────────────────────────
def check_yfinance_bab_cache(as_of: datetime.date) -> SourceCheckResult:
    """File mtime check on engine.factors.bab_compat cache parquet.

    Returns is_breach=True iff cache mtime is older than today by more
    than threshold trading days. Trading day approximation: weekdays
    only (no NYSE holiday adjustment; over-conservative is acceptable
    for a data-quality gate).
    """
    threshold_days = DQ_THRESHOLDS.yfinance_bab_cache_max_trading_days
    # Common candidate paths — bab_compat may have evolved location
    candidates = [
        Path("data/factor_library_in_sample/bab_compat_cache.parquet"),
        Path("data/factor_cache/bab_compat.parquet"),
        Path("data/cache/bab_compat.parquet"),
    ]
    cache_path = next((p for p in candidates if p.exists()), None)
    if cache_path is None:
        return SourceCheckResult(
            source_id      = "yfinance:bab_compat_cache",
            is_breach      = True,   # missing cache = data unavailable = HARD HALT
            observed_value = float("inf"),
            threshold      = float(threshold_days),
            extra          = {
                "error":          "cache_file_not_found",
                "search_paths":   [str(p) for p in candidates],
            },
        )
    mtime = datetime.datetime.fromtimestamp(cache_path.stat().st_mtime).date()
    bdays_stale = _bdays_between(mtime, as_of)
    return SourceCheckResult(
        source_id      = "yfinance:bab_compat_cache",
        is_breach      = bdays_stale > threshold_days,
        observed_value = float(bdays_stale),
        threshold      = float(threshold_days),
        extra          = {
            "cache_path":      str(cache_path),
            "mtime":           mtime.isoformat(),
            "cache_size_kb":   cache_path.stat().st_size // 1024,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3 — D-PEAD panel cache freshness
# ──────────────────────────────────────────────────────────────────────────────
def check_pead_panel_cache(as_of: datetime.date) -> SourceCheckResult:
    """File mtime check on data/path_c_dhs/_pead_ts_signal_panel.parquet."""
    threshold_days = DQ_THRESHOLDS.pead_panel_max_calendar_days
    panel_path = Path("data/path_c_dhs/_pead_ts_signal_panel.parquet")
    if not panel_path.exists():
        return SourceCheckResult(
            source_id      = "internal_parquet:pead_panel",
            is_breach      = False,  # missing → SOFT WARN by spec (mode 3 is WARN)
            observed_value = float("inf"),
            threshold      = float(threshold_days),
            extra          = {"error": "file_not_found", "path": str(panel_path)},
        )
    mtime = datetime.datetime.fromtimestamp(panel_path.stat().st_mtime).date()
    days_stale = (as_of - mtime).days
    return SourceCheckResult(
        source_id      = "internal_parquet:pead_panel",
        is_breach      = days_stale > threshold_days,
        observed_value = float(days_stale),
        threshold      = float(threshold_days),
        extra          = {
            "path":            str(panel_path),
            "mtime":           mtime.isoformat(),
            "panel_size_mb":   panel_path.stat().st_size // (1024 * 1024),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mode 4 — S&P 500 reconstitution feed freshness
# ──────────────────────────────────────────────────────────────────────────────
def check_sp500_feed_freshness(as_of: datetime.date) -> SourceCheckResult:
    """SELECT MAX(detected_at) FROM SP500AnnouncementEvent."""
    threshold_days = DQ_THRESHOLDS.sp500_feed_max_calendar_days
    try:
        from engine.db_models import SP500AnnouncementEvent
        from engine.memory import SessionFactory
        from sqlalchemy import func as sql_func
        sess = SessionFactory()
        try:
            max_detected = sess.query(
                sql_func.max(SP500AnnouncementEvent.detected_at)
            ).scalar()
        finally:
            sess.close()
        if max_detected is None:
            return SourceCheckResult(
                source_id      = "wikipedia+edgar:sp500_feed",
                is_breach      = False,        # empty table = no detection history yet
                observed_value = float("inf"),
                threshold      = float(threshold_days),
                extra          = {"error": "no_rows_in_table"},
            )
        last_date = max_detected.date() if hasattr(max_detected, "date") else max_detected
        days_stale = (as_of - last_date).days
        return SourceCheckResult(
            source_id      = "wikipedia+edgar:sp500_feed",
            is_breach      = days_stale > threshold_days,
            observed_value = float(days_stale),
            threshold      = float(threshold_days),
            extra          = {
                "last_detected_at": last_date.isoformat(),
            },
        )
    except Exception as exc:
        logger.warning("check_sp500_feed_freshness failed: %s", exc)
        return SourceCheckResult(
            source_id      = "wikipedia+edgar:sp500_feed",
            is_breach      = False,
            observed_value = float("nan"),
            threshold      = float(threshold_days),
            extra          = {"error": str(exc)},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Mode 5 / 6 — universe coverage
# ──────────────────────────────────────────────────────────────────────────────
def check_universe_coverage(
    universe_name:  str,
    n_with_data:    int,
    expected_n:     int,
    min_frac:       float,
) -> SourceCheckResult:
    """Pure functional — count comparison only. Caller supplies the
    actual coverage count from universe-specific source (yfinance
    snapshot / rdq panel cache / etc.).

    Returns is_breach=True iff (n_with_data / expected_n) < min_frac.
    """
    if expected_n <= 0:
        return SourceCheckResult(
            source_id      = f"universe:{universe_name}",
            is_breach      = False,
            observed_value = 1.0,
            threshold      = float(min_frac),
            extra          = {"error": "expected_n is zero or negative"},
        )
    coverage = float(n_with_data) / float(expected_n)
    return SourceCheckResult(
        source_id      = f"universe:{universe_name}",
        is_breach      = coverage < min_frac,
        observed_value = coverage,
        threshold      = float(min_frac),
        extra          = {
            "universe_name":  universe_name,
            "n_with_data":    n_with_data,
            "expected_n":     expected_n,
            "n_missing":      max(0, expected_n - n_with_data),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mode 7 — class-aware price anomaly
# ──────────────────────────────────────────────────────────────────────────────
def classify_ticker(
    ticker:             str,
    ticker_to_sleeves:  Optional[dict[str, set[str]]] = None,
    registry=None,
) -> str:
    """Map a ticker to its anomaly-cap class.

    Senior upgrade (mirrors Risk Manager Upgrade #1): if
    ``ticker_to_sleeves`` is supplied by the orchestrator (recommended
    production usage), classification routes through the live registry
    sleeve_class — institutional Basel-style "use the strictest class
    when in doubt" when a ticker is in multiple sleeves.

    Without ticker_to_sleeves (smoke testing / ad-hoc use), falls back
    to a heuristic ladder:
      1. CTA fund whitelist (PQTIX hard-coded)
      2. Insurance + K1-universe ETFs
      3. Default → single_stock (conservative 50% cap = LESS likely to
         halt on unknown tickers; matches institutional "innocent until
         proven anomalous" stance for tickers outside known universes)
    """
    ticker = ticker.upper()

    # Production path: orchestrator supplied registry mapping
    if ticker_to_sleeves and ticker in ticker_to_sleeves:
        sleeve_ids = ticker_to_sleeves[ticker]
        if registry is None:
            try:
                from engine.strategies import get_registry
                registry = get_registry()
            except Exception:
                registry = None
        if registry is not None:
            # Sleeve_class → DQ class mapping
            from engine.strategies import SleeveClass
            sleeve_class_to_dq = {
                SleeveClass.ALPHA_EQUITY_LS:    "etf",
                SleeveClass.ALPHA_SINGLE_STOCK: "single_stock",
                SleeveClass.INSURANCE:          "etf",
                SleeveClass.CTA_OVERLAY:        "fund_of_funds",
            }
            # Choose STRICTEST cap (smallest) across all sleeves the ticker is in
            candidates: list[tuple[float, str]] = []
            for sid in sleeve_ids:
                try:
                    sleeve = registry.get_sleeve(sid)
                except KeyError:
                    continue
                dq_class = sleeve_class_to_dq.get(sleeve.sleeve_class, "unknown")
                cap = MODE_7_CAP_BY_TICKER_CLASS[dq_class]
                candidates.append((cap, dq_class))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                return candidates[0][1]

    # Heuristic fallback (smoke testing / ticker outside registry)
    if ticker == "PQTIX":
        return "fund_of_funds"
    if ticker in {"TLT", "GLD"}:
        return "etf"
    try:
        from engine.path_c.k1_universe import get_k1_universe
        k1_tickers = set(get_k1_universe().values())
        if ticker in k1_tickers:
            return "etf"
    except Exception:
        pass
    return "single_stock"


def check_price_anomaly(
    ticker:             str,
    daily_return:       float,
    ticker_to_sleeves:  Optional[dict[str, set[str]]] = None,
    registry=None,
) -> SourceCheckResult:
    """Class-aware Mode 7 detector. Cap depends on classify_ticker."""
    cls = classify_ticker(ticker, ticker_to_sleeves=ticker_to_sleeves, registry=registry)
    cap = MODE_7_CAP_BY_TICKER_CLASS.get(cls, MODE_7_CAP_BY_TICKER_CLASS["unknown"])
    return SourceCheckResult(
        source_id      = f"price_anomaly:{ticker}",
        is_breach      = abs(daily_return) > cap,
        observed_value = abs(daily_return),
        threshold      = cap,
        extra          = {
            "ticker":         ticker,
            "ticker_class":   cls,
            "signed_return":  daily_return,
            "used_registry":  ticker_to_sleeves is not None and ticker in (ticker_to_sleeves or {}),
        },
    )
