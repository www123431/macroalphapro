"""engine/research/liveness_heartbeat.py — P0 of the liveness layer
(2026-06-02 senior ops protocol).

Doctrine:
  Existing monitoring (circuit_breaker / risk_manager / dq_inspector /
  decay_sentinel / watchdog) tells you *whether the system is HEALTHY*.
  This module tells you whether it IS RUNNING AT ALL — the silent
  failure that no quality gate catches because no quality gate fires
  when the cron itself didn't fire.

  Heartbeat is written at the *end* of every daily run, success or
  graceful halt. A separate watcher (scripts/check_liveness.py) reads
  the ledger and raises a NO_SHOW alarm if today's expected heartbeat
  is missing past a wall-clock deadline.

Ledger: data/research/liveness_heartbeat.jsonl, one row per daily run.
Append-only, newest-last (read newest-first via read_recent).

Row shape (stable contract — UI + check_liveness depend on it):
  {
    "ts":               ISO 8601 UTC (write moment),
    "as_of":            "YYYY-MM-DD" trade date,
    "exit_code":        int from main() return (0 = success),
    "status":           "success" | "partial" | "halt_cb" | "halt_risk" |
                        "halt_dq" | "orchestrator_failed" | "feed_partial",
    "n_orders":         int submitted to broker,
    "n_fills":          int reported by broker,
    "equity_before":    float account equity at start of submit, USD,
    "n_strategies":     int from PaperTradeRunResult.signals,
    "gross_weight":     float abs sum of intended weights,
    "halted_at_step":   str | null  (which step blocked, if any),
    "broker_ack":       "alpaca_paper" | "dryrun" | null,
    "log_file":         path to the daily_run_*.log produced this run,
    "errors":           [str, ...] non-blocking error summaries,
  }

The check_liveness script reads this. Frontend reads via
/api/research/liveness/* endpoint (P2).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVENESS_LEDGER = REPO_ROOT / "data" / "research" / "liveness_heartbeat.jsonl"


# ── Stable contract: status enums ──────────────────────────────────


STATUS_SUCCESS              = "success"
STATUS_FEED_PARTIAL         = "feed_partial"       # exit 3
STATUS_DB_PARTIAL           = "db_partial"          # exit 1
STATUS_ORCHESTRATOR_FAILED  = "orchestrator_failed" # exit 2
STATUS_HALT_CB              = "halt_cb"             # exit 4 circuit-breaker SEVERE
STATUS_HALT_RISK            = "halt_risk"           # exit 5 risk manager pre-trade
STATUS_HALT_DQ              = "halt_dq"             # exit 6 DQ inspector pre/post

_EXIT_TO_STATUS = {
    0: STATUS_SUCCESS,
    1: STATUS_DB_PARTIAL,
    2: STATUS_ORCHESTRATOR_FAILED,
    3: STATUS_FEED_PARTIAL,
    4: STATUS_HALT_CB,
    5: STATUS_HALT_RISK,
    6: STATUS_HALT_DQ,
}


def status_from_exit(exit_code: int) -> str:
    """Map daily-run exit code to a heartbeat status enum."""
    return _EXIT_TO_STATUS.get(int(exit_code), f"unknown_exit_{exit_code}")


# ── Recording side (called by run_paper_trade_daily.py tail) ───────


def record_run(
    *,
    as_of:           _dt.date,
    exit_code:       int,
    n_orders:        Optional[int]   = None,
    n_fills:         Optional[int]   = None,
    equity_before:   Optional[float] = None,
    n_strategies:    Optional[int]   = None,
    gross_weight:    Optional[float] = None,
    halted_at_step:  Optional[str]   = None,
    broker_ack:      Optional[str]   = None,
    log_file:        Optional[Path]  = None,
    errors:          Optional[list[str]] = None,
    broker_echo:     Optional[dict]  = None,   # P1a: broker_reconciliation.reconcile() output
    nav_anomaly:     Optional[dict]  = None,   # P1b: nav_anomaly.record_nav() output
    data_freshness:  Optional[dict]  = None,   # P0c: data_freshness.summarize() output
    data_sources:    Optional[list]  = None,   # P0c: per-source freshness rows
) -> dict:
    """Append one row to liveness_heartbeat.jsonl. Returns the row.

    Designed to be called from a `finally:` block so it fires whether
    the run succeeded or any step raised. Best-effort: any IO failure
    here is logged and swallowed; this layer must NEVER mask the real
    exit code or crash the cron."""
    row = {
        "ts":             _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "as_of":          as_of.isoformat() if isinstance(as_of, _dt.date) else str(as_of),
        "exit_code":      int(exit_code),
        "status":         status_from_exit(exit_code),
        "n_orders":       int(n_orders)   if n_orders   is not None else None,
        "n_fills":        int(n_fills)    if n_fills    is not None else None,
        "equity_before":  float(equity_before) if equity_before is not None else None,
        "n_strategies":   int(n_strategies)    if n_strategies is not None else None,
        "gross_weight":   float(gross_weight)  if gross_weight is not None else None,
        "halted_at_step": halted_at_step,
        "broker_ack":     broker_ack,
        "log_file":       str(log_file) if log_file is not None else None,
        "errors":         list(errors or []),
        "broker_echo":    broker_echo,
        "nav_anomaly":    nav_anomaly,
        "data_freshness": data_freshness,
        "data_sources":   data_sources,
    }
    try:
        LIVENESS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LIVENESS_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        logger.exception("liveness heartbeat append failed (non-fatal)")
    return row


# ── Reading side (UI + check_liveness) ─────────────────────────────


def _iter_rows() -> Iterable[dict]:
    if not LIVENESS_LEDGER.is_file():
        return
    with LIVENESS_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def read_recent(limit: int = 30) -> list[dict]:
    """Return up to `limit` most recent heartbeat rows, newest first."""
    rows = list(_iter_rows())
    rows.reverse()
    return rows[: max(1, int(limit))]


def latest_heartbeat() -> Optional[dict]:
    """Newest row, or None if ledger empty."""
    rows = list(_iter_rows())
    return rows[-1] if rows else None


def heartbeat_for_date(as_of: _dt.date) -> Optional[dict]:
    """Find the most recent row for a specific trade date."""
    target = as_of.isoformat() if isinstance(as_of, _dt.date) else str(as_of)
    last_match: Optional[dict] = None
    for row in _iter_rows():
        if str(row.get("as_of")) == target:
            last_match = row
    return last_match


# ── No-show alarm ──────────────────────────────────────────────────


# Default expectations (China-Standard-Time aligned per existing 06:00
# SGT paper-trade cron). Adjust if cron is moved.
DEFAULT_EXPECTED_HOUR_UTC = 22    # 06:00 SGT = 22:00 prev-day UTC
DEFAULT_NO_SHOW_GRACE_MIN = 90    # raise if heartbeat is 90 min late


def _most_recent_expected_run_date(
    now_utc: _dt.datetime, *, expected_hour_utc: int, grace_min: int,
    trading_days_only: bool,
) -> Optional[_dt.date]:
    """The most recent weekday whose run deadline has already passed
    relative to now_utc. None if no deadline in the last 7 days has
    passed (essentially never — but guard anyway)."""
    for back in range(8):
        candidate = now_utc.date() - _dt.timedelta(days=back)
        if trading_days_only and candidate.weekday() >= 5:
            continue
        deadline = (
            _dt.datetime.combine(candidate, _dt.time(hour=expected_hour_utc))
            + _dt.timedelta(minutes=grace_min)
        )
        if now_utc >= deadline:
            return candidate
    return None


def assess_liveness(
    *,
    now_utc:                _dt.datetime,
    expected_hour_utc:      int   = DEFAULT_EXPECTED_HOUR_UTC,
    no_show_grace_min:      int   = DEFAULT_NO_SHOW_GRACE_MIN,
    trading_days_only:      bool  = True,
) -> dict:
    """Return a verdict dict describing the most recent expected-run
    state. Looks at the MOST RECENT WEEKDAY whose deadline has passed,
    not just today — catches "yesterday's cron didn't fire and we
    haven't noticed yet".

    Verdicts:
      OK              — last expected run's heartbeat is present + success
      WARN_STATUS     — heartbeat present but status is HALT or partial
      ALERT_NO_SHOW   — last expected run's deadline passed; no heartbeat
      INFO_OFF_HOURS  — no run-deadline has passed yet (very early in
                        the trading day before the run window opens)
      INFO_WEEKEND    — current day is weekend AND no recent weekday
                        deadline has passed (e.g. checked Sat morning
                        before any weekday this week)

    Designed cheap (single ledger scan) so check_liveness can run every
    15 min from cron with negligible overhead."""
    today = now_utc.date()
    expected = _most_recent_expected_run_date(
        now_utc,
        expected_hour_utc=expected_hour_utc,
        grace_min=no_show_grace_min,
        trading_days_only=trading_days_only,
    )

    # No expected weekday deadline has passed (very early morning UTC
    # of a Monday after a weekend, or test fixture before the deadline).
    if expected is None:
        if trading_days_only and today.weekday() >= 5:
            return {
                "verdict":     "INFO_WEEKEND",
                "explanation": (
                    f"{today} is a weekend and no weekday run-deadline has "
                    f"passed yet."
                ),
                "as_of":       today.isoformat(),
                "checked_at":  now_utc.isoformat(),
            }
        return {
            "verdict":     "INFO_OFF_HOURS",
            "explanation": "Before today's run deadline; nothing to assess yet.",
            "as_of":       today.isoformat(),
            "checked_at":  now_utc.isoformat(),
        }

    latest = latest_heartbeat()
    age_min: Optional[float] = None
    if latest is not None:
        try:
            last_ts = _dt.datetime.fromisoformat(
                str(latest.get("ts", "")).rstrip("Z")
            )
            age_min = round((now_utc - last_ts).total_seconds() / 60.0, 1)
        except Exception:
            age_min = None

    expected_hb = heartbeat_for_date(expected)
    if expected_hb is None:
        return {
            "verdict":     "ALERT_NO_SHOW",
            "explanation": (
                f"Expected a heartbeat for {expected.isoformat()} but none "
                f"recorded. The {expected.isoformat()} run-deadline has "
                f"passed; the cron likely failed to start. "
                + (f"Last heartbeat was for as_of={latest.get('as_of')} "
                   f"({age_min} min ago)." if latest else
                   "No prior heartbeats in the ledger.")
            ),
            "as_of":      expected.isoformat(),
            "checked_at": now_utc.isoformat(),
            "latest":     latest,
            "age_min":    age_min,
        }

    if expected_hb.get("status") == STATUS_SUCCESS:
        # P0c (2026-06-02): even when cron-OK, downgrade to WARN if any
        # data source is DEAD — that's the "cron runs, data dies" silent
        # failure the user surfaced via the 21-day-stale NAV screenshot.
        # 2026-06-14 fix: the heartbeat's embedded `data_freshness`
        # field is a SNAPSHOT taken at heartbeat-write time and goes
        # stale; e.g. fixing a freshness probe today doesn't change a
        # heartbeat written yesterday. Re-compute LIVE on every read so
        # the banner reflects reality, not history.
        try:
            from engine.research.data_freshness import check_sources, summarize
            df = summarize(check_sources())
        except Exception:
            df = expected_hb.get("data_freshness") or {}
        worst = df.get("worst_status")
        if worst in ("dead",):
            return {
                "verdict":     "WARN_STATUS",
                "explanation": (
                    f"Cron ran successfully for {expected.isoformat()} but "
                    f"{df.get('headline','data source DEAD')}. "
                    f"Investigate the upstream writer."
                ),
                "as_of":      expected.isoformat(),
                "checked_at": now_utc.isoformat(),
                "latest":     expected_hb,
                "age_min":    age_min,
            }
        return {
            "verdict":     "OK",
            "explanation": (
                f"Run for {expected.isoformat()} succeeded "
                f"(n_orders={expected_hb.get('n_orders')}, "
                f"n_fills={expected_hb.get('n_fills')})."
            ),
            "as_of":      expected.isoformat(),
            "checked_at": now_utc.isoformat(),
            "latest":     expected_hb,
            "age_min":    age_min,
        }

    return {
        "verdict":     "WARN_STATUS",
        "explanation": (
            f"Run for {expected.isoformat()} recorded non-success status="
            f"{expected_hb.get('status')!r}, halted_at_step="
            f"{expected_hb.get('halted_at_step')!r}. Investigate before next "
            f"run."
        ),
        "as_of":      expected.isoformat(),
        "checked_at": now_utc.isoformat(),
        "latest":     expected_hb,
        "age_min":    age_min,
    }
