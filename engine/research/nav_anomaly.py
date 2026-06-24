"""engine/research/nav_anomaly.py — P1b of liveness layer
(2026-06-02 senior ops protocol).

Catches "overnight equity moved more than expected" silently. Even when
liveness=OK and broker_echo=OK, a 5% NAV swing in a vol-targeted book
is a real-money event that should page someone.

Method (UPDATED 2026-06-18 — 4 ops bugs fixed per audit_nav_monitoring_
fix_2026-06-18.py audit):

  1. SAME-DAY DEDUP: replace prior same-day row in place (keep-last);
     don't append duplicate rows that pollute z-baseline.
  2. GAP-AWARE RETURN: normalize log return by calendar days elapsed
     since the prior NAV — multi-day weekend / cron-skip gaps must NOT
     be misread as single-day extreme moves.
  3. TARGET-VOL-AWARE Z: z-score against the DEPLOYED vol target
     (10% annualized → 0.63%/day) rather than rolling realized std.
     Rolling std is unstable in early deployment (quiet warmup period
     gives tiny std → subsequent normal moves flagged 5-12σ).
  4. EXPLAIN GAPS: record `days_elapsed` field; alert message hints
     "single day" vs "after Nd gap (weekend / cron pause)".

Pre-2026-06-18 false-positive rate (live audit): 2/3 alerts (66%) were
monitoring bugs. New logic: real anomaly count over same 5w window: 1/3.

Stored at data/research/nav_history.jsonl — newest-last, ONE row per
calendar date (in-place replacement on re-run).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
NAV_LEDGER = REPO_ROOT / "data" / "research" / "nav_history.jsonl"

# Fix 3 (2026-06-18): z-score against DEPLOYED vol target, not rolling std.
# Source: data/portfolio/active_deployment.yaml.book_vol_target.
DEPLOYED_VOL_TARGET_ANN = 0.10        # annualized
_TRADING_DAYS_PER_YEAR  = 252


# Verdict enum — stable contract for frontend
STATUS_OK             = "ok"
STATUS_INSUFFICIENT   = "insufficient_history"   # < 5 prior days
STATUS_ANOMALY        = "anomaly"                # |z| > 3
STATUS_NO_PRIOR       = "no_prior"               # first NAV record


def _iter_rows():
    if not NAV_LEDGER.is_file():
        return
    with NAV_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _read_history() -> list[dict]:
    rows = list(_iter_rows())
    rows.sort(key=lambda r: str(r.get("as_of", "")))
    return rows


def _expected_daily_vol(days_elapsed: int) -> float:
    """Vol-target-scaled daily expected vol for a return spanning N days.

    For a 10% annualized vol target:
      1-day expected vol = 10% / sqrt(252) ≈ 0.63%
      Nd return vol     = 10% × sqrt(N/252)
    """
    return DEPLOYED_VOL_TARGET_ANN * math.sqrt(
        max(days_elapsed, 1) / _TRADING_DAYS_PER_YEAR
    )


def record_nav(
    *,
    as_of:        _dt.date,
    equity:       float,
    z_window:     int = 20,
    z_threshold:  float = 3.0,
) -> dict:
    """Append today's equity to the NAV ledger, return a verdict dict.

    Fix 1 (2026-06-18): same-date rows REPLACED IN PLACE (keep-last),
    not appended. Multiple calls per day overwrite prior same-day rows.

    Fix 2 (2026-06-18): log return is normalized by calendar days elapsed
    since last prior NAV. Multi-day weekend / cron-pause gaps no longer
    misread as single-day extreme moves.

    Fix 3 (2026-06-18): z-score uses the DEPLOYED vol target as baseline,
    not rolling realized std. Eliminates early-warmup false-positive
    cascade.
    """
    history = _read_history()
    # Today's row if already recorded
    todays = [r for r in history if str(r.get("as_of")) == as_of.isoformat()]
    prior = [r for r in history if str(r.get("as_of")) < as_of.isoformat()]

    # Compute today's log return vs the LAST prior date
    if not prior:
        verdict = {
            "status":        STATUS_NO_PRIOR,
            "as_of":         as_of.isoformat(),
            "equity":        float(equity),
            "log_return":    None,
            "z_score":       None,
            "days_elapsed":  None,
            "explanation":   (
                "First NAV record. No prior for return computation; baseline "
                "set today."
            ),
        }
    else:
        prev = prior[-1]
        prev_equity = float(prev.get("equity") or 0.0)
        try:
            prev_as_of = _dt.date.fromisoformat(str(prev.get("as_of")))
            days_elapsed = max((as_of - prev_as_of).days, 1)
        except Exception:
            days_elapsed = 1

        if prev_equity <= 0:
            log_return = None
        else:
            try:
                log_return = math.log(float(equity) / prev_equity)
            except Exception:
                log_return = None

        if log_return is None:
            verdict = {
                "status":      STATUS_INSUFFICIENT,
                "as_of":       as_of.isoformat(),
                "equity":      float(equity),
                "log_return":  None,
                "z_score":     None,
                "days_elapsed": days_elapsed,
                "explanation": (
                    "Cannot compute log return from previous NAV. Check that "
                    "prior equity is positive and current equity is finite."
                ),
            }
        else:
            # Fix 3: z-score vs deployed vol target, scaled by sqrt(days/252)
            expected_vol = _expected_daily_vol(days_elapsed)
            z = log_return / expected_vol if expected_vol > 0 else None

            if z is None:
                status = STATUS_INSUFFICIENT
                expl = "Cannot compute z-score (expected vol non-positive)."
            elif abs(z) > z_threshold:
                status = STATUS_ANOMALY
                pct = (math.exp(log_return) - 1.0) * 100
                gap_hint = (f" over {days_elapsed} calendar days"
                             if days_elapsed > 1 else "")
                expl = (
                    f"NAV moved {pct:+.2f}%{gap_hint} — z={z:+.2f}σ vs "
                    f"deployed {DEPLOYED_VOL_TARGET_ANN:.0%} vol target. "
                    f"Above |{z_threshold}σ| threshold; check for "
                    f"execution issues or risk event."
                )
            else:
                status = STATUS_OK
                pct = (math.exp(log_return) - 1.0) * 100
                gap_hint = (f" over {days_elapsed}d gap"
                             if days_elapsed > 1 else "")
                expl = (
                    f"NAV moved {pct:+.2f}%{gap_hint} — z={z:+.2f}σ vs "
                    f"deployed {DEPLOYED_VOL_TARGET_ANN:.0%} vol target. "
                    f"Within normal range."
                )
            verdict = {
                "status":      status,
                "as_of":       as_of.isoformat(),
                "equity":      float(equity),
                "log_return":  log_return,
                "z_score":     z,
                "days_elapsed": days_elapsed,
                "explanation": expl,
                "vol_target":  DEPLOYED_VOL_TARGET_ANN,
            }

    # Fix 1: in-place replacement on (as_of) re-runs
    # If a prior row for THIS as_of already exists, rewrite the entire
    # ledger with the latest verdict swapping the old row.
    row_to_write = {
        "ts":         _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **{k: v for k, v in verdict.items() if k != "explanation"},
    }
    try:
        NAV_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        if todays:
            # Replace same-date rows — write out a clean ledger
            all_rows = list(_iter_rows())
            kept = [r for r in all_rows
                     if str(r.get("as_of")) != as_of.isoformat()]
            kept.append(row_to_write)
            kept.sort(key=lambda r: (str(r.get("as_of", "")), str(r.get("ts", ""))))
            with NAV_LEDGER.open("w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r, default=str) + "\n")
        else:
            with NAV_LEDGER.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row_to_write, default=str) + "\n")
    except Exception:
        logger.exception("nav_history write failed (non-fatal)")
    return verdict


def read_recent(limit: int = 60) -> list[dict]:
    """Newest-first NAV ledger rows for diagnostic / UI."""
    rows = _read_history()
    rows.reverse()
    return rows[: max(1, int(limit))]
