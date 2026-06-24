"""engine/research/data_freshness.py — P0c of the liveness layer
(2026-06-02 — added after the user caught a silent failure the existing
heartbeat couldn't see).

Liveness P0 + P1 answer "did the cron run?" and "did the broker echo
the orders?". They DO NOT answer "is the data the cron produced
actually fresh?" The 2026-06-02 dashboard screenshot showed NAV PATH
from 2026-05-04 → 2026-05-12 while the heartbeat read "OK · 0m ago" —
exactly the silent failure the layer is supposed to catch.

This module probes the critical data sources (NAV history, decay
sentinel ledger, paper-trade DB, UI artifact directory) for staleness
relative to wall-clock now. Best-effort: a probe that errors out is
recorded as status="unknown" with the exception detail rather than
raising — a single stuck source must NOT crash the freshness check.

Status enum (kept stable for UI consumption):
  fresh       — age ≤ 1 trading day
  aging       — age 1-3 days (informational, no UI alert)
  stale       — age 3-7 days (WARN)
  dead        — age > 7 days (DANGER)
  unknown     — probe raised; surfaced honestly rather than masked

Caller (typically the daily-script heartbeat emitter) attaches the
list of source results to the heartbeat row so assess_liveness can
downgrade an otherwise-OK verdict when data is dead.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


# Status enum
STATUS_FRESH   = "fresh"
STATUS_AGING   = "aging"
STATUS_STALE   = "stale"
STATUS_DEAD    = "dead"
STATUS_UNKNOWN = "unknown"
STATUS_MISSING = "missing"


# Bucket thresholds (calendar days). Trading-day awareness is left to
# the source-specific probes that have access to a calendar.
THRESHOLDS = {
    "fresh_max_days": 1.0,
    "aging_max_days": 3.0,
    "stale_max_days": 7.0,
}


def _classify(age_days: float) -> str:
    if age_days <= THRESHOLDS["fresh_max_days"]:
        return STATUS_FRESH
    if age_days <= THRESHOLDS["aging_max_days"]:
        return STATUS_AGING
    if age_days <= THRESHOLDS["stale_max_days"]:
        return STATUS_STALE
    return STATUS_DEAD


def _age_days_from(date_str: str, *, now: Optional[_dt.date] = None) -> Optional[float]:
    """Parse an ISO date and return the wall-clock age in calendar days,
    or None if the date is unparsable."""
    if not date_str:
        return None
    try:
        d = _dt.date.fromisoformat(str(date_str)[:10])
    except Exception:
        return None
    today = now or _dt.date.today()
    return float((today - d).days)


# ── Per-source probes ─────────────────────────────────────────────


def check_nav_history(now: Optional[_dt.date] = None) -> dict:
    """Latest row in the PortfolioNavSnapshot DB table.

    This was the source that exposed the gap on 2026-06-02 — NAV stuck
    at 2026-05-12 while heartbeat was OK."""
    out = {
        "source":      "nav_history",
        "description": "PortfolioNavSnapshot daily NAV table (DB)",
        "latest_date": None,
        "age_days":    None,
        "status":      STATUS_UNKNOWN,
        "n_rows":      None,
        "error":       None,
    }
    try:
        from engine.db_models import PortfolioNavSnapshot
        from engine.memory import SessionFactory
        with SessionFactory() as s:
            n = s.query(PortfolioNavSnapshot).count()
            row = (
                s.query(PortfolioNavSnapshot)
                 .order_by(PortfolioNavSnapshot.snapshot_date.desc())
                 .first()
            )
        out["n_rows"] = int(n)
        if row is None:
            out["status"] = STATUS_MISSING
            return out
        out["latest_date"] = str(row.snapshot_date)
        out["age_days"] = _age_days_from(out["latest_date"], now=now)
        if out["age_days"] is None:
            return out
        out["status"] = _classify(out["age_days"])
    except Exception as exc:
        logger.warning("nav_history probe failed", exc_info=True)
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def check_decay_sentinel(now: Optional[_dt.date] = None) -> dict:
    """Latest row in the decay history ledger.

    2026-06-14 fix: legacy `decay_sentinel_history.jsonl` writer was
    retired (last row 2026-05-31) but the freshness check still pointed
    at it, surfacing a false "data source DEAD" banner. The LIVE writer
    is `engine.research.decay_history_log.run_history_audit` which
    appends to `decay_history.jsonl`. Prefer the live path; fall back
    to the legacy file only if the live one hasn't been bootstrapped
    yet (so the probe still reports against SOMETHING)."""
    out = {
        "source":      "decay_sentinel",
        "description": "Decay sentinel audit history ledger",
        "latest_date": None,
        "age_days":    None,
        "status":      STATUS_UNKNOWN,
        "n_rows":      None,
        "error":       None,
    }
    live_path   = REPO_ROOT / "data" / "research" / "decay_history.jsonl"
    legacy_path = REPO_ROOT / "data" / "research" / "decay_sentinel_history.jsonl"
    path = live_path if live_path.is_file() else legacy_path
    if not path.is_file():
        out["status"] = STATUS_MISSING
        return out
    try:
        max_date = ""
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                n += 1
                d = str(row.get("audit_date") or "")
                if d > max_date:
                    max_date = d
        out["n_rows"] = n
        if not max_date:
            out["status"] = STATUS_MISSING
            return out
        out["latest_date"] = max_date
        out["age_days"] = _age_days_from(max_date, now=now)
        if out["age_days"] is None:
            return out
        out["status"] = _classify(out["age_days"])
    except Exception as exc:
        logger.warning("decay_sentinel probe failed", exc_info=True)
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def check_paper_trade_log(now: Optional[_dt.date] = None) -> dict:
    """Latest row in PaperTradeStrategyLog (DB), the authoritative book
    state log written by the daily script."""
    out = {
        "source":      "paper_trade_log",
        "description": "PaperTradeStrategyLog book state (DB)",
        "latest_date": None,
        "age_days":    None,
        "status":      STATUS_UNKNOWN,
        "n_rows":      None,
        "error":       None,
    }
    try:
        from engine.db_models import PaperTradeStrategyLog
        from engine.memory import SessionFactory
        with SessionFactory() as s:
            n = s.query(PaperTradeStrategyLog).count()
            row = (
                s.query(PaperTradeStrategyLog)
                 .order_by(PaperTradeStrategyLog.date.desc())
                 .first()
            )
        out["n_rows"] = int(n)
        if row is None:
            out["status"] = STATUS_MISSING
            return out
        out["latest_date"] = str(row.date)
        out["age_days"] = _age_days_from(out["latest_date"], now=now)
        if out["age_days"] is None:
            return out
        out["status"] = _classify(out["age_days"])
    except Exception as exc:
        logger.warning("paper_trade_log probe failed", exc_info=True)
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def check_ui_artifact(now: Optional[_dt.date] = None) -> dict:
    """Latest dated file in data/ui_artifact/ — what the frontend
    reads for the dashboard hero strip."""
    out = {
        "source":      "ui_artifact",
        "description": "UI artifact JSON the dashboard reads",
        "latest_date": None,
        "age_days":    None,
        "status":      STATUS_UNKNOWN,
        "n_rows":      None,
        "error":       None,
    }
    art_dir = REPO_ROOT / "data" / "ui_artifact"
    if not art_dir.is_dir():
        out["status"] = STATUS_MISSING
        return out
    try:
        # Filenames are YYYY-MM-DD.json — sort lexicographically picks
        # the latest. Date in filename is the authoritative "as_of".
        files = sorted([
            p for p in art_dir.glob("*.json") if p.stem and not p.stem.startswith("_")
        ])
        out["n_rows"] = len(files)
        if not files:
            out["status"] = STATUS_MISSING
            return out
        latest_stem = files[-1].stem    # "YYYY-MM-DD"
        out["latest_date"] = latest_stem
        out["age_days"] = _age_days_from(latest_stem, now=now)
        if out["age_days"] is None:
            return out
        out["status"] = _classify(out["age_days"])
    except Exception as exc:
        logger.warning("ui_artifact probe failed", exc_info=True)
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ── Public API ────────────────────────────────────────────────────


def check_sources(now: Optional[_dt.date] = None) -> list[dict]:
    """Probe all known sources. Each entry is best-effort; failures
    are surfaced as status="unknown" with the exception text."""
    probes = [
        check_nav_history,
        check_decay_sentinel,
        check_paper_trade_log,
        check_ui_artifact,
    ]
    out: list[dict] = []
    for fn in probes:
        try:
            out.append(fn(now=now))
        except Exception as exc:
            # Belt-and-suspenders: even if a probe forgets its own
            # try/except, never let it crash the freshness sweep.
            out.append({
                "source":      fn.__name__.removeprefix("check_"),
                "description": "(probe raised at sweep level)",
                "status":      STATUS_UNKNOWN,
                "error":       f"{type(exc).__name__}: {exc}",
            })
    return out


def summarize(sources: Iterable[dict]) -> dict:
    """Aggregate the per-source list into the small dict the UI reads.

    Returns:
      {
        worst_status:  most severe status seen across sources
        n_dead:        count of sources with status="dead"
        n_stale:       count with status="stale"
        n_fresh:       count with status="fresh"
        worst_source:  name of the source driving worst_status (or None)
        headline:      1-line explanation suitable for a banner
      }
    """
    severity_rank = {
        STATUS_FRESH:   0,
        STATUS_AGING:   1,
        STATUS_UNKNOWN: 2,
        STATUS_MISSING: 3,
        STATUS_STALE:   4,
        STATUS_DEAD:    5,
    }

    rows = list(sources)
    worst = STATUS_FRESH
    worst_source: Optional[str] = None
    counts: dict[str, int] = {}
    for r in rows:
        st = r.get("status") or STATUS_UNKNOWN
        counts[st] = counts.get(st, 0) + 1
        if severity_rank.get(st, 0) > severity_rank.get(worst, 0):
            worst = st
            worst_source = r.get("source")

    if worst == STATUS_DEAD:
        wc = counts.get(STATUS_DEAD, 0)
        headline = (
            f"{wc} data source{'s' if wc != 1 else ''} DEAD — "
            f"oldest: {worst_source}. Cron may be running but writing "
            f"to a dead pipe."
        )
    elif worst == STATUS_STALE:
        wc = counts.get(STATUS_STALE, 0)
        headline = f"{wc} data source{'s' if wc != 1 else ''} stale (>3 days old)"
    elif worst == STATUS_MISSING:
        headline = f"{counts.get(STATUS_MISSING, 0)} source(s) missing entirely"
    elif worst == STATUS_UNKNOWN:
        headline = f"{counts.get(STATUS_UNKNOWN, 0)} source(s) failed to probe"
    elif worst == STATUS_AGING:
        headline = f"{counts.get(STATUS_AGING, 0)} source(s) aging (1-3d) — informational"
    else:
        headline = f"All {len(rows)} data sources fresh"

    return {
        "worst_status":  worst,
        "worst_source":  worst_source,
        "n_dead":        counts.get(STATUS_DEAD, 0),
        "n_stale":       counts.get(STATUS_STALE, 0),
        "n_aging":       counts.get(STATUS_AGING, 0),
        "n_fresh":       counts.get(STATUS_FRESH, 0),
        "n_missing":     counts.get(STATUS_MISSING, 0),
        "n_unknown":     counts.get(STATUS_UNKNOWN, 0),
        "n_total":       len(rows),
        "headline":      headline,
    }
