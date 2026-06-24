"""
engine/scheduler.py — Lightweight cycle auto-trigger scheduler.

Spec: docs/spec_agentic_orchestration_v1.md Sub-task 3 (2026-05-03).

Schedule rules (frozen):
  daily      : every weekday (last_run < today's weekday close)
  weekly     : every Friday (last_run before this Friday)
  monthly    : last weekday of each calendar month
  quarterly  : first weekday of Apr / Jul / Oct / Jan

NOT a cron daemon — runs once per CLI invocation. Recommended deployment:

  # OS cron (Linux/Mac), every weekday at 17:00:
  0 17 * * 1-5 cd /path/to/intern && PYTHONPATH=. python -m engine.scheduler --check

  # Windows Task Scheduler equivalent:
  D:\python\python.exe -m engine.scheduler --check
  (run as Triggered Task daily at 17:00, working dir = project root)

Or call from streamlit dashboard "Run due cycles" button (see pages/agent_observability.py).
"""
from __future__ import annotations

import calendar
import datetime
import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)


# ── Schedule constants (frozen per spec §3.2) ─────────────────────────────────

CYCLE_TYPES: tuple[str, ...] = ("daily", "weekly", "monthly", "quarterly")

# Quarterly months: first weekday of Apr / Jul / Oct / Jan
_QUARTERLY_MONTHS = (1, 4, 7, 10)

# Weekly: trigger on Friday (weekday 4)
_WEEKLY_DAY_OF_WEEK = 4   # Mon=0 ... Sun=6


# ── Date arithmetic helpers ───────────────────────────────────────────────────

def _is_weekday(d: datetime.date) -> bool:
    return d.weekday() < 5   # Mon-Fri


def _last_weekday_of_month(year: int, month: int) -> datetime.date:
    """Return last weekday (Mon-Fri) of given month."""
    last_day = calendar.monthrange(year, month)[1]
    d = datetime.date(year, month, last_day)
    while not _is_weekday(d):
        d -= datetime.timedelta(days=1)
    return d


def _first_weekday_of_month(year: int, month: int) -> datetime.date:
    """Return first weekday (Mon-Fri) of given month."""
    d = datetime.date(year, month, 1)
    while not _is_weekday(d):
        d += datetime.timedelta(days=1)
    return d


def _next_friday_on_or_after(d: datetime.date) -> datetime.date:
    """Return next Friday on or after d."""
    diff = (_WEEKLY_DAY_OF_WEEK - d.weekday()) % 7
    return d + datetime.timedelta(days=diff)


# ── Scheduler ─────────────────────────────────────────────────────────────────

class CycleScheduler:
    """
    Determines whether each cycle is due based on cycle_states table history.

    No background thread — call `is_due()` / `run_due_cycles()` from cron or button.
    """

    def __init__(self) -> None:
        from engine.memory import init_db
        init_db()

    # ── Last run lookup ───────────────────────────────────────────────────────

    def _last_run_date(self, cycle_type: str) -> Optional[datetime.date]:
        """Most-recent as_of_date for cycle_type, regardless of status."""
        from engine.memory import SessionFactory, CycleState
        with SessionFactory() as s:
            latest = (
                s.query(CycleState)
                 .filter(CycleState.cycle_type == cycle_type)
                 .order_by(CycleState.as_of_date.desc())
                 .first()
            )
            return latest.as_of_date if latest else None

    # ── Next-due computation ──────────────────────────────────────────────────

    def next_scheduled_date(
        self,
        cycle_type: str,
        last_run:   Optional[datetime.date] = None,
    ) -> datetime.date:
        """
        Compute next scheduled date for cycle_type given last_run.

        If last_run is None, returns today (first-time run).
        """
        if last_run is None:
            today = datetime.date.today()
            if cycle_type == "daily":
                return today if _is_weekday(today) else today + datetime.timedelta(
                    days=(7 - today.weekday()) if today.weekday() == 6 else (1 if today.weekday() == 5 else 0)
                )
            if cycle_type == "weekly":
                return _next_friday_on_or_after(today)
            if cycle_type == "monthly":
                return _last_weekday_of_month(today.year, today.month)
            if cycle_type == "quarterly":
                # Find next quarterly month
                for m in _QUARTERLY_MONTHS:
                    candidate = _first_weekday_of_month(today.year, m)
                    if candidate >= today:
                        return candidate
                return _first_weekday_of_month(today.year + 1, _QUARTERLY_MONTHS[0])
            raise ValueError(f"Unknown cycle_type: {cycle_type}")

        # last_run is set: compute next instance after last_run
        if cycle_type == "daily":
            d = last_run + datetime.timedelta(days=1)
            while not _is_weekday(d):
                d += datetime.timedelta(days=1)
            return d

        if cycle_type == "weekly":
            return _next_friday_on_or_after(last_run + datetime.timedelta(days=1))

        if cycle_type == "monthly":
            # Next month's last weekday
            year = last_run.year
            month = last_run.month + 1
            if month > 12:
                year += 1
                month = 1
            return _last_weekday_of_month(year, month)

        if cycle_type == "quarterly":
            # Next quarterly month after last_run
            for m in _QUARTERLY_MONTHS:
                candidate = _first_weekday_of_month(last_run.year, m)
                if candidate > last_run:
                    return candidate
            return _first_weekday_of_month(last_run.year + 1, _QUARTERLY_MONTHS[0])

        raise ValueError(f"Unknown cycle_type: {cycle_type}")

    # ── Due check ─────────────────────────────────────────────────────────────

    def is_due(self, cycle_type: str, today: Optional[datetime.date] = None) -> bool:
        """True if next scheduled date for cycle_type ≤ today."""
        today = today or datetime.date.today()
        last_run = self._last_run_date(cycle_type)
        next_due = self.next_scheduled_date(cycle_type, last_run)
        return today >= next_due

    def get_next_run_time(self, cycle_type: str) -> datetime.date:
        """Public accessor for next scheduled date."""
        return self.next_scheduled_date(cycle_type, self._last_run_date(cycle_type))

    def status_summary(self) -> list[dict]:
        """Return per-cycle (last_run, next_due, is_due) summary for dashboard."""
        today = datetime.date.today()
        rows = []
        for ct in CYCLE_TYPES:
            last_run = self._last_run_date(ct)
            next_due = self.next_scheduled_date(ct, last_run)
            rows.append({
                "cycle_type": ct,
                "last_run":   str(last_run) if last_run else "—",
                "next_due":   str(next_due),
                "is_due":     today >= next_due,
            })
        return rows

    # ── Trigger ───────────────────────────────────────────────────────────────

    def run_due_cycles(
        self,
        model=None,
        dry_run: bool = False,
        force_cycle: Optional[str] = None,
    ) -> list[dict]:
        """
        Check all cycles, run any due. If force_cycle is set, run it regardless.

        Returns list of {cycle_type, status, error, summary} dicts.
        """
        from engine.orchestrator import TradingCycleOrchestrator
        orch = TradingCycleOrchestrator()
        results = []

        cycles_to_run = [force_cycle] if force_cycle else [
            ct for ct in CYCLE_TYPES if self.is_due(ct)
        ]

        if not cycles_to_run:
            logger.info("No cycles due as of %s", datetime.date.today())
            return []

        as_of = datetime.date.today()
        for ct in cycles_to_run:
            if dry_run:
                logger.info("DRY RUN: would trigger %s for as_of=%s", ct, as_of)
                results.append({
                    "cycle_type": ct,
                    "status": "dry_run",
                    "as_of": str(as_of),
                    "error": None,
                    "summary": "would have triggered",
                })
                continue
            try:
                logger.info("Triggering %s cycle for as_of=%s", ct, as_of)
                if ct == "daily":
                    cr = orch.run_daily(as_of=as_of, model=model)
                elif ct == "weekly":
                    cr = orch.run_weekly(as_of=as_of, model=model)
                elif ct == "monthly":
                    cr = orch.run_monthly(as_of=as_of, model=model, require_approval=False)
                elif ct == "quarterly":
                    cr = orch.run_quarterly(as_of=as_of, model=model)
                else:
                    raise ValueError(f"Unknown cycle_type: {ct}")
                results.append({
                    "cycle_type": ct,
                    "status": "completed" if cr.ok else "failed",
                    "as_of": str(as_of),
                    "error": "; ".join(cr.errors) if cr.errors else None,
                    "summary": f"steps={len(cr.steps)} regime={cr.regime}",
                })
            except Exception as exc:
                logger.error("%s cycle failed: %s", ct, exc, exc_info=True)
                results.append({
                    "cycle_type": ct,
                    "status": "exception",
                    "as_of": str(as_of),
                    "error": str(exc),
                    "summary": None,
                })
        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> int:
    import argparse, json
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Cycle auto-trigger scheduler.")
    parser.add_argument("--check", action="store_true",
                        help="Run any cycles that are due. Without this, only print status.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run, don't actually trigger.")
    parser.add_argument("--force", type=str, default=None, choices=list(CYCLE_TYPES) + [""],
                        help="Force-run a specific cycle regardless of schedule.")
    parser.add_argument("--with-llm", action="store_true",
                        help="Pass real Gemini model to cycles (for sector debate / LLM proposer).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    sched = CycleScheduler()

    print("=== Cycle Schedule Status ===")
    for row in sched.status_summary():
        marker = "[DUE]" if row["is_due"] else "[OK]"
        # Use ASCII fallback for "—" so Windows cmd (GBK) doesn't render mojibake
        last_run_str = str(row["last_run"]) if row["last_run"] != "—" else "(never)"
        print(f"  {row['cycle_type']:<10}  last_run={last_run_str:<12}  next_due={row['next_due']}  {marker}")

    if not args.check and not args.force:
        return 0

    model = None
    if args.with_llm:
        try:
            from engine.key_pool import get_pool
            model = get_pool().get_model()
            print(f"\n[INFO] LLM model loaded: {type(model).__name__}")
        except Exception as exc:
            print(f"[WARN] failed to load LLM model: {exc} — cycles will run without LLM")

    print("\n=== Triggering due cycles ===")
    results = sched.run_due_cycles(
        model=model, dry_run=args.dry_run,
        force_cycle=args.force or None,
    )
    if not results:
        print("(no cycles due)")
        return 0
    for r in results:
        print(f"\n{r['cycle_type']} [{r['status']}] as_of={r['as_of']}")
        if r['error']:
            print(f"  error: {r['error'][:200]}")
        if r['summary']:
            print(f"  {r['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
