"""scripts/cron/daily_belief_refresh.py — daily $0 belief layer refresh.

Built 2026-06-22 (W6-rigor-A peel from burndown_run.py).

Rationale (per principal's cron audit 2026-06-22):
  The burndown_run.py cron (Mon+Thu 09:00) is correctly weekly — Bailey-LdP
  family caps put the verdict ceiling at ~6/week, so running daily would
  waste wall-clock checking the same caps. But the belief layer's read
  side (autopsy backfill + track record regeneration) is FREE and benefits
  from daily refresh: any verdict events emitted async (manual dispatch,
  on-demand, etc.) get joined into autopsies the next morning, and the
  user-visible markdown is never stale.

Three steps, all $0 LLM:
  1. belief_autopsy.backfill_all() — join any new prediction/verdict pairs
  2. report_belief_track_record.py — Phase-3 markdown + JSON
  3. report_belief_track_record_rigor.py — W6 rigor markdown + JSON

Total runtime: ~5 seconds typical. Failure of any single step is logged
but does not block the others (defensive cron pattern).

Usage:
  python scripts/cron/daily_belief_refresh.py
  python scripts/cron/daily_belief_refresh.py --quiet  (suppress info logs)

Installed via scripts/install_daily_belief_refresh_cron.py
(future). For now: manual run or schtasks one-off.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def step_autopsy_backfill() -> tuple[bool, str]:
    """Run autopsy backfill. Returns (ok, message)."""
    try:
        from engine.research.belief_autopsy import backfill_all
        produced = backfill_all()
        return True, f"backfill_all OK: {len(produced)} new autopsies"
    except Exception as exc:
        return False, f"backfill_all FAILED: {type(exc).__name__}: {exc}"


def step_track_record_report() -> tuple[bool, str]:
    """Run Phase-3 track record markdown report."""
    script = REPO_ROOT / "scripts" / "reports" / "report_belief_track_record.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return False, (f"track_record_report exit {result.returncode}: "
                              f"{(result.stderr or '')[:200]}")
        return True, "track_record_report OK"
    except Exception as exc:
        return False, f"track_record_report FAILED: {exc}"


def step_rigor_report() -> tuple[bool, str]:
    """Run W6 rigor markdown report."""
    script = REPO_ROOT / "scripts" / "reports" / "report_belief_track_record_rigor.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            return False, (f"rigor_report exit {result.returncode}: "
                              f"{(result.stderr or '')[:200]}")
        return True, "rigor_report OK"
    except Exception as exc:
        return False, f"rigor_report FAILED: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.quiet:
        print("[daily_belief_refresh] start")

    n_ok = 0
    n_fail = 0
    for label, fn in [
        ("autopsy_backfill",      step_autopsy_backfill),
        ("track_record_report",   step_track_record_report),
        ("rigor_report",          step_rigor_report),
    ]:
        ok, msg = fn()
        if not args.quiet:
            tag = "OK " if ok else "FAIL"
            print(f"[daily_belief_refresh] {tag}  {label}: {msg}")
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    if not args.quiet:
        print(f"[daily_belief_refresh] done: {n_ok}/3 OK, {n_fail}/3 failed")

    # Exit 0 even if individual steps failed — cron should not crash
    # on a single bad step (next day's run will retry naturally).
    return 0


if __name__ == "__main__":
    sys.exit(main())
