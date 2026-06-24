"""scripts/burndown_run.py — burn-1b/c verdict batch entry point.

CADENCE NOTE (2026-06-14 architecture reset)
============================================
This is the VERDICT PRODUCTION batch — runs WEEKLY (or on-demand),
NOT daily. Verdict throughput is hard-capped at ~9/week by Bailey-LdP
n_trials family-cap doctrine; running daily just wastes wall-clock
checking the same caps. The DAILY heartbeat is
`scripts/papers_curator_daily.py` which grows the hypothesis queue
via paper ingestion.

Recommended cadence:
  - Manual: `python scripts/burndown_run.py --top-k 10` when queue
    has been freshly populated (check Inbox digest for ingest health)
  - Scheduled: 2x/week (Mon + Thu morning) at --top-k 8-12 to absorb
    a week's worth of fresh hypotheses
  - Auto-triggered: when queue_depth > 200 AND last_burndown > 7d
    (not yet wired; manual for now)

Burndown does NOT generate new substrate. Don't expect more GREENs
by running it more often; you'll just exhaust the queue faster. The
substrate pump is the daily ingest.

Safety chain (top to bottom)
============================
1. Kill switch — data/cron_burndown/_disabled present → no-op.
2. Enable flag — data/cron_burndown/_enabled MUST be present for
   actual execution UNLESS --force is given with --reason ≥ 10 chars.
   Without it (and without --force), the script falls back to DRY-RUN.

Safety chain (top to bottom)
============================
1. Kill switch — data/cron_burndown/_disabled present → no-op.
2. Enable flag — data/cron_burndown/_enabled MUST be present for
   actual execution UNLESS --force is given with --reason ≥ 10 chars.
   Without it (and without --force), the script falls back to DRY-RUN.
3. --dry-run flag overrides enable flag (force dry-run).
4. --max-k cap — never dispatches more than this in one invocation
   regardless of the plan (defense against runaway cron loop).
5. Hard cap auto-shutoff — burndown_caps.global_hard_cap_breached →
   skip execution + write a halt outcome file.

Source tagging (burn-1c, 2026-06-11)
=====================================
--source auto    → invoked by Windows Task Scheduler / cron daemon
--source manual  → principal invoked from terminal (default)
The source string is attached to dispatch_log rows + the cron_run_id
record so post-hoc audit ("what was auto vs manual yesterday")
becomes a one-line query.

Usage
=====
  # Dry run (always safe):
  python scripts/burndown_run.py --dry-run

  # Real execution (requires _enabled flag):
  touch data/cron_burndown/_enabled
  python scripts/burndown_run.py --top-k 3

  # FORCE execution bypassing _enabled — requires audit reason:
  python scripts/burndown_run.py --top-k 1 --force \\
      --reason "smoke testing new portfolio_overlay template"

  # Halt all cron at once:
  touch data/cron_burndown/_disabled

  # Install daily auto-cron (Windows Task Scheduler):
  python scripts/install_burndown_cron.py windows
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.research import burndown_planner, burndown_executor, burndown_caps  # noqa: E402


KILL_SWITCH = REPO_ROOT / "data" / "cron_burndown" / "_disabled"
ENABLE_FLAG = REPO_ROOT / "data" / "cron_burndown" / "_enabled"


def main() -> int:
    parser = argparse.ArgumentParser(description="burn-1b/c daily burndown runner")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=5,
                        help="hard ceiling on dispatches per invocation (default 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="never dispatch, even if _enabled is present")
    parser.add_argument("--source", choices=("manual", "auto"), default="manual",
                        help=("invocation source for audit. Task Scheduler "
                              "should pass --source auto; principal terminal "
                              "runs default to manual."))
    parser.add_argument("--force", action="store_true",
                        help="bypass _enabled gate; requires --reason ≥ 10 chars")
    parser.add_argument("--reason", default=None,
                        help="audit string for --force; recorded on every dispatch row")
    args = parser.parse_args()

    # 1. Kill switch
    if KILL_SWITCH.is_file():
        print(f"[burndown] kill switch present at "
              f"{KILL_SWITCH.relative_to(REPO_ROOT)} — no-op.")
        return 0

    # --force validation
    if args.force:
        if not args.reason or len(args.reason.strip()) < 10:
            print(f"[burndown] --force requires --reason ≥ 10 chars (got "
                  f"{len(args.reason or '')}). Refusing.")
            return 2

    # Effective execute decision
    will_execute = (not args.dry_run) and (ENABLE_FLAG.is_file() or args.force)

    # 2. Hard cap pre-check
    usage_now = burndown_caps.usage_last_7d()
    if will_execute and burndown_caps.global_hard_cap_breached(usage_now):
        print(
            f"[burndown] GLOBAL_HARD_CAP_BREACHED "
            f"({usage_now.global_count}/{burndown_caps.WEEKLY_GLOBAL_HARD_CAP}) "
            f"— refusing to execute. Investigate or wait."
        )
        return 1

    # 3. Build plan (max-k applies to plan target too, cron should never
    #    rank past max-k just to throw away later)
    effective_k = min(args.top_k, args.max_k)
    plan = burndown_planner.plan(target_k=effective_k, dry_run=not will_execute)
    print(burndown_planner.format_plan_human(plan))
    plan_path = burndown_planner.write_plan(plan)
    print(f"\n[burndown] plan written: {plan_path.relative_to(REPO_ROOT)}")

    if not will_execute:
        if not ENABLE_FLAG.is_file():
            print(f"\n[burndown] _enabled flag absent — DRY-RUN only. "
                  f"To enable execution:\n  touch {ENABLE_FLAG.relative_to(REPO_ROOT)}")
        else:
            print("\n[burndown] --dry-run passed — execution skipped.")
        return 0

    # 4. Execute
    cron_run_id = str(uuid.uuid4())
    print(f"\n[burndown] executing plan {plan.plan_id[:8]} "
          f"as cron_run_id={cron_run_id[:8]} source={args.source}"
          + (f" force=true reason={args.reason!r}" if args.force else "")
          + " ...")
    executor = burndown_executor.BurndownExecutor(
        cron_run_id = cron_run_id,
        source      = args.source,
        force_reason = (args.reason if args.force else None),
    )
    outcomes = executor.execute_plan(plan)
    print()
    print(burndown_executor.summarize_outcomes(outcomes))

    out_path = burndown_executor.write_outcomes(outcomes, plan.plan_id)
    print(f"\n[burndown] outcomes written: {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
