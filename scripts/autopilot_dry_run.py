"""scripts/autopilot_dry_run.py — daily F14a dry-run entry.

Designed to be invoked from a cron / scheduled task each morning:
  python scripts/autopilot_dry_run.py [--top N]

Writes data/autopilot/<date>.md + latest.md. Read-only — never calls
compose() / pipeline / LLM. Pure metadata transformation over the
F13 catalog + redundancy reports.

Per A+B substrate-first roadmap (memory file 2026-06-05):
  Run this for ~1 week, validate the selection logic matches your
  manual judgment, then advance to F14b (limited live auto-run).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5,
                    help="Max WOULD_TEST candidates (default 5)")
    ap.add_argument("--print", action="store_true",
                    help="also print markdown to stdout")
    args = ap.parse_args()

    from engine.agents.autopilot import (
        compute_dry_run_plan, render_markdown, write_dry_run_to_disk,
    )

    plan = compute_dry_run_plan(top_n=args.top)
    out = write_dry_run_to_disk(plan)
    print(f"Plan written: {out}")
    print(f"  ready specs: {plan.n_ready_specs}")
    print(f"  WOULD_TEST: {plan.n_would_test}")
    print(f"  WOULD_SKIP_REDUNDANCY: {plan.n_would_skip}")
    print(f"  estimated cost: ${plan.estimated_cost_usd:.2f}")
    print(f"  estimated wall: {plan.estimated_wall_s}s")
    if args.print:
        print()
        print(render_markdown(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
