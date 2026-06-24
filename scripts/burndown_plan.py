"""scripts/burndown_plan.py — DRY-RUN cron plan generator (burn-1a).

Usage:
  python scripts/burndown_plan.py                  # default target_k=3
  python scripts/burndown_plan.py --top-k 5
  python scripts/burndown_plan.py --top-k 5 --print-only   # don't write file

Kill switch:
  Touch  data/cron_burndown/_disabled   to make this script no-op
  (the file content is ignored; presence is sufficient).

burn-1a ships ONLY the planner. Execution (actually calling
dispatch_factor_spec on selected candidates) lands in burn-1b after
the principal has reviewed a few plans + flipped data/cron_burndown/_enabled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.research import burndown_planner  # noqa: E402


KILL_SWITCH = REPO_ROOT / "data" / "cron_burndown" / "_disabled"
ENABLE_FLAG = REPO_ROOT / "data" / "cron_burndown" / "_enabled"


def main() -> int:
    parser = argparse.ArgumentParser(description="burn-1a daily burndown plan")
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="target candidate count (default 3; family + global caps may reduce)",
    )
    parser.add_argument(
        "--print-only", action="store_true",
        help="don't write the plan JSON file; print human summary only",
    )
    args = parser.parse_args()

    if KILL_SWITCH.is_file():
        print(f"[burndown] kill switch present at {KILL_SWITCH.relative_to(REPO_ROOT)} — no-op.")
        return 0

    plan = burndown_planner.plan(target_k=args.top_k, dry_run=True)
    print(burndown_planner.format_plan_human(plan))

    if not args.print_only:
        out = burndown_planner.write_plan(plan)
        print(f"\n[burndown] plan written: {out.relative_to(REPO_ROOT)}")

    if ENABLE_FLAG.is_file():
        print(
            "\n[burndown] _enabled flag PRESENT — burn-1b execution path "
            "would run here once shipped.\n"
            "burn-1a is dry-run only; no dispatches made."
        )
    else:
        print(
            "\n[burndown] _enabled flag absent — burn-1b execution would "
            "be gated even if shipped. To enable later:\n"
            f"  touch {ENABLE_FLAG.relative_to(REPO_ROOT)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
