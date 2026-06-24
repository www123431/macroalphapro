"""scripts/run_forward_oos.py — daily watchlist simulation runner.

Per Tier 1 cadence ③ + Senior loop ①: cron entry that scans
forward_oos_watchlist and simulates each ready mechanism. Writes one
SimulationRun per (mechanism, day) to data/paper_trade/forward_oos_runs/.

USAGE:
  python scripts/run_forward_oos.py             # run today
  python scripts/run_forward_oos.py --date 2026-05-30  # backfill specific day
  python scripts/run_forward_oos.py --verbose

Idempotent — same mechanism only runs once per calendar day.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default=None,
                        help="ISO date for the run (default: today UTC)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    today = (datetime.date.fromisoformat(args.date) if args.date
             else datetime.date.today())

    from engine.research.discovery.forward_oos_runner import run_watchlist_pass
    summary = run_watchlist_pass(today=today)

    print("=" * 56)
    print(f"FORWARD OOS WATCHLIST PASS — {today.isoformat()}")
    print("=" * 56)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
