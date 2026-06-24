"""
scripts/run_auto_audit.py — Cron entry point for the Auto-Audit Loop (R-1.A 2026-05-06)

Daily critical sweep:
  python scripts/run_auto_audit.py --scope critical

Weekly slow-drift sweep:
  python scripts/run_auto_audit.py --scope weekly

Exit code 0 on clean run, non-zero if any rule errored. Intended for
Windows Task Scheduler / cron — the JSON summary on stdout makes per-run
auditing easy via tee + log rotation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.auto_audit import run_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--scope",
        choices=["critical", "weekly"],
        required=True,
        help="critical = daily sweep; weekly = slow-drift sweep",
    )
    args = parser.parse_args()

    summary = run_audit(scope=args.scope)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["exit_status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
