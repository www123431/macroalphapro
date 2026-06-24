"""scripts/run_sleeve_fix_proposer.py — Stage B P2 piece 1 CLI.

Scans recent doctrine_signal_detected events and proposes
deterministic-template sleeve_fix Hypotheses for B's review pipeline.

Usage:
  python scripts/run_sleeve_fix_proposer.py
  python scripts/run_sleeve_fix_proposer.py --days 7 --max 3
  python scripts/run_sleeve_fix_proposer.py --dry-run
  python scripts/run_sleeve_fix_proposer.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.agents.strengthener.sleeve_fix_proposer import (
    propose_sleeve_fixes,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=30,
                    help="Look back N days (default 30)")
    p.add_argument("--max", type=int, default=10,
                    help="Max fix-proposals per run (default 10)")
    p.add_argument("--dry-run", action="store_true",
                    help="Build proposals but don't persist")
    p.add_argument("--json", action="store_true",
                    help="Output JSON")
    args = p.parse_args()

    result = propose_sleeve_fixes(
        days        = args.days,
        max_signals = args.max,
        dry_run     = args.dry_run,
    )

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"Sleeve-fix proposer @ {result['run_ts']}")
    print(f"  dry_run:        {result['dry_run']}")
    print(f"  signals_seen:   {result['n_signals_seen']}")
    print(f"  already_done:   {result['n_already_done']} (idempotent skip)")
    print(f"  proposed:       {result['n_proposed']}")
    print(f"  persisted:      {result['n_persisted']}")
    if result["errors"]:
        print(f"  errors:")
        for e in result["errors"]:
            print(f"    {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
