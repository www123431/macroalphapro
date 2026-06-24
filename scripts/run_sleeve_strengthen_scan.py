"""scripts/run_sleeve_strengthen_scan.py — Stage B P3b CLI.

Active-B per-sleeve weekly strengthen scan.

Usage:
  python scripts/run_sleeve_strengthen_scan.py
  python scripts/run_sleeve_strengthen_scan.py --max 3 --dry-run
  python scripts/run_sleeve_strengthen_scan.py --force         # ignore week dedup
  python scripts/run_sleeve_strengthen_scan.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.agents.strengthener.sleeve_strengthen_scan import (
    run_sleeve_strengthen_scan,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max", type=int, default=5,
                    help="Max sleeves to scan this run (default 5; "
                          "rotation picks oldest-scanned first)")
    p.add_argument("--lookback-days", type=int, default=30,
                    help="Recent-state lookback for context build")
    p.add_argument("--force", action="store_true",
                    help="Ignore same-week dedup (re-scan everything)")
    p.add_argument("--dry-run", action="store_true",
                    help="Build context + call LLM but skip persist")
    p.add_argument("--json", action="store_true",
                    help="Output JSON")
    args = p.parse_args()

    result = run_sleeve_strengthen_scan(
        max_sleeves   = args.max,
        lookback_days = args.lookback_days,
        force         = args.force,
        dry_run       = args.dry_run,
    )

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"Active-B sleeve scan @ {result['run_ts']}")
    print(f"  iso_week:               {result['iso_week']}")
    print(f"  dry_run:                {result['dry_run']}")
    print(f"  eligible/scanned/skip:  "
          f"{result['n_sleeves_eligible']}/"
          f"{result['n_sleeves_scanned']}/"
          f"{result['n_sleeves_skipped']}")
    print(f"  proposals total/persisted: "
          f"{result['n_proposals_total']}/"
          f"{result['n_proposals_persisted']}")
    print()
    if result["per_sleeve"]:
        print("  per-sleeve:")
        for ps in result["per_sleeve"]:
            line = (f"    {ps['sleeve_id']:30}  "
                     f"{ps['n_proposals']} proposals")
            if ps["errors"]:
                line += f"   [{len(ps['errors'])} err]"
            print(line)
    if result["errors"]:
        print()
        print("  ERRORS:")
        for e in result["errors"]:
            print(f"    {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
