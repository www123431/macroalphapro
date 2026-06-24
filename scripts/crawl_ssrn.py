"""scripts/crawl_ssrn.py — Stage A piece 5 follow-up CLI.

Pulls recent SSRN papers via CrossRef DOI-prefix workaround.

Usage
-----
  python scripts/crawl_ssrn.py
  python scripts/crawl_ssrn.py --lookback-days 14 --max-results 200
  python scripts/crawl_ssrn.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.agents.papers_curator.ssrn_crossref_crawler import (
    crawl_and_persist_ssrn,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lookback-days", type=int, default=7,
                    help="Pull SSRN papers deposited in last N days "
                          "(default 7 — weekly cadence)")
    p.add_argument("--max-results", type=int, default=100,
                    help="Cap on candidates returned (default 100). "
                          "CrossRef supports up to 1000 per call.")
    p.add_argument("--json", action="store_true",
                    help="Output JSON instead of human-readable")
    args = p.parse_args()

    result = crawl_and_persist_ssrn(
        lookback_days=args.lookback_days,
        max_results=args.max_results,
    )

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"SSRN (via CrossRef) crawl @ {result['run_ts']}")
    print(f"  fetched: {result['n_fetched']}")
    print(f"  new:     {result['n_new']} (after dedup)")
    if result["errors"]:
        print("  errors:")
        for e in result["errors"]:
            print(f"    {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
