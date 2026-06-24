"""scripts/crawl_forward_citations.py — Stage A piece 4 CLI.

Runs one forward-citation crawl over all seeds in
data/papers_curator/forward_seeds.yaml.

Usage
-----
  python scripts/crawl_forward_citations.py
  python scripts/crawl_forward_citations.py --max-per-seed 30
  python scripts/crawl_forward_citations.py --lookback-years 5
  python scripts/crawl_forward_citations.py --force   # ignore last_crawled_ts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.agents.papers_curator.forward_citation_crawler import (
    crawl_forward_citations,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-per-seed", type=int, default=20,
                    help="Max citers to fetch per seed (default 20)")
    p.add_argument("--lookback-years", type=int, default=3,
                    help="Filter to citers published within last N years "
                          "(default 3)")
    p.add_argument("--skip-recent-hours", type=int, default=24,
                    help="Skip seeds crawled within last N hours "
                          "(default 24)")
    p.add_argument("--force", action="store_true",
                    help="Ignore last_crawled_ts (override --skip-recent-hours)")
    p.add_argument("--json", action="store_true",
                    help="Output JSON instead of human-readable summary")
    args = p.parse_args()

    skip_hours = 0 if args.force else args.skip_recent_hours
    result = crawl_forward_citations(
        max_per_seed      = args.max_per_seed,
        lookback_years    = args.lookback_years,
        skip_recent_hours = skip_hours,
    )

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"Forward-citation crawl @ {result['run_ts']}")
    print(f"  seeds:        {result['n_seeds_crawled']}/"
          f"{result['n_seeds_total']} crawled, "
          f"{result['n_seeds_skipped']} skipped")
    print(f"  citations:    {result['n_citations_fetched']} fetched, "
          f"{result['n_citations_new']} new (after dedup)")
    if result["unresolved_seeds"]:
        print(f"  unresolved:   {result['unresolved_seeds']}")
    if result["errors"]:
        print(f"  errors:")
        for e in result["errors"]:
            print(f"    {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
