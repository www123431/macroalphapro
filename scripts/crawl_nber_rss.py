"""scripts/crawl_nber_rss.py — Stage A piece 5 CLI.

Pull the NBER working-papers RSS, dedup against cache.jsonl, persist.

Usage
-----
  python scripts/crawl_nber_rss.py
  python scripts/crawl_nber_rss.py --url https://other.feed.example
  python scripts/crawl_nber_rss.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.agents.papers_curator.nber_rss_crawler import (
    crawl_and_persist_nber, _NBER_RSS_URL,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=_NBER_RSS_URL,
                    help=f"RSS endpoint (default {_NBER_RSS_URL})")
    p.add_argument("--json", action="store_true",
                    help="Output JSON")
    args = p.parse_args()

    result = crawl_and_persist_nber(url=args.url)

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"NBER crawl @ {result['run_ts']}")
    print(f"  fetched: {result['n_fetched']}")
    print(f"  new:     {result['n_new']} (after dedup)")
    if result["errors"]:
        print(f"  errors:")
        for e in result["errors"]:
            print(f"    {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
