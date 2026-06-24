"""scripts/run_watchlist_crawl.py — Stage A piece 2 CLI entry.

  python scripts/run_watchlist_crawl.py [--papers-per-author 10]
                                          [--lookback-years 2]
                                          [--skip-recent-hours 24]
                                          [--json]

Walks the adversarial author watchlist (data/papers_curator/watchlist.yaml),
fetches recent papers via Semantic Scholar, persists new candidates
to cache.jsonl (with existing dedup).

Cost: $0 — Semantic Scholar API is free; only embedding cost is
downstream (in summarizer, which runs separately).

Designed for the chief_of_staff weekly cron — fires before D / A so
the new substrate is visible to A on the same session.

See: project_anti_rut_doctrine_2026-06-07.md for the design intent.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-per-author", type=int, default=10,
                     help="Max papers per author per crawl (cost gate).")
    ap.add_argument("--lookback-years", type=int, default=2,
                     help="Only include papers from last N years "
                          "(older work isn't anti-rut signal — it's stale).")
    ap.add_argument("--skip-recent-hours", type=int, default=24,
                     help="Skip authors crawled within the last N hours "
                          "(cost discipline against daily re-runs).")
    ap.add_argument("--json", action="store_true",
                     help="Machine-readable JSON output.")
    ap.add_argument("--verbose", action="store_true",
                     help="DEBUG logging.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from engine.agents.papers_curator.watchlist_crawler import (
        crawl_watchlist,
    )
    result = crawl_watchlist(
        papers_per_author = args.papers_per_author,
        lookback_years    = args.lookback_years,
        skip_recent_hours = args.skip_recent_hours,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[watchlist] run_ts             : {result['run_ts']}")
        print(f"[watchlist] authors_total       : {result['n_authors_total']}")
        print(f"[watchlist] authors_crawled     : {result['n_authors_crawled']}")
        print(f"[watchlist] authors_skipped     : {result['n_authors_skipped']}")
        print(f"[watchlist] papers_fetched      : {result['n_papers_fetched']}")
        print(f"[watchlist] papers_new (cached) : {result['n_papers_new']}")
        if result["unresolved_names"]:
            print(f"")
            print(f"[watchlist] unresolved_names    : "
                  f"{result['unresolved_names']}")
        if result["errors"]:
            print(f"")
            print(f"[watchlist] errors              : {result['errors']}")

    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
