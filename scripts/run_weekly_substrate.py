"""scripts/run_weekly_substrate.py — Stage A piece 7a CLI.

Runs the weekly substrate refresh (all 5 crawlers).

Usage
-----
  python scripts/run_weekly_substrate.py
  python scripts/run_weekly_substrate.py --dry-run
  python scripts/run_weekly_substrate.py --sources arxiv,nber
  python scripts/run_weekly_substrate.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.agents.chief_of_staff.substrate import (
    ALL_SOURCES, run_weekly_substrate,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                    help="Skip network + persist; return zero-counts "
                          "shell")
    p.add_argument("--sources", default=",".join(ALL_SOURCES),
                    help=(f"Comma-separated source list "
                           f"(default: {','.join(ALL_SOURCES)}). "
                           "Pass a subset to disable some — e.g. "
                           "--sources arxiv,nber to skip SS-dependent "
                           "sources."))
    p.add_argument("--json", action="store_true",
                    help="Output JSON instead of human-readable")
    p.add_argument("--arxiv-max", type=int, default=50)
    p.add_argument("--ssrn-lookback-days", type=int, default=7)
    p.add_argument("--ssrn-max", type=int, default=100)
    args = p.parse_args()

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())

    result = run_weekly_substrate(
        dry_run             = args.dry_run,
        enabled_sources     = sources,
        arxiv_max           = args.arxiv_max,
        ssrn_lookback_days  = args.ssrn_lookback_days,
        ssrn_max_results    = args.ssrn_max,
    )

    if args.json:
        sys.stdout.write(json.dumps(result.to_dict(), indent=2,
                                       ensure_ascii=False))
        return 0

    print(f"Weekly substrate run @ {result.run_ts}")
    print(f"  dry_run:  {result.dry_run}")
    print(f"  sources:  {','.join(result.enabled_sources)}")
    print(f"  TOTAL:    {result.total_fetched} fetched, "
          f"{result.total_new} new")
    print()
    # Per-source breakdown
    pairs = [
        ("arxiv",             result.arxiv_result),
        ("nber",              result.nber_result),
        ("ssrn",              result.ssrn_result),
        ("watchlist",         result.watchlist_result),
        ("forward_citations", result.forward_citation_result),
    ]
    for name, r in pairs:
        if not r:
            continue
        from engine.agents.chief_of_staff.substrate import _extract_counts
        f, n = _extract_counts(r)
        line = f"  {name:18}: {f:>5} fetched, {n:>5} new"
        if r.get("errors"):
            line += f"   [{len(r['errors'])} err]"
        print(line)
    if result.errors:
        print()
        print("  ERRORS:")
        for e in result.errors:
            print(f"    {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
