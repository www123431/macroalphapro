"""
Fetch CFTC Commitments of Traders data into the local database.

Usage
-----
  # Fetch one year, both report types (Disaggregated + TFF):
  python scripts/fetch_cftc_cot.py --year 2024

  # Backfill 5 years:
  python scripts/fetch_cftc_cot.py --years 2020 2021 2022 2023 2024

  # Single report type:
  python scripts/fetch_cftc_cot.py --year 2024 --report-type tff_fut

Both report types share one ``cftc_cot_weekly`` table with a
``report_type`` discriminator. Idempotent — re-running for the same
year overwrites in place.

Reference: see engine/data_sources/cftc_cot.py module docstring for
URL patterns + trader-category schemas.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.data_sources.cftc_cot import (  # noqa: E402
    upsert_year, upsert_year_both,
)

_VALID_RT = ("disagg_fut", "tff_fut")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--year",  type=int,
                   help="Single year to fetch.")
    g.add_argument("--years", type=int, nargs="+",
                   help="Multiple years to fetch in sequence.")
    p.add_argument("--report-type", choices=_VALID_RT, default=None,
                   help="Restrict to one report type (default: both).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    years = args.years if args.years else [args.year]
    output: dict = {"years": {}}

    for year in years:
        if args.report_type:
            output["years"][year] = {
                args.report_type: upsert_year(year, report_type=args.report_type)
            }
        else:
            output["years"][year] = upsert_year_both(year)["by_report_type"]

    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
