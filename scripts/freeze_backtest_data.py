"""
scripts/freeze_backtest_data.py — CLI to capture a backtest data snapshot
(S-3 reproducibility, 2026-05-06).

Captures all 4 external data dependencies (yfinance monthly + daily ETF
prices, ^VIX, FRED macros) into a parquet bundle that can later be
replayed via `engine.backtest.run_backtest(snapshot=...)` for full
reproducibility.

Usage:
    # Default tickers (project's 18-ETF universe) + default FRED series:
    python scripts/freeze_backtest_data.py --start=2009-01-01 --end=2026-05-06 --name=thesis_v1

    # Custom set:
    python scripts/freeze_backtest_data.py --start=... --end=... --name=... \\
        --tickers=XLF,XLE,XLK --fred-series=DGS10,BAMLH0A0HYM2

    # List existing snapshots:
    python scripts/freeze_backtest_data.py --list
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# Default 18-ETF universe (current project active set per UniverseETF).
DEFAULT_TICKERS = [
    "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE",
    "XLC", "EEM", "EFA", "GLD", "SLV", "TLT", "AGG", "DBC",
]
# Default FRED series captured for regime / macro context.
DEFAULT_FRED_SERIES = [
    "DGS10",          # 10Y Treasury yield
    "DGS2",           # 2Y Treasury yield
    "BAMLH0A0HYM2",   # HY OAS (credit spread proxy)
    "VIXCLS",         # VIX (closing) — backup if yfinance ^VIX fails
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--start", help="ISO start date (e.g. 2009-01-01)")
    parser.add_argument("--end",   help="ISO end date (default: today)")
    parser.add_argument("--name",  help="snapshot name (becomes <name>_<YYYY-MM-DD>)")
    parser.add_argument("--tickers", help="comma-separated tickers (default: 18 ETFs)")
    parser.add_argument("--fred-series", help="comma-separated FRED IDs")
    parser.add_argument("--notes", default=None,
                        help="free-text note stored in manifest.json")
    parser.add_argument("--list", action="store_true",
                        help="just list known snapshots and exit")
    args = parser.parse_args()

    from engine.data_snapshot import freeze_snapshot, list_snapshots

    if args.list:
        snaps = list_snapshots()
        if not snaps:
            print("No snapshots found.")
            return 0
        for s in snaps:
            print(f"{s['snapshot_id']:40s}  {s.get('fetch_start')} → {s.get('fetch_end')}  "
                  f"({len(s.get('tickers', []))} tickers, "
                  f"{len(s.get('fred_series', []))} FRED, "
                  f"created {s.get('created_at')})")
        return 0

    if not args.start or not args.name:
        parser.error("--start and --name are required (or use --list)")
    start = datetime.date.fromisoformat(args.start)
    end = datetime.date.fromisoformat(args.end) if args.end else datetime.date.today()
    tickers = [t.strip() for t in (args.tickers or "").split(",") if t.strip()] or DEFAULT_TICKERS
    fred_series = (
        [t.strip() for t in (args.fred_series or "").split(",") if t.strip()]
        or DEFAULT_FRED_SERIES
    )

    snap = freeze_snapshot(
        start_date  = start,
        end_date    = end,
        name        = args.name,
        tickers     = tickers,
        fred_series = fred_series,
        notes       = args.notes,
    )
    print(json.dumps({
        "snapshot_id":   snap.snapshot_id,
        "n_tickers":     len(snap.tickers),
        "n_fred":        len(snap.fred_series),
        "monthly_rows":  int(len(snap.yf_monthly_etf)),
        "daily_rows":    int(len(snap.yf_daily_etf)),
        "vix_rows":      int(len(snap.yf_vix)),
        "fred_rows":     int(len(snap.fred_macros)),
        "path":          str(snap.path),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
