"""
scripts/run_backtest.py — CLI runner for engine.backtest.run_backtest
(S-3 reproducibility, 2026-05-06).

Headless backtest entry point so a thesis examiner can reproduce numbers
without spinning up the Streamlit UI. Pairs with
`scripts/freeze_backtest_data.py`:

    # 1) freeze the inputs you want to anchor the result on
    python scripts/freeze_backtest_data.py --start=2009-01-01 --name=thesis_v1

    # 2) reproduce a headline number
    python scripts/run_backtest.py \\
        --start=2010-01-01 --end=2024-12-31 \\
        --use-snapshot=thesis_v1_2026-05-06

Output is JSON to stdout: BacktestMetrics for portfolio A (TSMOM only),
B (TSMOM + regime), and BM (60/40 SPY/AGG). Pipe to jq / save to file
as needed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _metrics_to_dict(m) -> dict:
    if m is None:
        return {}
    if dataclasses.is_dataclass(m):
        return dataclasses.asdict(m)
    if isinstance(m, dict):
        return m
    return {"repr": repr(m)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--start",     required=True, help="ISO start date")
    parser.add_argument("--end",       required=True, help="ISO end date")
    parser.add_argument("--lookback-months",  type=int, default=12)
    parser.add_argument("--skip-months",      type=int, default=1)
    parser.add_argument("--regime-scale",     type=float, default=1.0,
                        help="REGIME_SCALE override; production baseline is 1.0")
    parser.add_argument("--transaction-cost", type=float, default=None)
    parser.add_argument("--use-snapshot",     default=None,
                        help="snapshot_id to use (zero live network calls)")
    parser.add_argument("--verify-hashes",    action="store_true", default=True,
                        help="Verify snapshot file sha256 on load (default on)")
    args = parser.parse_args()

    snapshot = None
    if args.use_snapshot:
        from engine.data_snapshot import load_snapshot
        snapshot = load_snapshot(args.use_snapshot, verify_hashes=args.verify_hashes)
        print(f"# Loaded snapshot {snapshot.snapshot_id} "
              f"({len(snapshot.tickers)} tickers, "
              f"{snapshot.fetch_start} → {snapshot.fetch_end})",
              file=sys.stderr)

    from engine.backtest import run_backtest
    kwargs = dict(
        start_date       = args.start,
        end_date         = args.end,
        lookback_months  = args.lookback_months,
        skip_months      = args.skip_months,
        regime_scale     = args.regime_scale,
        snapshot         = snapshot,
    )
    if args.transaction_cost is not None:
        kwargs["transaction_cost"] = args.transaction_cost

    result = run_backtest(**kwargs)

    out = {
        "config": {
            "start":           args.start,
            "end":             args.end,
            "lookback_months": args.lookback_months,
            "skip_months":     args.skip_months,
            "regime_scale":    args.regime_scale,
            "snapshot_id":     args.use_snapshot,
        },
        "metrics_a":   _metrics_to_dict(getattr(result, "metrics_a", None)),
        "metrics_b":   _metrics_to_dict(getattr(result, "metrics_b", None)),
        "metrics_bm":  _metrics_to_dict(getattr(result, "metrics_bm", None)),
        "warnings":    list(getattr(result, "warnings_log", []) or []),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
