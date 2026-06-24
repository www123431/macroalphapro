"""
scripts/run_counterfactual_pnl_daily.py — Daily P&L delta computation.

Pre-registration: docs/spec_etf_holdings_llm_risk_monitor.md (id=49)
Spec section: §2.9 Counterfactual Tracking + §3.1 L1 Mechanism integrity

Purpose
-------
Daily cron entry — reads latest dual-track snapshot, fetches per-ETF returns
for as_of, computes Track A vs Track B P&L delta, persists to
data/etf_holdings_risk_monitor/counterfactual_pnl.parquet.

Resilience: never crashes — logs and skips on missing snapshot / fetch failures.

Usage
-----
  python scripts/run_counterfactual_pnl_daily.py
  python scripts/run_counterfactual_pnl_daily.py --as-of 2026-05-09
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("counterfactual_pnl_daily")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(as_of: Optional[datetime.date] = None, *, verbose: bool = False) -> dict:
    _setup_logging(verbose)
    if as_of is None:
        as_of = datetime.date.today()

    logger.info("=" * 70)
    logger.info("Counterfactual P&L daily delta as_of=%s (spec id=49)", as_of)
    logger.info("=" * 70)

    from engine.etf_holdings_counterfactual import (
        compute_daily_pnl_delta,
        persist_daily_pnl_delta,
        compute_cumulative_metrics,
        get_latest_dual_track_snapshot,
    )

    snapshot = get_latest_dual_track_snapshot(as_of)
    if snapshot is None:
        logger.warning("No dual-track snapshot exists. Skipping. Run "
                       "scripts/run_dual_track_snapshot.py first.")
        result = {"status": "no_snapshot", "as_of": as_of.isoformat()}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    logger.info("Latest snapshot: %s with %d capped ETFs",
                snapshot["snapshot_date"], len(snapshot["capped_etfs"]))

    # Compute daily delta
    record = compute_daily_pnl_delta(as_of, snapshot=snapshot)
    logger.info("Daily delta: %s", json.dumps(record, ensure_ascii=False, default=str))

    persisted = persist_daily_pnl_delta(record)
    if persisted:
        logger.info("Persisted to data/etf_holdings_risk_monitor/counterfactual_pnl.parquet")
    else:
        logger.warning("Failed to persist daily delta")

    # Surface cumulative metrics
    cumulative = compute_cumulative_metrics()
    logger.info("Cumulative metrics: %s",
                json.dumps(cumulative, ensure_ascii=False, default=str))

    result = {
        "as_of":                as_of.isoformat(),
        "spec_id":              49,
        "daily_record":         record,
        "cumulative_metrics":   cumulative,
        "persisted":            persisted,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfactual P&L daily delta")
    parser.add_argument("--as-of", type=str, default=None,
                        help="ISO date (YYYY-MM-DD); default = today")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    as_of_arg = (
        datetime.date.fromisoformat(args.as_of) if args.as_of else None
    )
    try:
        main(as_of_arg, verbose=args.verbose)
        sys.exit(0)
    except Exception as exc:
        logger.error("Counterfactual daily run failed: %s", exc, exc_info=True)
        sys.exit(1)
