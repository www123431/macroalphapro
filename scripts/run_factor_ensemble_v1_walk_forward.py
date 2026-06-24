"""
scripts/run_factor_ensemble_v1_walk_forward.py — Framework E v1 walk-forward CLI.

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50, hash 1665945d2ca5)

Usage
-----
  # Full OOS verdict run (2011-2024)
  python scripts/run_factor_ensemble_v1_walk_forward.py

  # Custom range
  python scripts/run_factor_ensemble_v1_walk_forward.py --start 2015-01-01 --end 2024-12-31

  # Baseline-only (BAB) for Gate 0 reproducibility comparison
  python scripts/run_factor_ensemble_v1_walk_forward.py --baseline-only

Output
------
  data/factor_ensemble_v1/walk_forward.parquet           # per-period returns + diagnostics
  data/factor_ensemble_v1/per_factor_signals.parquet     # per-factor coverage stats per period
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

logger = logging.getLogger("factor_ensemble_v1_walk_forward")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(
    start_date:    Optional[datetime.date] = None,
    end_date:      Optional[datetime.date] = None,
    baseline_only: bool = False,
    verbose:       bool = False,
) -> dict:
    _setup_logging(verbose)

    from engine.factor_ensemble_walk_forward import (
        OOS_START_DATE,
        DEFAULT_END_DATE,
        run_walk_forward,
    )

    start = start_date or OOS_START_DATE
    end = end_date or DEFAULT_END_DATE

    logger.info("=" * 70)
    logger.info("Factor Ensemble v1 walk-forward")
    logger.info("Spec: id=50 hash=1665945d2ca5")
    logger.info("Window: %s → %s", start, end)
    logger.info("Mode: %s", "BAB-only baseline (Gate 0)" if baseline_only else "Full ensemble")
    logger.info("=" * 70)

    result = run_walk_forward(
        start_date=start,
        end_date=end,
        baseline_only=baseline_only,
        use_cache=True,
        persist=True,
    )

    summary = {
        "spec_id":           50,
        "spec_hash_prefix":  "1665945d2ca5",
        "mode":              "baseline_only" if baseline_only else "ensemble_v1",
        "start_date":        start.isoformat(),
        "end_date":          end.isoformat(),
        "n_periods":         result.n_periods,
        "annualized_sharpe": round(result.annualized_sharpe, 4),
        "annualized_vol":    round(result.annualized_vol, 4),
        "cumulative_return": round(result.cumulative_return, 4),
        "max_drawdown":      round(result.max_drawdown, 4),
        "completed_at":      datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("=" * 70)
    logger.info("Walk-forward DONE — Sharpe=%.4f, Vol=%.4f, Cumulative=%.4f, MaxDD=%.4f",
                result.annualized_sharpe, result.annualized_vol,
                result.cumulative_return, result.max_drawdown)
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework E v1 walk-forward")
    parser.add_argument("--start", type=str, default=None, help="ISO YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="ISO YYYY-MM-DD")
    parser.add_argument("--baseline-only", action="store_true",
                        help="BAB-only mode for Gate 0 reproducibility")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    s = datetime.date.fromisoformat(args.start) if args.start else None
    e = datetime.date.fromisoformat(args.end) if args.end else None

    try:
        main(start_date=s, end_date=e,
             baseline_only=args.baseline_only, verbose=args.verbose)
        sys.exit(0)
    except Exception as exc:
        logger.error("Walk-forward failed: %s", exc, exc_info=True)
        sys.exit(1)
