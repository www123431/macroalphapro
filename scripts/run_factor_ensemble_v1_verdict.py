"""
scripts/run_factor_ensemble_v1_verdict.py — Framework E v1 verdict CLI.

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50) §4.5

Per spec amendment 2026-05-09 (pre-Sprint-Week-4 audit Nit #5), this CLI is
deliberately PARAMETER-FREE for date inputs — the harness module constants
OOS_START_DATE / DEFAULT_END_DATE are the single source of truth. CLI does
NOT expose --start / --end overrides, to preserve apples-to-apples between
ensemble + baseline legs and to close HARKing R3 surface.

Usage
-----
  python scripts/run_factor_ensemble_v1_verdict.py
  python scripts/run_factor_ensemble_v1_verdict.py --no-cache    # disable yfinance cache
  python scripts/run_factor_ensemble_v1_verdict.py --no-persist  # don't write JSON/TXT files

Output
------
  data/factor_ensemble_v1/v1_verdict.json   # locked schema (11 mandatory fields)
  data/factor_ensemble_v1/v1_verdict.txt    # human-readable summary
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("factor_ensemble_v1_verdict")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(use_cache: bool, persist: bool, verbose: bool) -> dict:
    _setup_logging(verbose)
    from engine.factor_ensemble_verdict import compute_verdict, build_verdict_json_payload
    from engine.factor_ensemble_walk_forward import OOS_START_DATE, DEFAULT_END_DATE

    logger.info("=" * 70)
    logger.info("Factor Ensemble v1 — walk-forward verdict")
    logger.info("Spec: id=50, harness window LOCKED at %s → %s",
                OOS_START_DATE, DEFAULT_END_DATE)
    logger.info("CLI is parameter-free for dates per spec §4.5 amendment Nit #5")
    logger.info("Cache: %s | Persist: %s", use_cache, persist)
    logger.info("=" * 70)

    result = compute_verdict(use_cache=use_cache, persist=persist)
    payload = build_verdict_json_payload(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    logger.info("=" * 70)
    logger.info("Verdict label: %s", result.decision_label)
    logger.info("ΔSharpe: %+.4f | CI [%+.4f, %+.4f] | Memmel Z=%+.4f",
                result.delta_sharpe_walk_forward, result.ci_lower_95,
                result.ci_upper_95, result.memmel_z)
    logger.info("Ensemble Sharpe=%+.4f vs Baseline Sharpe=%+.4f over %d months",
                result.ensemble_sharpe, result.baseline_sharpe, result.n_oos_months)
    logger.info("Gate 0 baseline status: %s",
                result.harness_ensemble_only_baseline_consistency)
    logger.info("=" * 70)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework E v1 walk-forward verdict (date-param-free per spec)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", default=True,
                        help="Disable yfinance cache (slow path; for cold-start verdict)")
    parser.add_argument("--no-persist", dest="persist", action="store_false", default=True,
                        help="Skip writing JSON/TXT files (dry-run)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    try:
        main(use_cache=args.use_cache, persist=args.persist, verbose=args.verbose)
        sys.exit(0)
    except Exception as exc:
        logger.error("Verdict failed: %s", exc, exc_info=True)
        sys.exit(1)
