"""
scripts/run_factor_ensemble_v2_verdict.py — v2 robust verdict CLI.

Pre-registration: docs/spec_factor_ensemble_v2_robust.md (id=51, hash c6d395ad0fb7)

Per spec discipline (inherited from v1 spec §4.5 amendment Nit #5),
this CLI is parameter-free for date inputs.

Outputs:
  data/factor_ensemble_v2/v2_verdict.json
  data/factor_ensemble_v2/v2_verdict.txt
  data/factor_ensemble_v2/v2_per_baseline_diagnostics.parquet
  data/factor_ensemble_v2/v2_per_regime_diagnostics.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("factor_ensemble_v2_verdict")


def main(persist: bool, verbose: bool) -> dict:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from engine.factor_ensemble_v2 import compute_v2_verdict
    from engine.factor_ensemble_walk_forward import OOS_START_DATE, DEFAULT_END_DATE

    logger.info("=" * 70)
    logger.info("Factor Ensemble v2 Robust — walk-forward verdict")
    logger.info("Spec: id=51 (locked from harness module constants)")
    logger.info("Window: %s → %s | TC=8bps | β-neutral=TSMOM | 4 baselines × 4 regimes",
                OOS_START_DATE, DEFAULT_END_DATE)
    logger.info("=" * 70)

    result = compute_v2_verdict(persist=persist)

    summary = {
        "overall_decision":           result.overall_decision,
        "n_oos_months":               result.n_oos_months,
        "ensemble_sharpe_net":        result.ensemble_sharpe_net,
        "n_baselines_positive":       result.n_baselines_positive,
        "n_regimes_positive":         result.n_regimes_positive,
        "abs_net_sharpe_above_zero":  result.abs_net_sharpe_above_zero,
        "per_baseline":               result.per_baseline,
        "per_regime":                 result.per_regime,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("=" * 70)
    logger.info("v2 overall decision: %s | abs Sharpe net: %s",
                result.overall_decision, result.ensemble_sharpe_net)
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factor Ensemble v2 robust verdict")
    parser.add_argument("--no-persist", dest="persist", action="store_false", default=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    try:
        main(persist=args.persist, verbose=args.verbose)
        sys.exit(0)
    except Exception as exc:
        logger.error("v2 verdict failed: %s", exc, exc_info=True)
        sys.exit(1)
