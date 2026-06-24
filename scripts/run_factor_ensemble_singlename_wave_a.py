"""
scripts/run_factor_ensemble_singlename_wave_a.py — Stage 2 Wave A verdict CLI.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52)

Wave A uses mktcap_top500_proxy as primary universe (PURE SURVIVORSHIP, lower
bound; honest preliminary case study) + Wikipedia archive as robustness check.

Output:
  data/factor_ensemble_singlename/v1_wave_a_verdict.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("singlestock_wave_a")


def main(use_cache: bool, persist: bool, universe_source: str, verbose: bool) -> dict:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from engine.factor_ensemble_singlename import compute_singlestock_verdict
    from engine.universe_singlename import load_sp500_constituents_at_date

    # Universe lookup function — closes over the chosen source
    def universe_at_date_fn(d: datetime.date) -> list:
        return load_sp500_constituents_at_date(as_of=d, source=universe_source).tickers

    logger.info("=" * 70)
    logger.info("Factor Ensemble Single-Name Wave A — walk-forward verdict")
    logger.info("Universe source: %s (Wave A is PRELIMINARY, not publishable)", universe_source)
    logger.info("Spec: id=52 (sample period locked from harness const)")
    logger.info("=" * 70)

    result = compute_singlestock_verdict(
        universe_at_date_fn=universe_at_date_fn,
        wave="A",
        use_cache=use_cache,
        persist=persist,
    )

    summary = {
        "wave":                     "A",
        "universe_source":          universe_source,
        "overall_decision":         result.overall_decision,
        "n_oos_months":             result.n_oos_months,
        "ensemble_sharpe_net":      result.ensemble_sharpe_net,
        "n_baselines_positive":     result.n_baselines_positive,
        "n_regimes_positive":       result.n_regimes_positive,
        "abs_net_sharpe_above_zero":result.abs_net_sharpe_above_zero,
        "per_baseline":             result.per_baseline,
        "per_regime":               result.per_regime,
        "metadata":                 result.metadata,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    logger.info("=" * 70)
    logger.info("Wave A overall: %s | abs Sharpe net: %s",
                result.overall_decision, result.ensemble_sharpe_net)
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Wave A walk-forward verdict")
    parser.add_argument("--universe-source", choices=["mktcap_top500_proxy", "wikipedia_archive"],
                        default="mktcap_top500_proxy",
                        help="primary: proxy (PURE SURVIVORSHIP); robustness: wikipedia")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", default=True)
    parser.add_argument("--no-persist", dest="persist", action="store_false", default=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    try:
        main(use_cache=args.use_cache, persist=args.persist,
             universe_source=args.universe_source, verbose=args.verbose)
        sys.exit(0)
    except Exception as exc:
        logger.error("Wave A verdict failed: %s", exc, exc_info=True)
        sys.exit(1)
