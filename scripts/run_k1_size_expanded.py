"""
scripts/run_k1_size_expanded.py — Path K1 Size-Expanded B++ QL01 runner.

Pre-registration: docs/spec_path_k1_size_expanded_b_plus_v1.md (id=61) §4.1 + §八.

Pipeline:
  1. Validate spec id=61
  2. Build K1 universe (Tier-1 + 10 size/style ETFs = 43 ETFs)
  3. Run B++ QL01 strategy on K1 universe — weekly rebalance 2014-2023
  4. Run B++ QL01 strategy on Tier-1 universe (paired baseline)
  5. Compute paired Memmel Z-test on weekly Sharpe difference
  6. Build verdict.json with K1 absolute + Memmel comparison
  7. Persist outputs to data/path_c_k1/

Usage:
  py -3.11 scripts/run_k1_size_expanded.py
  py -3.11 scripts/run_k1_size_expanded.py --start 2014-01-01 --end 2023-12-31 --run-id v1_k1_size_expanded
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

logger = logging.getLogger("path_c.run_k1")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run_k1_size_expanded.py")
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end",   default="2023-12-31")
    p.add_argument("--run-id", default="v1_k1_size_expanded")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from engine.preregistration import (
        validate_reference, _compute_git_blob_hash, _resolve_to_abs,
    )
    from engine.b_plus_search import (
        get_universe_tier, get_strategy, run_single_strategy_weekly,
        fetch_universe_closes,
    )
    from engine.path_c.k1_universe import get_k1_universe
    from engine.multivariate_msm_verdict import memmel_z_paired_sharpe_diff

    SPEC_PATH = "docs/spec_path_k1_size_expanded_b_plus_v1.md"

    # Step 0: pre-reg validation
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason); return 2
    logger.info("Spec validated.")
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))

    # Step 1: universes
    tier1_universe = get_universe_tier(1)
    k1_universe    = get_k1_universe()
    logger.info("Tier-1 universe: %d ETFs", len(tier1_universe))
    logger.info("K1 universe: %d ETFs (=Tier-1 + 10 size/style)", len(k1_universe))

    # Pre-fetch closes once for both (K1 superset includes all Tier-1 tickers)
    fetch_start = pd.Timestamp(args.start) - pd.Timedelta(days=365 * 6)
    fetch_end   = pd.Timestamp(args.end)
    logger.info("Bulk-fetching closes for K1 universe over [%s, %s]",
                fetch_start.date(), fetch_end.date())
    closes_k1 = fetch_universe_closes(k1_universe, fetch_start.date(), fetch_end.date())
    logger.info("Closes: %d dates × %d tickers", *closes_k1.shape)

    # Step 2: QL01 strategy spec
    ql01_spec = get_strategy("QL01")
    logger.info("Strategy: %s, params=%s", ql01_spec.id, ql01_spec.params)

    # Step 3: Run K1
    logger.info("Running QL01 on K1 universe (43 ETFs)...")
    k1_result = run_single_strategy_weekly(
        spec=ql01_spec, universe=k1_universe,
        start_date=args.start, end_date=args.end,
        closes=closes_k1,
    )
    if "error" in k1_result:
        logger.error("K1 run failed: %s", k1_result["error"]); return 3
    logger.info("K1: n_obs=%d Sharpe=%.3f NW t=%.3f",
                k1_result["n_obs"], k1_result["sharpe"], k1_result["nw_t_stat"])

    # Step 4: Run Tier-1 (paired baseline)
    logger.info("Running QL01 on Tier-1 universe (33 ETFs) for paired baseline...")
    # Tier-1 is a subset of K1 universe → reuse closes
    tier1_closes = closes_k1[[t for t in tier1_universe.values() if t in closes_k1.columns]]
    t1_result = run_single_strategy_weekly(
        spec=ql01_spec, universe=tier1_universe,
        start_date=args.start, end_date=args.end,
        closes=tier1_closes,
    )
    if "error" in t1_result:
        logger.error("T1 run failed: %s", t1_result["error"]); return 4
    logger.info("T1: n_obs=%d Sharpe=%.3f NW t=%.3f",
                t1_result["n_obs"], t1_result["sharpe"], t1_result["nw_t_stat"])

    # Step 5: Memmel paired Z
    k1_returns = k1_result["weekly_returns"]
    t1_returns = t1_result["weekly_returns"]
    if not isinstance(k1_returns, pd.Series):
        k1_returns = pd.Series(k1_returns)
    if not isinstance(t1_returns, pd.Series):
        t1_returns = pd.Series(t1_returns)
    logger.info("Memmel paired Z-test on weekly Sharpe diff...")
    z, rho, V = memmel_z_paired_sharpe_diff(k1_returns, t1_returns, obs_per_year=52)
    logger.info("Memmel: Z=%.3f rho=%.3f V=%.3f", z, rho, V)

    # Step 6: Decision logic
    k1_sharpe = float(k1_result["sharpe"])
    k1_nw_t   = float(k1_result["nw_t_stat"])
    t1_sharpe = float(t1_result["sharpe"])
    t1_nw_t   = float(t1_result["nw_t_stat"])

    # Industry-grade gates (BHY demoted)
    if k1_sharpe >= 0.5 and k1_nw_t >= 2.0:
        absolute = "PASS"
    elif k1_sharpe >= 0.3 and k1_nw_t >= 1.5:
        absolute = "MARGINAL"
    else:
        absolute = "FAIL"

    # Deploy threshold per spec §3.3
    if absolute == "PASS" and z > 1.5:
        deploy = "deploy_k1_replace_t1"
    elif absolute == "PASS" and -1.0 <= z <= 1.5:
        deploy = "keep_t1_paper_trade_k1"
    elif absolute == "MARGINAL":
        deploy = "k1_paper_trade_24mo"
    elif absolute == "FAIL":
        deploy = "no_deploy_universe_dilutes_alpha"
    else:
        deploy = "no_deploy_negative_memmel"

    logger.info("Verdict: %s | Memmel Z %.3f | Decision: %s", absolute, z, deploy)

    # Step 7: persist
    out_dir = REPO_ROOT / "data" / "path_c_k1"
    out_dir.mkdir(parents=True, exist_ok=True)

    verdict = {
        "decision":                     absolute,
        "deploy_decision":              deploy,
        "spec_hash":                    spec_hash,
        "spec_path":                    SPEC_PATH,
        "run_at":                       datetime.datetime.utcnow().isoformat() + "Z",
        "wave":                         "K1-size-expanded",
        "window_start":                 args.start,
        "window_end":                   args.end,
        "k1_universe_size":             len(k1_universe),
        "t1_universe_size":             len(tier1_universe),
        "k1_n_obs":                     int(k1_result["n_obs"]),
        "k1_sharpe":                    k1_sharpe,
        "k1_nw_t":                      k1_nw_t,
        "k1_ann_return":                float(k1_result.get("ann_return", float("nan"))),
        "k1_ann_vol":                   float(k1_result.get("ann_vol", float("nan"))),
        "t1_sharpe":                    t1_sharpe,
        "t1_nw_t":                      t1_nw_t,
        "memmel_z":                     float(z) if np.isfinite(z) else None,
        "memmel_rho":                   float(rho) if np.isfinite(rho) else None,
        "memmel_V":                     float(V) if np.isfinite(V) else None,
        "delta_sharpe_k1_minus_t1":     k1_sharpe - t1_sharpe,
        "comparison_baseline":          "B++ QL01_T1 paired re-run on same weekly windows",
        "comparison_baseline_original": {
            "sharpe": 0.985,
            "nw_t":   2.31,
            "source": "data/b_plus_results/oos_verdict.json (2026-05-04)",
        },
    }
    verdict_path = out_dir / f"{args.run_id}_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Persist weekly returns parquet for reproducibility
    df = pd.DataFrame({
        "k1_weekly_returns": k1_returns.values,
        "t1_weekly_returns": t1_returns.values[:len(k1_returns)] if len(t1_returns) >= len(k1_returns) else np.concatenate([t1_returns.values, [np.nan] * (len(k1_returns) - len(t1_returns))]),
    })
    df.to_parquet(out_dir / f"{args.run_id}_paired_returns.parquet")

    logger.info("Artifacts: %s", verdict_path)
    logger.info("=" * 70)
    logger.info("FINAL VERDICT: %s", absolute)
    logger.info("  K1 Sharpe %.3f NW t %.3f", k1_sharpe, k1_nw_t)
    logger.info("  T1 Sharpe %.3f NW t %.3f (paired re-run)", t1_sharpe, t1_nw_t)
    logger.info("  ΔSharpe K1 - T1 = %+.3f, Memmel Z = %+.3f", k1_sharpe - t1_sharpe, z)
    logger.info("  Deploy decision: %s", deploy)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
