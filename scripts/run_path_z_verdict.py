"""
scripts/run_path_z_verdict.py — Path Z Short-Term Reversal verdict.

5 gates per spec §2.6 (same framework as V/W/X for clean sibling comparison).
"""
from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.run_path_v_verdict import (
    WEEKLY_RFR, G1_SHARPE_THRESHOLD, G2_T_THRESHOLD, G3_RHO_THRESHOLD,
    NW_LAG, BOOTSTRAP_N, BOOTSTRAP_BLOCK, CRISIS_WINDOWS,
    annualized_sharpe, newey_west_t, stationary_bootstrap_sharpe_ci,
    crisis_returns,
)


def main() -> None:
    print("=== Path Z Short-Term Reversal verdict run ===")
    print()

    pv_path = REPO_ROOT / "data" / "portfolio_replay" / "v1_path_z_str_weekly.parquet"
    pv = pd.read_parquet(pv_path)
    pv.index = pd.to_datetime(pv.index)
    pv_net = pv["net"].dropna()
    print(f"Path Z net returns: n={len(pv_net)} weeks, "
          f"{pv_net.index.min().date()} -> {pv_net.index.max().date()}")

    existing_path = REPO_ROOT / "data" / "portfolio_replay" / "v1_per_strategy_returns_weekly.parquet"
    existing = pd.read_parquet(existing_path).astype("float64").fillna(0.0)
    existing.index = pd.to_datetime(existing.index)
    print(f"Existing 4 sleeves: n={len(existing)} weeks")
    print()

    # G1
    sharpe_net = annualized_sharpe(pv_net)
    g1_pass = sharpe_net >= G1_SHARPE_THRESHOLD
    print(f"G1 PRIMARY   Sharpe net ann. = {sharpe_net:+.4f}   >= {G1_SHARPE_THRESHOLD}   -> {'PASS' if g1_pass else 'FAIL'}")

    # G2
    excess = (pv_net - WEEKLY_RFR).to_numpy()
    nw_t = newey_west_t(excess, lag=NW_LAG)
    g2_pass = (not math.isnan(nw_t)) and (nw_t > G2_T_THRESHOLD)
    print(f"G2           Newey-West t (lag-8) = {nw_t:+.4f}   > {G2_T_THRESHOLD}   -> {'PASS' if g2_pass else 'FAIL'}")

    # G3
    common_idx = pv_net.index.intersection(existing.index)
    pv_a = pv_net.loc[common_idx]
    ex_a = existing.loc[common_idx]
    rho_vec = {col: float(pv_a.corr(ex_a[col])) for col in ex_a.columns}
    max_abs_rho = max(abs(v) for v in rho_vec.values())
    g3_pass = max_abs_rho <= G3_RHO_THRESHOLD
    print(f"G3           max |rho| vs existing = {max_abs_rho:+.4f}   <= {G3_RHO_THRESHOLD}   -> {'PASS' if g3_pass else 'FAIL'}")
    for col, r in rho_vec.items():
        print(f"               rho(PathZ, {col:<12}) = {r:+.4f}")

    # G4
    ci_lo, ci_hi = stationary_bootstrap_sharpe_ci(pv_net)
    g4_pass = (not math.isnan(ci_lo)) and (ci_lo > 0)
    print(f"G4           Bootstrap 95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]   -> {'PASS' if g4_pass else 'FAIL'}")

    # G5
    crisis = crisis_returns(pv_net)
    n_pos = sum(1 for v in crisis.values() if (not math.isnan(v)) and v >= 0)
    g5_pass = n_pos >= 1
    print(f"G5           Crisis non-neg = {n_pos} of 3   >= 1   -> {'PASS' if g5_pass else 'FAIL'}")
    for label, r in crisis.items():
        sign = "PASS" if (not math.isnan(r) and r >= 0) else "FAIL"
        print(f"               [{sign}] {label}: {r*100:+.3f}%")
    print()

    n_pass = sum([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass])
    verdict = "PASS" if n_pass == 5 else "MARGINAL" if n_pass == 4 else "FAIL"
    print(f"=================================================================")
    print(f"  Path Z: {n_pass}/5 gates PASS  ->  VERDICT: {verdict}")
    print(f"=================================================================")

    today = datetime.date.today()
    payload = {
        "spec_id": 73,
        "spec_hash": "8845f771",
        "spec_name": "Short-Term Reversal (Lehmann 1990 / Jegadeesh 1990)",
        "run_date": today.isoformat(),
        "window": {"start": str(pv_net.index.min().date()),
                   "end":   str(pv_net.index.max().date()),
                   "n_weeks": int(len(pv_net))},
        "verdict": verdict,
        "n_pass":  int(n_pass),
        "gates": {
            "G1_sharpe_net_ann":   {"value": sharpe_net, "threshold": G1_SHARPE_THRESHOLD, "pass": g1_pass},
            "G2_newey_west_t":     {"value": nw_t, "threshold": G2_T_THRESHOLD, "pass": g2_pass},
            "G3_max_abs_rho":      {"value": max_abs_rho, "threshold": G3_RHO_THRESHOLD, "pass": g3_pass,
                                     "rho_by_sleeve": rho_vec},
            "G4_bootstrap_ci_95":  {"lo": ci_lo, "hi": ci_hi, "pass": g4_pass},
            "G5_crisis_non_neg":   {"count": n_pos, "threshold": 1, "pass": g5_pass,
                                     "returns": crisis},
        },
        "summary_stats": {
            "weekly_mean_net":   float(pv_net.mean()),
            "weekly_std_net":    float(pv_net.std()),
            "annualized_vol":    float(pv_net.std() * math.sqrt(52)),
            "weekly_mean_gross": float(pv["gross"].mean()),
            "annual_tc_drag":    float(pv["tc"].sum() / (len(pv) / 52.0)),
        },
    }
    out_path = REPO_ROOT / "data" / "portfolio_replay" / f"path_z_verdict_{today.isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Verdict saved: {out_path}")


if __name__ == "__main__":
    main()
