"""
scripts/run_path_wx_verdict.py — Path W (52WH) + Path X (IVOL) sibling verdicts.

Runs same 5 gates as Path V verdict (sibling test — clean comparison).
Outputs 2 verdict JSON + 2 capability evidence MDs.
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

# Reuse gate functions from Path V verdict script
from scripts.run_path_v_verdict import (
    WEEKLY_RFR,
    G1_SHARPE_THRESHOLD, G2_T_THRESHOLD, G3_RHO_THRESHOLD,
    NW_LAG, BOOTSTRAP_N, BOOTSTRAP_BLOCK,
    CRISIS_WINDOWS,
    annualized_sharpe, newey_west_t, stationary_bootstrap_sharpe_ci,
    crisis_returns,
)


def run_verdict(spec_name: str) -> dict:
    """Run 5-gate verdict for Path W or X. Returns verdict dict."""
    print(f"=== Path {spec_name} verdict run ===")
    print()

    pv_path = REPO_ROOT / "data" / "portfolio_replay" / f"v1_path_{spec_name.lower()}_weekly.parquet"
    pv = pd.read_parquet(pv_path)
    pv.index = pd.to_datetime(pv.index)
    pv_net = pv["net"].dropna()
    print(f"Path {spec_name} net returns: n={len(pv_net)} weeks, "
          f"{pv_net.index.min().date()} → {pv_net.index.max().date()}")

    existing_path = REPO_ROOT / "data" / "portfolio_replay" / "v1_per_strategy_returns_weekly.parquet"
    existing = pd.read_parquet(existing_path).astype("float64").fillna(0.0)
    existing.index = pd.to_datetime(existing.index)
    print(f"Existing 4 sleeves: n={len(existing)} weeks")
    print()

    # G1 PRIMARY
    sharpe_net = annualized_sharpe(pv_net)
    g1_pass = sharpe_net >= G1_SHARPE_THRESHOLD
    print(f"G1 PRIMARY   Sharpe (net, ann.) = {sharpe_net:+.4f}   threshold ≥ {G1_SHARPE_THRESHOLD}   →   {'PASS' if g1_pass else 'FAIL'}")

    # G2
    excess = (pv_net - WEEKLY_RFR).to_numpy()
    nw_t = newey_west_t(excess, lag=NW_LAG)
    g2_pass = (not math.isnan(nw_t)) and (nw_t > G2_T_THRESHOLD)
    print(f"G2           Newey-West t (lag-8) = {nw_t:+.4f}   threshold > {G2_T_THRESHOLD}   →   {'PASS' if g2_pass else 'FAIL'}")

    # G3
    common_idx = pv_net.index.intersection(existing.index)
    pv_aligned = pv_net.loc[common_idx]
    existing_aligned = existing.loc[common_idx]
    rho_vec = {col: float(pv_aligned.corr(existing_aligned[col]))
               for col in existing_aligned.columns}
    max_abs_rho = max(abs(v) for v in rho_vec.values())
    g3_pass = max_abs_rho <= G3_RHO_THRESHOLD
    print(f"G3           max |rho| vs existing sleeves = {max_abs_rho:+.4f}   threshold ≤ {G3_RHO_THRESHOLD}   →   {'PASS' if g3_pass else 'FAIL'}")
    for col, r in rho_vec.items():
        print(f"               rho(Path{spec_name}, {col:<12}) = {r:+.4f}")

    # G4
    ci_lo, ci_hi = stationary_bootstrap_sharpe_ci(pv_net)
    g4_pass = (not math.isnan(ci_lo)) and (ci_lo > 0)
    print(f"G4           Bootstrap 95% CI on Sharpe = [{ci_lo:+.4f}, {ci_hi:+.4f}]   excludes 0?   →   {'PASS' if g4_pass else 'FAIL'}")

    # G5
    crisis_ret = crisis_returns(pv_net)
    n_non_negative = sum(1 for v in crisis_ret.values()
                          if not math.isnan(v) and v >= 0)
    g5_pass = n_non_negative >= 1
    print(f"G5           Crisis non-negative count = {n_non_negative} of 3   threshold ≥ 1   →   {'PASS' if g5_pass else 'FAIL'}")
    for label, r in crisis_ret.items():
        sign = "PASS" if (not math.isnan(r) and r >= 0) else "FAIL"
        print(f"               [{sign}] {label}: {r*100:+.3f}%")
    print()

    n_pass = sum([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass])
    if n_pass == 5:
        verdict = "PASS"
    elif n_pass == 4:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Path {spec_name}: {n_pass}/5 gates PASS  →  VERDICT: {verdict}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    spec_meta = {
        "W": {"id": 70, "hash": "758206c5", "name": "52-Week High Momentum"},
        "X": {"id": 71, "hash": "0c71f9a5", "name": "Idiosyncratic Volatility"},
    }[spec_name]

    today = datetime.date.today()
    payload = {
        "spec_id":  spec_meta["id"],
        "spec_hash": spec_meta["hash"],
        "spec_name": spec_meta["name"],
        "run_date": today.isoformat(),
        "window": {"start": str(pv_net.index.min().date()),
                   "end":   str(pv_net.index.max().date()),
                   "n_weeks": int(len(pv_net))},
        "verdict": verdict,
        "n_pass":  int(n_pass),
        "gates": {
            "G1_sharpe_net_ann":   {"value": sharpe_net, "threshold": G1_SHARPE_THRESHOLD, "pass": g1_pass},
            "G2_newey_west_t":     {"value": nw_t,       "threshold": G2_T_THRESHOLD,      "pass": g2_pass},
            "G3_max_abs_rho":      {"value": max_abs_rho, "threshold": G3_RHO_THRESHOLD,   "pass": g3_pass,
                                     "rho_by_sleeve": rho_vec},
            "G4_bootstrap_ci_95":  {"lo": ci_lo, "hi": ci_hi, "pass": g4_pass},
            "G5_crisis_non_neg":   {"count": n_non_negative, "threshold": 1, "pass": g5_pass,
                                     "returns": crisis_ret},
        },
        "summary_stats": {
            "weekly_mean_net":   float(pv_net.mean()),
            "weekly_std_net":    float(pv_net.std()),
            "annualized_vol":    float(pv_net.std() * math.sqrt(52)),
            "weekly_mean_gross": float(pv["gross"].mean()),
            "weekly_mean_tc":    float(pv["tc"].mean()),
            "annual_tc_drag":    float(pv["tc"].sum() / (len(pv) / 52.0)),
        },
    }

    out_dir = REPO_ROOT / "data" / "portfolio_replay"
    out_path = out_dir / f"path_{spec_name.lower()}_verdict_{today.isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Verdict saved: {out_path}")
    return payload


def main():
    print()
    w_payload = run_verdict("W")
    print()
    print()
    x_payload = run_verdict("X")


if __name__ == "__main__":
    main()
