"""
scripts/run_path_ae_verdict.py — Path AE v3 alpha class verdict.

Daniel-Moskowitz 2016 risk-managed momentum on top-1500 per spec id=79
hash 8353c298. Same 5 v3 alpha gates as Path AD with G6 portfolio Sharpe lift.

Decision: 5/5 PASS - 4/5 MARGINAL - <=3/5 FAIL
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
    NW_LAG, CRISIS_WINDOWS,
    annualized_sharpe, newey_west_t,
)
from scripts.run_path_aa_verdict import peak_to_trough_dd, crisis_window_dd, load_spy_weekly
from scripts.run_v3_retro_analysis import (
    G6_ALPHA_SHARPE_IMPROVEMENT_THRESHOLD, ALPHA_WEIGHT_SWEEP,
    SLEEVE_WEIGHTS, compute_current_4_sleeve_returns, evaluate_g6_alpha,
)


def main() -> None:
    print("=" * 78)
    print("Path AE Risk-Managed Momentum on top-1500 — v3 ALPHA class verdict")
    print("=" * 78)
    print()

    pv_path = REPO_ROOT / "data" / "portfolio_replay" / "v1_path_ae_rm_momentum_top1500_weekly.parquet"
    pv = pd.read_parquet(pv_path)
    pv.index = pd.to_datetime(pv.index)
    pv_net = pv["net"].dropna()
    print(f"Path AE net returns: n={len(pv_net)} weeks")

    existing = pd.read_parquet(
        REPO_ROOT / "data" / "portfolio_replay" / "v1_per_strategy_returns_weekly.parquet"
    ).astype("float64").fillna(0.0)
    existing.index = pd.to_datetime(existing.index)
    baseline = compute_current_4_sleeve_returns(existing)

    spy_weekly = load_spy_weekly()
    print()

    # G1
    sharpe_net = annualized_sharpe(pv_net)
    g1_pass = sharpe_net >= G1_SHARPE_THRESHOLD
    print(f"G1           Sharpe net ann.  = {sharpe_net:+.4f}   (>= {G1_SHARPE_THRESHOLD})   -> {'PASS' if g1_pass else 'FAIL'}")

    # G2
    excess = (pv_net - WEEKLY_RFR).to_numpy()
    nw_t = newey_west_t(excess, lag=NW_LAG)
    g2_pass = (not math.isnan(nw_t)) and (nw_t > G2_T_THRESHOLD)
    print(f"G2           Newey-West t     = {nw_t:+.4f}   (>  {G2_T_THRESHOLD})   -> {'PASS' if g2_pass else 'FAIL'}")

    # G3
    common = pv_net.index.intersection(existing.index)
    pv_a = pv_net.loc[common]
    ex_a = existing.loc[common]
    rho_vec = {col: float(pv_a.corr(ex_a[col])) for col in ex_a.columns}
    max_abs_rho = max(abs(v) for v in rho_vec.values())
    g3_pass = max_abs_rho <= G3_RHO_THRESHOLD
    print(f"G3           max |rho|        = {max_abs_rho:+.4f}   (<= {G3_RHO_THRESHOLD})   -> {'PASS' if g3_pass else 'FAIL'}")
    for col, r in rho_vec.items():
        print(f"               rho(AE, {col:<12}) = {r:+.4f}")

    # G5-v2
    print()
    print("G5-v2        Crisis DD attenuation vs SPY:")
    g5_wins = 0
    g5_details = {}
    for label, (start, end) in CRISIS_WINDOWS.items():
        spy_dd = crisis_window_dd(spy_weekly, start, end)
        cand_dd = crisis_window_dd(pv_net, start, end)
        win = cand_dd >= spy_dd
        if win:
            g5_wins += 1
        g5_details[label] = {"spy_dd": spy_dd, "cand_dd": cand_dd, "win": win}
        marker = "WIN " if win else "LOSE"
        print(f"               [{marker}] {label}: AE DD = {cand_dd*100:+.2f}%, SPY DD = {spy_dd*100:+.2f}%")
    g5_pass = g5_wins >= 2
    print(f"             G5-v2 result: {g5_wins} of 3   (>= 2/3)   -> {'PASS' if g5_pass else 'FAIL'}")

    # G6
    print()
    print(f"G6           Portfolio Sharpe lift (4-sleeve baseline):")
    g6 = evaluate_g6_alpha(pv_net, baseline)
    print(f"               base Sharpe = {g6['base_sharpe']:+.4f}")
    for w, sh in g6["sharpe_by_w"].items():
        print(f"               w={int(w*100):2d}%: blend Sharpe = {sh:+.4f}")
    print(f"             Best lift = {g6['sharpe_improvement_pct']*100:+.2f}% at w={int(g6['best_w']*100)}%   "
          f"(>= {G6_ALPHA_SHARPE_IMPROVEMENT_THRESHOLD*100:.0f}%)   -> {'PASS' if g6['g6_pass'] else 'FAIL'}")
    print()

    # AE-specific stats
    avg_scale = float(pv["scale"].dropna().mean())
    pct_lo = float((pv["scale"] <= 0.501).sum() / len(pv) * 100)
    pct_hi = float((pv["scale"] >= 1.999).sum() / len(pv) * 100)
    print(f"Vol-scaling stats: avg_scale={avg_scale:.3f}, "
          f"clamped LO {pct_lo:.1f}%, HI {pct_hi:.1f}%")
    print()

    n_pass = sum([g1_pass, g2_pass, g3_pass, g5_pass, g6["g6_pass"]])
    verdict = "PASS" if n_pass == 5 else "MARGINAL" if n_pass == 4 else "FAIL"
    print("=" * 78)
    print(f"  Path AE: {n_pass}/5 v3 alpha gates PASS  ->  VERDICT: {verdict}")
    print("=" * 78)

    today = datetime.date.today()
    payload = {
        "spec_id":   79,
        "spec_hash": "8353c298ba982585a7f6971b86aba46eca058eef",
        "spec_name": "Daniel-Moskowitz Risk-Managed Momentum on top-1500",
        "framework": "v3 alpha (id=76 hash 7400b360)",
        "category":  "alpha",
        "run_date":  today.isoformat(),
        "window":    {"start": str(pv_net.index.min().date()),
                      "end":   str(pv_net.index.max().date()),
                      "n_weeks": int(len(pv_net))},
        "vol_scaling_params": {"sigma_target_ann": 0.18, "sigma_window_weeks": 22,
                                "scale_clamp": [0.5, 2.0],
                                "avg_scale_observed": avg_scale,
                                "pct_clamped_low": pct_lo,
                                "pct_clamped_high": pct_hi},
        "verdict": verdict,
        "n_pass":  int(n_pass),
        "gates": {
            "G1_sharpe":       {"value": float(sharpe_net), "threshold": G1_SHARPE_THRESHOLD, "pass": bool(g1_pass)},
            "G2_nw_t":         {"value": float(nw_t),       "threshold": G2_T_THRESHOLD,     "pass": bool(g2_pass)},
            "G3_max_abs_rho":  {"value": float(max_abs_rho),"threshold": G3_RHO_THRESHOLD,   "pass": bool(g3_pass), "rho_by_sleeve": rho_vec},
            "G5_v2_dd_atten":  {"wins":  int(g5_wins),      "threshold": 2,                  "pass": bool(g5_pass), "detail": g5_details},
            "G6_portfolio_sharpe": g6,
        },
        "summary_stats": {
            "weekly_mean_net":   float(pv_net.mean()),
            "weekly_std_net":    float(pv_net.std()),
            "annualized_vol":    float(pv_net.std() * math.sqrt(52)),
            "weekly_mean_gross": float(pv["gross"].mean()),
            "annual_tc_drag":    float(pv["tc"].sum() / (len(pv) / 52.0)),
            "max_dd_overall":    peak_to_trough_dd(pv_net),
        },
    }
    out_path = REPO_ROOT / "data" / "portfolio_replay" / f"path_ae_verdict_{today.isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Verdict saved: {out_path}")


if __name__ == "__main__":
    main()
