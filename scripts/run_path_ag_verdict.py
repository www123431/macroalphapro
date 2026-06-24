"""
scripts/run_path_ag_verdict.py — Path AG IVOL top-1500 v3 alpha class verdict.
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

import pandas as pd

from scripts.run_path_v_verdict import (
    WEEKLY_RFR, G1_SHARPE_THRESHOLD, G2_T_THRESHOLD, G3_RHO_THRESHOLD,
    NW_LAG, CRISIS_WINDOWS,
    annualized_sharpe, newey_west_t,
)
from scripts.run_path_aa_verdict import peak_to_trough_dd, crisis_window_dd, load_spy_weekly
from scripts.run_v3_retro_analysis import (
    G6_ALPHA_SHARPE_IMPROVEMENT_THRESHOLD,
    compute_current_4_sleeve_returns, evaluate_g6_alpha,
)


def main() -> None:
    print("=" * 78)
    print("Path AG IVOL on top-1500 — v3 ALPHA class verdict")
    print("=" * 78)
    print()

    pv = pd.read_parquet(REPO_ROOT / "data/portfolio_replay/v1_path_ag_ivol_top1500_weekly.parquet")
    pv.index = pd.to_datetime(pv.index)
    pv_net = pv["net"].dropna()
    print(f"Path AG net returns: n={len(pv_net)} weeks")
    print()

    existing = pd.read_parquet(
        REPO_ROOT / "data/portfolio_replay/v1_per_strategy_returns_weekly.parquet"
    ).astype("float64").fillna(0.0)
    existing.index = pd.to_datetime(existing.index)
    baseline = compute_current_4_sleeve_returns(existing)
    spy_weekly = load_spy_weekly()

    sharpe_net = annualized_sharpe(pv_net)
    g1_pass = sharpe_net >= G1_SHARPE_THRESHOLD
    print(f"G1     Sharpe net ann.  = {sharpe_net:+.4f}   (>= {G1_SHARPE_THRESHOLD})   -> {'PASS' if g1_pass else 'FAIL'}")

    excess = (pv_net - WEEKLY_RFR).to_numpy()
    nw_t = newey_west_t(excess, lag=NW_LAG)
    g2_pass = (not math.isnan(nw_t)) and (nw_t > G2_T_THRESHOLD)
    print(f"G2     Newey-West t     = {nw_t:+.4f}   (>  {G2_T_THRESHOLD})   -> {'PASS' if g2_pass else 'FAIL'}")

    common = pv_net.index.intersection(existing.index)
    pv_a = pv_net.loc[common]
    ex_a = existing.loc[common]
    rho_vec = {col: float(pv_a.corr(ex_a[col])) for col in ex_a.columns}
    max_abs_rho = max(abs(v) for v in rho_vec.values())
    g3_pass = max_abs_rho <= G3_RHO_THRESHOLD
    print(f"G3     max |rho|        = {max_abs_rho:+.4f}   (<= {G3_RHO_THRESHOLD})   -> {'PASS' if g3_pass else 'FAIL'}")
    for col, r in rho_vec.items():
        print(f"         rho(AG, {col:<12}) = {r:+.4f}")

    print()
    print("G5-v2  Crisis DD attenuation vs SPY:")
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
        print(f"         [{marker}] {label}: AG DD = {cand_dd*100:+.2f}%, SPY DD = {spy_dd*100:+.2f}%")
    g5_pass = g5_wins >= 2
    print(f"       result: {g5_wins}/3 (>= 2)   -> {'PASS' if g5_pass else 'FAIL'}")

    print()
    print("G6     Portfolio Sharpe lift:")
    g6 = evaluate_g6_alpha(pv_net, baseline)
    print(f"         base Sharpe = {g6['base_sharpe']:+.4f}")
    for w, sh in g6["sharpe_by_w"].items():
        print(f"         w={int(w*100):2d}%: blend Sharpe = {sh:+.4f}")
    print(f"       Best lift = {g6['sharpe_improvement_pct']*100:+.2f}% at w={int(g6['best_w']*100)}%   -> {'PASS' if g6['g6_pass'] else 'FAIL'}")
    print()

    n_pass = sum([g1_pass, g2_pass, g3_pass, g5_pass, g6["g6_pass"]])
    verdict = "PASS" if n_pass == 5 else "MARGINAL" if n_pass == 4 else "FAIL"
    print("=" * 78)
    print(f"  Path AG: {n_pass}/5 v3 alpha gates PASS  ->  VERDICT: {verdict}")
    print("=" * 78)

    today = datetime.date.today()
    payload = {
        "spec_id":   80,
        "spec_hash": "b1bd40cb37b5623344c10fd136251ca470b89def",
        "spec_name": "Idiosyncratic Volatility on top-1500 (AHXZ 2006)",
        "framework": "v3 alpha (id=76 hash 7400b360)",
        "category":  "alpha",
        "run_date":  today.isoformat(),
        "verdict":   verdict,
        "n_pass":    int(n_pass),
        "window":    {"start": str(pv_net.index.min().date()),
                      "end":   str(pv_net.index.max().date()),
                      "n_weeks": int(len(pv_net))},
        "gates": {
            "G1_sharpe":           {"value": float(sharpe_net), "threshold": G1_SHARPE_THRESHOLD, "pass": bool(g1_pass)},
            "G2_nw_t":             {"value": float(nw_t),       "threshold": G2_T_THRESHOLD,     "pass": bool(g2_pass)},
            "G3_max_abs_rho":      {"value": float(max_abs_rho),"threshold": G3_RHO_THRESHOLD,   "pass": bool(g3_pass), "rho_by_sleeve": rho_vec},
            "G5_v2_dd_atten":      {"wins":  int(g5_wins), "threshold": 2, "pass": bool(g5_pass), "detail": g5_details},
            "G6_portfolio_sharpe": g6,
        },
        "summary_stats": {
            "weekly_mean_net":   float(pv_net.mean()),
            "weekly_std_net":    float(pv_net.std()),
            "annualized_vol":    float(pv_net.std() * math.sqrt(52)),
            "annual_tc_drag":    float(pv["tc"].sum() / (len(pv) / 52.0)),
            "max_dd_overall":    peak_to_trough_dd(pv_net),
        },
    }
    out_path = REPO_ROOT / "data" / "portfolio_replay" / f"path_ag_verdict_{today.isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Verdict saved: {out_path}")


if __name__ == "__main__":
    main()
