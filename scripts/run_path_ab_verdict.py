"""
scripts/run_path_ab_verdict.py — Path AB TLT/GLD Crisis Hedge Sleeve verdict.

Uses Gate Framework v2 per docs/spec_gate_framework_v2_2026-05-14.md:
  G1 PRIMARY  Sharpe (net 4bp TC x 2 sides, ann.) >= 0.30
  G2          Newey-West HAC t-stat (lag-8) > 1.96
  G3          max |rho| vs K1/D-PEAD/PATH_N/CTA <= 0.25
  G5-v2       Strategy peak-to-trough DD <= SPY peak-to-trough DD x 1.0
              in >= 2 of 3 crisis windows

Decision: 4/4 PASS - 3/4 MARGINAL - <=2/4 FAIL
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
from scripts.run_path_aa_verdict import (
    peak_to_trough_dd, crisis_window_dd, load_spy_weekly,
)


def main() -> None:
    print("=== Path AB TLT/GLD Crisis Hedge Sleeve verdict (v2 framework) ===")
    print()

    pv_path = REPO_ROOT / "data" / "portfolio_replay" / "v1_path_ab_tlt_gld_crisis_hedge_weekly.parquet"
    pv = pd.read_parquet(pv_path)
    pv.index = pd.to_datetime(pv.index)
    pv_net = pv["net"].dropna()
    print(f"Path AB net returns: n={len(pv_net)} weeks")

    existing_path = REPO_ROOT / "data" / "portfolio_replay" / "v1_per_strategy_returns_weekly.parquet"
    existing = pd.read_parquet(existing_path).astype("float64").fillna(0.0)
    existing.index = pd.to_datetime(existing.index)

    spy_weekly = load_spy_weekly()
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
        print(f"               rho(AB, {col:<12}) = {r:+.4f}")

    # G5-v2: relative DD attenuation
    print()
    print("G5-v2        Relative DD attenuation vs SPY in crisis windows:")
    g5_wins = 0
    g5_details = {}
    for label, (start, end) in CRISIS_WINDOWS.items():
        spy_dd = crisis_window_dd(spy_weekly, start, end)
        ab_dd  = crisis_window_dd(pv_net, start, end)
        win = ab_dd >= spy_dd   # ab_dd less negative = better attenuation
        if win:
            g5_wins += 1
        g5_details[label] = {"spy_dd": spy_dd, "ab_dd": ab_dd, "win": win}
        marker = "WIN " if win else "LOSE"
        print(f"               [{marker}] {label}: AB peak-to-trough DD = {ab_dd*100:+.2f}%, "
              f"SPY DD = {spy_dd*100:+.2f}%")
    g5_pass = g5_wins >= 2
    print(f"             G5-v2 result: {g5_wins} of 3 crisis windows attenuated   "
          f">= 2   -> {'PASS' if g5_pass else 'FAIL'}")
    print()

    n_pass = sum([g1_pass, g2_pass, g3_pass, g5_pass])
    verdict = "PASS" if n_pass == 4 else "MARGINAL" if n_pass == 3 else "FAIL"
    print(f"=================================================================")
    print(f"  Path AB: {n_pass}/4 gates PASS  ->  VERDICT: {verdict}")
    print(f"=================================================================")

    today = datetime.date.today()
    payload = {
        "spec_id":  75,
        "spec_hash": "d79261179add3fcfd87f3156d6044eafd0c13810",
        "spec_name": "TLT/GLD Crisis Hedge Sleeve (Brunnermeier-Pedersen 2009 + Baur-Lucey 2010)",
        "framework": "v2 (commit 5019d6d)",
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
            "G5v2_relative_dd_attenuation": {"wins": g5_wins, "threshold": 2, "pass": g5_pass,
                                              "detail": g5_details},
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
    out_path = REPO_ROOT / "data" / "portfolio_replay" / f"path_ab_verdict_{today.isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Verdict saved: {out_path}")


if __name__ == "__main__":
    main()
