"""
scripts/run_path_ac_verdict.py — Path AC TLT/GLD v3 insurance class verdict.

Extended window 2005-2023, 60/40 SPY/AGG institutional baseline per
docs/spec_path_ac_tlt_gld_extended_v3_v1.md and v3 doctrine
docs/spec_gate_framework_v3_2026-05-15.md §2.3.

4 v3 insurance gates:
  G1'         Sharpe (net 4bp TC, ann.) >= -0.30
  G3          max |rho| vs 60/40 baseline weekly <= 0.25
  G5-insurance Crisis DD attenuation vs baseline >= 3/5 windows
  G7          Blended portfolio max DD reduction >= 3pp at any w in {5,10,15}%

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
    WEEKLY_RFR, G3_RHO_THRESHOLD, NW_LAG,
    annualized_sharpe, newey_west_t,
)
from scripts.run_path_aa_verdict import peak_to_trough_dd, crisis_window_dd

# v3 insurance thresholds (per v3 §2.3 LOCKED)
G1_PRIME_INSURANCE_FLOOR:        float = -0.30
G7_DD_REDUCTION_THRESHOLD:       float = 0.03  # >= 3pp
INSURANCE_WEIGHT_SWEEP:          list[float] = [0.05, 0.10, 0.15]

# Extended-window crisis windows (5 total per Path AC spec §2.7)
EXTENDED_CRISIS_WINDOWS = {
    "2008_GFC":             ("2008-09-15", "2009-03-09"),
    "2011_Euro_USDowngrade":("2011-07-25", "2011-10-04"),
    "2018_Q4":              ("2018-10-01", "2018-12-31"),
    "2020_COVID":           ("2020-02-15", "2020-04-30"),
    "2022_full":            ("2022-01-01", "2022-12-31"),
}
G5_INSURANCE_THRESHOLD_COUNT = 3  # majority of 5


def main() -> None:
    print("=" * 78)
    print("Path AC TLT/GLD Crisis Hedge Sleeve — v3 INSURANCE class verdict")
    print("Extended window 2005-2023 · 60/40 SPY/AGG institutional baseline")
    print("=" * 78)
    print()

    pv_path = REPO_ROOT / "data" / "portfolio_replay" / "v1_path_ac_tlt_gld_extended_weekly.parquet"
    pv = pd.read_parquet(pv_path)
    pv.index = pd.to_datetime(pv.index)
    pv_net = pv["net"].dropna()
    baseline = pv["baseline_60_40"].dropna()
    print(f"Path AC net returns: n={len(pv_net)} weeks")
    print(f"60/40 baseline: n={len(baseline)} weeks")
    print(f"Window: {pv_net.index.min().date()} -> {pv_net.index.max().date()}")
    print()

    # G1': Sharpe >= -0.30
    sharpe_net = annualized_sharpe(pv_net)
    g1_prime_pass = sharpe_net >= G1_PRIME_INSURANCE_FLOOR
    print(f"G1' INSURANCE  Sharpe net ann. = {sharpe_net:+.4f}   (>= {G1_PRIME_INSURANCE_FLOOR})   -> {'PASS' if g1_prime_pass else 'FAIL'}")

    # G3: max |rho| vs 60/40 baseline <= 0.25
    common = pv_net.index.intersection(baseline.index)
    pv_a = pv_net.loc[common]
    bl_a = baseline.loc[common]
    rho_baseline = float(pv_a.corr(bl_a))
    max_abs_rho = abs(rho_baseline)
    g3_pass = max_abs_rho <= G3_RHO_THRESHOLD
    print(f"G3             rho vs 60/40 baseline = {rho_baseline:+.4f}   |rho|=({max_abs_rho:.4f}) <= {G3_RHO_THRESHOLD}   -> {'PASS' if g3_pass else 'FAIL'}")

    # G5-insurance: crisis DD attenuation vs baseline >= 3/5
    print()
    print("G5-insurance   Crisis DD attenuation vs 60/40 baseline:")
    g5_wins = 0
    g5_details = {}
    for label, (start, end) in EXTENDED_CRISIS_WINDOWS.items():
        bl_dd = crisis_window_dd(baseline, start, end)
        ac_dd = crisis_window_dd(pv_net, start, end)
        win = ac_dd >= bl_dd
        if win:
            g5_wins += 1
        g5_details[label] = {"baseline_dd": bl_dd, "ac_dd": ac_dd, "win": win}
        marker = "WIN " if win else "LOSE"
        print(f"                 [{marker}] {label}: AC DD = {ac_dd*100:+.2f}%, baseline DD = {bl_dd*100:+.2f}%")
    g5_pass = g5_wins >= G5_INSURANCE_THRESHOLD_COUNT
    print(f"               G5-insurance result: {g5_wins} of 5 crisis windows attenuated   "
          f">= {G5_INSURANCE_THRESHOLD_COUNT}/5 (majority)   -> {'PASS' if g5_pass else 'FAIL'}")

    # G7: blended portfolio max DD reduction >= 3pp at any w in {5,10,15}%
    print()
    print("G7             Blended portfolio max DD reduction at w in {5,10,15}%:")
    base_max_dd = peak_to_trough_dd(baseline)
    print(f"                 baseline_60_40 max DD = {base_max_dd*100:+.2f}%")
    best_w = None
    best_blend_max_dd = -1e9
    g7_details = {}
    for w in INSURANCE_WEIGHT_SWEEP:
        blend = (1.0 - w) * baseline + w * pv_net
        blend_max_dd = peak_to_trough_dd(blend)
        dd_reduction_pp = abs(base_max_dd) - abs(blend_max_dd)
        g7_details[w] = {"blend_max_dd": float(blend_max_dd), "reduction_pp": float(dd_reduction_pp)}
        print(f"                 w={int(w*100):2d}%: blend max DD = {blend_max_dd*100:+.2f}%, reduction = {dd_reduction_pp*100:+.2f}pp")
        if blend_max_dd > best_blend_max_dd:
            best_blend_max_dd = blend_max_dd
            best_w = w
    dd_reduction_pp = abs(base_max_dd) - abs(best_blend_max_dd)
    g7_pass = dd_reduction_pp >= G7_DD_REDUCTION_THRESHOLD
    print(f"               Best DD reduction = {dd_reduction_pp*100:+.2f}pp at w={int(best_w*100)}%   "
          f">= {G7_DD_REDUCTION_THRESHOLD*100:.0f}pp   -> {'PASS' if g7_pass else 'FAIL'}")
    print()

    n_pass = sum([g1_prime_pass, g3_pass, g5_pass, g7_pass])
    verdict = "PASS" if n_pass == 4 else "MARGINAL" if n_pass == 3 else "FAIL"
    print("=" * 78)
    print(f"  Path AC: {n_pass}/4 v3 insurance gates PASS  ->  VERDICT: {verdict}")
    print("=" * 78)

    today = datetime.date.today()
    payload = {
        "spec_id":  77,
        "spec_hash": "4db40176056a882d0e365d45fea335599bed5182",
        "spec_name": "TLT/GLD Crisis Hedge Sleeve · v3 insurance class · extended window 2005-2023",
        "framework": "v3 (id=76 hash 7400b3607337d1289bddf5468e0f401d60719cb0)",
        "run_date": today.isoformat(),
        "category": "insurance",
        "window": {"start": str(pv_net.index.min().date()),
                   "end":   str(pv_net.index.max().date()),
                   "n_weeks": int(len(pv_net))},
        "baseline": {"composition": "60% SPY + 40% AGG monthly rebalance",
                     "sharpe": float(annualized_sharpe(baseline)),
                     "max_dd": float(base_max_dd)},
        "verdict": verdict,
        "n_pass":  int(n_pass),
        "gates": {
            "G1_prime_sharpe":  {"value": float(sharpe_net), "threshold": G1_PRIME_INSURANCE_FLOOR, "pass": bool(g1_prime_pass)},
            "G3_rho_baseline":  {"value": float(rho_baseline), "max_abs": float(max_abs_rho), "threshold": G3_RHO_THRESHOLD, "pass": bool(g3_pass)},
            "G5_insurance_dd_atten": {"wins": int(g5_wins), "threshold": G5_INSURANCE_THRESHOLD_COUNT, "n_windows": len(EXTENDED_CRISIS_WINDOWS), "pass": bool(g5_pass), "detail": g5_details},
            "G7_portfolio_dd_reduction": {"reduction_pp": float(dd_reduction_pp), "threshold_pp": float(G7_DD_REDUCTION_THRESHOLD), "best_w": float(best_w), "best_blend_max_dd": float(best_blend_max_dd), "pass": bool(g7_pass), "by_w": g7_details},
        },
        "summary_stats": {
            "weekly_mean_net":   float(pv_net.mean()),
            "weekly_std_net":    float(pv_net.std()),
            "annualized_vol":    float(pv_net.std() * math.sqrt(52)),
            "weekly_mean_gross": float(pv["gross"].mean()),
            "annual_tc_drag":    float(pv["tc"].sum() / (len(pv) / 52.0)),
            "max_dd_overall":    float(peak_to_trough_dd(pv_net)),
        },
    }
    out_path = REPO_ROOT / "data" / "portfolio_replay" / f"path_ac_verdict_{today.isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Verdict saved: {out_path}")


if __name__ == "__main__":
    main()
