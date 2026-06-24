"""
scripts/run_v3_retro_analysis.py — Informational retro under v3 framework.

Per docs/spec_gate_framework_v3_2026-05-15.md §三 + §六.

Computes for each prior candidate (V/W/X/Z/AA/AB) all v3 gates per declared
category. Output is INFORMATIONAL ONLY — does NOT modify locked v1/v2
verdicts published in capability evidence MDs.

Gates evaluated:
  Alpha class candidates (V/W/X/Z/AA):
    G1 Sharpe >= 0.30
    G2 NW-t > 1.96
    G3 max |rho| vs K1/D-PEAD/PATH_N/CTA <= 0.25
    G5-v2 DD attenuation >= 2/3 crisis
    G6 NEW: blended portfolio Sharpe improvement >= 2% at any w in {2,5,10,15,20}%

  Insurance class candidates (AB):
    G1' Sharpe >= -0.30
    G3 max |rho| <= 0.25
    G5-insurance DD attenuation >= 2/3 crisis
    G7 NEW: blended portfolio max DD reduction >= 3pp at any w in {5,10,15}%

Decision rules:
  alpha:     5/5 PASS - 4/5 MARGINAL - <=3/5 FAIL
  insurance: 4/4 PASS - 3/4 MARGINAL - <=2/4 FAIL
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


# v3 LOCKED thresholds (per docs/spec_gate_framework_v3_2026-05-15.md §2.2 + §2.3)
G6_ALPHA_SHARPE_IMPROVEMENT_THRESHOLD: float = 0.02  # >= 2% portfolio Sharpe improvement
G7_INSURANCE_DD_REDUCTION_THRESHOLD:   float = 0.03  # >= 3pp portfolio max DD reduction
G1_PRIME_INSURANCE_SHARPE_FLOOR:       float = -0.30 # tolerable insurance carry

ALPHA_WEIGHT_SWEEP:     list[float] = [0.02, 0.05, 0.10, 0.15, 0.20]
INSURANCE_WEIGHT_SWEEP: list[float] = [0.05, 0.10, 0.15]

# Sleeve weights per PAPER_TRADE_SLEEVE_ALLOCATION mapped to strategy level
SLEEVE_WEIGHTS = {
    "K1_BAB":    0.36,  # etf_l1
    "D_PEAD":    0.27,  # ss_sp500 split half
    "PATH_N":    0.27,  # ss_sp500 split half
    "CTA_PQTIX": 0.10,  # cta_defensive
}


def compute_current_4_sleeve_returns(per_strategy: pd.DataFrame) -> pd.Series:
    """Compose current 4-sleeve portfolio weekly returns at production weights."""
    aligned = per_strategy.fillna(0.0)
    weighted = pd.Series(0.0, index=aligned.index)
    for col, w in SLEEVE_WEIGHTS.items():
        if col in aligned.columns:
            weighted = weighted + w * aligned[col].astype(float)
    return weighted.rename("current_4_sleeve")


def evaluate_g6_alpha(candidate_net: pd.Series, baseline_4sleeve: pd.Series) -> dict:
    """G6: blended portfolio Sharpe improvement at any w in {2,5,10,15,20}%."""
    common = candidate_net.index.intersection(baseline_4sleeve.index)
    cand = candidate_net.loc[common]
    base = baseline_4sleeve.loc[common]
    base_sharpe = annualized_sharpe(base)

    best_w = None
    best_blend_sharpe = -1e9
    detail = {}
    for w in ALPHA_WEIGHT_SWEEP:
        blend = (1.0 - w) * base + w * cand
        blend_sharpe = annualized_sharpe(blend)
        detail[w] = float(blend_sharpe)
        if blend_sharpe > best_blend_sharpe:
            best_blend_sharpe = blend_sharpe
            best_w = w

    improvement = (best_blend_sharpe - base_sharpe) / abs(base_sharpe) if abs(base_sharpe) > 1e-6 else 0.0
    g6_pass = improvement >= G6_ALPHA_SHARPE_IMPROVEMENT_THRESHOLD
    return {
        "g6_pass": g6_pass,
        "base_sharpe": float(base_sharpe),
        "best_blend_sharpe": float(best_blend_sharpe),
        "best_w": float(best_w) if best_w is not None else None,
        "sharpe_improvement_pct": float(improvement),
        "sharpe_by_w": detail,
    }


def evaluate_g7_insurance(candidate_net: pd.Series, baseline_4sleeve: pd.Series) -> dict:
    """G7: blended portfolio max DD reduction at any w in {5,10,15}%."""
    common = candidate_net.index.intersection(baseline_4sleeve.index)
    cand = candidate_net.loc[common]
    base = baseline_4sleeve.loc[common]
    base_max_dd = peak_to_trough_dd(base)  # negative number

    best_w = None
    best_blend_max_dd = -1e9
    detail = {}
    for w in INSURANCE_WEIGHT_SWEEP:
        blend = (1.0 - w) * base + w * cand
        blend_max_dd = peak_to_trough_dd(blend)  # negative number
        detail[w] = float(blend_max_dd)
        if blend_max_dd > best_blend_max_dd:  # less negative is better
            best_blend_max_dd = blend_max_dd
            best_w = w

    dd_reduction_pp = abs(base_max_dd) - abs(best_blend_max_dd)
    g7_pass = dd_reduction_pp >= G7_INSURANCE_DD_REDUCTION_THRESHOLD
    return {
        "g7_pass": g7_pass,
        "base_max_dd": float(base_max_dd),
        "best_blend_max_dd": float(best_blend_max_dd),
        "best_w": float(best_w) if best_w is not None else None,
        "dd_reduction_pp": float(dd_reduction_pp),
        "dd_by_w": detail,
    }


def evaluate_alpha_class(name: str, candidate_path: str,
                         baseline: pd.Series, existing: pd.DataFrame,
                         spy_weekly: pd.Series) -> dict:
    """Apply v3 alpha-class 5 gates."""
    pv = pd.read_parquet(candidate_path)
    pv.index = pd.to_datetime(pv.index)
    pv_net = pv["net"].dropna()

    # G1
    sharpe_net = annualized_sharpe(pv_net)
    g1_pass = sharpe_net >= G1_SHARPE_THRESHOLD

    # G2
    excess = (pv_net - WEEKLY_RFR).to_numpy()
    nw_t = newey_west_t(excess, lag=NW_LAG)
    g2_pass = (not math.isnan(nw_t)) and (nw_t > G2_T_THRESHOLD)

    # G3
    common_idx = pv_net.index.intersection(existing.index)
    pv_a = pv_net.loc[common_idx]
    ex_a = existing.loc[common_idx]
    rho_vec = {col: float(pv_a.corr(ex_a[col])) for col in ex_a.columns}
    max_abs_rho = max(abs(v) for v in rho_vec.values())
    g3_pass = max_abs_rho <= G3_RHO_THRESHOLD

    # G5-v2
    g5_wins = 0
    g5_details = {}
    for label, (start, end) in CRISIS_WINDOWS.items():
        spy_dd = crisis_window_dd(spy_weekly, start, end)
        cand_dd = crisis_window_dd(pv_net, start, end)
        win = cand_dd >= spy_dd
        if win:
            g5_wins += 1
        g5_details[label] = {"spy_dd": spy_dd, "cand_dd": cand_dd, "win": win}
    g5_pass = g5_wins >= 2

    # G6 NEW
    g6 = evaluate_g6_alpha(pv_net, baseline)

    n_pass = sum([g1_pass, g2_pass, g3_pass, g5_pass, g6["g6_pass"]])
    verdict = "PASS" if n_pass == 5 else "MARGINAL" if n_pass == 4 else "FAIL"

    return {
        "name":     name,
        "category": "alpha",
        "verdict":  verdict,
        "n_pass":   int(n_pass),
        "gates": {
            "G1_sharpe":       {"value": float(sharpe_net), "threshold": G1_SHARPE_THRESHOLD, "pass": bool(g1_pass)},
            "G2_nw_t":         {"value": float(nw_t),       "threshold": G2_T_THRESHOLD,     "pass": bool(g2_pass)},
            "G3_max_abs_rho":  {"value": float(max_abs_rho),"threshold": G3_RHO_THRESHOLD,   "pass": bool(g3_pass), "rho_by_sleeve": rho_vec},
            "G5_v2_dd_atten":  {"wins":  int(g5_wins),      "threshold": 2,                  "pass": bool(g5_pass), "detail": g5_details},
            "G6_portfolio_sharpe": g6,
        },
    }


def evaluate_insurance_class(name: str, candidate_path: str,
                              baseline: pd.Series, existing: pd.DataFrame,
                              spy_weekly: pd.Series) -> dict:
    """Apply v3 insurance-class 4 gates."""
    pv = pd.read_parquet(candidate_path)
    pv.index = pd.to_datetime(pv.index)
    pv_net = pv["net"].dropna()

    # G1'
    sharpe_net = annualized_sharpe(pv_net)
    g1_prime_pass = sharpe_net >= G1_PRIME_INSURANCE_SHARPE_FLOOR

    # G3
    common_idx = pv_net.index.intersection(existing.index)
    pv_a = pv_net.loc[common_idx]
    ex_a = existing.loc[common_idx]
    rho_vec = {col: float(pv_a.corr(ex_a[col])) for col in ex_a.columns}
    max_abs_rho = max(abs(v) for v in rho_vec.values())
    g3_pass = max_abs_rho <= G3_RHO_THRESHOLD

    # G5-insurance (same threshold as alpha v2: >= 2/3)
    g5_wins = 0
    g5_details = {}
    for label, (start, end) in CRISIS_WINDOWS.items():
        spy_dd = crisis_window_dd(spy_weekly, start, end)
        cand_dd = crisis_window_dd(pv_net, start, end)
        win = cand_dd >= spy_dd
        if win:
            g5_wins += 1
        g5_details[label] = {"spy_dd": spy_dd, "cand_dd": cand_dd, "win": win}
    g5_pass = g5_wins >= 2

    # G7 NEW
    g7 = evaluate_g7_insurance(pv_net, baseline)

    n_pass = sum([g1_prime_pass, g3_pass, g5_pass, g7["g7_pass"]])
    verdict = "PASS" if n_pass == 4 else "MARGINAL" if n_pass == 3 else "FAIL"

    return {
        "name":     name,
        "category": "insurance",
        "verdict":  verdict,
        "n_pass":   int(n_pass),
        "gates": {
            "G1_prime_sharpe": {"value": float(sharpe_net),  "threshold": G1_PRIME_INSURANCE_SHARPE_FLOOR, "pass": bool(g1_prime_pass)},
            "G3_max_abs_rho":  {"value": float(max_abs_rho), "threshold": G3_RHO_THRESHOLD,               "pass": bool(g3_pass), "rho_by_sleeve": rho_vec},
            "G5_insurance_dd_atten": {"wins": int(g5_wins),  "threshold": 2,                              "pass": bool(g5_pass), "detail": g5_details},
            "G7_portfolio_dd_reduction": g7,
        },
    }


CANDIDATE_CONFIG = [
    ("V",  "alpha",     "data/portfolio_replay/v1_path_v_csm_weekly.parquet"),
    ("W",  "alpha",     "data/portfolio_replay/v1_path_w_weekly.parquet"),
    ("X",  "alpha",     "data/portfolio_replay/v1_path_x_weekly.parquet"),
    ("Z",  "alpha",     "data/portfolio_replay/v1_path_z_str_weekly.parquet"),
    ("AA", "alpha",     "data/portfolio_replay/v1_path_aa_sector_mom_ls_weekly.parquet"),
    ("AB", "insurance", "data/portfolio_replay/v1_path_ab_tlt_gld_crisis_hedge_weekly.parquet"),
]


def main() -> None:
    print("=" * 78)
    print("v3 INFORMATIONAL RETRO — gate framework v3 (id=76 hash 7400b360)")
    print("=" * 78)
    print("This is INFORMATIONAL ONLY. v1/v2 verdicts published in capability")
    print("evidence MDs STAND unchanged. Promotion requires NEW spec under v3.")
    print()

    # Load baseline
    existing = pd.read_parquet(
        REPO_ROOT / "data" / "portfolio_replay" / "v1_per_strategy_returns_weekly.parquet"
    ).astype("float64").fillna(0.0)
    existing.index = pd.to_datetime(existing.index)
    baseline = compute_current_4_sleeve_returns(existing)
    print(f"Current 4-sleeve baseline: n={len(baseline)} weeks, "
          f"Sharpe = {annualized_sharpe(baseline):+.4f}, "
          f"Max DD = {peak_to_trough_dd(baseline)*100:+.2f}%")
    print(f"Weights: K1={SLEEVE_WEIGHTS['K1_BAB']:.0%}, "
          f"D-PEAD={SLEEVE_WEIGHTS['D_PEAD']:.0%}, "
          f"Path N={SLEEVE_WEIGHTS['PATH_N']:.0%}, "
          f"CTA-PQTIX={SLEEVE_WEIGHTS['CTA_PQTIX']:.0%}")
    print()

    # Load SPY
    spy_weekly = load_spy_weekly()
    print(f"SPY benchmark loaded: n={len(spy_weekly)} weeks")
    print()

    # Run retro on each candidate
    results = []
    for name, category, path in CANDIDATE_CONFIG:
        print("-" * 78)
        print(f"Candidate: Path {name}  ({category} class)")
        print("-" * 78)
        if category == "alpha":
            res = evaluate_alpha_class(name, str(REPO_ROOT / path), baseline, existing, spy_weekly)
            g = res["gates"]
            print(f"  G1 Sharpe        = {g['G1_sharpe']['value']:+.4f}   (>= {g['G1_sharpe']['threshold']})    -> {'PASS' if g['G1_sharpe']['pass'] else 'FAIL'}")
            print(f"  G2 NW-t          = {g['G2_nw_t']['value']:+.4f}   (>  {g['G2_nw_t']['threshold']})   -> {'PASS' if g['G2_nw_t']['pass'] else 'FAIL'}")
            print(f"  G3 max |rho|     = {g['G3_max_abs_rho']['value']:+.4f}   (<= {g['G3_max_abs_rho']['threshold']})   -> {'PASS' if g['G3_max_abs_rho']['pass'] else 'FAIL'}")
            print(f"  G5-v2 DD atten   = {g['G5_v2_dd_atten']['wins']}/3        (>= 2/3)         -> {'PASS' if g['G5_v2_dd_atten']['pass'] else 'FAIL'}")
            g6 = g["G6_portfolio_sharpe"]
            print(f"  G6 portfolio Sharpe lift = {g6['sharpe_improvement_pct']*100:+.2f}%   "
                  f"(>= +2.00%)   -> {'PASS' if g6['g6_pass'] else 'FAIL'}")
            print(f"     base Sharpe {g6['base_sharpe']:+.4f}  ->  best blend Sharpe {g6['best_blend_sharpe']:+.4f} at w={g6['best_w']:.0%}")
        else:
            res = evaluate_insurance_class(name, str(REPO_ROOT / path), baseline, existing, spy_weekly)
            g = res["gates"]
            print(f"  G1' Sharpe       = {g['G1_prime_sharpe']['value']:+.4f}   (>= {g['G1_prime_sharpe']['threshold']})   -> {'PASS' if g['G1_prime_sharpe']['pass'] else 'FAIL'}")
            print(f"  G3 max |rho|     = {g['G3_max_abs_rho']['value']:+.4f}   (<= {g['G3_max_abs_rho']['threshold']})   -> {'PASS' if g['G3_max_abs_rho']['pass'] else 'FAIL'}")
            print(f"  G5-ins DD atten  = {g['G5_insurance_dd_atten']['wins']}/3        (>= 2/3)         -> {'PASS' if g['G5_insurance_dd_atten']['pass'] else 'FAIL'}")
            g7 = g["G7_portfolio_dd_reduction"]
            print(f"  G7 portfolio max DD reduction = {g7['dd_reduction_pp']*100:+.2f}pp   "
                  f"(>= +3.00pp)   -> {'PASS' if g7['g7_pass'] else 'FAIL'}")
            print(f"     base max DD {g7['base_max_dd']*100:+.2f}%  ->  best blend max DD {g7['best_blend_max_dd']*100:+.2f}% at w={g7['best_w']:.0%}")

        print()
        print(f"  v3 RETRO VERDICT: {res['n_pass']}/{'5' if category=='alpha' else '4'}  ->  {res['verdict']}")
        print()
        results.append(res)

    # Summary
    print("=" * 78)
    print("v3 RETRO SUMMARY (informational only)")
    print("=" * 78)
    print(f"{'Path':<6} {'Category':<10} {'v3 Verdict':<10} {'v1/v2 Verdict':<15}")
    print("-" * 78)
    v1v2_verdicts = {
        "V":  "FAIL (v1 3/5)",
        "W":  "FAIL (v1 1/5)",
        "X":  "FAIL (v1 1/5)",
        "Z":  "FAIL (v1 2/5)",
        "AA": "FAIL (v2 1/4)",
        "AB": "FAIL (v2 2/4)",
    }
    for r in results:
        print(f"{r['name']:<6} {r['category']:<10} "
              f"{r['verdict']+' ('+str(r['n_pass'])+'/'+('5' if r['category']=='alpha' else '4')+')':<15} "
              f"{v1v2_verdicts.get(r['name'], '?'):<15}")
    print()

    today = datetime.date.today()
    out_path = REPO_ROOT / "data" / "portfolio_replay" / f"v3_retro_analysis_{today.isoformat()}.json"
    out_path.write_text(json.dumps({
        "framework":      "v3 (spec id=76 hash 7400b3607337d1289bddf5468e0f401d60719cb0)",
        "run_date":       today.isoformat(),
        "is_informational": True,
        "anti_hark_note": "v3 doctrine hash-stamped 2026-05-15 BEFORE this retro. Prior v1/v2 verdicts STAND unchanged.",
        "baseline_4sleeve": {
            "weights":    SLEEVE_WEIGHTS,
            "n_weeks":    int(len(baseline)),
            "sharpe":     float(annualized_sharpe(baseline)),
            "max_dd":     float(peak_to_trough_dd(baseline)),
        },
        "retro_results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"v3 retro saved: {out_path}")


if __name__ == "__main__":
    main()
