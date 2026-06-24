"""scripts/audit_rmw_overlay_weight_sweep.py — Stage 3 follow-up.

Stage 3 of FEGD integration roadmap: take the 5% RMW overlay result
(commit e760740a, ΔSh +0.024 t=+1.72 p=0.04, NOISE marginal) and scan
across weights 5/10/15/20/25/30% to see whether t-stat or ΔSharpe
crosses the IMPROVEMENT threshold.

IMPROVEMENT requires ALL THREE simultaneously:
  ΔSharpe > +0.15  AND  t-stat >= +1.96  AND  p < 0.05

Linear extrapolation predicts t crosses ~20% but ΔSharpe doesn't cross
until ~30%+ overlay. The actual scaling may show non-linearity
(regime concentration, vol-target rescaling effects) — let the data say.

Also runs 3-seed replication at the most promising weight to verify
the marginal positive signal is seed-stable, not bootstrap noise.

Run:
    python scripts/audit_rmw_overlay_weight_sweep.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from engine.portfolio.combined_book import build_combined_book
from engine.research.enhance import dispatch_enhance_hypothesis

KF_DAILY = _REPO_ROOT / "data" / "cache" / "ken_french_ff5_mom_daily.parquet"
OUT_DIR  = _REPO_ROOT / "data" / "research_store" / "audit" / "rmw_weight_sweep_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG_PATH = OUT_DIR / "enhance_verdict_log.jsonl"


def _ann_sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 12 or s.std(ddof=1) <= 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * math.sqrt(12))


def _vol_target(s: pd.Series, target_ann: float = 0.10) -> pd.Series:
    s = s.dropna()
    v = s.std(ddof=1) * math.sqrt(12)
    if not math.isfinite(v) or v <= 0:
        return s
    return s * (target_ann / v)


def main():
    print("Building combined book + KF factors...")
    book = build_combined_book(
        crisis_risk_weight=0.05, mom_hedge_risk_weight=0.02,
        regime_conditional=True, book_vol_target=0.10,
    ).dropna()
    book.index = pd.to_datetime(book.index).to_period("M").to_timestamp("M")

    kf = pd.read_parquet(KF_DAILY)
    kf_monthly = ((1.0 + kf).resample("ME").prod() - 1.0).dropna(how="all")
    kf_monthly.index = pd.to_datetime(kf_monthly.index).to_period("M").to_timestamp("M")
    rmw_vt = _vol_target(kf_monthly["RMW"].dropna(), 0.10)

    common = book.index.intersection(rmw_vt.index)
    baseline = _vol_target(book.loc[common], 0.10)
    print(f"  overlap: {common.min().date()} → {common.max().date()} "
          f"({len(common)} mo)  baseline Sharpe={_ann_sharpe(baseline):+.3f}")
    print()

    # WEIGHT SCAN
    print("=" * 100)
    print("WEIGHT SCAN — RMW overlay at 5 / 10 / 15 / 20 / 25 / 30%, seed=42")
    print("=" * 100)
    print(f"{'weight':<8}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}"
          f"{'CI low':>11}{'CI high':>11}{'IMPRV?':>10}")
    print("-" * 100)
    weight_scan = []
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        variant = (1 - w) * baseline + w * rmw_vt.loc[common]
        variant = _vol_target(variant.dropna(), 0.10)
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"rmw_overlay_w{int(w*100):02d}_seed42",
            sleeve_id        = "combined_book",
            variant_returns  = variant,
            baseline_returns = baseline,
            n_iterations     = 2000, block_size=6, seed=42,
            log_path         = ENHANCE_LOG_PATH,
        )
        b = r.bootstrap_result or {}
        ds = b.get("sharpe_diff_observed"); t = b.get("sharpe_diff_t_stat")
        p  = b.get("sharpe_diff_p_value");  lo= b.get("sharpe_diff_ci_lo")
        hi = b.get("sharpe_diff_ci_hi")
        v_label = r.refusal_reason or r.verdict
        # Check the 3 IMPROVEMENT criteria individually
        all3 = ("✓✓✓" if (ds and ds > 0.15 and t and t >= 1.96 and p and p < 0.05)
                else f"{'D' if ds and ds > 0.15 else '-'}"
                     f"{'T' if t  and t  >= 1.96 else '-'}"
                     f"{'P' if p  and p  < 0.05 else '-'}")
        ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
        t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
        p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
        lo_s = f"{lo:>+10.4f}" if lo is not None else "     n/a"
        hi_s = f"{hi:>+10.4f}" if hi is not None else "     n/a"
        print(f"{int(w*100)}%      {v_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}{all3:>10}")
        weight_scan.append({
            "weight": w, "verdict": r.verdict, "bootstrap": b,
            "improv_criteria_met": all3,
        })

    # SEED REPLICATION (at most promising weight from scan)
    best = max(weight_scan, key=lambda x: x["bootstrap"].get("sharpe_diff_t_stat") or 0)
    best_weight = best["weight"]
    print()
    print("=" * 100)
    print(f"SEED REPLICATION at w={int(best_weight*100)}% (highest t-stat from weight scan)")
    print("=" * 100)
    print(f"{'seed':<8}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}")
    print("-" * 60)
    seed_replication = []
    for seed in [42, 123, 456, 789, 2026]:
        variant = (1 - best_weight) * baseline + best_weight * rmw_vt.loc[common]
        variant = _vol_target(variant.dropna(), 0.10)
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"rmw_overlay_w{int(best_weight*100):02d}_seed{seed}",
            sleeve_id        = "combined_book",
            variant_returns  = variant,
            baseline_returns = baseline,
            n_iterations     = 2000, block_size=6, seed=seed,
            log_path         = ENHANCE_LOG_PATH,
        )
        b = r.bootstrap_result or {}
        ds = b.get("sharpe_diff_observed"); t = b.get("sharpe_diff_t_stat")
        p  = b.get("sharpe_diff_p_value")
        v_label = r.refusal_reason or r.verdict
        ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
        t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
        p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
        print(f"{seed:<8}{v_label:<14}{ds_s}{t_s}{p_s}")
        seed_replication.append({
            "seed": seed, "verdict": r.verdict, "bootstrap": b,
        })

    # Persist
    out_json = OUT_DIR / "rmw_weight_sweep_results.json"
    out_json.write_text(json.dumps({
        "baseline":           "build_combined_book(5-sleeve, vt 10%)",
        "factor":             "Ken French RMW (vt 10%)",
        "method":             "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months":    int(len(baseline)),
        "weight_scan":        weight_scan,
        "best_weight_for_replication": best_weight,
        "seed_replication":   seed_replication,
        "improvement_thresholds": {
            "delta_sharpe": 0.15, "t_stat": 1.96, "p_value": 0.05,
        },
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
