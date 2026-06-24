"""scripts/audit_bab_book_overlay_vs_deployed.py — first FEGD-Stage-2-cleared enhance.

The FEGD pipeline (Stage 1 + Stage 2) prioritized this test:
  Stage 1: LOW_VOL family auto-boosted in burndown_ranker (FEGD found
    cross_asset_tsmom and book-level have BAB gaps).
  Stage 2 pre-check: PROCEED at t=+1.648 (just below 1.65 threshold —
    combined book has no significant BAB loading, true gap).

Tests AQR BAB factor (Frazzini-Pedersen 2014, USA 1930-2026) as a
book-level overlay at 5/10/15/20% weights, paired Politis-Romano 1994
block bootstrap (B=2000, block=6mo) vs the deployed combined book.

This is the first candidate this session that has cleared BOTH:
  - Ranker prioritization (LOW_VOL boosted by FEGD)
  - Pre-enhance filter (PROCEED, not WARN/SKIP)

Run:
    python scripts/audit_bab_book_overlay_vs_deployed.py
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
from engine.research.factor_exposure_gap_detector import (
    build_canonical_factor_matrix, pre_enhance_check,
)

AQR_BAB = _REPO_ROOT / "data" / "cache" / "aqr_bab_usa_monthly.parquet"
OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "bab_book_overlay_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG = OUT_DIR / "enhance_verdict_log.jsonl"


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
    print("Building deployed combined book + AQR BAB factor...")
    book = build_combined_book(
        crisis_risk_weight=0.05, mom_hedge_risk_weight=0.02,
        regime_conditional=True, book_vol_target=0.10,
    ).dropna()
    book.index = pd.to_datetime(book.index).to_period("M").to_timestamp("M")

    bab_df = pd.read_parquet(AQR_BAB)
    bab = (bab_df["BAB"] if "BAB" in bab_df.columns else bab_df.iloc[:, 0]).dropna()
    bab.index = pd.to_datetime(bab.index).to_period("M").to_timestamp("M")
    bab_vt = _vol_target(bab, 0.10)

    common = book.index.intersection(bab_vt.index)
    baseline = _vol_target(book.loc[common], 0.10)
    print(f"  book:    n={len(book)}  range={book.index.min().date()} → {book.index.max().date()}")
    print(f"  AQR BAB: n={len(bab)}   range={bab.index.min().date()} → {bab.index.max().date()}")
    print(f"  overlap: {common.min().date()} → {common.max().date()} ({len(common)} mo)")
    print(f"  baseline Sharpe (vt 10%) on overlap: {_ann_sharpe(baseline):+.3f}")
    print()

    # Stage 2 pre-check ON RECORD for audit trail
    fm = build_canonical_factor_matrix()
    dec = pre_enhance_check(
        sleeve_id="combined_book",
        candidate_mechanism_family="LOW_VOL",
        sleeve_pnl=book, factor_matrix=fm,
    )
    print("Stage 2 pre-enhance filter:")
    print(f"  recommendation:    {dec.recommendation}")
    print(f"  matched factor:    {dec.matched_gap_factor}")
    print(f"  t-stat:            {dec.factor_t_stat:+.3f}")
    print(f"  reason:            {dec.reason[:120]}")
    print()

    # Sweep weights
    print("Paired enhance test (Politis-Romano 1994, B=2000, block=6mo):")
    print("=" * 105)
    print(f"{'weight':<8}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}"
          f"{'CI low':>11}{'CI high':>11}{'corr':>8}{'IMPRV':>10}")
    print("-" * 105)

    results = []
    for w in [0.05, 0.10, 0.15, 0.20]:
        variant = (1 - w) * baseline + w * bab_vt.loc[common]
        variant = _vol_target(variant.dropna(), 0.10)
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"bab_book_overlay_w{int(w*100):02d}",
            sleeve_id        = "combined_book",
            variant_returns  = variant,
            baseline_returns = baseline,
            n_iterations     = 2000, block_size=6, seed=42,
            log_path         = ENHANCE_LOG,
        )
        b = r.bootstrap_result or {}
        ds = b.get("sharpe_diff_observed"); t = b.get("sharpe_diff_t_stat")
        p  = b.get("sharpe_diff_p_value");  lo= b.get("sharpe_diff_ci_lo")
        hi = b.get("sharpe_diff_ci_hi");    c = b.get("correlation")
        v_label = r.refusal_reason or r.verdict
        criteria = (f"{'D' if ds and ds > 0.15 else '-'}"
                     f"{'T' if t  and t  >= 1.96 else '-'}"
                     f"{'P' if p  and p  < 0.05 else '-'}")
        ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
        t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
        p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
        lo_s = f"{lo:>+10.4f}" if lo is not None else "     n/a"
        hi_s = f"{hi:>+10.4f}" if hi is not None else "     n/a"
        c_s  = f"{c:>+7.3f}"  if c  is not None else "   n/a"
        print(f"{int(w*100)}%     {v_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}{c_s}{criteria:>10}")
        results.append({
            "weight": w, "verdict": r.verdict, "bootstrap": b,
            "improv_criteria_met": criteria,
        })

    out_json = OUT_DIR / "bab_book_overlay_results.json"
    out_json.write_text(json.dumps({
        "subject":            "combined_book",
        "candidate_factor":   "AQR BAB USA (Frazzini-Pedersen 2014)",
        "stage2_decision":    {
            "recommendation": dec.recommendation,
            "t_stat":         dec.factor_t_stat,
            "matched_factor": dec.matched_gap_factor,
        },
        "method":             "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months":    int(len(baseline)),
        "results":            results,
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
