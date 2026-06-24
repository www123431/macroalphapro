"""scripts/audit_bond_lsc_substrate_vs_book.py — WRDS-style bonds substrate.

Phase 1 Step 4: build a bond curve carry sleeve from cached FRED
constant-maturity Treasury yields, test standalone Sharpe + composite
with CFTC for cross-data-source multi-substrate.

Substrate construction (Cochrane-Piazzesi 2005 / Fama-Bliss 1987):
  Level     = mean(yields across 2Y/5Y/10Y/30Y)
  Slope     = 10Y - 3M (bond carry — capture roll-down)
  Curvature = 2*5Y - 2Y - 10Y

Bond carry strategy: long-duration position when slope > 0, short when
slope < 0. Approximation of duration-adjusted holding period excess return
via lagged yield change (Δy * duration ≈ price return).

Composite test:
  combined_book baseline (5-sleeve regime-conditional, vol-targeted 10%)
  variant = blend with [CFTC 2-substrate composite + bond_lsc_strategy]

If bond Sharpe >= 0.30 standalone AND adds incremental edge to composite,
this is the FIRST cross-data-source 3-substrate composite test.

Run:
    python scripts/audit_bond_lsc_substrate_vs_book.py
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

from engine.research.enhance import dispatch_enhance_hypothesis
from scripts.audits.audit_cftc_hedging_pressure_vs_deployed import (
    _ann_sharpe, _vol_target, _build_ls_sleeve,
)
from scripts.audits.audit_cftc_composite_pressure_plus_money_vs_deployed import (
    build_cftc_signals_monthly, _standardize_cross_sectional,
)

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "bond_lsc_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG = OUT_DIR / "enhance_verdict_log.jsonl"

FRED_YIELDS = _REPO_ROOT / "data" / "cache" / "_fred_cmt_yields.parquet"

# Modified Duration (years) per maturity for excess return approximation
# (duration < maturity due to coupon reinvestment; we use 0.85 × maturity)
DURATION_APPROX = {
    "DGS3MO": 0.25, "DGS2": 1.85, "DGS5": 4.50,
    "DGS10": 8.50, "DGS30": 18.00,
}


def build_bond_lsc_signals_monthly() -> dict[str, pd.Series]:
    """Build bond curve factors + carry-driven strategy series.

    Returns dict:
      "carry":       monthly slope (10Y-3M) — bond carry signal
      "ls_curve":    monthly L/S of long-end vs short-end (slope strategy)
      "10y_excess":  monthly excess return of 10Y over 3M (compounded carry)
    """
    df = pd.read_parquet(FRED_YIELDS).copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how="any")    # require all 5 yields each day

    # Monthly: take last business day of month
    monthly = df.resample("ME").last()

    out: dict[str, pd.Series] = {}
    # 1. Carry signal: 10Y - 3M slope (positive when curve upward sloping)
    out["carry"] = (monthly["DGS10"] - monthly["DGS3MO"]).rename("bond_carry_slope")

    # 2. Bond excess return approximation:
    # Long 10Y, short 3M → realized monthly excess return ≈
    #   (slope at t-1) / 12  -  duration_10y × Δyield_10y_t
    # In yield units; we want fractional returns.
    # Approximation: bond excess return ≈ (slope/12) - duration*(Δy)
    # where Δy is in yield basis points / 100
    slope_lag = out["carry"].shift(1) / 100   # convert % to decimal
    dy_10y = monthly["DGS10"].diff() / 100
    duration_10y = DURATION_APPROX["DGS10"]
    excess_ret = slope_lag / 12 - duration_10y * dy_10y
    out["10y_excess"] = excess_ret.rename("bond_10y_excess_ret").dropna()

    # 3. L/S strategy: long 10Y vs short 3M, vary EXPOSURE by signal
    # (a "carry timing" strategy: when slope is high, go long; low/negative, neutral)
    signal_z = (out["carry"] - out["carry"].rolling(36).mean()) / out["carry"].rolling(36).std()
    signal_z = signal_z.clip(-2, 2)   # cap exposure
    # Strategy return = signal × next-month excess return
    out["ls_curve"] = (signal_z.shift(1) * out["10y_excess"]).rename("bond_lsc_strategy")

    return out


def main():
    print("Step 1 — Build bond LSC factors from FRED yields...")
    signals = build_bond_lsc_signals_monthly()

    for nm, s in signals.items():
        s_clean = s.dropna()
        sh = _ann_sharpe(s_clean) if s_clean.std() > 0 else float("nan")
        print(f"  {nm}: n={len(s_clean)} range={s_clean.index.min().date()} → "
              f"{s_clean.index.max().date()} Sharpe={sh:+.3f}")
    print()

    # Bond carry strategy standalone Sharpe — the critical metric
    bond_strat = signals["ls_curve"].dropna()
    bond_strat.index = pd.to_datetime(bond_strat.index).to_period("M").to_timestamp("M")
    bond_sharpe = _ann_sharpe(bond_strat)
    print(f"  Bond LSC carry strategy standalone Sharpe: {bond_sharpe:+.3f}")
    print(f"  (Substrate strength filter: needs ≥ 0.30 to compose constructively)")
    print()

    print("Step 2 — Load CFTC 2-substrate composite + commodity baseline...")
    cftc_panels = build_cftc_signals_monthly()
    pm_z = _standardize_cross_sectional(cftc_panels["prod_merc"])
    mm_z = _standardize_cross_sectional(cftc_panels["m_money"])
    cftc_panel = 0.3 * pm_z + 0.7 * mm_z   # 30/70 from prior breakthrough

    from engine.validation.commodity_carry import build_carry_and_returns, build_carry_sleeve
    _, rwide = build_carry_and_returns()
    rwide.index = pd.to_datetime(rwide.index).to_period("M").to_timestamp("M")
    cftc_ls = _build_ls_sleeve(cftc_panel, rwide)
    cftc_ls.index = pd.to_datetime(cftc_ls.index).to_period("M").to_timestamp("M")

    deployed_ls, _, _ = build_carry_sleeve()
    deployed_ls.index = pd.to_datetime(deployed_ls.index).to_period("M").to_timestamp("M")

    # Combined book — bonds is cross-asset so test at book level too
    from engine.portfolio.combined_book import build_combined_book
    book = build_combined_book(
        crisis_risk_weight=0.05, mom_hedge_risk_weight=0.02,
        regime_conditional=True, book_vol_target=0.10,
    ).dropna()
    book.index = pd.to_datetime(book.index).to_period("M").to_timestamp("M")

    print(f"  CFTC composite Sharpe: {_ann_sharpe(cftc_ls):+.3f}")
    print(f"  Bond LSC strategy Sharpe: {bond_sharpe:+.3f}")
    print(f"  Deployed cmdty carry Sharpe: {_ann_sharpe(deployed_ls):+.3f}")
    print(f"  Combined book Sharpe: {_ann_sharpe(book):+.3f}")
    print()

    # Cross-data-source correlations
    common_cb = cftc_ls.index.intersection(bond_strat.index)
    if len(common_cb) > 24:
        c_corr = cftc_ls.loc[common_cb].corr(bond_strat.loc[common_cb])
        print(f"  corr(CFTC, bond_LSC) in overlap = {c_corr:+.3f} (cross-data-source orthogonality)")
    print()

    # Test 1: bond_strat alone as overlay on combined book
    print("=" * 105)
    print("Test 1 — Bond LSC alone as overlay on COMBINED BOOK")
    print("=" * 105)
    common = book.index.intersection(bond_strat.index)
    if len(common) < 24:
        print(f"Too little overlap: {len(common)} mo. Skipping book-level test.")
    else:
        b_vt = _vol_target(book.loc[common], 0.10)
        s_vt = _vol_target(bond_strat.loc[common], 0.10)
        print(f"  overlap: {common.min().date()} → {common.max().date()} ({len(common)} mo)")
        print(f"{'weight':<8}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}{'CI low':>11}{'CI high':>11}")
        print("-" * 70)
        for w in [0.10, 0.20, 0.30, 0.40, 0.50]:
            variant = (1 - w) * b_vt + w * s_vt
            variant = _vol_target(variant.dropna(), 0.10)
            r = dispatch_enhance_hypothesis(
                hypothesis_id    = f"bond_lsc_book_w{int(w*100):02d}",
                sleeve_id        = "combined_book",
                variant_returns  = variant,
                baseline_returns = b_vt,
                n_iterations     = 2000, block_size=6, seed=42,
                log_path         = ENHANCE_LOG,
            )
            b = r.bootstrap_result or {}
            ds, t, p = b.get("sharpe_diff_observed"), b.get("sharpe_diff_t_stat"), b.get("sharpe_diff_p_value")
            lo, hi = b.get("sharpe_diff_ci_lo"), b.get("sharpe_diff_ci_hi")
            v_label = r.refusal_reason or r.verdict
            ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
            t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
            p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
            lo_s = f"{lo:>+10.4f}" if lo is not None else "     n/a"
            hi_s = f"{hi:>+10.4f}" if hi is not None else "     n/a"
            print(f"{int(w*100)}%     {v_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}")

    # Test 2: 3-substrate composite (CFTC + bond) overlay on cmdty carry leg
    print()
    print("=" * 105)
    print("Test 2 — 3-substrate cross-data-source composite (CFTC + Bond LSC) overlay on commodity carry leg")
    print("=" * 105)
    common_3 = deployed_ls.index.intersection(cftc_ls.index).intersection(bond_strat.index)
    if len(common_3) < 24:
        print(f"Too little overlap: {len(common_3)} mo. Skipping 3-substrate test.")
        return
    print(f"  overlap: {common_3.min().date()} → {common_3.max().date()} ({len(common_3)} mo)")
    # Z-blend at PnL level — but only equal-weight since strengths differ
    cftc_z = (cftc_ls.loc[common_3] - cftc_ls.loc[common_3].mean()) / cftc_ls.loc[common_3].std()
    bond_z = (bond_strat.loc[common_3] - bond_strat.loc[common_3].mean()) / bond_strat.loc[common_3].std()
    # Test multiple weight schemes
    composites = {
        "50/50 (CFTC + Bond)":     (cftc_z + bond_z) / 2,
        "70/30 (CFTC-heavy)":      0.7 * cftc_z + 0.3 * bond_z,
        "30/70 (Bond-heavy)":      0.3 * cftc_z + 0.7 * bond_z,
    }
    d_vt = _vol_target(deployed_ls.loc[common_3], 0.10)
    print(f"{'composite':<24}{'w':<6}{'verdict':<14}{'ΔSh':>10}{'t':>9}{'p':>8}{'IMPROV?':>10}")
    print("-" * 90)
    results = []
    for cname, c in composites.items():
        c_vt = _vol_target(c, 0.10)
        for w in [0.10, 0.20, 0.30]:
            variant = (1 - w) * d_vt + w * c_vt
            variant = _vol_target(variant.dropna(), 0.10)
            r = dispatch_enhance_hypothesis(
                hypothesis_id    = f"3sub_{cname.replace('/','_').replace(' ','_')}_w{int(w*100):02d}",
                sleeve_id        = "cmdty_carry_leg",
                variant_returns  = variant,
                baseline_returns = d_vt,
                n_iterations     = 2000, block_size=6, seed=42,
                log_path         = ENHANCE_LOG,
            )
            b = r.bootstrap_result or {}
            ds, t, p = b.get("sharpe_diff_observed"), b.get("sharpe_diff_t_stat"), b.get("sharpe_diff_p_value")
            v_label = r.refusal_reason or r.verdict
            all3 = ("YES" if (ds and ds > 0.15 and t and t >= 1.96 and p and p < 0.05)
                    else f"{'D' if ds and ds > 0.15 else '-'}"
                         f"{'T' if t and t >= 1.96 else '-'}"
                         f"{'P' if p and p < 0.05 else '-'}")
            ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
            t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
            p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
            print(f"{cname:<24}{int(w*100)}%   {v_label:<14}{ds_s}{t_s}{p_s}{all3:>10}")
            results.append({
                "composite": cname, "weight": w, "verdict": r.verdict, "bootstrap": b,
                "improv_criteria": all3,
            })

    out_json = OUT_DIR / "bond_lsc_results.json"
    out_json.write_text(json.dumps({
        "subject":             "cmdty_carry_leg + combined_book",
        "substrate":           "Bond LSC carry strategy (FRED CMT yields, 2002-2026)",
        "method":              "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "bond_standalone_sharpe":  float(bond_sharpe),
        "cftc_composite_sharpe":   float(_ann_sharpe(cftc_ls)),
        "results":             results,
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
