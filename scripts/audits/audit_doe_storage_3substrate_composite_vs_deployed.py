"""scripts/audit_doe_storage_3substrate_composite_vs_deployed.py

Phase 1 Step 2 of substrate inventory work — DOE inventory substrate.

Tests Pindyck 2001 storage theory directly: long high-storage-deficit
commodities, short low-storage-deficit. Storage deficit defined as
deviation from 5-year same-week-of-year baseline.

Energy-only MVP scope (3 commodities): CL_WTI, RB_Gasoline, HO_HeatOil.
Then combines with CFTC 2-substrate composite for a 3-substrate test:
the critical experiment to see if t-stat scales toward 1.96 IMPROVEMENT.

Theoretical projection (commit f2b3e... validated):
  1 substrate: t ~ 1.18
  2 substrates (CFTC composite): t ~ 1.74 (observed)
  3 substrates (+ DOE): t ~ 1.18 * sqrt(3) ~ 2.04 → CROSSES 1.96

Run:
    python scripts/audit_doe_storage_3substrate_composite_vs_deployed.py
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
from engine.data_sources.eia_stocks import (
    fetch_petroleum_stocks, compute_storage_deficit_signal,
)
from scripts.audits.audit_cftc_hedging_pressure_vs_deployed import (
    _ann_sharpe, _vol_target, _build_ls_sleeve,
)
from scripts.audits.audit_cftc_composite_pressure_plus_money_vs_deployed import (
    build_cftc_signals_monthly, _standardize_cross_sectional,
)

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "doe_3substrate_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG = OUT_DIR / "enhance_verdict_log.jsonl"


def build_doe_storage_signal_monthly() -> pd.DataFrame:
    """Wide monthly storage-deficit panel: month-end × {CL_WTI, RB_Gasoline, HO_HeatOil}."""
    print("Fetching EIA weekly stocks...")
    stocks = fetch_petroleum_stocks()
    signals: dict[str, pd.Series] = {}
    for sym, s in stocks.items():
        deficit = compute_storage_deficit_signal(s)
        # Resample to monthly: take last week of month
        deficit.index = pd.to_datetime(deficit.index)
        monthly = deficit.resample("ME").last()
        signals[sym] = monthly
        print(f"  {sym}: monthly_obs={len(monthly.dropna())} "
              f"first_nonan={monthly.dropna().index.min().date() if monthly.notna().any() else 'none'}")
    return pd.DataFrame(signals).sort_index()


def main():
    print("Step 1 — Build DOE storage deficit signal (3 petroleum commodities)...")
    doe = build_doe_storage_signal_monthly()
    print(f"  panel shape: {doe.shape}")
    print()

    print("Step 2 — Load deployed commodity returns panel...")
    from engine.validation.commodity_carry import build_carry_and_returns
    _, rwide = build_carry_and_returns()
    rwide.index = pd.to_datetime(rwide.index).to_period("M").to_timestamp("M")
    print()

    print("Step 3 — Build DOE-only L/S sleeve (3-commodity)...")
    # For 3 commodities, do top-1 long / bottom-1 short
    doe_idx = pd.to_datetime(doe.index).to_period("M").to_timestamp("M")
    doe.index = doe_idx
    ls_doe = []
    allm = sorted(set(doe.index) | set(rwide.index))
    for i in range(len(allm) - 1):
        m, nxt = allm[i], allm[i + 1]
        if m not in doe.index or nxt not in rwide.index:
            continue
        c = doe.loc[m].dropna()
        if len(c) < 3:
            continue
        c = c.reindex(c.index.intersection(rwide.columns)).dropna()
        if len(c) < 3:
            continue
        # Long high-deficit, short low-deficit
        hi_sym = c.idxmax()
        lo_sym = c.idxmin()
        nr = rwide.loc[nxt]
        r_hi = nr.get(hi_sym)
        r_lo = nr.get(lo_sym)
        if pd.notna(r_hi) and pd.notna(r_lo):
            ls_doe.append((nxt, float(r_hi - r_lo)))
    ls_doe_series = pd.Series(dict(ls_doe)).sort_index().rename("doe_storage_ls")
    print(f"  DOE storage L/S: n={len(ls_doe_series)} "
          f"Sharpe={_ann_sharpe(ls_doe_series):+.3f}")
    print()

    print("Step 4 — Build CFTC 2-substrate composite (per prior breakthrough)...")
    cftc_panels = build_cftc_signals_monthly()
    pm_z = _standardize_cross_sectional(cftc_panels["prod_merc"])
    mm_z = _standardize_cross_sectional(cftc_panels["m_money"])
    # Best composite from prior audit: 30/70 (money-heavy)
    cftc_composite_panel = 0.3 * pm_z + 0.7 * mm_z
    ls_cftc_comp = _build_ls_sleeve(cftc_composite_panel, rwide)
    ls_cftc_comp.index = pd.to_datetime(ls_cftc_comp.index).to_period("M").to_timestamp("M")
    print(f"  CFTC composite 30/70 L/S: n={len(ls_cftc_comp)} "
          f"Sharpe={_ann_sharpe(ls_cftc_comp):+.3f}")
    print()

    print("Step 5 — Load deployed commodity carry (baseline)...")
    from engine.validation.commodity_carry import build_carry_sleeve
    deployed_ls, _, _ = build_carry_sleeve()
    deployed_ls.index = pd.to_datetime(deployed_ls.index).to_period("M").to_timestamp("M")
    print(f"  deployed: n={len(deployed_ls)} "
          f"Sharpe={_ann_sharpe(deployed_ls):+.3f}")
    print()

    # 3-substrate composite: deploy + CFTC_composite + DOE
    # Note: we BLEND the new composite (CFTC + DOE) into deployed
    # First combine CFTC composite + DOE into a single substrate-composite series
    common_subs = ls_cftc_comp.index.intersection(ls_doe_series.index)
    if len(common_subs) < 60:
        print(f"Warning: short DOE-CFTC overlap {len(common_subs)} mo")
    cftc_z = (ls_cftc_comp.loc[common_subs] - ls_cftc_comp.loc[common_subs].mean()) / ls_cftc_comp.loc[common_subs].std()
    doe_z = (ls_doe_series.loc[common_subs] - ls_doe_series.loc[common_subs].mean()) / ls_doe_series.loc[common_subs].std()

    # 3-substrate composite weighted by inverse-vol — equal-weight after vol-target is similar
    # Try 50/50 and 70/30 (CFTC-heavy, DOE-light)
    composite_50_50 = (cftc_z + doe_z) / 2.0
    composite_70_30 = 0.7 * cftc_z + 0.3 * doe_z

    common = deployed_ls.index.intersection(composite_50_50.index)
    d_vt = _vol_target(deployed_ls.loc[common], 0.10)

    print(f"Step 6 — 3-SUBSTRATE COMPOSITE paired enhance vs deployed")
    print(f"   Overlap window: {common.min().date()} → {common.max().date()} "
          f"({len(common)} mo)")
    print(f"   (DOE start window is later — limited by 5y storage baseline warmup)")
    print()

    print("=" * 110)
    print(f"{'composite':<24}{'w':<6}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}"
          f"{'CI low':>11}{'CI high':>11}")
    print("-" * 110)
    results = []
    for comp_name, comp_z in [("50/50 CFTC+DOE", composite_50_50),
                                ("70/30 CFTC-heavy", composite_70_30)]:
        comp_vt = _vol_target(comp_z.loc[common], 0.10)
        for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
            variant = (1 - w) * d_vt + w * comp_vt
            variant = _vol_target(variant.dropna(), 0.10)
            r = dispatch_enhance_hypothesis(
                hypothesis_id    = f"doe_3sub_composite_{comp_name.replace('/','_').replace(' ','_')}_w{int(w*100):02d}",
                sleeve_id        = "cmdty_carry_leg",
                variant_returns  = variant,
                baseline_returns = d_vt,
                n_iterations     = 2000, block_size=6, seed=42,
                log_path         = ENHANCE_LOG,
            )
            b = r.bootstrap_result or {}
            ds, t, p = (b.get("sharpe_diff_observed"),
                          b.get("sharpe_diff_t_stat"),
                          b.get("sharpe_diff_p_value"))
            lo, hi = b.get("sharpe_diff_ci_lo"), b.get("sharpe_diff_ci_hi")
            v_label = r.refusal_reason or r.verdict
            ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
            t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
            p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
            lo_s = f"{lo:>+10.4f}" if lo is not None else "     n/a"
            hi_s = f"{hi:>+10.4f}" if hi is not None else "     n/a"
            print(f"{comp_name:<24}{int(w*100)}%   {v_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}")
            results.append({
                "composite": comp_name, "weight": w,
                "verdict": r.verdict, "bootstrap": b,
            })

    out_json = OUT_DIR / "doe_3substrate_results.json"
    out_json.write_text(json.dumps({
        "subject":             "cmdty_carry_leg",
        "substrates":          ["CFTC prod_merc", "CFTC m_money", "DOE storage (energy)"],
        "method":              "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months":     int(len(common)),
        "doe_standalone_sharpe":   float(_ann_sharpe(ls_doe_series)),
        "cftc_composite_sharpe":   float(_ann_sharpe(ls_cftc_comp)),
        "deployed_sharpe":         float(_ann_sharpe(deployed_ls.loc[common])),
        "results":             results,
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
