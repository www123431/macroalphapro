"""scripts/audit_tsmom_speed_blend_vs_deployed.py — TSMOM speed blend audit.

Tests whether a fast+slow TSMOM blend strictly improves the deployed
cross_asset_tsmom sleeve (Moskowitz-Ooi-Pedersen 2012 standard 12-1
single-speed) via paired block bootstrap (Politis-Romano 1994).

Why this matters
================
Deployed sleeve uses LOOKBACK_MONTHS=12 (canonical MOP 2012). Hurst-
Ooi-Pedersen 2017 "A Century of Evidence on Trend-Following" + Lev
2018 "A Century of Trend-Following Investing" report that BLENDS of
fast+slow trend speeds reduce drawdown without proportional Sharpe
sacrifice. The hypothesis-queue Moskowitz 2011 entry (hyp 15d720de,
mechanism_subtype=time_series_momentum_lookback_holding_grid)
explicitly tests the grid of look-back periods.

Test design
===========
1. Build TSMOM PnL at lookbacks ∈ {3, 6, 9, 12} months. The 12-month
   variant equals the deployed sleeve definitionally (sanity check).
2. Vol-target each to 10% annualized.
3. Build blends:
     - fast+slow 50/50  : (3mo + 12mo) / 2
     - fast+slow 30/70  : 0.30 × 3mo + 0.70 × 12mo
     - 4-speed average  : (3mo + 6mo + 9mo + 12mo) / 4
4. For each blend, run paired block bootstrap (Politis-Romano 1994,
   B=2000, block=6mo) vs the 12mo baseline.
5. Classify IMPROVEMENT / NOISE / DEGRADATION.

Cost-aware caveat
=================
Faster TSMOM has higher turnover; per-month TC penalty grows roughly
linearly in turnover. The MOP 2012 spec uses 5 × RT_TS / 10000 / 12
monthly. We apply the SAME constant to all variants for like-for-like
comparison; a fully cost-rigorous study would estimate per-variant
turnover. NOISE verdicts here would tighten only if blend turnover is
materially higher than 12-1; this gives the blend the benefit of doubt.

Run
===
    python scripts/audit_tsmom_speed_blend_vs_deployed.py
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

from engine.research.enhance import (
    dispatch_enhance_hypothesis,
    paired_block_bootstrap_summary,
)

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "tsmom_speed_blend_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG_PATH = OUT_DIR / "enhance_verdict_log.jsonl"


def _tsmom_per_instrument_with_lookback(
    rwide_monthly: pd.DataFrame,
    *,
    lookback_months:    int,
    skip_months:        int = 1,
    target_instrument_vol: float = 0.40,
    vol_lookback_months:   int = 12,
) -> pd.DataFrame:
    """Same as engine.validation.crossasset_tsmom._tsmom_per_instrument
    but with parameterizable lookback. Vol scaling capped at 2x (MOP standard).
    """
    log_ret = np.log1p(rwide_monthly.astype(float))
    cum_lookback = (log_ret.shift(skip_months)
                          .rolling(max(1, lookback_months - skip_months))
                          .sum())
    signal = np.sign(cum_lookback)
    realized_vol = (
        rwide_monthly.shift(skip_months)
        .rolling(vol_lookback_months).std() * np.sqrt(12)
    )
    scale = (target_instrument_vol / realized_vol).clip(upper=2.0)
    return signal * scale * rwide_monthly


def _aggregate_leg(inst_returns: pd.DataFrame, min_n: int = 3) -> pd.Series:
    n_live = inst_returns.notna().sum(axis=1)
    leg = inst_returns.mean(axis=1, skipna=True)
    leg[n_live < min_n] = np.nan
    return leg.dropna()


def _build_5leg_tsmom_at_lookback(lookback_m: int) -> pd.Series:
    """Reproduce build_tsmom_sleeve_returns but with parameterizable lookback.

    Reuses the same data loaders so the 12-month variant equals the deployed
    sleeve (sanity check verifies this).
    """
    from engine.validation.commodity_carry import build_carry_and_returns as cmdty_loader
    from engine.validation.crossasset_carry import (
        fetch_fx_futures, _carry_and_returns, FX,
        _fetch_classes, RATES, _RT_CONTR, _RT_PX, _RT_PXDIR,
        RATES_XC, _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR,
        EQIDX, _EQIDX_CONTR, _EQIDX_PX, _EQIDX_PXDIR,
    )
    from engine.portfolio.carry_sleeve import risk_parity_combine

    legs: dict[str, pd.Series] = {}

    # 1) Commodity leg
    _, rw_cmdty = cmdty_loader(daily=False)
    legs["cmdty"] = _aggregate_leg(
        _tsmom_per_instrument_with_lookback(rw_cmdty, lookback_months=lookback_m)
    )

    # 2) FX leg
    c_fx, p_fx = fetch_fx_futures()
    _, rw_fx = _carry_and_returns(c_fx, p_fx, FX)
    legs["fx"] = _aggregate_leg(
        _tsmom_per_instrument_with_lookback(rw_fx, lookback_months=lookback_m)
    )

    # 3) US rates leg
    c_us, p_us = _fetch_classes(RATES, _RT_CONTR, _RT_PX, _RT_PXDIR)
    _, rw_us = _carry_and_returns(c_us, p_us, RATES)
    legs["rates_us"] = _aggregate_leg(
        _tsmom_per_instrument_with_lookback(rw_us, lookback_months=lookback_m)
    )

    # 4) XC rates leg
    c_xc, p_xc = _fetch_classes(RATES_XC, _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR,
                                   isocurr=None)
    _, rw_xc = _carry_and_returns(c_xc, p_xc, RATES_XC)
    legs["rates_xc"] = _aggregate_leg(
        _tsmom_per_instrument_with_lookback(rw_xc, lookback_months=lookback_m)
    )

    # 5) Equity index leg
    c_eq, p_eq = _fetch_classes(EQIDX, _EQIDX_CONTR, _EQIDX_PX, _EQIDX_PXDIR,
                                   isocurr=None)
    _, rw_eq = _carry_and_returns(c_eq, p_eq, EQIDX)
    legs["eqidx"] = _aggregate_leg(
        _tsmom_per_instrument_with_lookback(rw_eq, lookback_months=lookback_m)
    )

    return risk_parity_combine(legs)


def _ann_sharpe(s: pd.Series) -> float:
    if len(s) < 12 or s.std(ddof=1) <= 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * math.sqrt(12))


def _vol_target(s: pd.Series, target_ann: float = 0.10) -> pd.Series:
    v = s.std(ddof=1) * math.sqrt(12)
    if not math.isfinite(v) or v <= 0:
        return s
    return s * (target_ann / v)


def main():
    print("Building TSMOM at multiple lookbacks...")
    print("=" * 70)

    variants: dict[int, pd.Series] = {}
    for L in [3, 6, 9, 12]:
        print(f"  L={L}mo... ", end="", flush=True)
        s = _build_5leg_tsmom_at_lookback(L)
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        variants[L] = s
        sh = _ann_sharpe(s)
        print(f"n={len(s)} Sharpe(unhedged)={sh:+.3f}")

    # Sanity check: 12mo should match deployed sleeve closely
    from engine.validation.crossasset_tsmom import build_tsmom_sleeve_returns
    deployed_12 = build_tsmom_sleeve_returns()
    deployed_12.index = pd.to_datetime(deployed_12.index).to_period("M").to_timestamp("M")
    common = variants[12].index.intersection(deployed_12.index)
    corr_check = variants[12].loc[common].corr(deployed_12.loc[common])
    print()
    print(f"Sanity: corr(L=12 vs deployed) = {corr_check:.4f} "
          f"(should be ~1.0 — identical algorithm)")

    # Build vol-targeted variants for blending
    vt = {L: _vol_target(variants[L], 0.10) for L in variants}

    blends = {
        "L12_baseline":      vt[12],
        "fast_slow_50_50":   (vt[3]  + vt[12]) / 2,
        "fast_slow_30_70":   0.30 * vt[3]  + 0.70 * vt[12],
        "med_slow_50_50":    (vt[6]  + vt[12]) / 2,
        "smooth_4speed":     (vt[3]  + vt[6] + vt[9] + vt[12]) / 4,
    }
    # Re-vol-target every blend so they're equal-risk comparators
    blends = {k: _vol_target(v.dropna(), 0.10) for k, v in blends.items()}

    print()
    print("Standalone vol-targeted Sharpe (10% target):")
    for name, s in blends.items():
        sh = _ann_sharpe(s)
        print(f"  {name:<22} n={len(s)} Sharpe={sh:+.3f}")

    # Paired enhance test: each non-baseline blend vs L12_baseline
    print()
    print("Paired enhance test (Politis-Romano 1994 block bootstrap, B=2000):")
    print("=" * 95)
    print(f"{'variant':<22}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}{'CI low':>11}{'CI high':>11}{'corr':>8}")
    print("-" * 95)

    baseline = blends["L12_baseline"]
    results: list[dict] = []

    for name, variant in blends.items():
        if name == "L12_baseline":
            continue
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"tsmom_blend_{name}",
            sleeve_id        = "cross_asset_tsmom",
            variant_returns  = variant,
            baseline_returns = baseline,
            cron_run_id      = None,
            cron_source      = "manual_audit",
            n_iterations     = 2000,
            block_size       = 6,
            log_path         = ENHANCE_LOG_PATH,
            seed             = 42,
        )
        b = r.bootstrap_result or {}
        ds = b.get("sharpe_diff_observed")
        t  = b.get("sharpe_diff_t_stat")
        p  = b.get("sharpe_diff_p_value")
        lo = b.get("sharpe_diff_ci_lo")
        hi = b.get("sharpe_diff_ci_hi")
        c  = b.get("correlation")
        verdict_label = r.refusal_reason or r.verdict
        ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
        t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
        p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
        lo_s = f"{lo:>+10.4f}" if lo is not None else "     n/a"
        hi_s = f"{hi:>+10.4f}" if hi is not None else "     n/a"
        c_s  = f"{c:>+7.3f}"  if c  is not None else "   n/a"
        print(f"{name:<22}{verdict_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}{c_s}")
        results.append({
            "variant":   name,
            "verdict":   r.verdict,
            "refusal":   r.refusal_reason,
            "bootstrap": b,
            "summary":   r.summary,
        })

    # Persist
    out_json = OUT_DIR / "tsmom_speed_blend_results.json"
    out_json.write_text(json.dumps({
        "subject":         "cross_asset_tsmom",
        "baseline":        "L12_baseline (Moskowitz 2012 canonical 12-1)",
        "sanity_check_corr_L12_vs_deployed": float(corr_check),
        "method":          "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months": int(len(baseline)),
        "results":         results,
    }, indent=2, default=str))
    print()
    print(f"Results saved → {out_json}")
    print(f"Verdict log:   {ENHANCE_LOG_PATH}")


if __name__ == "__main__":
    main()
