"""scripts/audit_carry_regime_conditional_sizing_vs_deployed.py

Tests whether regime-conditional scaling of the deployed 4-leg cross-asset
carry sleeve produces a statistically significant Sharpe improvement vs
the current static-weight deployment.

Why this matters
================
Two prior TSMOM enhance audits (commits 4c6517a6 + feeb5774) ruled out
per-instrument signal-level changes — speed blend and trend-strength
both failed. The remaining valid TSMOM-direction was REGIME-CONDITIONAL
sizing (portfolio-level, not signal-level). The same logic applies to
CARRY: yaml `improvement_directions` explicitly lists "regime-aware
carry weighting" as the canonical untapped direction.

Academic anchors
================
  - Koijen-Moskowitz-Pedersen-Vrugt 2018 §liquidity & volatility risk:
    carry returns load positively on global liquidity shocks,
    negatively on volatility (VXO) shocks. Implies carry should be
    SCALED DOWN in stress regimes.
  - Asness 2014 "Quality minus junk" §quality crashes: defensive
    factors should be scaled by regime indicator.
  - Daniel-Moskowitz 2016 "Momentum Crashes" §dynamic strategy:
    same regime-conditional framework, applied to momentum.

Test design
===========
Baseline: build_carry_book() — current deployed 4-leg static sleeve
Regime classifier: build_vix_regime_monthly (VIX 1y rolling z-score,
   CALM=z<-1, NORMAL=-1≤z≤+1, STRESS=z>+1)

Variants (regime sizing multipliers):
  conservative   : CALM 1.20  NORMAL 1.00  STRESS 0.70
  balanced       : CALM 1.30  NORMAL 1.00  STRESS 0.50
  aggressive     : CALM 1.50  NORMAL 1.00  STRESS 0.30
  stress_throttle: CALM 1.00  NORMAL 1.00  STRESS 0.30  (only throttle, no upside)

Each variant uses LAGGED regime (regime @ t-1 sizes carry @ t) — no
look-ahead. Each variant is re-vol-targeted to baseline annualized vol
so we compare equal-risk strategies (swap framing, not add framing).

Paired Politis-Romano 1994 block bootstrap (B=2000, block=6mo) on the
overlapping window.

Run
===
    python scripts/audit_carry_regime_conditional_sizing_vs_deployed.py
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

from engine.portfolio.combined_book import build_carry_book, build_vix_regime_monthly
from engine.research.enhance import dispatch_enhance_hypothesis

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "carry_regime_sizing_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG_PATH = OUT_DIR / "enhance_verdict_log.jsonl"


REGIME_SIZING_GRIDS = {
    # name              CALM   NORMAL  STRESS
    "conservative":    {"CALM": 1.20, "NORMAL": 1.00, "STRESS": 0.70},
    "balanced":        {"CALM": 1.30, "NORMAL": 1.00, "STRESS": 0.50},
    "aggressive":      {"CALM": 1.50, "NORMAL": 1.00, "STRESS": 0.30},
    "stress_throttle": {"CALM": 1.00, "NORMAL": 1.00, "STRESS": 0.30},
}


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


def _apply_regime_sizing(carry_pnl: pd.Series, regime: pd.Series,
                          grid: dict) -> pd.Series:
    """Multiply each month's carry PnL by the LAGGED regime sizing factor.

    LAGGED = use regime classified at month t-1 to scale PnL at month t.
    No look-ahead.
    """
    # Align indices (both should be month-end)
    carry_pnl = carry_pnl.copy()
    carry_pnl.index = pd.to_datetime(carry_pnl.index).to_period("M").to_timestamp("M")
    regime = regime.copy()
    regime.index = pd.to_datetime(regime.index).to_period("M").to_timestamp("M")
    # Lag regime by one month: regime classified at t-1 sizes PnL at t
    regime_lagged = regime.shift(1)
    # Default to NORMAL when unknown
    sizes = pd.Series(index=carry_pnl.index, dtype=float)
    for t in carry_pnl.index:
        r = regime_lagged.get(t)
        if r is None or (isinstance(r, float) and pd.isna(r)):
            r = "NORMAL"
        sizes.loc[t] = grid.get(r, 1.0)
    return carry_pnl * sizes


def main():
    print("Building deployed carry sleeve PnL...")
    carry = build_carry_book().dropna()
    carry.index = pd.to_datetime(carry.index).to_period("M").to_timestamp("M")
    print(f"  carry sleeve: n={len(carry)}  "
          f"range={carry.index.min().date()} → {carry.index.max().date()}  "
          f"Sharpe={_ann_sharpe(carry):+.3f}")

    print("Building VIX regime classifier (1y rolling z, ±1σ thresholds)...")
    regime = build_vix_regime_monthly()
    regime.index = pd.to_datetime(regime.index).to_period("M").to_timestamp("M")
    common = carry.index.intersection(regime.index)
    print(f"  regime n={len(regime)}  overlap with carry: n={len(common)}")
    # Sanity: regime distribution
    reg_in_common = regime.loc[common].shift(1).dropna()
    counts = reg_in_common.value_counts()
    print(f"  regime distribution (lagged, in overlap window):")
    for r, n in counts.items():
        pct = 100 * n / len(reg_in_common)
        print(f"    {r}: {n} ({pct:.1f}%)")

    # Vol-target the baseline so all subsequent variants compare at 10% ann vol
    baseline = _vol_target(carry.loc[common], 0.10)
    print()
    print(f"Baseline (vol-targeted 10%):  Sharpe={_ann_sharpe(baseline):+.3f}")

    # Build variants
    print()
    print("Building regime-conditional variants...")
    variants = {}
    for name, grid in REGIME_SIZING_GRIDS.items():
        sized = _apply_regime_sizing(carry, regime, grid).loc[common]
        # Re-vol-target to 10% so the comparison is equal-risk
        variants[name] = _vol_target(sized, 0.10)
        print(f"  {name:<18} grid={grid}  Sharpe={_ann_sharpe(variants[name]):+.3f}")

    # Paired enhance test
    print()
    print("Paired enhance test (Politis-Romano 1994, B=2000, block=6mo):")
    print("=" * 95)
    print(f"{'variant':<22}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}{'CI low':>11}{'CI high':>11}{'corr':>8}")
    print("-" * 95)

    results = []
    for name, variant in variants.items():
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"carry_regime_{name}",
            sleeve_id        = "cross_asset_carry",
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
        ds = b.get("sharpe_diff_observed"); t = b.get("sharpe_diff_t_stat")
        p  = b.get("sharpe_diff_p_value");  lo= b.get("sharpe_diff_ci_lo")
        hi = b.get("sharpe_diff_ci_hi");    c = b.get("correlation")
        v_label = r.refusal_reason or r.verdict
        ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
        t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
        p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
        lo_s = f"{lo:>+10.4f}" if lo is not None else "     n/a"
        hi_s = f"{hi:>+10.4f}" if hi is not None else "     n/a"
        c_s  = f"{c:>+7.3f}"  if c  is not None else "   n/a"
        print(f"{name:<22}{v_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}{c_s}")
        results.append({
            "variant":   name, "grid":     REGIME_SIZING_GRIDS[name],
            "verdict":   r.verdict, "refusal":  r.refusal_reason,
            "bootstrap": b, "summary":  r.summary,
        })

    # Persist
    out_json = OUT_DIR / "carry_regime_results.json"
    out_json.write_text(json.dumps({
        "subject":           "cross_asset_carry",
        "baseline":          "build_carry_book() — deployed 4-leg static (cmdty + FX + US-rates + G10-rates-XC)",
        "regime_classifier": "build_vix_regime_monthly (1y rolling z, ±1σ)",
        "regime_distribution": {str(k): int(v) for k, v in counts.items()},
        "method":            "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months":   int(len(baseline)),
        "lagged_regime":     True,
        "results":           results,
    }, indent=2, default=str))
    print()
    print(f"Results saved → {out_json}")
    print(f"Verdict log:   {ENHANCE_LOG_PATH}")


if __name__ == "__main__":
    main()
