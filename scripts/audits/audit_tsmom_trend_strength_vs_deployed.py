"""scripts/audit_tsmom_trend_strength_vs_deployed.py — TSMOM trend-strength audit.

Tests whether replacing the canonical SIGN-based position scaling
(Moskowitz-Ooi-Pedersen 2012 §2.1: signal_{i,t} = sign(cum_log_return))
with a CONTINUOUS trend-strength score improves the deployed 5-leg
cross_asset_tsmom sleeve.

Why this matters
================
The 2026-06-17 TSMOM speed blend audit (commit 4c6517a6) closed the
"faster lookbacks help?" question (REJECT). The remaining yaml-listed
TSMOM improvement direction with academic backing is:

  "vol-targeted trend strength scoring"

Sign-based sizing treats a +1% / 12mo trend the same as a +25% / 12mo
trend. The continuous-strength hypothesis: load more aggressively into
high-conviction trends and lightly into weak ones, modulating
exposure via the SIGNAL itself (the per-instrument vol target stays
the same — this changes WHEN we get to full vol-target, not the cap).

Academic anchors:
  - Asness 2014 "Quality Minus Junk" — continuous strength scoring
    framework (positive but bounded signal magnitudes outperform binary)
  - Bauer-Frijns 2014 "Trend Strength in TSMOM" — explicit treatment
    of |cum_return / σ_cum_return| as position-sizing factor
  - Moskowitz 2012 §3.2 — discusses sign vs magnitude but defaults to
    sign for parsimony; explicit comparison left open

Variants tested
===============
All variants use the SAME 12-1 lookback (canonical), SAME per-instrument
40% vol target (capped at 2x), SAME risk-parity 5-leg combine. Only the
inner `signal` term differs:

  baseline_sign       : signal = sign(cum_log_return)                  ← deployed
  zscore_strength     : signal = clip(cum_log_return / σ_cum, -3, +3) / 3
  tanh_strength       : signal = tanh(cum_log_return / 0.05)
  magnitude_capped    : signal = clip(cum_log_return / 0.10, -1, +1)

Each is paired-tested (Politis-Romano 1994, B=2000, block=6mo) against
the deployed sign-baseline.

Run
===
    python scripts/audit_tsmom_trend_strength_vs_deployed.py
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

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "tsmom_trend_strength_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG_PATH = OUT_DIR / "enhance_verdict_log.jsonl"


LOOKBACK_MONTHS         = 12
SKIP_MONTHS             = 1
TARGET_INSTRUMENT_VOL   = 0.40
VOL_LOOKBACK_MONTHS     = 12
ZSCORE_LOOKBACK_MONTHS  = 36   # for zscore_strength variant
MIN_INSTRUMENTS_PER_LEG = 3


def _make_per_instrument_returns(
    rwide_monthly: pd.DataFrame,
    *,
    signal_kind: str,
) -> pd.DataFrame:
    """Per-instrument vol-scaled signed returns with parameterizable signal.

    signal_kind ∈ {baseline_sign, zscore_strength, tanh_strength, magnitude_capped}
    """
    log_ret = np.log1p(rwide_monthly.astype(float))
    cum_lookback = (log_ret.shift(SKIP_MONTHS)
                          .rolling(LOOKBACK_MONTHS - SKIP_MONTHS)
                          .sum())

    if signal_kind == "baseline_sign":
        signal = np.sign(cum_lookback)
    elif signal_kind == "zscore_strength":
        # Rolling std of cum_lookback over a longer window
        cum_std = cum_lookback.rolling(ZSCORE_LOOKBACK_MONTHS, min_periods=12).std()
        z = (cum_lookback / cum_std.replace(0, np.nan)).clip(lower=-3, upper=3)
        signal = z / 3.0   # normalize back to roughly [-1, +1]
    elif signal_kind == "tanh_strength":
        # tanh squashing: 5% 12mo trend → tanh(1) ≈ 0.76; 10% → tanh(2) ≈ 0.96
        signal = np.tanh(cum_lookback / 0.05)
    elif signal_kind == "magnitude_capped":
        # Linear in cum_return up to 10% threshold, clipped beyond
        signal = (cum_lookback / 0.10).clip(lower=-1, upper=1)
    else:
        raise ValueError(f"unknown signal_kind={signal_kind!r}")

    realized_vol = (rwide_monthly.shift(SKIP_MONTHS)
                                  .rolling(VOL_LOOKBACK_MONTHS)
                                  .std() * np.sqrt(12))
    scale = (TARGET_INSTRUMENT_VOL / realized_vol).clip(upper=2.0)
    return signal * scale * rwide_monthly


def _aggregate_leg(inst_returns: pd.DataFrame) -> pd.Series:
    n_live = inst_returns.notna().sum(axis=1)
    leg = inst_returns.mean(axis=1, skipna=True)
    leg[n_live < MIN_INSTRUMENTS_PER_LEG] = np.nan
    return leg.dropna()


def _build_sleeve(signal_kind: str) -> pd.Series:
    from engine.validation.commodity_carry import build_carry_and_returns as cmdty_loader
    from engine.validation.crossasset_carry import (
        fetch_fx_futures, _carry_and_returns, FX,
        _fetch_classes, RATES, _RT_CONTR, _RT_PX, _RT_PXDIR,
        RATES_XC, _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR,
        EQIDX, _EQIDX_CONTR, _EQIDX_PX, _EQIDX_PXDIR,
    )
    from engine.portfolio.carry_sleeve import risk_parity_combine

    legs: dict[str, pd.Series] = {}
    _, rw_cmdty = cmdty_loader(daily=False)
    legs["cmdty"] = _aggregate_leg(
        _make_per_instrument_returns(rw_cmdty, signal_kind=signal_kind))

    c_fx, p_fx = fetch_fx_futures()
    _, rw_fx = _carry_and_returns(c_fx, p_fx, FX)
    legs["fx"] = _aggregate_leg(
        _make_per_instrument_returns(rw_fx, signal_kind=signal_kind))

    c_us, p_us = _fetch_classes(RATES, _RT_CONTR, _RT_PX, _RT_PXDIR)
    _, rw_us = _carry_and_returns(c_us, p_us, RATES)
    legs["rates_us"] = _aggregate_leg(
        _make_per_instrument_returns(rw_us, signal_kind=signal_kind))

    c_xc, p_xc = _fetch_classes(RATES_XC, _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR,
                                   isocurr=None)
    _, rw_xc = _carry_and_returns(c_xc, p_xc, RATES_XC)
    legs["rates_xc"] = _aggregate_leg(
        _make_per_instrument_returns(rw_xc, signal_kind=signal_kind))

    c_eq, p_eq = _fetch_classes(EQIDX, _EQIDX_CONTR, _EQIDX_PX, _EQIDX_PXDIR,
                                   isocurr=None)
    _, rw_eq = _carry_and_returns(c_eq, p_eq, EQIDX)
    legs["eqidx"] = _aggregate_leg(
        _make_per_instrument_returns(rw_eq, signal_kind=signal_kind))

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
    print("Building TSMOM at canonical L=12 with different signal-strength formulations...")
    print("=" * 70)

    variants = {}
    for kind in ["baseline_sign", "zscore_strength", "tanh_strength", "magnitude_capped"]:
        print(f"  signal={kind} ... ", end="", flush=True)
        s = _build_sleeve(kind)
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        variants[kind] = s
        sh = _ann_sharpe(s)
        print(f"n={len(s)} Sharpe(unhedged)={sh:+.3f}")

    # Sanity: baseline_sign should match deployed exactly
    from engine.validation.crossasset_tsmom import build_tsmom_sleeve_returns
    deployed = build_tsmom_sleeve_returns()
    deployed.index = pd.to_datetime(deployed.index).to_period("M").to_timestamp("M")
    common = variants["baseline_sign"].index.intersection(deployed.index)
    sanity_corr = variants["baseline_sign"].loc[common].corr(deployed.loc[common])
    print()
    print(f"Sanity: corr(baseline_sign self-built vs deployed) = {sanity_corr:.4f}")

    # Vol-target everything to 10% for paired comparison
    vt = {k: _vol_target(v.dropna(), 0.10) for k, v in variants.items()}
    print()
    print("Standalone vol-targeted Sharpe (10% target):")
    for kind, s in vt.items():
        sh = _ann_sharpe(s)
        print(f"  {kind:<22} n={len(s)} Sharpe={sh:+.3f}")

    print()
    print("Paired enhance test (Politis-Romano 1994 block bootstrap, B=2000):")
    print("=" * 95)
    print(f"{'variant':<22}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}{'CI low':>11}{'CI high':>11}{'corr':>8}")
    print("-" * 95)

    baseline = vt["baseline_sign"]
    results = []
    for kind, variant in vt.items():
        if kind == "baseline_sign":
            continue
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"tsmom_strength_{kind}",
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
        print(f"{kind:<22}{v_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}{c_s}")
        results.append({
            "variant":   kind, "verdict":  r.verdict, "refusal":  r.refusal_reason,
            "bootstrap": b, "summary":  r.summary,
        })

    out_json = OUT_DIR / "tsmom_strength_results.json"
    out_json.write_text(json.dumps({
        "subject":             "cross_asset_tsmom",
        "baseline":            "baseline_sign (Moskowitz 2012 §2.1 canonical)",
        "sanity_check_corr":   float(sanity_corr),
        "method":              "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months":     int(len(baseline)),
        "lookback_months":     LOOKBACK_MONTHS,
        "results":             results,
    }, indent=2, default=str))
    print()
    print(f"Results saved → {out_json}")


if __name__ == "__main__":
    main()
