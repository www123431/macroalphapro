"""scripts/audit_cftc_composite_pressure_plus_money_vs_deployed.py

Phase 1 Step 1.5 — multi-substrate composite hypothesis test.

The previous CFTC audit (commit 7c..., extended window) confirmed:
  - prod_merc hedging pressure produces a directional positive signal
    (ΔSharpe +0.029 at 10% weight, t=+1.18 over 180mo)
  - Magnitude is too small for IMPROVEMENT verdict at any single weight
  - The path forward is COMPOSITE of multiple orthogonal substrates

This audit tests the SIMPLEST composite hypothesis: combine
prod_merc (commercial hedger pressure, Bhardwaj-Gorton 2014) with
m_money (managed money positioning, Hong-Yogo 2012) into a single
blended signal.

Mechanism interpretation:
  prod_merc_pressure = risk premium signal (commercial demand for
    hedging insurance)
  m_money_pressure   = speculator positioning signal (trend/sentiment)
  These should be approximately ORTHOGONAL (commercial and
    speculative views are typically mirror images)
  Composite = average of standardized signals → captures both

If composite ΔSharpe > either single signal alone → validates the
multi-substrate roadmap toward Sharpe 1.5+ within the rigorous
statistical framework. If not → tells us composite needs more
sophisticated weighting / regime conditioning.

Run:
    python scripts/audit_cftc_composite_pressure_plus_money_vs_deployed.py
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from engine.research.enhance import dispatch_enhance_hypothesis

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "cftc_composite_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG = OUT_DIR / "enhance_verdict_log.jsonl"

# Reuse the curated CFTC mapping from the prior audit
from scripts.audits.audit_cftc_hedging_pressure_vs_deployed import (
    _CFTC_MAPPING, _resolve_cftc_market, _ann_sharpe, _vol_target,
    _build_ls_sleeve,
)


def build_cftc_signals_monthly() -> dict[str, pd.DataFrame]:
    """Build BOTH prod_merc pressure AND m_money pressure panels."""
    conn = sqlite3.connect(_REPO_ROOT / "macro_alpha_memory.db")
    df = pd.read_sql(
        "SELECT report_date, market_name, open_interest, "
        "prod_merc_long, prod_merc_short, "
        "m_money_long, m_money_short FROM cftc_cot_weekly "
        "WHERE report_type='disagg_fut'",
        conn,
    )
    conn.close()
    df["report_date"] = pd.to_datetime(df["report_date"])

    all_markets = df["market_name"].unique().tolist()
    sym_to_market: dict[str, str] = {}
    for sym, (pat, exclude) in _CFTC_MAPPING.items():
        m = _resolve_cftc_market(pat, exclude, all_markets)
        if m is not None:
            sym_to_market[sym] = m

    df = df[df["market_name"].isin(sym_to_market.values())].copy()
    mkt_to_sym = {v: k for k, v in sym_to_market.items()}
    df["sym"] = df["market_name"].map(mkt_to_sym)
    df["m"] = df["report_date"].dt.to_period("M").dt.to_timestamp("M")

    # prod_merc pressure (commercial hedger)
    df["prod_merc_pressure"] = (
        (df["prod_merc_short"] - df["prod_merc_long"])
        / df["open_interest"].replace(0, np.nan)
    )
    # m_money pressure (managed money / speculator)
    # Sign convention: m_money LONG net is bullish, but for cross-section
    # we use net long − short (positive = speculator bullish)
    df["m_money_pressure"] = (
        (df["m_money_long"] - df["m_money_short"])
        / df["open_interest"].replace(0, np.nan)
    )

    prod_merc = (df.sort_values(["sym", "m", "report_date"])
                    .groupby(["sym", "m"])["prod_merc_pressure"].last()
                    .unstack("sym").sort_index())
    m_money   = (df.sort_values(["sym", "m", "report_date"])
                    .groupby(["sym", "m"])["m_money_pressure"].last()
                    .unstack("sym").sort_index())
    return {"prod_merc": prod_merc, "m_money": m_money}


def _standardize_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score within each month across the cross-section of commodities."""
    return panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1), axis=0)


def main():
    print("Step 1 — Build BOTH CFTC signals (prod_merc + m_money)...")
    panels = build_cftc_signals_monthly()
    pm = panels["prod_merc"]
    mm = panels["m_money"]
    print(f"  prod_merc: shape={pm.shape}  range={pm.index.min().date()} → {pm.index.max().date()}")
    print(f"  m_money:   shape={mm.shape}")

    # Inter-signal correlation across cross-section
    common_t = pm.index.intersection(mm.index)
    common_c = pm.columns.intersection(mm.columns)
    pm_a = pm.loc[common_t, common_c]
    mm_a = mm.loc[common_t, common_c]
    # Average per-commodity time-series correlation
    corrs = []
    for c in common_c:
        if pm_a[c].std() > 0 and mm_a[c].std() > 0:
            corrs.append(pm_a[c].corr(mm_a[c]))
    avg_inter_signal_corr = float(np.mean(corrs))
    print(f"  avg corr(prod_merc, m_money) across commodities = {avg_inter_signal_corr:+.3f}")
    print()

    print("Step 2 — Z-standardize each signal cross-sectionally + build composite...")
    pm_z = _standardize_cross_sectional(pm)
    # m_money sign: positive net long = bullish speculator, but Bhardwaj-Gorton
    # framework says risk premium accrues OPPOSITE to speculator sentiment
    # (when speculators are net-long, risk premium should be LOWER).
    # So invert m_money sign for risk-premium-aligned signal.
    mm_z = _standardize_cross_sectional(-mm)   # NEGATED — speculator net long → low risk premium

    # Composite = 0.5 × prod_merc_z + 0.5 × m_money_z
    composite = (pm_z + mm_z) / 2.0
    print(f"  composite: shape={composite.shape}")
    print()

    print("Step 3 — Load deployed commodity returns panel + build 3 L/S sleeves...")
    from engine.validation.commodity_carry import build_carry_and_returns
    cwide, rwide = build_carry_and_returns()
    rwide.index = pd.to_datetime(rwide.index).to_period("M").to_timestamp("M")

    ls_prod_merc = _build_ls_sleeve(pm, rwide)
    ls_m_money   = _build_ls_sleeve(-mm, rwide)   # negated speculator signal
    ls_composite = _build_ls_sleeve(composite, rwide)

    for nm, s in [("prod_merc", ls_prod_merc),
                   ("m_money (negated)", ls_m_money),
                   ("composite", ls_composite)]:
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        print(f"  {nm:<22}: n={len(s):>3}  Sharpe={_ann_sharpe(s):+.3f}")
    print()

    print("Step 4 — Load deployed commodity carry (baseline)...")
    from engine.validation.commodity_carry import build_carry_sleeve
    deployed_ls, _, _ = build_carry_sleeve()
    deployed_ls.index = pd.to_datetime(deployed_ls.index).to_period("M").to_timestamp("M")
    print(f"  deployed: Sharpe={_ann_sharpe(deployed_ls):+.3f}")
    print()

    common = deployed_ls.index.intersection(ls_composite.index)
    d_c = deployed_ls.loc[common]
    print(f"Overlap window: {common.min().date()} → {common.max().date()} ({len(common)} mo)")
    print()

    # Paired enhance test — composite vs deployed
    d_vt = _vol_target(d_c, 0.10)
    comp_vt = _vol_target(ls_composite.loc[common], 0.10)

    print("Step 5 — Paired enhance test: composite blend vs deployed")
    print("=" * 105)
    print(f"{'weight':<8}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}"
          f"{'CI low':>11}{'CI high':>11}{'corr':>8}")
    print("-" * 105)

    results = []
    for w in [0.10, 0.20, 0.30, 0.40, 0.50]:
        variant = (1 - w) * d_vt + w * comp_vt
        variant = _vol_target(variant.dropna(), 0.10)
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"cftc_composite_blend_w{int(w*100):02d}",
            sleeve_id        = "cmdty_carry_leg",
            variant_returns  = variant,
            baseline_returns = d_vt,
            n_iterations     = 2000, block_size=6, seed=42,
            log_path         = ENHANCE_LOG,
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
        print(f"{int(w*100)}%     {v_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}{c_s}")
        results.append({
            "weight": w, "verdict": r.verdict, "bootstrap": b,
        })

    out_json = OUT_DIR / "cftc_composite_results.json"
    out_json.write_text(json.dumps({
        "subject":             "cmdty_carry_leg",
        "baseline":            "deployed commodity carry (F1-F2 basis)",
        "variant":             "CFTC composite (prod_merc + (-m_money)) z-blend",
        "method":              "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months":     int(len(common)),
        "avg_inter_signal_corr": avg_inter_signal_corr,
        "composite_sharpe":    float(_ann_sharpe(ls_composite.loc[common])),
        "prod_merc_sharpe":    float(_ann_sharpe(ls_prod_merc.reindex(common).dropna())),
        "m_money_sharpe":      float(_ann_sharpe(ls_m_money.reindex(common).dropna())),
        "results":             results,
        "academic_anchor":     "Bhardwaj-Gorton 2014 + Hong-Yogo 2012",
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
