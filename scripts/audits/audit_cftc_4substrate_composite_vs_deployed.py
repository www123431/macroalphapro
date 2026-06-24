"""scripts/audit_cftc_4substrate_composite_vs_deployed.py

Phase 1 Step 3: CFTC within-table 4-substrate composite — the
projected-IMPROVEMENT experiment.

After the DOE-weak audit (commit ...), refined composite doctrine:
  sqrt(N) scaling requires substrates of SIMILAR strength.
  CFTC table holds 4 disagg_fut trader categories — likely similar
  strength because all are CFTC-quality institutional data on the
  same commodity universe.

Tests composite of:
  1. prod_merc (commercial hedger pressure)        Bhardwaj-Gorton 2014
  2. m_money (managed money positioning)            Hong-Yogo 2012
  3. swap (swap dealer positioning)                 institutional flow
  4. other_rept (other reportable positioning)      non-commercial speculation

For each signal: verify empirical SIGN via standalone Sharpe before
composing (per the m_money-sign lesson from prior composite audit).

If composite scales as theoretically projected:
  prior 2-substrate composite t = +1.74 (observed)
  4-substrate composite t ≈ 1.74 × sqrt(2) ≈ 2.46 → CROSSES 1.96
  → Session's FIRST IMPROVEMENT verdict

Run:
    python scripts/audit_cftc_4substrate_composite_vs_deployed.py
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
from scripts.audits.audit_cftc_hedging_pressure_vs_deployed import (
    _CFTC_MAPPING, _resolve_cftc_market, _ann_sharpe, _vol_target,
    _build_ls_sleeve,
)
from scripts.audits.audit_cftc_composite_pressure_plus_money_vs_deployed import (
    _standardize_cross_sectional,
)

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "cftc_4substrate_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG = OUT_DIR / "enhance_verdict_log.jsonl"


def build_4_cftc_signals_monthly() -> dict[str, pd.DataFrame]:
    """Build all 4 CFTC trader-category pressure signals as monthly wide panels.

    pressure_i,t = (long - short) / open_interest
    Returns dict[name → wide DataFrame].
    """
    conn = sqlite3.connect(_REPO_ROOT / "macro_alpha_memory.db")
    df = pd.read_sql(
        "SELECT report_date, market_name, open_interest, "
        "prod_merc_long, prod_merc_short, "
        "m_money_long, m_money_short, "
        "swap_long, swap_short, "
        "other_rept_long, other_rept_short "
        "FROM cftc_cot_weekly WHERE report_type='disagg_fut'",
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

    # Signal convention: net = LONG − SHORT, with sign verification below
    signal_defs = {
        "prod_merc":  ("prod_merc_long", "prod_merc_short"),
        "m_money":    ("m_money_long",   "m_money_short"),
        "swap":       ("swap_long",      "swap_short"),
        "other_rept": ("other_rept_long", "other_rept_short"),
    }
    panels = {}
    for nm, (long_col, short_col) in signal_defs.items():
        df[f"{nm}_net"] = (df[long_col] - df[short_col]) / df["open_interest"].replace(0, np.nan)
        wide = (df.sort_values(["sym", "m", "report_date"])
                  .groupby(["sym", "m"])[f"{nm}_net"].last()
                  .unstack("sym").sort_index())
        panels[nm] = wide
    return panels


def main():
    print("Step 1 — Build 4 CFTC signal panels...")
    panels = build_4_cftc_signals_monthly()
    for nm, p in panels.items():
        print(f"  {nm}: shape={p.shape}")
    print()

    print("Step 2 — Determine empirical SIGN per signal (avoid m_money mistake)...")
    from engine.validation.commodity_carry import build_carry_and_returns
    _, rwide = build_carry_and_returns()
    rwide.index = pd.to_datetime(rwide.index).to_period("M").to_timestamp("M")

    # For each signal, test BOTH signs, pick the positive-Sharpe one
    sign_decisions: dict[str, int] = {}
    standalone_sharpes: dict[str, float] = {}
    for nm, p in panels.items():
        ls_pos = _build_ls_sleeve(p, rwide)
        ls_neg = _build_ls_sleeve(-p, rwide)
        s_pos = _ann_sharpe(ls_pos)
        s_neg = _ann_sharpe(ls_neg)
        if s_pos >= s_neg:
            sign_decisions[nm] = +1
            standalone_sharpes[nm] = s_pos
        else:
            sign_decisions[nm] = -1
            standalone_sharpes[nm] = s_neg
        print(f"  {nm}: sign={sign_decisions[nm]:+d}  "
              f"standalone_Sharpe={standalone_sharpes[nm]:+.3f}")
    print()

    # Filter to substrates of SIMILAR strength (per refined doctrine)
    strongest = max(standalone_sharpes.values())
    sim_strength_set = {nm for nm, s in standalone_sharpes.items()
                         if s >= 0.5 * strongest}
    print(f"  Strongest standalone Sharpe = {strongest:+.3f}")
    print(f"  Substrates ≥ 50% of strongest (similar-strength set): {sim_strength_set}")
    print()

    print("Step 3 — Build composite signal (z-blend of similar-strength substrates)...")
    z_panels = {nm: _standardize_cross_sectional(sign_decisions[nm] * panels[nm])
                  for nm in sim_strength_set}
    composite_eq = sum(z_panels.values()) / len(z_panels)
    print(f"  Composite uses {len(z_panels)} substrates: equal-weight z-blend")
    print()

    print("Step 4 — Build deployed baseline + composite L/S...")
    from engine.validation.commodity_carry import build_carry_sleeve
    deployed_ls, _, _ = build_carry_sleeve()
    deployed_ls.index = pd.to_datetime(deployed_ls.index).to_period("M").to_timestamp("M")
    comp_ls = _build_ls_sleeve(composite_eq, rwide)
    comp_ls.index = pd.to_datetime(comp_ls.index).to_period("M").to_timestamp("M")
    print(f"  deployed: Sharpe={_ann_sharpe(deployed_ls):+.3f}")
    print(f"  N-substrate composite: Sharpe={_ann_sharpe(comp_ls):+.3f}")

    common = deployed_ls.index.intersection(comp_ls.index)
    print(f"  overlap: {common.min().date()} → {common.max().date()} ({len(common)} mo)")
    d_c = deployed_ls.loc[common]
    c_c = comp_ls.loc[common]
    corr = d_c.corr(c_c)
    print(f"  corr(deployed, composite) = {corr:+.3f}")
    print()

    print("Step 5 — Paired enhance test")
    print("=" * 110)
    print(f"{'weight':<8}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}"
          f"{'CI low':>11}{'CI high':>11}{'IMPROV?':>10}")
    print("-" * 110)

    d_vt = _vol_target(d_c, 0.10)
    c_vt = _vol_target(c_c, 0.10)
    results = []
    for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        variant = (1 - w) * d_vt + w * c_vt
        variant = _vol_target(variant.dropna(), 0.10)
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"cftc_4sub_composite_w{int(w*100):02d}",
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
        # IMPROVEMENT criteria
        all3 = ("YES" if (ds and ds > 0.15 and t and t >= 1.96 and p and p < 0.05)
                else f"{'D' if ds and ds > 0.15 else '-'}"
                     f"{'T' if t  and t  >= 1.96 else '-'}"
                     f"{'P' if p  and p  < 0.05 else '-'}")
        ds_s = f"{ds:>+9.4f}" if ds is not None else "    n/a"
        t_s  = f"{t:>+8.3f}"  if t  is not None else "   n/a"
        p_s  = f"{p:>7.3f}"   if p  is not None else "   n/a"
        lo_s = f"{lo:>+10.4f}" if lo is not None else "     n/a"
        hi_s = f"{hi:>+10.4f}" if hi is not None else "     n/a"
        print(f"{int(w*100)}%     {v_label:<14}{ds_s}{t_s}{p_s}{lo_s}{hi_s}{all3:>10}")
        results.append({
            "weight": w, "verdict": r.verdict, "bootstrap": b,
            "improv_criteria_met": all3,
        })

    out_json = OUT_DIR / "cftc_4substrate_results.json"
    out_json.write_text(json.dumps({
        "subject":             "cmdty_carry_leg",
        "substrates_tested":   list(panels.keys()),
        "sign_decisions":      sign_decisions,
        "standalone_sharpes":  standalone_sharpes,
        "substrates_in_composite": list(sim_strength_set),
        "method":              "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months":     int(len(common)),
        "corr_deployed_composite": float(corr),
        "deployed_sharpe":     float(_ann_sharpe(d_c)),
        "composite_sharpe":    float(_ann_sharpe(c_c)),
        "results":             results,
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
