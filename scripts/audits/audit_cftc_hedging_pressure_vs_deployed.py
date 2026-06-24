"""scripts/audit_cftc_hedging_pressure_vs_deployed.py — CFTC COT substrate audit.

First TRUE NEW SUBSTRATE experiment. The deployed commodity carry leg
uses near-next futures basis (F1-F2/F2 annualized). This substrate is
ORTHOGONAL: weekly producer/merchant hedging positioning from the CFTC
Disaggregated Commitment of Traders report (Bhardwaj-Gorton 2014).

Mechanism (Hirshleifer 1990 risk premium theory):
  pressure_i,t = (prod_merc_short - prod_merc_long) / open_interest
Producers (commercial hedgers) systematically short futures to hedge
output; speculators must be long for market clearing. The risk premium
they demand → higher expected future returns for commodities where
producer hedging is heaviest.

Empirical evidence (Bhardwaj-Gorton 2014, Cong-Eckblad 2015):
  Long-short on hedging pressure → Sharpe 0.3-0.5 in US commodity
  futures, 1980-2010 sample.

Variant test: paired Politis-Romano 1994 bootstrap on
  variant = blend of deployed commodity carry + hedging pressure L/S
  baseline = deployed commodity carry alone

Data: 2020-2024 cached in macro_alpha_memory.db.cftc_cot_weekly
(72,790 weekly rows, 556 markets). ~5-year window — shorter than ideal
for paired bootstrap statistical power, but enough for first signal.

Run:
    python scripts/audit_cftc_hedging_pressure_vs_deployed.py
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

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "cftc_hedging_pressure_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG = OUT_DIR / "enhance_verdict_log.jsonl"


# Carefully-curated commodity → exact CFTC market_name pattern.
# Each entry: (sym, exact_pattern_to_match) with explicit
# EXCLUDE_PATTERNS for false-positive avoidance (MINI, BLACK SEA, BRENT vs WTI).
_CFTC_MAPPING = {
    "CL_WTI":          ("CRUDE OIL, LIGHT SWEET", []),
    "BRN_Brent":       ("BRENT LAST DAY",         []),
    "HO_HeatOil":      ("#2 HEATING OIL",         []),
    "NG_NatGas":       ("NATURAL GAS -",          ["E-MINI"]),
    "RB_Gasoline":     ("GASOLINE BLENDSTOCK",    []),
    "GC_Gold":         ("GOLD -",                 []),
    "SI_Silver":       ("SILVER -",               []),
    "HG_Copper":       ("COPPER- #1",             []),
    "PL_Platinum":     ("PLATINUM -",             []),
    "PA_Palladium":    ("PALLADIUM -",            []),
    "ZC_Corn":         ("CORN -",                 ["MINI"]),
    "ZS_Soybean":      ("SOYBEANS -",             ["MINI", "MEAL", "OIL"]),
    "ZM_SoyMeal":      ("SOYBEAN MEAL",           []),
    "ZL_SoyOil":       ("SOYBEAN OIL",            []),
    "CC_Cocoa":        ("COCOA -",                []),
    "LE_LiveCattle":   ("LIVE CATTLE",            []),
    "GF_FeederCattle": ("FEEDER CATTLE",          []),
    "HE_LeanHogs":     ("LEAN HOGS",              []),
    "ZW_Wheat":        ("WHEAT-SRW",              ["BLACK SEA"]),
    "KC_Coffee":       ("COFFEE C",               []),
    "SB_Sugar":        ("SUGAR NO. 11",           []),
    "CT_Cotton":       ("COTTON NO. 2",           []),
}


def _resolve_cftc_market(target_pattern: str, exclude_patterns: list[str],
                          all_markets: list[str]) -> str | None:
    """Find best market_name matching pattern + not matching excludes."""
    cands = [m for m in all_markets
              if target_pattern in m.upper() and
                 not any(ex.upper() in m.upper() for ex in exclude_patterns)]
    if not cands:
        return None
    # Prefer shortest match (most canonical)
    return min(cands, key=len)


def build_hedging_pressure_signal_monthly() -> pd.DataFrame:
    """Build monthly (last-week-of-month) hedging pressure panel from CFTC.

    Returns wide DataFrame: month-end index × commodity columns.
    pressure_i,t = (prod_merc_short - prod_merc_long) / open_interest.
    """
    conn = sqlite3.connect(_REPO_ROOT / "macro_alpha_memory.db")
    df = pd.read_sql(
        "SELECT report_date, market_name, open_interest, prod_merc_long, "
        "prod_merc_short FROM cftc_cot_weekly "
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

    print(f"Resolved {len(sym_to_market)}/{len(_CFTC_MAPPING)} CFTC mappings")

    # Filter to mapped markets, compute pressure
    df = df[df["market_name"].isin(sym_to_market.values())].copy()
    df["pressure"] = (
        (df["prod_merc_short"] - df["prod_merc_long"])
        / df["open_interest"].replace(0, np.nan)
    )
    # Reverse map
    mkt_to_sym = {v: k for k, v in sym_to_market.items()}
    df["sym"] = df["market_name"].map(mkt_to_sym)

    # Monthly: take last week of month per (sym, month)
    df["m"] = df["report_date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = (df.sort_values(["sym", "m", "report_date"])
                  .groupby(["sym", "m"])["pressure"].last()
                  .unstack("sym").sort_index())
    return monthly


def _build_ls_sleeve(signal_wide: pd.DataFrame, returns_wide: pd.DataFrame,
                       q: float = 0.3) -> pd.Series:
    """Long top-q signal commodities, short bottom-q, next-month return."""
    signal_wide.index = pd.to_datetime(signal_wide.index).to_period("M").to_timestamp("M")
    returns_wide.index = pd.to_datetime(returns_wide.index).to_period("M").to_timestamp("M")
    ls = []
    allm = sorted(set(signal_wide.index) | set(returns_wide.index))
    for i in range(len(allm) - 1):
        m, nxt = allm[i], allm[i + 1]
        if m not in signal_wide.index or nxt not in returns_wide.index:
            continue
        c = signal_wide.loc[m].dropna()
        if len(c) < 6:
            continue
        # Use only commodities present in BOTH signal and next-month returns
        c = c.reindex(c.index.intersection(returns_wide.columns)).dropna()
        if len(c) < 6:
            continue
        hi = c[c >= c.quantile(1 - q)].index
        loq = c[c <= c.quantile(q)].index
        nr = returns_wide.loc[nxt]
        rl = nr.reindex(hi).dropna()
        rs = nr.reindex(loq).dropna()
        if len(rl) < 2 or len(rs) < 2:
            continue
        ls.append((nxt, float(rl.mean() - rs.mean())))
    return pd.Series(dict(ls)).sort_index().rename("cftc_pressure_ls")


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
    print("Step 1 — Build CFTC hedging pressure signal panel (monthly)...")
    pressure = build_hedging_pressure_signal_monthly()
    print(f"  panel: {pressure.shape}  range={pressure.index.min().date()} → "
          f"{pressure.index.max().date()}")
    print()

    print("Step 2 — Load deployed commodity carry's monthly returns panel...")
    from engine.validation.commodity_carry import build_carry_and_returns
    cwide, rwide = build_carry_and_returns()
    rwide.index = pd.to_datetime(rwide.index).to_period("M").to_timestamp("M")
    print(f"  returns: {rwide.shape}  range={rwide.index.min().date()} → "
          f"{rwide.index.max().date()}")
    print()

    print("Step 3 — Build CFTC hedging-pressure L/S sleeve PnL...")
    pressure_ls = _build_ls_sleeve(pressure, rwide)
    print(f"  pressure_ls: n={len(pressure_ls)}  "
          f"range={pressure_ls.index.min().date()} → "
          f"{pressure_ls.index.max().date()}  "
          f"Sharpe={_ann_sharpe(pressure_ls):+.3f}")
    print()

    print("Step 4 — Build deployed commodity carry L/S (baseline)...")
    from engine.validation.commodity_carry import build_carry_sleeve
    deployed_ls, _, _ = build_carry_sleeve()
    deployed_ls.index = pd.to_datetime(deployed_ls.index).to_period("M").to_timestamp("M")
    print(f"  deployed: n={len(deployed_ls)}  "
          f"range={deployed_ls.index.min().date()} → "
          f"{deployed_ls.index.max().date()}  "
          f"Sharpe={_ann_sharpe(deployed_ls):+.3f}")
    print()

    # Align + compare
    common = pressure_ls.index.intersection(deployed_ls.index)
    if len(common) < 36:
        print(f"Insufficient overlap: {len(common)} months. Stopping.")
        return
    p_c = pressure_ls.loc[common]
    d_c = deployed_ls.loc[common]
    corr = p_c.corr(d_c)
    print(f"Common window: {common.min().date()} → {common.max().date()} "
          f"({len(common)} mo)")
    print(f"  Sharpe(deployed):  {_ann_sharpe(d_c):+.3f}")
    print(f"  Sharpe(CFTC):      {_ann_sharpe(p_c):+.3f}")
    print(f"  corr(deployed, CFTC pressure):  {corr:+.3f}")
    print()

    # Vol-target both
    d_vt = _vol_target(d_c, 0.10)
    p_vt = _vol_target(p_c, 0.10)

    print("Step 5 — Paired enhance test (Politis-Romano 1994, B=2000, block=6mo):")
    print("Variant = (1-w) * deployed_cmdty_carry + w * cftc_pressure_ls")
    print("Baseline = deployed_cmdty_carry (the leg currently in carry sleeve)")
    print("=" * 105)
    print(f"{'weight':<8}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}"
          f"{'CI low':>11}{'CI high':>11}{'corr':>8}")
    print("-" * 105)

    results = []
    for w in [0.10, 0.20, 0.30, 0.40, 0.50]:
        variant = (1 - w) * d_vt + w * p_vt
        variant = _vol_target(variant.dropna(), 0.10)
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"cftc_pressure_blend_w{int(w*100):02d}",
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

    out_json = OUT_DIR / "cftc_hedging_pressure_results.json"
    out_json.write_text(json.dumps({
        "subject":             "cmdty_carry_leg",
        "baseline":            "deployed commodity carry (F1-F2 basis, build_carry_sleeve)",
        "variant":             "CFTC hedging pressure blend (prod_merc net / OI)",
        "method":              "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months":     int(len(common)),
        "signal_correlation":  float(corr),
        "deployed_sharpe":     float(_ann_sharpe(d_c)),
        "cftc_sharpe":         float(_ann_sharpe(p_c)),
        "results":             results,
        "academic_anchor":     "Bhardwaj-Gorton 2014, Hirshleifer 1990",
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
