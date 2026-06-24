"""scripts/audit_commodity_curve_depth_vs_deployed.py — first new substrate experiment.

Tests whether a FULL-CURVE convenience yield carry signal (F1 vs deepest
available contract F_last) produces a meaningfully different signal than
the deployed near-next basis carry (F1 vs F2), and whether blending the
two improves the deployed cross_asset_carry sleeve via paired bootstrap.

Mechanism
=========
Deployed commodity carry uses (F1-F2)/F2 annualized — near-next basis.
Convenience yield (Hull-McDonald) is more fully captured by the slope of
the ENTIRE forward curve, not just the front two contracts. Koijen 2018
§commodity reports that long-dated forward curve shape contains
predictability beyond the near basis.

Variant construction
====================
1. Re-use existing per-commodity multi-contract data from
   _cmdty_settle.parquet (already cached, 4.2M rows, 9k contracts)
2. For each (commodity, date), compute carry_depth = (F1 - F_last) / F_last
   annualized by months-to-expiry of F_last
3. Build L/S sleeve sorted on carry_depth (mirroring build_carry_sleeve
   from engine.validation.commodity_carry)
4. Paired bootstrap test:
   variant = blend of deployed commodity carry leg + depth signal sleeve

This is the FIRST substrate-expansion experiment in the session — gives
a real test of whether NEW signal construction on existing data can
break the 0/18 PROMOTE pattern.

Run:
    python scripts/audit_commodity_curve_depth_vs_deployed.py
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

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "cmdty_curve_depth_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENHANCE_LOG = OUT_DIR / "enhance_verdict_log.jsonl"


def _build_curve_depth_signal_and_returns():
    """Build (depth_carry_wide, returns_wide) using F1 vs F_LAST instead of F1-F2.

    Mirrors engine.validation.commodity_carry.build_carry_and_returns
    but computes carry as the slope from F1 to the deepest available
    contract for that (sym, date), annualized over the full gap.
    """
    from engine.validation.commodity_carry import (
        fetch_commodity_futures, COMMODITIES,
    )
    contracts, prices = fetch_commodity_futures()
    prices = prices.drop_duplicates(["futcode", "date_"])
    contracts = contracts.dropna(subset=["lasttrddate"]).copy()
    contracts["sym"] = contracts["clscode"].map(COMMODITIES)
    px = prices.merge(
        contracts[["futcode", "sym", "lasttrddate"]], on="futcode", how="inner",
    )
    px = px[px["settlement"] > 0]
    # Keep only contracts not yet expired at date_
    px = px[px["lasttrddate"] > px["date_"]].sort_values(
        ["sym", "date_", "lasttrddate"],
    )
    px["rank"] = px.groupby(["sym", "date_"]).cumcount()

    # F1 = rank=0
    f1 = px[px["rank"] == 0][
        ["sym", "date_", "futcode", "settlement", "lasttrddate"]
    ].rename(columns={"settlement": "f1_px", "lasttrddate": "f1_exp",
                       "futcode": "front_fut"})

    # F_last = max rank per (sym, date) — the deepest available contract
    # Use idxmax on lasttrddate
    flast_idx = px.groupby(["sym", "date_"])["lasttrddate"].idxmax()
    flast = px.loc[flast_idx][
        ["sym", "date_", "settlement", "lasttrddate"]
    ].rename(columns={"settlement": "flast_px", "lasttrddate": "flast_exp"})

    # Merge
    m = f1.merge(flast, on=["sym", "date_"])
    gap_days = (m["flast_exp"] - m["f1_exp"]).dt.days
    m = m[(gap_days > 30) & (m["flast_px"] > 0) & (m["f1_px"] > 0)]
    # Annualize by total gap
    m["carry_depth"] = (m["f1_px"] - m["flast_px"]) / m["flast_px"] * (
        365.0 / (m["flast_exp"] - m["f1_exp"]).dt.days
    )

    # Daily front return + monthly compounded (mirror existing carry code)
    fr = m.sort_values(["sym", "date_"])
    fr["ret"] = fr.groupby("sym")["f1_px"].pct_change()
    rolled = fr.groupby("sym")["front_fut"].shift(1) != fr["front_fut"]
    fr.loc[rolled, "ret"] = np.nan
    fr = fr[fr["ret"].abs() < 0.5]

    fr["m"] = fr["date_"].dt.to_period("M").dt.to_timestamp("M")
    cwide_depth = fr.groupby(["m", "sym"])["carry_depth"].last().unstack("sym").sort_index()
    rwide = (fr.set_index("date_").groupby("sym")["ret"]
             .apply(lambda x: (1 + x).resample("ME").prod() - 1)
             .unstack("sym").sort_index())
    return cwide_depth, rwide


def _build_ls_from_signal(cwide, rwide, q: float = 0.3) -> pd.Series:
    """Cross-sectional L/S sleeve from a carry signal panel.

    Long top-q signal commodities, short bottom-q, next-month return.
    Mirrors build_carry_sleeve in commodity_carry.
    """
    ls = []
    allm = sorted(set(cwide.index) | set(rwide.index))
    for i in range(len(allm) - 1):
        m, nxt = allm[i], allm[i + 1]
        if m not in cwide.index or nxt not in rwide.index:
            continue
        c = cwide.loc[m].dropna()
        if len(c) < 8:
            continue
        hi = c[c >= c.quantile(1 - q)].index
        loq = c[c <= c.quantile(q)].index
        nr = rwide.loc[nxt]
        rl = nr.reindex(hi).dropna()
        rs = nr.reindex(loq).dropna()
        if len(rl) < 2 or len(rs) < 2:
            continue
        ls.append((nxt, float(rl.mean() - rs.mean())))
    return pd.Series(dict(ls)).sort_index().rename("cmdty_curve_depth_ls")


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
    print("Building deployed commodity carry (F1-F2 basis)...")
    from engine.validation.commodity_carry import build_carry_sleeve
    deployed_ls, _, _ = build_carry_sleeve()
    deployed_ls.index = pd.to_datetime(deployed_ls.index).to_period("M").to_timestamp("M")
    print(f"  deployed: n={len(deployed_ls)} "
          f"range={deployed_ls.index.min().date()} → "
          f"{deployed_ls.index.max().date()} "
          f"Sharpe={_ann_sharpe(deployed_ls):+.3f}")

    print("Building NEW curve-depth carry (F1-F_last full curve)...")
    cwide_depth, rwide = _build_curve_depth_signal_and_returns()
    depth_ls = _build_ls_from_signal(cwide_depth, rwide)
    depth_ls.index = pd.to_datetime(depth_ls.index).to_period("M").to_timestamp("M")
    print(f"  depth:    n={len(depth_ls)} "
          f"range={depth_ls.index.min().date()} → "
          f"{depth_ls.index.max().date()} "
          f"Sharpe={_ann_sharpe(depth_ls):+.3f}")

    # Compare the two signals
    common = deployed_ls.index.intersection(depth_ls.index)
    if len(common) < 36:
        print(f"\nInsufficient overlap: {len(common)} months. Stopping.")
        return
    dep_c = deployed_ls.loc[common]
    dep_d = depth_ls.loc[common]
    corr = dep_c.corr(dep_d)
    print()
    print(f"Common window: {common.min().date()} → {common.max().date()} "
          f"({len(common)} mo)")
    print(f"  Sharpe(deployed F1-F2): {_ann_sharpe(dep_c):+.3f}")
    print(f"  Sharpe(curve depth):    {_ann_sharpe(dep_d):+.3f}")
    print(f"  corr(deployed, depth):  {corr:.3f}")
    print()

    # Paired enhance test — blend deployed + depth at multiple weights
    # Vol-target both to 10% so blend weights are interpretable
    dep_vt = _vol_target(dep_c, 0.10)
    dep_d_vt = _vol_target(dep_d, 0.10)

    print("Paired enhance test (Politis-Romano 1994, B=2000, block=6mo):")
    print("Variant = (1-w) * deployed_cmdty_carry + w * curve_depth_cmdty_carry")
    print("Baseline = deployed_cmdty_carry alone (the leg currently in carry sleeve)")
    print("=" * 105)
    print(f"{'weight':<8}{'verdict':<14}{'ΔSharpe':>10}{'t-stat':>9}{'p':>8}"
          f"{'CI low':>11}{'CI high':>11}{'corr':>8}")
    print("-" * 105)

    results = []
    for w in [0.10, 0.20, 0.30, 0.40, 0.50]:
        variant = (1 - w) * dep_vt + w * dep_d_vt
        variant = _vol_target(variant.dropna(), 0.10)
        r = dispatch_enhance_hypothesis(
            hypothesis_id    = f"cmdty_curve_depth_blend_w{int(w*100):02d}",
            sleeve_id        = "cmdty_carry_leg",
            variant_returns  = variant,
            baseline_returns = dep_vt,
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

    out_json = OUT_DIR / "cmdty_curve_depth_results.json"
    out_json.write_text(json.dumps({
        "subject":             "cmdty_carry_leg",
        "baseline":            "deployed commodity carry (F1-F2 basis, build_carry_sleeve)",
        "variant":             "curve depth blend (F1 vs F_last full curve)",
        "method":              "Politis-Romano 1994 paired circular block bootstrap, B=2000, block=6mo",
        "n_paired_months":     int(len(common)),
        "signal_correlation":  float(corr),
        "deployed_sharpe":     float(_ann_sharpe(dep_c)),
        "depth_sharpe":        float(_ann_sharpe(dep_d)),
        "results":             results,
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
