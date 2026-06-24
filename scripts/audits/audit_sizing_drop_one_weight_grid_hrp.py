"""scripts/audit_sizing_drop_one_weight_grid_hrp.py

P0 Sizing-side audit per [[feedback-sizing-before-signal-2026-06-17]].
Doesn't add new alpha; tests if deployed book SIZING is optimal at all.

Four tests:
  1. Standalone Sharpe per sleeve (sanity)
  2. Drop-one: book Sharpe with each sleeve removed
  3. Weight grid: 5×5×3 over (equity, carry, tsmom) holding
     hedges at deployed weights
  4. HRP allocation (López de Prado 2016) for comparison

Output: ranked configurations + best vs deployed baseline.
"""
from __future__ import annotations

import json
import math
import sys
from itertools import product
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from engine.portfolio.combined_book import (
    build_equity_book, build_carry_book, build_tsmom_book,
    build_crisis_hedge_book, build_mom_hedge_book,
    build_vix_regime_monthly, scale_to_book_vol,
)

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "sizing_audit_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ann_sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 12 or s.std(ddof=1) <= 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * math.sqrt(12))


def _ann_dd(s: pd.Series) -> float:
    """Max drawdown of compounded PnL."""
    s = s.dropna()
    if len(s) < 12:
        return float("nan")
    cum = (1 + s).cumprod()
    peak = cum.cummax()
    return float((cum / peak - 1).min())


def _vol_target(s: pd.Series, target_ann: float = 0.10) -> pd.Series:
    s = s.dropna()
    v = s.std(ddof=1) * math.sqrt(12)
    if not math.isfinite(v) or v <= 0:
        return s
    return s * (target_ann / v)


def _build_book(weights: dict[str, float], sleeves: dict[str, pd.Series],
                   target_vol: float = 0.10) -> pd.Series:
    """Mix vol-targeted sleeves by weights, re-vol-target to book level."""
    # Each sleeve already vol-targeted to 10%; blend at given weights
    J = pd.concat(sleeves, axis=1).dropna()
    w = pd.Series(weights).reindex(J.columns).fillna(0.0)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum   # normalize to 1.0 (so removing a sleeve redistributes)
    book = (J * w).sum(axis=1)
    return _vol_target(book, target_vol)


def _hrp_weights(sleeves: dict[str, pd.Series]) -> dict[str, float]:
    """Hierarchical Risk Parity weights (López de Prado 2016).
    Simple implementation: bisect by cluster, allocate inversely by vol.
    """
    J = pd.concat(sleeves, axis=1).dropna()
    cov = J.cov()
    corr = J.corr()
    # Distance via correlation
    dist = ((1 - corr) / 2.0) ** 0.5
    # Single-linkage clustering
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    link = linkage(squareform(dist.values, checks=False), method="single")
    order = leaves_list(link)
    cols = [J.columns[i] for i in order]

    # Recursive bisection
    w = pd.Series(1.0, index=cols)
    clusters = [cols]
    while clusters:
        new_clusters = []
        for c in clusters:
            if len(c) <= 1:
                continue
            mid = len(c) // 2
            left, right = c[:mid], c[mid:]
            cov_L = cov.loc[left, left]
            cov_R = cov.loc[right, right]
            # Inverse-variance within each cluster
            iv_L = 1.0 / np.diag(cov_L.values)
            iv_L /= iv_L.sum()
            iv_R = 1.0 / np.diag(cov_R.values)
            iv_R /= iv_R.sum()
            var_L = iv_L @ cov_L.values @ iv_L
            var_R = iv_R @ cov_R.values @ iv_R
            alpha = 1.0 - var_L / (var_L + var_R)
            for s in left:  w[s] *= alpha
            for s in right: w[s] *= (1 - alpha)
            new_clusters.append(left)
            new_clusters.append(right)
        clusters = new_clusters
    return {c: float(w[c]) for c in J.columns}


def main():
    print("=" * 80)
    print("P0 SIZING-SIDE AUDIT — drop-one / weight grid / HRP")
    print("=" * 80)

    # 1. Build all 5 sleeves, vol-target each to 10%
    print("Building 5 sleeves (vol-target 10% each)...")
    raw = {
        "equity":     build_equity_book(),
        "carry":      build_carry_book(),
        "tsmom":      build_tsmom_book(),
        "crisis":     build_crisis_hedge_book(),
        "mom_hedge":  build_mom_hedge_book(),
    }
    sleeves: dict[str, pd.Series] = {}
    print()
    print("Standalone Sharpe (sanity check):")
    for name, s in raw.items():
        s = s.dropna()
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        vt = _vol_target(s, 0.10)
        sleeves[name] = vt
        sh = _ann_sharpe(vt)
        dd = _ann_dd(vt)
        print(f"  {name:<12} Sharpe={sh:+.3f}  maxDD={dd:.1%}  n={len(vt)}")
    print()

    # Common overlap
    J = pd.concat(sleeves, axis=1).dropna()
    print(f"Common overlap: {J.index.min().date()} → {J.index.max().date()} ({len(J)} mo)")
    print()

    # Deployed weights
    DEPLOYED = {"equity": 0.63, "carry": 0.25, "tsmom": 0.05,
                 "crisis": 0.05, "mom_hedge": 0.02}
    book_deployed = _build_book(DEPLOYED, sleeves, 0.10)
    sh_dep = _ann_sharpe(book_deployed)
    dd_dep = _ann_dd(book_deployed)
    print(f"DEPLOYED ({DEPLOYED}): Sharpe={sh_dep:+.3f}  maxDD={dd_dep:.1%}")
    print()

    # 2. Drop-one test
    print("=" * 80)
    print("TEST 1 — DROP-ONE (zero a sleeve's weight, redistribute pro-rata)")
    print("=" * 80)
    drop_results = []
    for drop_name in sleeves.keys():
        w = {k: v for k, v in DEPLOYED.items() if k != drop_name}
        # Pro-rata redistribute the dropped weight
        # (so total stays at 1.0 across remaining sleeves — vol target will rescale anyway)
        book = _build_book(w, sleeves, 0.10)
        sh = _ann_sharpe(book)
        dd = _ann_dd(book)
        delta = sh - sh_dep
        marker = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
        print(f"  drop {drop_name:<12} → Sharpe={sh:+.3f} ({delta:+.3f}) {marker}  "
              f"maxDD={dd:.1%}")
        drop_results.append({
            "dropped": drop_name, "sharpe": sh, "delta": delta, "maxDD": dd,
        })
    print()

    # Drop-two test
    print("=" * 80)
    print("TEST 1b — DROP-TWO (combined insurance leg removal)")
    print("=" * 80)
    drop_combos = [
        ("crisis + mom_hedge", ["crisis", "mom_hedge"]),
        ("crisis only",         ["crisis"]),
        ("mom_hedge only",     ["mom_hedge"]),
        ("tsmom + crisis + mom_hedge (all small)", ["tsmom", "crisis", "mom_hedge"]),
    ]
    for label, drops in drop_combos:
        w = {k: v for k, v in DEPLOYED.items() if k not in drops}
        book = _build_book(w, sleeves, 0.10)
        sh = _ann_sharpe(book)
        dd = _ann_dd(book)
        delta = sh - sh_dep
        marker = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
        print(f"  drop [{label:<40}] → Sharpe={sh:+.3f} ({delta:+.3f}) {marker}  maxDD={dd:.1%}")
    print()

    # 3. Weight grid search (over equity, carry, tsmom; keep hedges fixed)
    print("=" * 80)
    print("TEST 2 — WEIGHT GRID 5×5×3 (equity / carry / tsmom; hedges fixed)")
    print("=" * 80)
    EQUITY_GRID = [0.50, 0.60, 0.70, 0.80, 0.90]
    CARRY_GRID  = [0.10, 0.20, 0.30, 0.40, 0.50]
    TSMOM_GRID  = [0.00, 0.05, 0.15]
    grid = []
    for eq, cr, ts in product(EQUITY_GRID, CARRY_GRID, TSMOM_GRID):
        # Hedges fixed at deployed (5% + 2% = 7%)
        # Normalize main legs to sum to 0.93
        main_sum = eq + cr + ts
        if main_sum <= 0:
            continue
        w = {
            "equity":    0.93 * eq / main_sum,
            "carry":     0.93 * cr / main_sum,
            "tsmom":     0.93 * ts / main_sum,
            "crisis":    0.05,
            "mom_hedge": 0.02,
        }
        book = _build_book(w, sleeves, 0.10)
        sh = _ann_sharpe(book)
        dd = _ann_dd(book)
        grid.append({"eq": eq, "cr": cr, "ts": ts, "sharpe": sh, "maxDD": dd,
                      "delta": sh - sh_dep})
    grid_sorted = sorted(grid, key=lambda x: -x["sharpe"])
    print(f"  Top 10 configurations (sorted by Sharpe):")
    print(f"  {'eq':<6}{'cr':<6}{'ts':<6}{'Sharpe':>10}{'Δ vs deployed':>16}{'maxDD':>10}")
    for r in grid_sorted[:10]:
        print(f"  {r['eq']:<6.2f}{r['cr']:<6.2f}{r['ts']:<6.2f}"
              f"{r['sharpe']:>+10.3f}{r['delta']:>+16.3f}{r['maxDD']:>+10.1%}")
    print()

    # 4. HRP allocation
    print("=" * 80)
    print("TEST 3 — HRP allocation (López de Prado 2016)")
    print("=" * 80)
    try:
        hrp_w = _hrp_weights(sleeves)
        # Normalize to sum 1
        s = sum(hrp_w.values())
        hrp_w = {k: v / s for k, v in hrp_w.items()}
        book_hrp = _build_book(hrp_w, sleeves, 0.10)
        sh_hrp = _ann_sharpe(book_hrp)
        dd_hrp = _ann_dd(book_hrp)
        print(f"  HRP weights:")
        for k, v in hrp_w.items():
            dep_w = DEPLOYED.get(k, 0)
            print(f"    {k:<12}: HRP={v:.3f}   (deployed {dep_w:.3f}, Δ {v-dep_w:+.3f})")
        print(f"  HRP book Sharpe = {sh_hrp:+.3f} ({sh_hrp - sh_dep:+.3f} vs deployed)  maxDD={dd_hrp:.1%}")
    except Exception as exc:
        print(f"  HRP failed: {exc}")
        sh_hrp = None
        hrp_w = None

    # 5. Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    best_grid = grid_sorted[0]
    print(f"  Deployed Sharpe:          {sh_dep:+.3f}")
    print(f"  Best drop-one Sharpe:    {max(r['sharpe'] for r in drop_results):+.3f}")
    print(f"  Best weight-grid Sharpe: {best_grid['sharpe']:+.3f}  "
           f"(eq={best_grid['eq']:.2f}, cr={best_grid['cr']:.2f}, ts={best_grid['ts']:.2f})")
    if sh_hrp is not None:
        print(f"  HRP Sharpe:               {sh_hrp:+.3f}")

    # Save
    out_json = OUT_DIR / "sizing_audit_results.json"
    out_json.write_text(json.dumps({
        "deployed_weights":     DEPLOYED,
        "deployed_sharpe":      sh_dep,
        "deployed_maxDD":       dd_dep,
        "standalone_sharpes":   {k: _ann_sharpe(v) for k, v in sleeves.items()},
        "drop_one_results":     drop_results,
        "grid_top10":           grid_sorted[:10],
        "hrp_weights":          hrp_w,
        "hrp_sharpe":           sh_hrp,
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
