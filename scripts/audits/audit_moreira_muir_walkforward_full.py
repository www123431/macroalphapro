"""scripts/audit_moreira_muir_walkforward_full.py

Walk-forward validation of Moreira-Muir 2017 vol-managed book.

Tests on multiple book definitions for robustness:
  A. Full 5-sleeve deployed book (97mo overlap)
  B. Core 3-sleeve book (equity + carry + tsmom only, longer history)
  C. Equity-only book (longest history)

Plus:
  - Sub-period stability (first half vs second half of each sample)
  - TC drag from monthly leverage rebalancing (5% annualized borrow cost
    on excess leverage above 1.0x)
  - Leverage usage statistics (how often is 3x cap binding?)

Reports whether +0.244 Sharpe lift in 97mo audit holds out-of-sample.
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

from engine.portfolio.combined_book import (
    build_equity_book, build_carry_book, build_tsmom_book,
    build_combined_book,
)

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "mm_walkforward_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ann_sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 12 or s.std(ddof=1) <= 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * math.sqrt(12))


def _ann_dd(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 12:
        return float("nan")
    cum = (1 + s).cumprod()
    return float((cum / cum.cummax() - 1).min())


def _vol_target(s: pd.Series, target_ann: float = 0.10) -> pd.Series:
    s = s.dropna()
    v = s.std(ddof=1) * math.sqrt(12)
    if not math.isfinite(v) or v <= 0:
        return s
    return s * (target_ann / v)


def moreira_muir(returns: pd.Series, lookback_months: int = 12,
                  target_vol: float = 0.10, leverage_cap: float = 3.0,
                  borrow_rate: float = 0.05) -> dict:
    """Walk-forward Moreira-Muir 2017 with TC drag estimate.

    Returns dict with:
      scaled_returns: vol-managed PnL series
      scale_series:   the per-month scaling factor (lagged)
      tc_drag:        per-month borrow cost on excess leverage
      gross_sharpe:   Sharpe WITHOUT TC drag (pure vol-management effect)
      net_sharpe:     Sharpe WITH TC drag (realistic)
    """
    realized_vol = returns.rolling(lookback_months).std() * np.sqrt(12)
    raw_scale = target_vol / realized_vol
    scale = raw_scale.shift(1).clip(upper=leverage_cap).dropna()
    aligned_returns = returns.loc[scale.index]
    gross = (scale * aligned_returns).dropna()

    # TC drag: borrow cost on excess leverage (scale > 1.0)
    excess_lev = (scale - 1.0).clip(lower=0)
    tc_drag = excess_lev * (borrow_rate / 12)   # monthly
    net = gross - tc_drag

    return {
        "scaled_returns": net,
        "gross_returns":  gross,
        "scale_series":   scale,
        "tc_drag":        tc_drag,
        "gross_sharpe":   _ann_sharpe(gross),
        "net_sharpe":     _ann_sharpe(net),
        "max_dd_net":     _ann_dd(net),
        "max_dd_gross":   _ann_dd(gross),
        "avg_leverage":   float(scale.mean()),
        "median_leverage": float(scale.median()),
        "cap_binding_pct": float((scale >= leverage_cap * 0.99).mean()),
    }


def evaluate_book(name: str, raw: pd.Series, vol_target: float = 0.10):
    """Evaluate static vs Moreira-Muir on a book PnL series."""
    raw = raw.dropna()
    raw.index = pd.to_datetime(raw.index).to_period("M").to_timestamp("M")

    # Static vol-target baseline
    static = _vol_target(raw, vol_target)
    static_sh = _ann_sharpe(static)
    static_dd = _ann_dd(static)

    # Moreira-Muir at multiple lookbacks
    mm_results = {}
    for lookback in [6, 12, 24, 36]:
        mm = moreira_muir(static, lookback_months=lookback,
                            target_vol=vol_target, leverage_cap=3.0,
                            borrow_rate=0.05)
        mm_results[lookback] = mm

    # Sub-period stability — split sample in half, test 12mo MM
    mid = len(static) // 2
    first_half = static.iloc[:mid]
    second_half = static.iloc[mid:]
    fh = moreira_muir(first_half, 12, vol_target, 3.0, 0.05)
    sh2 = moreira_muir(second_half, 12, vol_target, 3.0, 0.05)
    fh_static_sh = _ann_sharpe(first_half)
    sh2_static_sh = _ann_sharpe(second_half)

    return {
        "name":             name,
        "n_months":         len(static),
        "window":           (static.index.min().date().isoformat(),
                              static.index.max().date().isoformat()),
        "static_sharpe":    static_sh,
        "static_dd":        static_dd,
        "mm_results":       mm_results,
        "subperiod": {
            "first_half_static_sharpe": fh_static_sh,
            "first_half_mm12_sharpe":   fh["net_sharpe"],
            "first_half_n":              len(first_half),
            "second_half_static_sharpe": sh2_static_sh,
            "second_half_mm12_sharpe":   sh2["net_sharpe"],
            "second_half_n":             len(second_half),
        },
    }


def main():
    print("=" * 80)
    print("Moreira-Muir 2017 walk-forward validation")
    print("=" * 80)
    print()

    # A. Full 5-sleeve deployed book
    print("Building deployed combined book (5-sleeve regime-conditional)...")
    book_full = build_combined_book(
        crisis_risk_weight=0.05, mom_hedge_risk_weight=0.02,
        regime_conditional=True, book_vol_target=None,
    ).dropna()
    res_A = evaluate_book("A. Full 5-sleeve book", book_full)

    # B. Core 3-sleeve book (longer history)
    print("Building 3-sleeve core book (equity + carry + tsmom)...")
    eq = build_equity_book().dropna()
    cr = build_carry_book().dropna()
    ts = build_tsmom_book().dropna()
    J = pd.concat({"e": eq, "c": cr, "t": ts}, axis=1).dropna()
    # Use deployed-style risk weights (excluding hedges)
    # equity 70 / carry 25 / tsmom 5 (renormalized)
    w = pd.Series({"e": 0.70, "c": 0.25, "t": 0.05})
    J_vt = J.apply(lambda x: _vol_target(x, 0.10))
    book_3 = (J_vt * w).sum(axis=1)
    res_B = evaluate_book("B. 3-sleeve core book", book_3)

    # C. Equity-only baseline
    print("Building equity-only baseline...")
    res_C = evaluate_book("C. Equity-only", eq)

    print()
    for res in [res_A, res_B, res_C]:
        print()
        print("=" * 80)
        print(f"{res['name']}")
        print("=" * 80)
        print(f"  window: {res['window'][0]} → {res['window'][1]}  "
               f"n={res['n_months']} months")
        print(f"  Static vol-target Sharpe: {res['static_sharpe']:+.3f}  "
               f"maxDD: {res['static_dd']:+.1%}")
        print(f"  Moreira-Muir results (NET of borrow cost):")
        for lb in [6, 12, 24, 36]:
            mm = res["mm_results"][lb]
            delta = mm["net_sharpe"] - res["static_sharpe"]
            marker = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "≈")
            print(f"    lookback {lb:>2}mo: "
                   f"net Sharpe={mm['net_sharpe']:+.3f} "
                   f"(gross {mm['gross_sharpe']:+.3f}, Δ vs static: {delta:+.3f}) {marker}  "
                   f"DD={mm['max_dd_net']:+.1%}  "
                   f"avg_lev={mm['avg_leverage']:.2f}x  "
                   f"cap_binding={mm['cap_binding_pct']:.0%}")
        print()
        print(f"  SUB-PERIOD STABILITY (12mo lookback):")
        sp = res["subperiod"]
        print(f"    First half  (n={sp['first_half_n']}): "
               f"static={sp['first_half_static_sharpe']:+.3f}  "
               f"mm12={sp['first_half_mm12_sharpe']:+.3f}  "
               f"Δ={sp['first_half_mm12_sharpe']-sp['first_half_static_sharpe']:+.3f}")
        print(f"    Second half (n={sp['second_half_n']}): "
               f"static={sp['second_half_static_sharpe']:+.3f}  "
               f"mm12={sp['second_half_mm12_sharpe']:+.3f}  "
               f"Δ={sp['second_half_mm12_sharpe']-sp['second_half_static_sharpe']:+.3f}")

    # Summary
    print()
    print("=" * 80)
    print("WALK-FORWARD VERDICT")
    print("=" * 80)
    for res in [res_A, res_B, res_C]:
        mm12_delta = res["mm_results"][12]["net_sharpe"] - res["static_sharpe"]
        mm36_delta = res["mm_results"][36]["net_sharpe"] - res["static_sharpe"]
        print(f"  {res['name']:<28}: 12mo Δ={mm12_delta:+.3f}  36mo Δ={mm36_delta:+.3f}  "
               f"(n={res['n_months']})")

    # Save
    out_json = OUT_DIR / "moreira_muir_walkforward_results.json"
    out_json.write_text(json.dumps({
        "results": [
            {k: v for k, v in res.items() if k != "mm_results"} for res in [res_A, res_B, res_C]
        ],
        "mm_details": {
            res["name"]: {
                str(lb): {kk: vv for kk, vv in mm.items()
                          if kk not in ("scaled_returns", "gross_returns", "scale_series", "tc_drag")}
                for lb, mm in res["mm_results"].items()
            }
            for res in [res_A, res_B, res_C]
        },
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
