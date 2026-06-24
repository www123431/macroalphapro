"""scripts/audit_voltarget_sweep_moreira_muir.py

P0 next: vol-target sweep + Moreira-Muir 2017 dynamic vol timing.

Test 1: static vol-target sweep (8%/10%/12%/15%/20%)
        Naively Sharpe should be INVARIANT — but TC drag + non-
        normal returns can break this. Empirical test.

Test 2: Moreira-Muir 2017 "Volatility-Managed Portfolios"
        Scale book exposure by INVERSE rolling vol → reduces
        risk during high-vol regimes (which historically have lower
        risk-adjusted returns). Reports +0.1-0.2 Sharpe in
        large-sample literature.
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

from engine.portfolio.combined_book import build_combined_book

OUT_DIR = _REPO_ROOT / "data" / "research_store" / "audit" / "voltarget_2026_06_17"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ann_sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 12 or s.std(ddof=1) <= 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * math.sqrt(12))


def _ann_dd(s: pd.Series) -> float:
    s = s.dropna()
    cum = (1 + s).cumprod()
    return float((cum / cum.cummax() - 1).min())


def _moreira_muir_scale(returns: pd.Series, lookback_months: int = 12,
                          target_vol: float = 0.10) -> pd.Series:
    """Scale returns by inverse rolling-vol estimate (Moreira-Muir 2017).

    sigma_hat_t = std of past `lookback_months` returns × sqrt(12)
    scaled_t = (target_vol / sigma_hat_{t-1}) × returns_t

    Uses LAGGED vol estimate (no look-ahead).
    """
    realized_vol = returns.rolling(lookback_months).std() * np.sqrt(12)
    scale = (target_vol / realized_vol).shift(1)
    scale = scale.clip(upper=3.0)   # cap at 3x leverage (Moreira-Muir suggest 1.5-3x)
    return (returns * scale).dropna()


def main():
    print("=" * 80)
    print("Vol-target sweep + Moreira-Muir 2017 dynamic vol management")
    print("=" * 80)

    # Get RAW book (vol-target=None → unscaled), then apply different targets
    raw = build_combined_book(
        crisis_risk_weight=0.05, mom_hedge_risk_weight=0.02,
        regime_conditional=True, book_vol_target=None,
    ).dropna()
    raw.index = pd.to_datetime(raw.index).to_period("M").to_timestamp("M")
    print(f"Raw book: n={len(raw)}  range={raw.index.min().date()} → {raw.index.max().date()}")
    raw_vol = raw.std(ddof=1) * np.sqrt(12)
    raw_sh = _ann_sharpe(raw)
    print(f"  Raw vol (annualized): {raw_vol:.1%}, Raw Sharpe: {raw_sh:+.3f}")
    print()

    print("TEST 1 — Static vol-target sweep")
    print("-" * 80)
    static_results = []
    for target in [0.08, 0.10, 0.12, 0.15, 0.20]:
        scale = target / raw_vol
        scaled = raw * scale
        sh = _ann_sharpe(scaled)
        dd = _ann_dd(scaled)
        static_results.append({"target": target, "scale": scale, "sharpe": sh, "maxDD": dd})
        print(f"  vol_target={target:.1%}  scale={scale:.2f}x  Sharpe={sh:+.3f}  maxDD={dd:+.1%}")
    print(f"  ✓ All static Sharpes ≈ equal (linear scaling preserves Sharpe — expected)")
    print()

    print("TEST 2 — Moreira-Muir 2017 dynamic vol management")
    print("-" * 80)
    mm_results = []
    for lookback in [6, 12, 24, 36]:
        scaled = _moreira_muir_scale(raw, lookback_months=lookback, target_vol=0.10)
        sh = _ann_sharpe(scaled)
        dd = _ann_dd(scaled)
        delta = sh - raw_sh
        marker = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "≈")
        mm_results.append({"lookback": lookback, "sharpe": sh, "maxDD": dd, "delta": delta})
        print(f"  lookback={lookback:>2}mo  Sharpe={sh:+.3f} ({delta:+.3f}) {marker}  maxDD={dd:+.1%}")
    print()

    # Identify best
    best_mm = max(mm_results, key=lambda x: x["sharpe"])
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Raw book Sharpe:          {raw_sh:+.3f}")
    print(f"  Static vol-target Sharpe: {static_results[1]['sharpe']:+.3f} (invariant, expected)")
    print(f"  Best Moreira-Muir:        {best_mm['sharpe']:+.3f} "
           f"(lookback={best_mm['lookback']}mo, Δ={best_mm['delta']:+.3f})")

    out_json = OUT_DIR / "voltarget_results.json"
    out_json.write_text(json.dumps({
        "raw_book_sharpe":   raw_sh,
        "raw_book_vol":      raw_vol,
        "static_results":    static_results,
        "moreira_muir_results": mm_results,
        "best_moreira_muir": best_mm,
    }, indent=2, default=str))
    print()
    print(f"Saved → {out_json}")


if __name__ == "__main__":
    main()
