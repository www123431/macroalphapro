"""Compare 3 deploy configurations side-by-side:
  A. 3-sleeve baseline (no hedge)
  B. 5-sleeve STATIC hedged (crisis 5% + mom_hedge 2%)
  C. 5-sleeve REGIME-CONDITIONAL hedged (per-regime allocation grids)

Uses cached VIX as regime signal (FRED BAA10Y timed out in this session;
VIX alone is the dominant component of vix_oas composite per AN-1 spec).

USAGE: python scripts/compare_static_vs_regime_hedged_book.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from engine.portfolio.combined_book import (
    DEFAULT_BOOK_VOL_TARGET,
    DEFAULT_CARRY_RISK_WEIGHT,
    DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
    DEFAULT_MOM_HEDGE_RISK_WEIGHT,
    DEFAULT_TARGET_VOL,
    DEFAULT_TSMOM_RISK_WEIGHT,
    blend_five,
    book_stats,
    build_carry_book,
    build_combined_book,
    build_crisis_hedge_book,
    build_equity_book,
    build_mom_hedge_book,
    build_tsmom_book,
    scale_to_book_vol,
    voltarget,
)

# Regime grids per spec patterns (CALM less hedge, STRESS more hedge)
REGIME_GRIDS = {
    "CALM":   {"crisis": 0.00, "mom_hedge": 0.00},
    "NORMAL": {"crisis": 0.05, "mom_hedge": 0.02},
    "STRESS": {"crisis": 0.10, "mom_hedge": 0.05},
}

VIX_Z_WINDOW = 252   # 1-year rolling z for the regime signal
REGIME_THRESHOLD = 1.0


def build_vix_regime_monthly() -> pd.Series:
    """Classify monthly regime via VIX rolling z-score (1y).
    CALM if z < -1, STRESS if z > +1, NORMAL otherwise."""
    df = pd.read_parquet("data/cache/_vix_spx_daily.parquet")
    df.index = pd.to_datetime(df.index)
    vix = df["VIX"].dropna()
    med = vix.rolling(VIX_Z_WINDOW, min_periods=VIX_Z_WINDOW).median()
    std = vix.rolling(VIX_Z_WINDOW, min_periods=VIX_Z_WINDOW).std()
    z = ((vix - med) / std).dropna()
    monthly_z = z.resample("ME").last()

    def _classify(z):
        if pd.isna(z): return "NORMAL"
        if z > REGIME_THRESHOLD: return "STRESS"
        if z < -REGIME_THRESHOLD: return "CALM"
        return "NORMAL"
    return monthly_z.apply(_classify)


def build_regime_conditional_book(target_vol: float = DEFAULT_TARGET_VOL,
                                       book_vol_target: float = DEFAULT_BOOK_VOL_TARGET):
    """Build the regime-conditional 5-sleeve book: per-month, look up
    the active regime and apply that regime's hedge weights."""
    eq_vt = voltarget(build_equity_book(), target_vol)
    cy_vt = voltarget(build_carry_book(), target_vol)
    ts_vt = voltarget(build_tsmom_book(), target_vol)
    crisis_vt = voltarget(build_crisis_hedge_book(), target_vol)
    momh_vt = voltarget(build_mom_hedge_book(), target_vol)
    J = pd.concat([eq_vt.rename("e"), cy_vt.rename("c"), ts_vt.rename("t"),
                       crisis_vt.rename("h"), momh_vt.rename("m")], axis=1).dropna()

    regime = build_vix_regime_monthly()
    book = pd.Series(index=J.index, dtype=float, name="book")
    for t in J.index:
        # Find nearest regime label <= t
        nearest = regime.index[regime.index <= t]
        r = "NORMAL" if len(nearest) == 0 else regime.loc[nearest[-1]]
        g = REGIME_GRIDS[r]
        crisis_w = g["crisis"]
        mom_w = g["mom_hedge"]
        eq_w = (1.0 - DEFAULT_CARRY_RISK_WEIGHT - DEFAULT_TSMOM_RISK_WEIGHT
                  - crisis_w - mom_w)
        book.loc[t] = (eq_w * J.loc[t, "e"]
                          + DEFAULT_CARRY_RISK_WEIGHT * J.loc[t, "c"]
                          + DEFAULT_TSMOM_RISK_WEIGHT * J.loc[t, "t"]
                          + crisis_w * J.loc[t, "h"]
                          + mom_w * J.loc[t, "m"])
    book = book.dropna()
    if book_vol_target:
        book = scale_to_book_vol(book, book_vol_target)
    # Regime histogram over book period
    regime_hist = (regime.reindex(book.index, method="ffill")
                          .value_counts(normalize=True))
    return book, regime_hist


def main():
    print("=" * 90)
    print(" 5-SLEEVE DEPLOY: STATIC vs REGIME-CONDITIONAL hedge")
    print(" Regime signal: VIX 1y rolling z, ±1σ thresholds (AN-1 spec)")
    print("=" * 90)
    print()

    book_a = build_combined_book(book_vol_target=DEFAULT_BOOK_VOL_TARGET)
    book_b = build_combined_book(
        book_vol_target=DEFAULT_BOOK_VOL_TARGET,
        crisis_risk_weight=DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
        mom_hedge_risk_weight=DEFAULT_MOM_HEDGE_RISK_WEIGHT,
    )
    book_c, regime_hist = build_regime_conditional_book()

    sa = book_stats(book_a)
    sb = book_stats(book_b)
    sc = book_stats(book_c)

    print("-" * 90)
    print(" REGIME HISTOGRAM (over book period)")
    print("-" * 90)
    for r in ["CALM", "NORMAL", "STRESS"]:
        pct = regime_hist.get(r, 0.0) * 100
        print(f"  {r:>7}: {pct:5.1f}% of months")
    print()

    print("-" * 90)
    print(" PERFORMANCE METRICS (10% vol-target)")
    print("-" * 90)
    print(f"  {'metric':<14}  {'(A) baseline':>14}  {'(B) static':>14}  "
          f"{'(C) regime':>14}")
    print(f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")
    print(f"  {'Sharpe':<14}  {sa['sharpe']:>+14.3f}  "
          f"{sb['sharpe']:>+14.3f}  {sc['sharpe']:>+14.3f}")
    print(f"  {'ann return':<14}  {sa['ann']:>+14.3%}  "
          f"{sb['ann']:>+14.3%}  {sc['ann']:>+14.3%}")
    print(f"  {'vol':<14}  {sa['vol']:>14.3%}  "
          f"{sb['vol']:>14.3%}  {sc['vol']:>14.3%}")
    print(f"  {'maxDD':<14}  {sa['maxdd']:>+14.3%}  "
          f"{sb['maxdd']:>+14.3%}  {sc['maxdd']:>+14.3%}")
    print(f"  {'n_months':<14}  {sa['n']:>14d}  "
          f"{sb['n']:>14d}  {sc['n']:>14d}")
    print()

    print("=" * 90)
    print(" INSTITUTIONAL READ")
    print("=" * 90)
    sb_vs_sa = sb["sharpe"] - sa["sharpe"]
    sc_vs_sa = sc["sharpe"] - sa["sharpe"]
    sc_vs_sb = sc["sharpe"] - sb["sharpe"]
    print(f"  (B) static hedge vs (A) baseline:")
    print(f"    Sharpe {sb_vs_sa:+.3f}, maxDD {sb['maxdd'] - sa['maxdd']:+.3%}")
    print(f"  (C) regime-conditional vs (A) baseline:")
    print(f"    Sharpe {sc_vs_sa:+.3f}, maxDD {sc['maxdd'] - sa['maxdd']:+.3%}")
    print(f"  (C) regime-conditional vs (B) static:")
    print(f"    Sharpe {sc_vs_sb:+.3f}, maxDD {sc['maxdd'] - sb['maxdd']:+.3%}")
    print()
    if sc["sharpe"] > sb["sharpe"]:
        print("  C > B: regime-conditional is the better hedge approach.")
    else:
        print("  C <= B: static hedge holds up better than regime-conditional ")
        print("    in this sample.  Possible reason: regime classifier whipsaw")
        print("    + insufficient stress periods in 2014-2024 window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
