"""scripts/strategy_config_research_2026-05-31.py — honest comparison
of book configurations after PIT sector-neutral D_PEAD discovery.

Computes book Sharpe / maxDD / vol for:
  A. Current deploy: equity (orig D_PEAD + revision) / carry / tsmom (3-sleeve)
  B. Current 5-sleeve regime-conditional (deploy C from 2026-05-30)
  C. Replace D_PEAD with PIT sector-neutral (within equity_book), 3-sleeve
  D. Replace + 5-sleeve regime-conditional

Uses HONEST numbers (PIT FF12 sectors, net of 25bp/mo realistic cost
for sector-neutral; 12.5bp for original D_PEAD).
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
    DEFAULT_BOOK_VOL_TARGET, DEFAULT_CARRY_RISK_WEIGHT,
    DEFAULT_CRISIS_HEDGE_RISK_WEIGHT, DEFAULT_MOM_HEDGE_RISK_WEIGHT,
    DEFAULT_TARGET_VOL, DEFAULT_TSMOM_RISK_WEIGHT,
    REGIME_HEDGE_GRIDS,
    book_stats, build_carry_book, build_combined_book,
    build_crisis_hedge_book, build_equity_book, build_mom_hedge_book,
    build_tsmom_book, build_vix_regime_monthly,
    scale_to_book_vol, voltarget,
)

RT_EQ_ORIG = 30.0     # 30bp/side × 5 turnovers/12 months for original
RT_EQ_SN = 30.0       # same per-trade cost but turnover ~2x higher
SN_MONTHLY_COST_BP = 25.0   # ≈ 30bp × 10 turnovers / 12 months (realistic)


def build_equity_book_sn() -> pd.Series:
    """Equity book using PIT sector-neutral D_PEAD instead of original."""
    from engine.validation.analyst_revision import build_revision_sleeve_buffered
    sn = pd.read_parquet("data/cache/_dpead_sector_neutral_pit.parquet").iloc[:, 0]
    sn.index = pd.to_datetime(sn.index)
    sn_m = ((1 + sn.clip(-0.2, 0.2)).resample("ME").prod() - 1)
    # Apply realistic 25bp/mo cost (higher within-sector turnover)
    sn_net = (sn_m - SN_MONTHLY_COST_BP / 10000.0).rename("sn")
    # Revision unchanged
    rev, rev_turn = build_revision_sleeve_buffered(
        q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5,
    )
    rev_net = (rev - rev_turn * RT_EQ_ORIG / 10000.0 / 12).rename("rev")
    E = pd.concat([sn_net, rev_net], axis=1).dropna()
    v_sn = E["sn"].rolling(12).std().shift(1)
    v_re = E["rev"].rolling(12).std().shift(1)
    w = (1 / v_sn) / (1 / v_sn + 1 / v_re)
    return (w * E["sn"] + (1 - w) * E["rev"]).dropna().rename("equity_sn_book")


def build_combined_3sleeve(equity_func, book_vol_target=DEFAULT_BOOK_VOL_TARGET):
    eq_vt = voltarget(equity_func(), DEFAULT_TARGET_VOL)
    cy_vt = voltarget(build_carry_book(), DEFAULT_TARGET_VOL)
    ts_vt = voltarget(build_tsmom_book(), DEFAULT_TARGET_VOL)
    eq_w = 1.0 - DEFAULT_CARRY_RISK_WEIGHT - DEFAULT_TSMOM_RISK_WEIGHT
    J = pd.concat([eq_vt.rename("e"), cy_vt.rename("c"), ts_vt.rename("t")],
                       axis=1).dropna()
    book = (eq_w * J["e"]
              + DEFAULT_CARRY_RISK_WEIGHT * J["c"]
              + DEFAULT_TSMOM_RISK_WEIGHT * J["t"])
    return scale_to_book_vol(book, book_vol_target).rename("book")


def build_combined_5sleeve_regime(equity_func,
                                       grids=REGIME_HEDGE_GRIDS,
                                       book_vol_target=DEFAULT_BOOK_VOL_TARGET):
    eq_vt = voltarget(equity_func(), DEFAULT_TARGET_VOL)
    cy_vt = voltarget(build_carry_book(), DEFAULT_TARGET_VOL)
    ts_vt = voltarget(build_tsmom_book(), DEFAULT_TARGET_VOL)
    crisis_vt = voltarget(build_crisis_hedge_book(), DEFAULT_TARGET_VOL)
    momh_vt = voltarget(build_mom_hedge_book(), DEFAULT_TARGET_VOL)
    J = pd.concat([eq_vt.rename("e"), cy_vt.rename("c"), ts_vt.rename("t"),
                       crisis_vt.rename("h"), momh_vt.rename("m")],
                       axis=1).dropna()
    regime = build_vix_regime_monthly()
    book = pd.Series(index=J.index, dtype=float, name="book")
    for t in J.index:
        nearest = regime.index[regime.index <= t]
        r = "NORMAL" if len(nearest) == 0 else regime.loc[nearest[-1]]
        g = grids[r]
        crisis_w = g["crisis"]
        mom_w = g["mom_hedge"]
        eq_w = (1.0 - DEFAULT_CARRY_RISK_WEIGHT - DEFAULT_TSMOM_RISK_WEIGHT
                  - crisis_w - mom_w)
        book.loc[t] = (eq_w * J.loc[t, "e"]
                          + DEFAULT_CARRY_RISK_WEIGHT * J.loc[t, "c"]
                          + DEFAULT_TSMOM_RISK_WEIGHT * J.loc[t, "t"]
                          + crisis_w * J.loc[t, "h"]
                          + mom_w * J.loc[t, "m"])
    return scale_to_book_vol(book.dropna(), book_vol_target).rename("book")


def main():
    print("=" * 90)
    print(" STRATEGY CONFIG RESEARCH 2026-05-31 — honest PIT sector-neutral")
    print("=" * 90)
    print()
    print(" Equity book variants:")
    print("   ORIGINAL: D_PEAD orig + analyst rev (RT_EQ=30bp×5/12=12.5bp/mo)")
    print("   PIT SN:   sector-neutral PIT FF12 + analyst rev (25bp/mo)")
    print()

    configs = []
    # A. 3-sleeve original (current legacy, pre-amendment)
    print("Building A (3-sleeve original)...")
    a = build_combined_book(book_vol_target=DEFAULT_BOOK_VOL_TARGET)
    configs.append(("A. 3-sleeve original (legacy)", a))

    # B. 5-sleeve static current (config B from 2026-05-30 — not deployed)
    print("Building B (5-sleeve static)...")
    b = build_combined_book(
        book_vol_target=DEFAULT_BOOK_VOL_TARGET,
        crisis_risk_weight=DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
        mom_hedge_risk_weight=DEFAULT_MOM_HEDGE_RISK_WEIGHT,
    )
    configs.append(("B. 5-sleeve static (orig D_PEAD)", b))

    # C. 5-sleeve regime-conditional (CURRENT DEPLOYED)
    print("Building C (5-sleeve regime-conditional, CURRENT DEPLOY)...")
    c = build_combined_book(regime_conditional=True)
    configs.append(("C. 5-sleeve regime-cond (DEPLOYED)", c))

    # D. 3-sleeve with PIT sector-neutral D_PEAD
    print("Building D (3-sleeve with PIT sector-neutral)...")
    d = build_combined_3sleeve(build_equity_book_sn)
    configs.append(("D. 3-sleeve with PIT SN D_PEAD", d))

    # E. 5-sleeve regime-conditional with PIT sector-neutral
    print("Building E (5-sleeve regime-cond with PIT SN)...")
    e = build_combined_5sleeve_regime(build_equity_book_sn)
    configs.append(("E. 5-sleeve regime-cond + PIT SN", e))

    print()
    print("-" * 90)
    print(" FULL CONFIG COMPARISON (10% vol-target)")
    print("-" * 90)
    print(f"  {'config':<42}  {'n':>4} {'Sharpe':>8} {'ann':>8} {'vol':>8} {'maxDD':>8}")
    print(f"  {'-'*42}  {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, book in configs:
        s = book_stats(book)
        print(f"  {name:<42}  {s['n']:>4} {s['sharpe']:>+8.3f} "
              f"{s['ann']:>+8.3%} {s['vol']:>8.3%} {s['maxdd']:>+8.3%}")

    # Pairwise deltas
    print()
    print("-" * 90)
    print(" KEY DELTAS")
    print("-" * 90)
    s_C = book_stats(configs[2][1])   # current deploy
    s_E = book_stats(configs[4][1])   # proposed deploy
    print(f"  E vs C (proposed deploy vs current):")
    print(f"    Sharpe  {s_E['sharpe'] - s_C['sharpe']:+.3f}  "
          f"({s_C['sharpe']:.3f} -> {s_E['sharpe']:.3f})")
    print(f"    ann     {s_E['ann'] - s_C['ann']:+.3%}")
    print(f"    maxDD   {s_E['maxdd'] - s_C['maxdd']:+.3%}  "
          f"({'BETTER' if s_E['maxdd'] > s_C['maxdd'] else 'WORSE'})")
    print()
    s_A = book_stats(configs[0][1])
    s_D = book_stats(configs[3][1])
    print(f"  D vs A (3-sleeve PIT SN vs 3-sleeve original):")
    print(f"    Sharpe  {s_D['sharpe'] - s_A['sharpe']:+.3f}  "
          f"({s_A['sharpe']:.3f} -> {s_D['sharpe']:.3f})")
    print(f"    ann     {s_D['ann'] - s_A['ann']:+.3%}")
    print(f"    maxDD   {s_D['maxdd'] - s_A['maxdd']:+.3%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
