"""scripts/ltr_long_history_rebuild.py — rebuild LTR on 35yr CRSP msf
+ re-apply pre-commit criteria.

Same methodology as engine.portfolio.long_term_reversal:
  formation: t-60 to t-13
  decile L/S losers-minus-winners
  monthly rebal
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from engine.portfolio.combined_book import build_tsmom_book
from engine.validation.deflated_sharpe import deflated_sharpe_ratio

OUT = Path("data/cache/_ltr_monthly_long.parquet")

FORMATION_END_LAG_MO = 12
FORMATION_MONTHS = 48
DECILE = 0.10
MIN_FIRMS_PER_DECILE = 100
MIN_PRICE = 5.0    # De Bondt-Thaler exclusion of penny stocks


def build_ltr_long() -> pd.Series:
    print("loading CRSP msf...")
    df = pd.read_parquet("data/cache/_crsp_msf_long_history.parquet")
    df["date"] = pd.to_datetime(df["date"])
    # Normalize month-end
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
    # Filter: price >= $5 to exclude penny stocks (DBT 1985)
    df["abs_prc"] = df["prc"].abs()
    df = df[df["abs_prc"] >= MIN_PRICE]

    panel = df.pivot_table(index="month", columns="permno", values="ret").sort_index()
    print(f"  monthly panel: {panel.shape}  range "
          f"{panel.index.min().date()} -> {panel.index.max().date()}")

    rows = []
    for i, t in enumerate(panel.index):
        if i < FORMATION_MONTHS + FORMATION_END_LAG_MO + 1:
            continue
        formation_start = i - FORMATION_MONTHS - FORMATION_END_LAG_MO
        formation_end = i - FORMATION_END_LAG_MO
        win = panel.iloc[formation_start:formation_end]
        full = win.notna().sum(axis=0) >= int(FORMATION_MONTHS * 0.9)
        valid = win.columns[full]
        if len(valid) < MIN_FIRMS_PER_DECILE * 2:
            continue
        formation_cum = (1 + win[valid].fillna(0)).prod(axis=0) - 1
        thr_lo = formation_cum.quantile(DECILE)
        thr_hi = formation_cum.quantile(1 - DECILE)
        losers = formation_cum[formation_cum <= thr_lo].index
        winners = formation_cum[formation_cum >= thr_hi].index
        if len(losers) < MIN_FIRMS_PER_DECILE or len(winners) < MIN_FIRMS_PER_DECILE:
            continue
        next_ret = panel.iloc[i]
        l_ret = float(next_ret.reindex(losers).dropna().mean())
        s_ret = float(next_ret.reindex(winners).dropna().mean())
        rows.append((t, l_ret - s_ret, len(losers), len(winners)))

    df_out = pd.DataFrame(rows, columns=["date", "ltr_ret", "n_loser", "n_winner"])
    df_out = df_out.set_index("date").sort_index()
    s = df_out["ltr_ret"].rename("ltr_long")
    s.to_frame("ltr").to_parquet(OUT)
    print(f"  built {len(s)} months from {s.index.min().date()} to {s.index.max().date()}")
    return s


def _sharpe(s: pd.Series) -> float:
    return float((s.mean() * 12) / (s.std() * math.sqrt(12))) if s.std() > 0 else 0.0


def _cosine(a: pd.Series, b: pd.Series) -> float:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 12:
        return float("nan")
    av, bv = j["a"].values, j["b"].values
    return float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv)))


def verdict(ltr: pd.Series):
    us_pit_sn = pd.read_parquet("data/cache/_dpead_sn_pit_monthly.parquet").iloc[:, 0]
    us_pit_sn.index = pd.to_datetime(us_pit_sn.index)
    tsmom = build_tsmom_book()
    tsmom.index = pd.to_datetime(tsmom.index)

    print("\n" + "=" * 80)
    print(" LTR PRE-COMMIT VERDICT — LONG HISTORY (1990-2024)")
    print("=" * 80)
    sharpe = _sharpe(ltr)
    cos_us = _cosine(ltr, us_pit_sn)
    cos_tsmom = _cosine(ltr, tsmom)
    dsr = deflated_sharpe_ratio(returns=ltr.values, n_trials=5, periods_per_year=12)

    print(f"\n  sample: {len(ltr)} months "
          f"({ltr.index.min().date()} → {ltr.index.max().date()})")
    print(f"  ann ret:  {ltr.mean()*12:+.2%}/yr")
    print(f"  ann vol:  {ltr.std()*math.sqrt(12):.2%}/yr")
    print(f"\n  [LTR1] Sharpe >= 0.5 ?")
    print(f"    Sharpe = {sharpe:+.3f}")
    ltr1 = sharpe >= 0.5
    print(f"    → {'PASS' if ltr1 else 'FAIL'}")
    print(f"\n  [LTR2] cosine US PIT SN < 0.4 ?")
    print(f"    cosine = {cos_us:+.3f}")
    ltr2 = cos_us < 0.4
    print(f"    → {'PASS' if ltr2 else 'FAIL'}")
    print(f"\n  [LTR3] cosine TSMOM <= 0 ?")
    print(f"    cosine = {cos_tsmom:+.3f}")
    ltr3 = cos_tsmom <= 0
    print(f"    → {'PASS' if ltr3 else 'FAIL'}")
    print(f"\n  [LTR4] DeflSR >= 0.6 ?")
    print(f"    DeflSR = {dsr.deflated_sr:.3f}  {dsr.verdict}")
    ltr4 = dsr.deflated_sr >= 0.6
    print(f"    → {'PASS' if ltr4 else 'FAIL'}")

    n_pass = sum([ltr1, ltr2, ltr3, ltr4])
    print(f"\n  PASSES: {n_pass}/4")
    if n_pass == 4:
        v = "GREEN — promote to PAPER_TRADE"
    elif n_pass >= 2:
        v = "YELLOW"
    else:
        v = "REJECTED"
    print(f"  VERDICT: {v}")


def main():
    s = build_ltr_long()
    verdict(s)


if __name__ == "__main__":
    main()
