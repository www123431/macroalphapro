"""engine/portfolio/long_term_reversal.py — De Bondt-Thaler 1985 LTR.

Canonical methodology:
  Formation window: months t-60 to t-13 (skip last 12 months to avoid
  contaminating with Jegadeesh-Titman momentum). Cumulative return.
  Rank cross-sectionally.
  Long bottom decile (losers), short top decile (winners).
  Equal-weighted, monthly rebalance.

Reference:
  De Bondt-Thaler 1985 "Does the Stock Market Overreact?" Journal of
    Finance. 36-60 month formation, 36-60 month hold, equal-weighted
    cross-section.
  Modern adaptation (HKK 2020 et al): monthly rebal, decile cuts,
    skip-12-months to isolate from momentum.

Anti-overfit:
  - Formation window 48 months (60-12) — published canonical
  - Skip 12 months — published canonical (vs JT 1993 momentum)
  - Decile (10%) cuts — published canonical
  - No look-ahead — all formation data ends BEFORE evaluation month
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CRSP_RET = REPO_ROOT / "data" / "cache" / "crsp_hist_daily_ret.parquet"
OUT_MONTHLY = REPO_ROOT / "data" / "cache" / "_ltr_monthly.parquet"

FORMATION_END_LAG_MO = 12   # skip last 12 months (avoid momentum contamination)
FORMATION_MONTHS = 48       # then look back 48 months (= total 60 months pre-eval)
DECILE = 0.10
MIN_HISTORY_REQUIRED = FORMATION_MONTHS + FORMATION_END_LAG_MO
MIN_FIRMS_PER_DECILE = 100  # power constraint


def build_ltr_returns() -> pd.Series:
    logger.info("loading CRSP daily returns")
    daily = pd.read_parquet(CRSP_RET)
    daily["date"] = pd.to_datetime(daily["date"])
    # Convert to monthly compounded returns per firm
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp("M")
    logger.info(f"  daily: {len(daily):,} rows")

    # Monthly compounded returns
    monthly = (daily.groupby(["permno", "month"])["ret"]
                    .apply(lambda r: (1 + r.clip(-0.5, 0.5)).prod() - 1)
                    .reset_index()
                    .rename(columns={"month": "date"}))
    monthly_pivot = monthly.pivot_table(
        index="date", columns="permno", values="ret",
    ).sort_index()
    logger.info(f"  monthly panel shape: {monthly_pivot.shape}")

    # For each month t, compute formation = product of returns from
    # t-60 to t-13 (inclusive). Need at least MIN_HISTORY_REQUIRED months
    # of history.
    n_months = len(monthly_pivot.index)
    rows = []

    for i, t in enumerate(monthly_pivot.index):
        # Need at least i >= MIN_HISTORY_REQUIRED months of past data
        if i < MIN_HISTORY_REQUIRED:
            continue

        formation_start_idx = i - FORMATION_MONTHS - FORMATION_END_LAG_MO
        formation_end_idx = i - FORMATION_END_LAG_MO  # exclusive
        formation_window = monthly_pivot.iloc[formation_start_idx:formation_end_idx]

        # Per firm: cumulative return over formation window; require
        # FULL formation history (no NaN tolerance — strict for cleanliness)
        full_history_mask = formation_window.notna().sum(axis=0) >= int(
            FORMATION_MONTHS * 0.9
        )
        valid_firms = formation_window.columns[full_history_mask]
        if len(valid_firms) < MIN_FIRMS_PER_DECILE * 2:
            continue

        # Cumulative return for each firm with valid history
        sub = formation_window[valid_firms].fillna(0)
        formation_cum = (1 + sub).prod(axis=0) - 1

        # Rank: long bottom decile (losers), short top decile (winners)
        thr_lo = formation_cum.quantile(DECILE)
        thr_hi = formation_cum.quantile(1 - DECILE)
        losers = formation_cum[formation_cum <= thr_lo].index
        winners = formation_cum[formation_cum >= thr_hi].index

        if len(losers) < MIN_FIRMS_PER_DECILE or len(winners) < MIN_FIRMS_PER_DECILE:
            continue

        # Holding return: next 1 month (t)
        next_month_ret = monthly_pivot.iloc[i]
        l_ret = float(next_month_ret.reindex(losers).dropna().mean())
        s_ret = float(next_month_ret.reindex(winners).dropna().mean())
        ls = l_ret - s_ret    # long losers, short winners
        rows.append((t, ls, len(losers), len(winners)))

    if not rows:
        raise RuntimeError("no LTR LS returns produced — insufficient history")

    df = pd.DataFrame(rows, columns=["date", "ltr_ret", "n_loser", "n_winner"])
    df = df.set_index("date").sort_index()
    df.to_parquet(OUT_MONTHLY.with_suffix(".diag.parquet"))

    series = df["ltr_ret"].rename("long_term_reversal")
    series.to_frame("ltr").to_parquet(OUT_MONTHLY)
    return series


def main():
    import math
    logging.basicConfig(level=logging.INFO)
    s = build_ltr_returns()
    ann_ret = float(s.mean() * 12)
    ann_vol = float(s.std() * math.sqrt(12))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    print("=" * 75)
    print(" Long-Term Reversal (De Bondt-Thaler 1985, t-60 to t-13 formation)")
    print("=" * 75)
    print(f"  n_months: {len(s)}")
    print(f"  date range: {s.index.min().date()} → {s.index.max().date()}")
    print(f"  ann return: {ann_ret:+.4f} ({ann_ret*100:+.2f}%/yr)")
    print(f"  ann vol:    {ann_vol:.4f}")
    print(f"  Sharpe:     {sharpe:+.3f}")
    print(f"  win rate:   {(s > 0).mean():.1%}")
    print(f"  saved:      {OUT_MONTHLY}")


if __name__ == "__main__":
    main()
