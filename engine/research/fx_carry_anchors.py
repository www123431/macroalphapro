"""engine.research.fx_carry_anchors — LRV 2011 HML_FX + DOL construction.

Third commit in Lustig-Roussanov-Verdelhan 2011 carry-anchor chain.
Combines the FX spot returns (Commit 1) and short-rate differentials
(Commit 2) to construct the academic-standard FX carry anchor
portfolios:

  DOL  = equal-weighted mean of all G10 FCY excess returns
  HML_FX = long top-3 high-carry currencies, short bottom-3 low-carry

This is the lens AQR / Two Sigma / institutional FX research uses
to attribute cross-asset carry strategies. Replacing the macro-lite
proxy (5 FRED regime variables) with LRV's actual portfolio
construction upgrades cross_asset_carry audit from "lite proxy" to
"academic-standard".

Methodology — Lustig-Roussanov-Verdelhan 2011 + MSSS 2012 §2
------------------------------------------------------------
Each month t:
  1. Compute total return of holding each FCY vs USD for one month:
        excess_return_i_t = log_spot_i_t - log_spot_i_{t-1}
                            + (rate_i_{t-1} - rate_USD_{t-1}) / 12
     Where rate is annualized %, /12 gives monthly equivalent.
     The first term is FX appreciation; the second is the carry.
     Per CIP, this approximates the actual return of going long FCY
     funded in USD over one month.
  2. Sort currencies into K=3 buckets by lagged rate differential
     rdiff_i_{t-1} (the "carry sort key" — known at start of month t)
  3. PORTFOLIO_HIGH = mean(excess_return) of top-tercile carry
     PORTFOLIO_LOW  = mean(excess_return) of bottom-tercile
     PORTFOLIO_MID  = mean(excess_return) of middle
  4. DOL_t = mean(excess_return) across all 9 non-USD G10 currencies
  5. HML_FX_t = PORTFOLIO_HIGH - PORTFOLIO_LOW

Lag handling
------------
The sort key (rdiff_i_{t-1}) MUST be lagged. Using same-month rdiff
introduces look-ahead bias (we wouldn't know t's differential at
the start of t in real trading). This is the same B0/B1-class bug
discipline as our PIT data work.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# G10 currencies (excluding USD — USD is the base)
G10_CURRENCIES = ("JPY", "EUR", "GBP", "CHF", "CAD",
                    "AUD", "NZD", "SEK", "NOK", "DKK")


def load_fx_spot_g10() -> Optional[pd.DataFrame]:
    """Read cached G10 FX spot from data/anchor_library/
    fx_spot_g10_monthly.parquet."""
    p = (Path(__file__).resolve().parents[2]
         / "data" / "anchor_library"
         / "fx_spot_g10_monthly.parquet")
    if not p.exists():
        logger.info("fx_carry: no FX spot library at %s "
                      "(run scripts/fetch_fx_spot_g10.py)", p)
        return None
    df = pd.read_parquet(p)
    if "date" not in df.columns:
        return None
    df = df.set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index)
    return df


def load_g10_short_rates() -> Optional[pd.DataFrame]:
    """Read cached G10 short rates from data/anchor_library/
    g10_short_rates_monthly.parquet."""
    p = (Path(__file__).resolve().parents[2]
         / "data" / "anchor_library"
         / "g10_short_rates_monthly.parquet")
    if not p.exists():
        logger.info("fx_carry: no rates library at %s "
                      "(run scripts/fetch_g10_short_rates.py)", p)
        return None
    df = pd.read_parquet(p)
    if "date" not in df.columns:
        return None
    df = df.set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index)
    return df


def build_carry_anchors(
    spot_df:    pd.DataFrame,
    rates_df:   pd.DataFrame,
    *,
    n_buckets:  int = 3,
) -> Optional[pd.DataFrame]:
    """Per LRV 2011 methodology, construct DOL + HML_FX + tercile
    portfolios from G10 spot returns + rate differentials.

    Args:
      spot_df: from load_fx_spot_g10() — month-end DatetimeIndex,
               columns including logret_<CCY> for each G10 currency
      rates_df: from load_g10_short_rates() — month-end DatetimeIndex,
                columns including rdiff_<CCY>_pct
      n_buckets: 3 (terciles) per LRV/MSSS canonical sort.
                 Could also use 5 (quintiles) but with only 9
                 non-USD currencies, 3 is the cleanest split.

    Returns DataFrame with month-end index and columns:
      DOL          mean FCY excess return (USD-funded), in % per month
      HML_FX       high carry - low carry tercile return (% per month)
      P_HIGH       high carry tercile mean return (% per month)
      P_MID        middle tercile (when n_buckets=3)
      P_LOW        low carry tercile

    Returns None if input data insufficient.
    """
    if spot_df is None or rates_df is None:
        return None

    # Align on date intersection
    aligned = spot_df.join(rates_df, how="inner")
    aligned = aligned.dropna(how="any")
    if len(aligned) < 24:
        logger.info("fx_carry: insufficient overlap %d months",
                      len(aligned))
        return None

    # Lag the sort key by ONE month — what we know at start of
    # period t is rdiff_{t-1}. Avoids look-ahead bias.
    rdiff_cols = [f"rdiff_{c}_pct" for c in G10_CURRENCIES
                    if f"rdiff_{c}_pct" in aligned.columns]
    if len(rdiff_cols) < n_buckets * 2:
        logger.warning("fx_carry: only %d currencies with rdiff; "
                          "need at least %d for %d-bucket sort",
                          len(rdiff_cols), n_buckets * 2, n_buckets)
        return None
    sort_keys = aligned[rdiff_cols].shift(1)
    logret_cols = [f"logret_{c}" for c in G10_CURRENCIES
                     if f"logret_{c}" in aligned.columns]

    # The carry component of return: rate differential of FCY relative
    # to USD, applied over a 1-month period. annualized % → monthly
    # decimal divisor 1200.
    monthly_carry_pct = aligned[rdiff_cols].shift(1) / 12.0  # in %
    # Combine FX log return (decimal, multiplied by 100 → %) +
    # monthly carry (already in %)
    excess_returns_pct = pd.DataFrame(index=aligned.index)
    for ccy in G10_CURRENCIES:
        lr_col = f"logret_{ccy}"
        rd_col = f"rdiff_{ccy}_pct"
        if lr_col not in aligned.columns or rd_col not in aligned.columns:
            continue
        excess_returns_pct[ccy] = (
            aligned[lr_col] * 100.0
            + monthly_carry_pct[rd_col]
        )

    # Drop first row (lagged sort key NaN)
    excess_returns_pct = excess_returns_pct.iloc[1:]
    sort_keys = sort_keys.iloc[1:]
    sort_keys.columns = [c.replace("rdiff_", "").replace("_pct", "")
                            for c in sort_keys.columns]

    # Build portfolio per month
    out_rows = []
    for date in excess_returns_pct.index:
        keys = sort_keys.loc[date].dropna()
        rets = excess_returns_pct.loc[date].dropna()
        common = keys.index.intersection(rets.index)
        if len(common) < n_buckets * 2:
            continue
        sorted_ccys = keys.loc[common].sort_values(ascending=False)
        n = len(sorted_ccys)
        bucket_size = n // n_buckets
        high_ccys = sorted_ccys.iloc[:bucket_size].index
        low_ccys  = sorted_ccys.iloc[-bucket_size:].index
        # Middle: everything in between (for n=3, this is sorted[bucket:2*bucket])
        mid_ccys  = sorted_ccys.iloc[bucket_size:n - bucket_size].index

        p_high = rets.loc[high_ccys].mean()
        p_low  = rets.loc[low_ccys].mean()
        p_mid  = rets.loc[mid_ccys].mean() if len(mid_ccys) else float("nan")
        dol    = rets.loc[common].mean()
        hml_fx = p_high - p_low

        out_rows.append({
            "date":   date,
            "DOL":    dol,
            "HML_FX": hml_fx,
            "P_HIGH": p_high,
            "P_MID":  p_mid,
            "P_LOW":  p_low,
        })

    if not out_rows:
        return None
    out_df = pd.DataFrame(out_rows).set_index("date")
    out_df.index = pd.DatetimeIndex(out_df.index)
    return out_df


def build_and_cache_carry_anchors(
    out_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Convenience: load + build + persist to parquet for re-use.
    Skips network; reads cached FX spot + rates parquets."""
    spot_df = load_fx_spot_g10()
    rates_df = load_g10_short_rates()
    if spot_df is None or rates_df is None:
        return None
    portfolios = build_carry_anchors(spot_df, rates_df)
    if portfolios is None:
        return None
    if out_path is None:
        out_path = (Path(__file__).resolve().parents[2]
                     / "data" / "anchor_library"
                     / "lrv_fx_carry_anchors_monthly.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist with date as a column for parquet round-trip
    portfolios_for_disk = portfolios.reset_index()
    portfolios_for_disk.to_parquet(out_path, index=False)
    return portfolios
