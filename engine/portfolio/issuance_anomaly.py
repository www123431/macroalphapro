"""engine/portfolio/issuance_anomaly.py — Pontiff-Woodgate 2008 net
share issuance anomaly.

Reference:
  Pontiff & Woodgate 2008 "Share Issuance and Cross-Sectional Returns"
    Journal of Finance — long firms reducing share count, short firms
    issuing new shares. Effect distinct from value/quality.
  Daniel & Titman 2006 — composite issuance (debt + equity).
  Hou-Karolyi-Kho 2020 — SURVIVES (large) replication category.

Methodology:
  - Signal: 12-month log-change in split-adjusted shares outstanding
  - Low signal = buyback firms (LONG); high signal = issuance firms (SHORT)
  - Decile cuts, equal-weighted, monthly rebal
  - Price filter: >= $5 (avoid penny stocks)

Anti-overfit:
  - 12-month lookback per published canonical
  - Decile (10%) cuts per published canonical
  - $5 price filter per published canonical
  - Uses CRSP shrout which is split-adjusted (no look-ahead)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUT_MONTHLY = Path("data/cache/_issuance_monthly.parquet")

LOOKBACK_MO = 12
DECILE = 0.10
MIN_FIRMS_PER_DECILE = 100
MIN_PRICE = 5.0


def build_issuance_returns() -> pd.Series:
    logger.info("loading CRSP msf long history")
    df = pd.read_parquet("data/cache/_crsp_msf_long_history.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
    df["abs_prc"] = df["prc"].abs()
    df = df[df["abs_prc"] >= MIN_PRICE]
    df = df.dropna(subset=["shrout", "ret"])

    # Pivot for shares outstanding
    shrout = df.pivot_table(index="month", columns="permno", values="shrout").sort_index()
    rets = df.pivot_table(index="month", columns="permno", values="ret").sort_index()
    logger.info(f"  shrout panel: {shrout.shape}, returns panel: {rets.shape}")

    # Signal: 12-month log change in shares outstanding
    log_shr = np.log(shrout.replace(0, np.nan))
    issuance_signal = log_shr - log_shr.shift(LOOKBACK_MO)
    logger.info(f"  issuance signal panel computed (12mo log-change)")

    rows = []
    for i, t in enumerate(rets.index):
        if i < LOOKBACK_MO + 1:
            continue
        # Use signal from PREVIOUS month-end (i-1) to predict month i
        sig = issuance_signal.iloc[i - 1].dropna()
        if len(sig) < MIN_FIRMS_PER_DECILE * 2:
            continue

        # Long low-issuance (buybacks); short high-issuance
        thr_lo = sig.quantile(DECILE)
        thr_hi = sig.quantile(1 - DECILE)
        long_firms = sig[sig <= thr_lo].index
        short_firms = sig[sig >= thr_hi].index
        if len(long_firms) < MIN_FIRMS_PER_DECILE or len(short_firms) < MIN_FIRMS_PER_DECILE:
            continue

        # Holding return: month t
        r = rets.iloc[i]
        l_ret = float(r.reindex(long_firms).dropna().mean())
        s_ret = float(r.reindex(short_firms).dropna().mean())
        rows.append((t, l_ret - s_ret, len(long_firms), len(short_firms)))

    df_out = pd.DataFrame(rows, columns=["date", "iss_ret", "n_long", "n_short"])
    df_out = df_out.set_index("date").sort_index()
    series = df_out["iss_ret"].rename("issuance_anomaly")
    series.to_frame("iss").to_parquet(OUT_MONTHLY)
    return series


def main():
    import math
    logging.basicConfig(level=logging.INFO)
    s = build_issuance_returns()
    ann_ret = float(s.mean() * 12)
    ann_vol = float(s.std() * math.sqrt(12))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    print("=" * 75)
    print(" Issuance anomaly (Pontiff-Woodgate 2008, 12mo shares change)")
    print("=" * 75)
    print(f"  n_months: {len(s)}  ({s.index.min().date()} -> {s.index.max().date()})")
    print(f"  ann return: {ann_ret:+.4f} ({ann_ret*100:+.2f}%/yr)")
    print(f"  ann vol:    {ann_vol:.4f}")
    print(f"  Sharpe:     {sharpe:+.3f}")
    print(f"  win rate:   {(s > 0).mean():.1%}")


if __name__ == "__main__":
    main()
