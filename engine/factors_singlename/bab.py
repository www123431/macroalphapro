"""
engine/factors_singlename/bab.py — single-stock BAB factor (FP 2014 真原版).

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.2

Literature: Frazzini-Pedersen 2014 *JFE* "Betting Against Beta"
  - Compute β against market (SPY) using 60-day rolling window
  - Tertile-rank: bottom β tercile → +1 (long low-β), top → -1 (short high-β)
  - Single-stock literature Sharpe ~0.78 (vs ETF level ~0.05-0.15)

This is single-stock variant. ETF wrapper is in engine.factors.bab_compat.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Locked per spec §2.2 — same window as v1/v2 (consistency with TSMOM/Quality vol)
BETA_WINDOW_DAYS_LOCKED:   int = 60
BETA_BENCHMARK_LOCKED:     str = "SPY"
TRADING_DAYS_PER_YEAR:     int = 252
MIN_VALID_BETA_RATIO:      float = 0.5  # need ≥ half the tickers with valid β to compute tertiles


def compute_bab_singlestock_signal(
    as_of:         datetime.date,
    universe:      list[str],
    asset_classes: Optional[dict[str, str]] = None,
    panel:         Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Single-stock BAB signal at as_of via β tertile rank.

    Args:
        as_of:          decision date
        universe:       list of tickers (single-stock symbols)
        asset_classes:  ignored
        panel:          pre-fetched price panel (must include benchmark SPY)

    Returns:
        pd.Series indexed by ticker, values ∈ {-1.0, 0.0, +1.0}
        +1.0 = bottom β tercile (long low-β)
        -1.0 = top β tercile    (short high-β)
        0.0  = middle tercile or insufficient β data
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)
    if panel is None or panel.empty:
        logger.warning("compute_bab_singlestock_signal: panel required → all-NaN")
        return pd.Series(np.nan, index=universe, dtype=float)
    if BETA_BENCHMARK_LOCKED not in panel.columns:
        logger.warning("BAB: benchmark %s not in panel → all-NaN", BETA_BENCHMARK_LOCKED)
        return pd.Series(np.nan, index=universe, dtype=float)

    end = as_of - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=120)  # 120 calendar days for ~60 trading days

    mask = (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    sub = panel.loc[mask].dropna(how="all")
    if sub.empty or len(sub) < BETA_WINDOW_DAYS_LOCKED:
        return pd.Series(np.nan, index=universe, dtype=float)

    bench_returns = sub[BETA_BENCHMARK_LOCKED].pct_change().dropna().tail(BETA_WINDOW_DAYS_LOCKED)
    if len(bench_returns) < BETA_WINDOW_DAYS_LOCKED // 2:
        return pd.Series(np.nan, index=universe, dtype=float)

    bench_var = float(bench_returns.var(ddof=0))
    if bench_var <= 1e-12:
        return pd.Series(np.nan, index=universe, dtype=float)

    # Compute per-ticker β
    betas: dict[str, float] = {}
    for ticker in universe:
        if ticker not in sub.columns:
            betas[ticker] = np.nan
            continue
        ticker_returns = sub[ticker].pct_change().dropna().tail(BETA_WINDOW_DAYS_LOCKED)
        if len(ticker_returns) < BETA_WINDOW_DAYS_LOCKED // 2:
            betas[ticker] = np.nan
            continue
        common = ticker_returns.index.intersection(bench_returns.index)
        if len(common) < BETA_WINDOW_DAYS_LOCKED // 2:
            betas[ticker] = np.nan
            continue
        cov = float(np.cov(
            ticker_returns.loc[common].values,
            bench_returns.loc[common].values,
            ddof=0,
        )[0, 1])
        betas[ticker] = cov / bench_var

    beta_series = pd.Series(betas, dtype=float)
    valid_betas = beta_series.dropna()
    if len(valid_betas) < int(len(universe) * MIN_VALID_BETA_RATIO):
        # Insufficient β coverage → all-NaN
        return pd.Series(np.nan, index=universe, dtype=float)

    # Tertile-rank — bottom 1/3 → +1, top 1/3 → -1, middle → 0
    q_low, q_high = valid_betas.quantile([1.0 / 3.0, 2.0 / 3.0]).values
    out: dict[str, float] = {}
    for ticker in universe:
        b = beta_series.get(ticker, np.nan)
        if not np.isfinite(b):
            out[ticker] = np.nan
            continue
        if b <= q_low:
            out[ticker] = 1.0
        elif b >= q_high:
            out[ticker] = -1.0
        else:
            out[ticker] = 0.0
    return pd.Series(out, dtype=float)
