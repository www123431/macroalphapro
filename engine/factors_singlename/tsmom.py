"""
engine/factors_singlename/tsmom.py — single-stock TSMOM factor.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.2 Wave A
Literature: Moskowitz-Ooi-Pedersen 2012 *JFE* "Time Series Momentum"
  - 12-month lookback, 1-month skip
  - Sign of cumulative return → ±1 signal
  - Cross-sectionally applied at single-stock level

Implementation note: this is the single-stock variant. ETF version is in
engine.factors.tsmom (which uses HOP 2017 multi-lookback ensemble per
v1 spec amendment 2026-05-09 Fix #1). Single-stock here uses MOP 2012
classic single-lookback per literature for cleaner Stage 2 alignment.

Reads from pre-fetched price panel (engine.factor_ensemble_walk_forward
panel-cache pattern); no yfinance call within compute function.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Locked per spec §2.2 — MOP 2012 standard
LOOKBACK_MONTHS_LOCKED: int = 12
SKIP_MONTHS_LOCKED:     int = 1
TRADING_DAYS_PER_MONTH: int = 21


def compute_tsmom_singlestock_signal(
    as_of:         datetime.date,
    universe:      list[str],
    asset_classes: Optional[dict[str, str]] = None,
    panel:         Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Single-stock TSMOM signal at as_of.

    Mechanism:
      - cumulative return over [t - 13mo, t - 1mo] (12-month return, 1-month skip)
      - signed: +1 if positive, -1 if negative, 0 if NaN/zero/insufficient history

    Args:
        as_of:          decision date (no look-ahead — uses prices ≤ t - SKIP_MONTHS)
        universe:       list of tickers
        asset_classes:  ignored (for signature consistency with ETF factors)
        panel:          pre-fetched price DataFrame (date index × ticker columns)
                        REQUIRED for single-stock; no internal yfinance fetch

    Returns:
        pd.Series indexed by ticker, values ∈ {-1.0, 0.0, +1.0} (NaN if insufficient)
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)
    if panel is None or panel.empty:
        logger.warning("compute_tsmom_singlestock_signal: panel required, returning all-NaN")
        return pd.Series(np.nan, index=universe, dtype=float)

    skip_days = SKIP_MONTHS_LOCKED * TRADING_DAYS_PER_MONTH
    lookback_days = LOOKBACK_MONTHS_LOCKED * TRADING_DAYS_PER_MONTH

    # Use as_of - skip_days as effective end date (no peeking past skip month)
    end_date = as_of - datetime.timedelta(days=int(skip_days * 1.5))  # 1.5x cushion for non-trading days
    start_date = end_date - datetime.timedelta(days=int(lookback_days * 1.5))

    out: dict[str, float] = {}
    for ticker in universe:
        if ticker not in panel.columns:
            out[ticker] = np.nan
            continue
        ts = panel[ticker]
        # Slice [start_date, end_date]
        mask = (ts.index >= pd.Timestamp(start_date)) & (ts.index <= pd.Timestamp(end_date))
        sub = ts.loc[mask].dropna()
        if len(sub) < lookback_days // 2:  # need at least half the expected obs
            out[ticker] = np.nan
            continue
        # First and last price in window
        try:
            p_first = float(sub.iloc[0])
            p_last = float(sub.iloc[-1])
        except (IndexError, ValueError):
            out[ticker] = np.nan
            continue
        if p_first <= 0:
            out[ticker] = np.nan
            continue
        ret = p_last / p_first - 1
        if not np.isfinite(ret):
            out[ticker] = np.nan
            continue
        # Sign
        if ret > 0:
            out[ticker] = 1.0
        elif ret < 0:
            out[ticker] = -1.0
        else:
            out[ticker] = 0.0

    return pd.Series(out, dtype=float)
