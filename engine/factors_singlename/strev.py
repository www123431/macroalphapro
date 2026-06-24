"""
engine/factors_singlename/strev.py — Tier 1 mining candidate: Short-Term Reversal.

Tier 1 mining lab content (registered into FACTOR_REGISTRY_SINGLENAME).
Pre-registration: factor_kind="infrastructure_spec" (P-LAB exempt, +0 trials).

Literature anchor
-----------------
Jegadeesh (1990). "Evidence of Predictable Behavior of Security Returns."
Journal of Finance 45(3):881-898.

Lehmann (1990). "Fads, Martingales, and Market Efficiency."
Quarterly Journal of Economics 105(1):1-28.

Both seminal papers document the **1-month return reversal** anomaly:
  - Buy stocks with the lowest prior 1-month return
  - Sell stocks with the highest prior 1-month return
  - Documented across multiple US sample periods (1934-1987 Jegadeesh)

Mechanism (this module)
-----------------------
  - Raw factor = trailing-21-trading-day total return per ticker
  - Cross-section z-score within universe at as_of
  - High z (high recent return) → expected LOW future return per Jegadeesh 1990
  - → expected_sign = -1 (short high-z, long low-z = "reversal")

Honest disclose for Tier 1 mining
---------------------------------
  - Implementation matches Jegadeesh 1990 §II "1-month sorting" exactly
    (no FF factor decomposition needed; raw return is the factor)
  - Wave A retail panel (yfinance) is sufficient — no Compustat dependency
  - Caveats:
    * Trading costs not deducted (Tier 1 mining is gross-of-cost; STREV is
      famously cost-sensitive — Lehmann 1990 estimates 0.5% / month transaction
      costs erode the profit; this is captured at Tier 2 promotion gate
      via TC drag testing)
    * Jegadeesh 1990 used CRSP delisting-bias-free sample; Wave A uses
      mktcap_top500_proxy survivorship-biased universe
    * Microstructure noise (bid-ask bounce) inflates 1-mo reversal at very
      short windows; 21-day window is conservative

API mirror of `engine/factors_singlename/dividend_yield.py` etc. so
mining_runner walk-forward consumes uniformly.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from engine.factor_library_singlename import (
    FactorSpecSinglename,
    register_factor,
)

logger = logging.getLogger(__name__)


# Locked Tier 1 mining constants
STREV_LOOKBACK_DAYS_LOCKED:     int   = 21       # 1 trading month per Jegadeesh 1990
STREV_MIN_OBS_RATIO_LOCKED:     float = 0.5      # need ≥ 50% of window with valid prices
MIN_UNIVERSE_FOR_ZSCORE_LOCKED: int   = 5        # mirror Wave A z-score gate


def compute_strev_singlestock_signal(
    as_of:         datetime.date,
    universe:      list[str],
    asset_classes: Optional[dict[str, str]] = None,
    panel:         Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Cross-section z-score of trailing 1-month return.

    Mechanism (per Jegadeesh 1990 §II):
      1. For each ticker: compute total return over [as_of - 21d, as_of - 1d]
         (skip last day to avoid look-ahead via potential same-day price use)
      2. Cross-section z-score across universe
      3. Return raw z-score; expected_sign = -1 in registry tells caller that
         high z → low future returns (reversal)

    Args:
        as_of:          decision date
        universe:       list of tickers
        asset_classes:  ignored (Wave A factor signature parity)
        panel:          pre-fetched price panel

    Returns:
        pd.Series indexed by ticker, continuous z-score of 1-mo return.
        NaN for tickers with insufficient data or universe < 5.
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)
    if panel is None or panel.empty:
        logger.warning("compute_strev_singlestock_signal: panel required → all-NaN")
        return pd.Series(np.nan, index=universe, dtype=float)

    end   = as_of - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=45)   # 45 calendar ≈ 21 trading + buffer

    mask = (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    sub  = panel.loc[mask].dropna(how="all")
    if sub.empty:
        return pd.Series(np.nan, index=universe, dtype=float)

    min_obs = max(int(STREV_LOOKBACK_DAYS_LOCKED * STREV_MIN_OBS_RATIO_LOCKED), 5)

    raw_returns: dict[str, float] = {}
    for ticker in universe:
        if ticker not in sub.columns:
            raw_returns[ticker] = np.nan
            continue
        prices = sub[ticker].dropna().tail(STREV_LOOKBACK_DAYS_LOCKED)
        if len(prices) < min_obs:
            raw_returns[ticker] = np.nan
            continue
        first = float(prices.iloc[0])
        last  = float(prices.iloc[-1])
        if first <= 0 or last <= 0:
            raw_returns[ticker] = np.nan
            continue
        ret_1mo = (last / first) - 1.0
        if not np.isfinite(ret_1mo):
            raw_returns[ticker] = np.nan
            continue
        raw_returns[ticker] = ret_1mo

    return _cross_section_zscore(raw_returns, universe)


def _cross_section_zscore(
    raw_values: dict[str, float],
    universe:   list[str],
) -> pd.Series:
    """Cross-section z-score within universe; min-5 gate."""
    raw_series = pd.Series(raw_values, dtype=float)
    valid = raw_series.dropna()
    if len(valid) < MIN_UNIVERSE_FOR_ZSCORE_LOCKED:
        return pd.Series(np.nan, index=universe, dtype=float)
    mean = float(valid.mean())
    std  = float(valid.std(ddof=1))
    if std <= 1e-9:
        return pd.Series(np.nan, index=universe, dtype=float)

    out: dict[str, float] = {}
    for ticker in universe:
        v = raw_series.get(ticker, np.nan)
        out[ticker] = (v - mean) / std if np.isfinite(v) else np.nan
    return pd.Series(out, dtype=float)


# ── Register into Tier 1 mining content layer ──────────────────────────────
register_factor(FactorSpecSinglename(
    factor_id        = "strev_singlestock",
    citation         = (
        "Jegadeesh (1990) Journal of Finance 45(3):881-898; "
        "Lehmann (1990) Quarterly Journal of Economics 105(1):1-28"
    ),
    asset_class      = "equity_singlename",
    formula_summary  = (
        "Trailing 21-trading-day total return → cross-section z-score. "
        "Raw factor sign-aligned with past return (no flip in compute); "
        "expected_sign=-1 in registry encodes reversal direction (high z → low future)."
    ),
    signal_fn        = compute_strev_singlestock_signal,
    expected_sign    = -1,   # past return → negative correlation with future per Jegadeesh 1990
))
