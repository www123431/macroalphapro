"""
engine/factors/tsmom.py — Time-Series Momentum factor (Moskowitz-Ooi-Pedersen 2012).

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50) §2.2.1
Spec lock:
  - Lookback 12 months
  - Skip 1 month (avoid 1-month reversal)
  - Vol window 60 trading days
  - VXX sign-flip (volatility short polarity)
  - Applies to ALL 45 ETFs (universal across asset classes)

This module wraps existing `engine.signal.get_signal_dataframe` which already
computes TSMOM via the same locked methodology (Hurst-Ooi-Pedersen 2017
multi-lookback ensemble). Per spec §rule-9 N7, no separate implementation —
reuse production-validated signal computation to maintain consistency.

NaN protocol (per spec §2.3):
  - Insufficient history (ETF inception > t - 13 months) → NaN
  - All NaN ticker → 0 fallback in ensemble combiner
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Locked parameters (per spec §2.2.1)
LOOKBACK_MONTHS: int = 12
SKIP_MONTHS: int = 1
VOL_WINDOW_DAYS: int = 60


def compute_tsmom_signal(
    as_of:          datetime.date,
    universe:       list[str],
    asset_classes:  Optional[dict[str, str]] = None,
    use_cache:      bool = False,
) -> pd.Series:
    """
    Compute per-ticker TSMOM signal at as_of.

    Returns:
        pd.Series indexed by ticker, values are raw TSMOM signal.
        NaN for tickers with insufficient history (< 13 months data).

    Args:
        as_of:         signal computation date (no look-ahead, uses ≤ t-1 data)
        universe:      list of ETF tickers to compute for
        asset_classes: optional {ticker: asset_class}; not used by TSMOM (universal)
        use_cache:     pass-through to engine.signal.get_signal_dataframe

    Note: signature matches all factor modules even though asset_classes is unused
    (architectural consistency for ensemble combiner).
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)

    try:
        from engine.signal import get_signal_dataframe
        df = get_signal_dataframe(
            as_of=as_of,
            lookback_months=LOOKBACK_MONTHS,
            skip_months=SKIP_MONTHS,
            use_cache=use_cache,
        )
    except Exception as exc:
        logger.warning(
            "tsmom: get_signal_dataframe failed for %s: %s — returning all-NaN",
            as_of, exc,
        )
        return pd.Series(np.nan, index=universe, dtype=float)

    if df is None or df.empty or "tsmom" not in df.columns or "ticker" not in df.columns:
        logger.warning(
            "tsmom: signal_df missing tsmom/ticker columns for %s — returning all-NaN",
            as_of,
        )
        return pd.Series(np.nan, index=universe, dtype=float)

    # Re-index by ticker (signal_df index is sector_name)
    df_by_ticker = df.set_index("ticker")
    raw_signal = df_by_ticker["tsmom"].reindex(universe)

    # ETFs absent from signal_df (insufficient history) → NaN per spec §2.3 protocol
    return raw_signal.astype(float)
