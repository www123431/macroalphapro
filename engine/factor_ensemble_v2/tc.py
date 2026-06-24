"""
engine/factor_ensemble_v2/tc.py — Transaction cost modeling.

Pre-registration: docs/spec_factor_ensemble_v2_robust.md §2.2
Locked: 8bps roundtrip per Frazzini-Pedersen 2014 §3 + Pedersen 2015 §6 mid-tier ETF convention.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Locked per spec §2.2 — do NOT data-tune
TC_BPS_ROUNDTRIP_LOCKED: float = 8.0


def compute_tc_drag(
    weights_new: pd.Series,
    weights_prev: Optional[pd.Series] = None,
    bps_roundtrip: float = TC_BPS_ROUNDTRIP_LOCKED,
) -> float:
    """Compute TC drag = turnover × (bps / 10000).

    Turnover = ½ Σ |w_new - w_prev|. First period (no prev) uses ½ Σ |w_new|
    (full establishment cost from cash).

    Returns: TC drag as fraction of NAV (e.g. 0.0005 = 5bps drag).
    """
    if weights_new is None or weights_new.empty:
        return 0.0
    if weights_prev is None or weights_prev.empty:
        gross_change = float(weights_new.abs().sum())
    else:
        # Align on union of tickers, treat missing as 0 weight
        all_idx = weights_new.index.union(weights_prev.index)
        diff = weights_new.reindex(all_idx).fillna(0.0) - weights_prev.reindex(all_idx).fillna(0.0)
        gross_change = float(diff.abs().sum())
    turnover = gross_change / 2.0
    drag = turnover * (bps_roundtrip / 10000.0)
    return drag


def apply_tc_to_realized_returns(
    realized_returns_gross: pd.Series,
    turnover_per_period:    pd.Series,
    bps_roundtrip:          float = TC_BPS_ROUNDTRIP_LOCKED,
) -> pd.Series:
    """Vectorized: subtract per-period TC drag from gross realized returns.

    Args:
        realized_returns_gross: pd.Series indexed by rebalance date
        turnover_per_period:    pd.Series indexed identically; 0-1 scale
        bps_roundtrip:          locked value, exposed for testing only

    Returns:
        Series of net realized returns same shape.
    """
    if realized_returns_gross is None or realized_returns_gross.empty:
        return pd.Series(dtype=float)
    if turnover_per_period is None or turnover_per_period.empty:
        return realized_returns_gross.copy()
    aligned_turnover = turnover_per_period.reindex(realized_returns_gross.index).fillna(0.0)
    drag = aligned_turnover * (bps_roundtrip / 10000.0)
    return realized_returns_gross - drag
