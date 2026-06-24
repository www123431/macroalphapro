"""
Shared transaction cost utility (P1.2 deliverable, 2026-05-07).

Single function `compute_cost_bps()` is the canonical entry point for
weight-delta → bps cost conversion. Three live sites (portfolio_tracker
trades, daily_batch auto_stops, memory.py human_stop approvals) plus
backtest's flat-fallback path delegate here.

Kept deliberately narrow: ATR-based dynamic cost lives in backtest.py
(_atr_transaction_cost) because it requires a multi-sector return window
that live single-trade writes do not have. This module covers the
single-leg case and provides a `vol_aware` branch when caller can supply
a daily return window for the sector being traded.

Numerical behavior unchanged from the prior `abs(weight_delta) * 10`
pattern at portfolio_tracker.py:413 / daily_batch.py:730 / memory.py:5941.
Refactor is structural, not value-changing.

See docs/transaction_cost_model.md for the 3-context rationale.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

# ── Module-level constants (mirror engine/backtest.py for consistency) ───
LIVE_FLAT_COST_BPS: float = 10.0      # one-way, ETF rebal mid-literature
TC_FLOOR_BPS:       float = 3.0       # minimum half-spread for liquid US ETFs
TC_VOL_SCALE:       float = 0.15      # spread ≈ 15% of daily vol
TC_ATR_WINDOW:      int   = 14        # trailing window in trading days


def compute_cost_bps(
    weight_delta:     float,
    daily_ret_window: Optional[pd.Series] = None,
    floor_bps:        float = TC_FLOOR_BPS,
    vol_scale:        float = TC_VOL_SCALE,
    flat_fallback_bps: float = LIVE_FLAT_COST_BPS,
) -> float:
    """
    One-way transaction cost in basis points for a single trade leg.

    Args:
        weight_delta : signed or unsigned weight change. Sign is irrelevant —
                       cost is computed on |weight_delta|.
        daily_ret_window : optional trailing daily-return series for the
                       sector being traded. When provided with ≥5 obs,
                       cost scales with vol; else flat fallback.
        floor_bps    : floor for vol-aware half-spread (default 3 bps).
        vol_scale    : multiplier on daily-vol for half-spread estimate
                       (default 0.15 ≈ 15% of daily vol).
        flat_fallback_bps : flat cost used when no return window is
                       provided or window has <5 observations
                       (default 10 bps, matches prior live behavior).

    Returns:
        cost in basis points (e.g. 3.5 means 3.5 bps), one-way.

    Numerical contract:
        compute_cost_bps(d) == abs(d) * 10.0  (when daily_ret_window is None)
        — preserves the prior `round(abs(Δw) * 10, 2)` behavior at all
        three live sites after caller applies its own rounding.
    """
    abs_delta = abs(weight_delta)

    if daily_ret_window is None or len(daily_ret_window) < 5:
        return abs_delta * flat_fallback_bps

    vol_daily = float(daily_ret_window.dropna().tail(TC_ATR_WINDOW).std())
    if vol_daily != vol_daily or vol_daily <= 0:   # NaN guard
        return abs_delta * flat_fallback_bps

    half_spread_bps = max(floor_bps, vol_daily * vol_scale * 10_000.0)
    return abs_delta * half_spread_bps
