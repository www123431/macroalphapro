"""
engine/path_f/vix_signal.py — VIX term structure signal + risk management.

Pre-registration: docs/spec_path_f_vix_term_structure_v1.md (id=65) §2.2 + §2.3
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spec §六 locked
CONTANGO_RATIO_THRESHOLD_LOCKED = 0.95
STOP_LOSS_DAILY_THRESHOLD_LOCKED = -0.20
COOLING_OFF_DAYS_LOCKED          = 30
WINSORIZE_LOWER_LOCKED           = -0.50
WINSORIZE_UPPER_LOCKED           = +0.50
TC_BPS_PER_POSITION_CHANGE_LOCKED = 6.0


def compute_signal(panel: pd.DataFrame) -> pd.Series:
    """Compute daily target position from VIX/VIX3M ratio.

    Returns Series indexed by trading dates with values ∈ {0.0, 1.0}.
    Position is "target for next trading day" (1-day execution lag).
    """
    ratio = panel["VIX"] / panel["VIX3M"]
    target = (ratio <= CONTANGO_RATIO_THRESHOLD_LOCKED).astype(float)
    # Shift by 1: signal at close d → position at open d+1
    target_lagged = target.shift(1).fillna(0.0)
    return target_lagged


def apply_risk_management(
    target_position:    pd.Series,
    svxy_daily_returns: pd.Series,
) -> tuple[pd.Series, list[dict]]:
    """Apply stop-loss + cooling-off + winsorize rules.

    Returns:
      effective_position: actual position after risk management
      stop_loss_events:   list of {date, svxy_return, cooling_off_until}
    """
    n = len(svxy_daily_returns)
    effective = target_position.copy()
    stop_loss_events = []

    dates = svxy_daily_returns.index
    cooling_off_until_idx = -1   # day index when cooling-off ends; -1 = no active

    for i in range(n):
        date_i = dates[i]
        ret_i  = svxy_daily_returns.iloc[i]

        # Apply cooling-off: if currently in cooling-off, force 0 position
        if i < cooling_off_until_idx:
            effective.iloc[i] = 0.0

        # Check for stop-loss trigger: SVXY single-day return < -20%
        # Use ACTUAL position * return (since trigger is based on what we'd have lost)
        pos_today = effective.iloc[i]
        realized_today = pos_today * ret_i
        if realized_today < STOP_LOSS_DAILY_THRESHOLD_LOCKED:
            cooling_off_end = i + COOLING_OFF_DAYS_LOCKED + 1   # +1 next day
            stop_loss_events.append({
                "date":                  date_i.date() if hasattr(date_i, "date") else date_i,
                "svxy_return":           float(ret_i),
                "position_before":       float(pos_today),
                "realized_pnl":          float(realized_today),
                "cooling_off_until_idx": cooling_off_end,
            })
            cooling_off_until_idx = max(cooling_off_until_idx, cooling_off_end)
            # Tomorrow's position forced to 0 (cooling-off in effect)
            if i + 1 < n:
                effective.iloc[i + 1] = 0.0

    return effective, stop_loss_events


def compute_strategy_returns(
    effective_position: pd.Series,
    svxy_daily_returns: pd.Series,
) -> pd.Series:
    """Compute net daily strategy returns:
       - Apply winsorize to SVXY returns
       - Multiply by position
       - Subtract TC on position-change days
    """
    svxy_clipped = svxy_daily_returns.clip(WINSORIZE_LOWER_LOCKED, WINSORIZE_UPPER_LOCKED)
    gross = effective_position * svxy_clipped

    # TC: charged when position changes from previous day
    pos_prev = effective_position.shift(1).fillna(0.0)
    position_changed = (effective_position != pos_prev).astype(float)
    tc_drag = position_changed * (TC_BPS_PER_POSITION_CHANGE_LOCKED / 10000.0)

    net = gross - tc_drag
    return net


def derive_trade_log(effective_position: pd.Series) -> pd.DataFrame:
    """Extract individual long-position trades from effective position series.

    Each contiguous run of position=1.0 → 1 trade.
    """
    pos_diff = effective_position.diff().fillna(effective_position.iloc[0])
    entries = effective_position.index[(pos_diff > 0) & (effective_position == 1.0)]
    exits   = effective_position.index[(pos_diff < 0) & (effective_position == 0.0)]

    # Align entries/exits
    trades = []
    for entry in entries:
        # Find next exit after this entry
        future_exits = [e for e in exits if e > entry]
        exit_date = future_exits[0] if future_exits else effective_position.index[-1]
        n_days = (exit_date - entry).days
        trades.append({
            "entry_date": entry.date() if hasattr(entry, "date") else entry,
            "exit_date":  exit_date.date() if hasattr(exit_date, "date") else exit_date,
            "n_calendar_days": int(n_days),
        })
    return pd.DataFrame(trades)
