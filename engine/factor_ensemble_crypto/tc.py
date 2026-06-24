"""
engine/factor_ensemble_crypto/tc.py — locked TC per spec §2.4.

Per spec id=71 hash 48db143d §六 forbidden modifications, TC is hard-locked
at 25 bp roundtrip per rebalance event. Modifying requires spec_amend with
HARKing R3 review.

Per `feedback_etf_tc_tier_model.md` tier-specific TC audit:
  - Retail Binance taker fee:    10 bp / leg
  - Bid-ask spread BTC/ETH spot:  0.01-0.05 bp (Tier 1 liquid)
  - Slippage $10k order:          < 2 bp
  - Per leg total:                ~12 bp
  - Roundtrip (2 legs):           24 bp
  - Locked:                       25 bp (1 bp buffer)
"""
from __future__ import annotations

import pandas as pd


# Spec §2.4 — hard-locked
TC_BPS_PER_EVENT_LOCKED: float = 25.0


def apply_tc_to_returns(
    monthly_gross_returns: pd.Series,
    position_changes:      pd.Series,
    tc_bps_per_event:      float = TC_BPS_PER_EVENT_LOCKED,
) -> pd.Series:
    """
    Subtract per-event transaction cost from monthly gross returns.

    A "rebalance event" occurs whenever position changes from the previous
    month. For the L/S 50%-each TSMOM portfolio, a sign flip on either asset
    triggers a full position swap = roundtrip TC on that asset's 50% leg.

    Args:
        monthly_gross_returns: pd.Series indexed by month-end date,
            values = portfolio gross return for that month
        position_changes: pd.Series same index,
            values = number of legs that flipped this month (0, 1, or 2 for BTC+ETH)
        tc_bps_per_event: TC per leg flip in bps

    Returns:
        Monthly net returns = gross - (n_legs_flipped × 25 bp × 0.5 leg weight)
    """
    if not monthly_gross_returns.index.equals(position_changes.index):
        raise ValueError("monthly_gross_returns and position_changes index mismatch")
    tc_per_leg_decimal = (tc_bps_per_event / 10_000.0) * 0.5
    tc_drag = position_changes.fillna(0).astype(float) * tc_per_leg_decimal
    return monthly_gross_returns - tc_drag
