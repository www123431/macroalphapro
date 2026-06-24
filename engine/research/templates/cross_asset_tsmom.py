"""engine/research/templates/cross_asset_tsmom.py — time-series momentum.

Per-instrument trend following (Moskowitz-Ooi-Pedersen 2012 form). Distinct
from cross-sectional momentum (equity_xsmom) and from cross-sectional bond
momentum (would be cross_asset_xsmom):

  TSMOM:  sign(instrument's own past return) → instrument's own position
  XSMOM:  rank across instruments → top minus bottom L/S

Pure composition of Layer 0 primitives.

Binding schema:
  lookback_months           — int (e.g. 12)
  skip_months               — int (e.g. 1 for canonical 12-1)
  per_instrument_vol_target — float (e.g. 0.40 — per-instrument vol target
                                       per MOP 2012 footnote 9)
  per_instrument_vol_lookback — int (default 36)
  rebal_freq                — "monthly" (v1 only)
  cost_bps_per_side         — float (e.g. 12.0)
  n_min_instruments         — int (skip months with < this many live)
  agg_method                — "equal_weight" — equal-weight across N live
                                instruments (MOP 2012 canonical)

Inputs (provided in data_kwargs):
  return_panel:  wide DataFrame, dates × instruments, monthly returns

Returns:
  pd.Series of monthly TSMOM-aggregated net-of-cost returns (vol-targeted at
  per-instrument level; book-level vol scaling NOT applied here — that
  belongs in book construction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.research import primitives as P

# Per [[project-gate-production-redesign-2026-05-30]]:
# 12-month TSMOM hold → strong residual autocorrelation → HAC lags=18.
# Cross-asset (futures/rates/FX) → pead_control=False (PEAD = US equity only).
# Grid: (lookback × vol_target × per_inst_vol_lookback) ≈ 25 trials.
GATE_PROFILE = {
    "hac_lags":         18,
    "cost_bps_default": 3,
    "pead_control":     False,
    "n_trials_base":    25,
}


def warmup_months(binding: dict) -> int:
    """Months of NaN-warmup. rolling_sum(lookback) + apply_lag(1) +
    per-instrument vol target lookback."""
    b = binding or {}
    warmup = int(b.get("lookback_months", 12)) + 1
    warmup += int(b.get("per_instrument_vol_lookback", 36))
    return warmup


def run_cross_asset_tsmom(*,
                            return_panel: pd.DataFrame,
                            universe: str | None = None,
                            lookback_months: int = 12,
                            skip_months: int = 1,
                            per_instrument_vol_target: float = 0.40,
                            per_instrument_vol_lookback: int = 36,
                            rebal_freq: str = "monthly",
                            cost_bps_per_side: float = 12.0,
                            n_min_instruments: int = 4,
                            agg_method: str = "equal_weight",
                            ) -> pd.Series:
    """Per-instrument 12-1 TSMOM, equal-weight aggregated.

    Pipeline:
    1. rolling_sum(returns, lookback, skip) → per-instrument momentum signal
       (anti-look-ahead lag built in)
    2. sign() → per-instrument position direction
    3. Per-instrument vol target: position scaled by target_vol / realized_vol
    4. Aggregate across instruments (equal-weight)
    5. apply_round_trip_cost

    Note: book-level vol target NOT applied — that's a sleeve composition
    decision, not a per-strategy decision.
    """
    if rebal_freq != "monthly":
        raise NotImplementedError("v1 supports monthly rebal only")
    if agg_method != "equal_weight":
        raise NotImplementedError("v1 supports equal_weight aggregation only")

    # 1. Per-instrument momentum signal
    signal = P.rolling_sum(return_panel,
                             window=lookback_months,
                             skip=skip_months)

    # 2. Direction sign
    direction = np.sign(signal)

    # 3. Per-instrument vol target
    # Realized vol uses shift(1).rolling(lookback) to prevent look-ahead
    realized_vol = (return_panel.rolling(per_instrument_vol_lookback).std()
                     * np.sqrt(12)).shift(1)
    vol_scale = per_instrument_vol_target / realized_vol.replace(0, np.nan)
    # Position size at month t = direction at t-1 × vol scale at t
    # (apply_lag enforces no look-ahead on direction)
    direction_lagged = P.apply_lag(direction, n_periods=1)
    position = direction_lagged * vol_scale

    # 4. Per-instrument contribution to month t = position × return at t
    per_inst_contrib = position * return_panel

    # 5. Equal-weight aggregation: NaN-skipping mean across live instruments,
    # then NaN out months with < n_min_instruments live signals
    n_live = (~position.isna()).sum(axis=1)
    agg = per_inst_contrib.mean(axis=1).where(n_live >= n_min_instruments)

    # 6. Cost (full-portfolio turnover ~ proportional to sign-flip frequency)
    # Conservative: assume 100% monthly turnover for cost calc
    agg = P.apply_round_trip_cost(agg, bps_per_side=cost_bps_per_side, turnover=1.0)

    return agg.rename("cross_asset_tsmom_ls")
