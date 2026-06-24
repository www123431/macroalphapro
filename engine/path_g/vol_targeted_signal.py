"""
engine/path_g/vol_targeted_signal.py — Vol-targeted SVXY position signal.

Pre-registration: docs/spec_path_g_vix_voltgt_v1.md (id=66) §2.2

Position = contango_signal × min(1.0, target_vol_annual / realized_vol_21d_annual)

Reuses Path F: VIX data fetch + risk management (stop-loss + cooling-off + winsorize)
+ TC drag + 5-gate backtest framework + IR-corrected incremental alpha test.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spec §六 locked
TARGET_VOL_ANNUAL_LOCKED       = 0.12       # Moskowitz-Ooi-Pedersen 2012 canonical
VOL_LOOKBACK_DAYS_LOCKED       = 21         # monthly convention
VOL_SCALE_CAP_LOCKED           = 1.0        # no leverage
CONTANGO_RATIO_THRESHOLD_LOCKED = 0.95      # same as Path F
TC_BPS_PER_POSITION_CHANGE_LOCKED = 6.0     # same as Path F (Tier 2 ETF)
WINSORIZE_LOWER_LOCKED         = -0.50      # same as Path F
WINSORIZE_UPPER_LOCKED         = +0.50      # same as Path F


def compute_vol_targeted_signal(
    panel:              pd.DataFrame,        # cols: VIX, VIX3M, SVXY
    svxy_daily_returns: pd.Series,
) -> pd.Series:
    """Compute daily target position with vol-targeting.

    position_d = contango_signal_d × min(1.0, TARGET_VOL / realized_vol_d)
    where realized_vol_d = std(svxy_returns[d-20:d+1], ddof=1) × √252.

    Returns Series indexed by trading dates with values ∈ [0, 1].
    Position is "target for next trading day" (1-day execution lag, same as Path F).
    """
    # Contango signal (same as Path F)
    ratio = panel["VIX"] / panel["VIX3M"]
    contango_signal = (ratio <= CONTANGO_RATIO_THRESHOLD_LOCKED).astype(float)

    # 21-day realized vol on SVXY returns (annualized)
    realized_vol_daily = svxy_daily_returns.rolling(window=VOL_LOOKBACK_DAYS_LOCKED,
                                                     min_periods=VOL_LOOKBACK_DAYS_LOCKED).std(ddof=1)
    realized_vol_ann = realized_vol_daily * np.sqrt(252)

    # Vol-scale (no leverage)
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_scale_raw = TARGET_VOL_ANNUAL_LOCKED / realized_vol_ann
    vol_scale_capped = vol_scale_raw.clip(upper=VOL_SCALE_CAP_LOCKED)
    vol_scale_capped = vol_scale_capped.fillna(0.0)  # early period before 21d lookback

    # Combine: position = contango × vol_scale
    target = contango_signal * vol_scale_capped

    # 1-day execution lag (signal at close d → position open d+1)
    target_lagged = target.shift(1).fillna(0.0)
    return target_lagged


def compute_strategy_returns_voltgt(
    effective_position: pd.Series,
    svxy_daily_returns: pd.Series,
) -> pd.Series:
    """Compute net daily strategy returns with vol-targeted position.

    Same TC + winsorize logic as Path F, but position now continuous (not binary)
    so TC paid on |Δposition| each day.
    """
    svxy_clipped = svxy_daily_returns.clip(WINSORIZE_LOWER_LOCKED, WINSORIZE_UPPER_LOCKED)
    gross = effective_position * svxy_clipped

    # TC on |Δposition| × 2 sides × 3bp roundtrip = 6bp × delta
    pos_prev = effective_position.shift(1).fillna(0.0)
    abs_delta = (effective_position - pos_prev).abs()
    tc_drag = abs_delta * (TC_BPS_PER_POSITION_CHANGE_LOCKED / 10000.0)

    net = gross - tc_drag
    return net
