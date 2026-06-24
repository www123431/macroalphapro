"""
engine/factor_ensemble_v2/regime.py — 4-regime classifier.

Pre-registration: docs/spec_factor_ensemble_v2_robust.md §2.5

Regime defined by SPY 252-day return + 60-day realized vol at as_of:
  - bull_low_vol:  ret_252d > 0  AND vol_60d ≤ 0.18
  - bull_high_vol: ret_252d > 0  AND vol_60d > 0.18
  - bear_low_vol:  ret_252d ≤ 0  AND vol_60d ≤ 0.18
  - bear_high_vol: ret_252d ≤ 0  AND vol_60d > 0.18

Thresholds locked pre-launch per Moreira-Muir 2017 + Goyal-Welch 2008 conventions.
"""
from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd

# Locked per spec §2.5 — DO NOT data-tune
REGIMES_LOCKED: tuple[str, ...] = (
    "bull_low_vol",
    "bull_high_vol",
    "bear_low_vol",
    "bear_high_vol",
)
REGIME_RETURN_THRESHOLD_LOCKED: float = 0.0    # SPY 252d return cut (Goyal-Welch 2008)
REGIME_VOL_THRESHOLD_LOCKED:    float = 0.18   # 18% annualized vol (Moreira-Muir 2017)
REGIME_RETURN_WINDOW_DAYS:      int   = 252
REGIME_VOL_WINDOW_DAYS:         int   = 60
TRADING_DAYS_PER_YEAR:          int   = 252


def classify_regime(
    spy_ret_252d:  float,
    spy_vol_60d:   float,
) -> str:
    """Return one of REGIMES_LOCKED based on locked thresholds.

    NaN inputs (insufficient history) → returns 'bull_low_vol' as default
    (least-restrictive bucket; caller should detect and exclude per spec §3.2).
    """
    if not np.isfinite(spy_ret_252d) or not np.isfinite(spy_vol_60d):
        return "bull_low_vol"  # default; caller checks
    is_bull = spy_ret_252d > REGIME_RETURN_THRESHOLD_LOCKED
    is_high_vol = spy_vol_60d > REGIME_VOL_THRESHOLD_LOCKED
    if is_bull and not is_high_vol:
        return "bull_low_vol"
    if is_bull and is_high_vol:
        return "bull_high_vol"
    if (not is_bull) and not is_high_vol:
        return "bear_low_vol"
    return "bear_high_vol"


def classify_regime_series(
    panel: pd.DataFrame,
    rebalance_dates: list[datetime.date],
    benchmark: str = "SPY",
) -> pd.Series:
    """Classify each rebalance date's regime using SPY price panel.

    Returns:
        pd.Series indexed by rebalance_date, value in REGIMES_LOCKED.
    """
    if panel is None or panel.empty or benchmark not in panel.columns:
        return pd.Series({d: "bull_low_vol" for d in rebalance_dates}, dtype="object")

    spy = panel[benchmark].dropna()
    if spy.empty:
        return pd.Series({d: "bull_low_vol" for d in rebalance_dates}, dtype="object")

    out: dict[datetime.date, str] = {}
    for d in rebalance_dates:
        end_ts = pd.Timestamp(d - datetime.timedelta(days=1))
        # 252-day return
        ret_start = pd.Timestamp(d - datetime.timedelta(days=REGIME_RETURN_WINDOW_DAYS + 30))
        ret_window = spy.loc[(spy.index >= ret_start) & (spy.index <= end_ts)]
        if len(ret_window) < REGIME_RETURN_WINDOW_DAYS // 2:
            ret_252d = float("nan")
        else:
            ret_252d = float(ret_window.iloc[-1] / ret_window.iloc[0] - 1)
        # 60-day realized vol
        vol_start = pd.Timestamp(d - datetime.timedelta(days=REGIME_VOL_WINDOW_DAYS + 30))
        vol_window = spy.loc[(spy.index >= vol_start) & (spy.index <= end_ts)]
        if len(vol_window) < REGIME_VOL_WINDOW_DAYS // 2:
            vol_60d = float("nan")
        else:
            rets = vol_window.pct_change().dropna().tail(REGIME_VOL_WINDOW_DAYS)
            vol_60d = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
        out[d] = classify_regime(ret_252d, vol_60d)
    return pd.Series(out, dtype="object")
