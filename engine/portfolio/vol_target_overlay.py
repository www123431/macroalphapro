"""
engine/portfolio/vol_target_overlay.py — Path Y MM 2017 vol-target overlay.

Spec: docs/spec_path_y_portfolio_vol_target_overlay_v1.md
Spec id=72, current hash 4fc22eb8 (active).

Implements Moreira-Muir 2017 *JF* "Volatility-Managed Portfolios" overlay
on the existing 4-sleeve combined portfolio (Sprint B replay output).

Algorithm (locked, no parameter tuning post-impl):
  1. Read combined weekly returns from
     data/portfolio_replay/v1_combined_returns_weekly.parquet
  2. For each weekly bar t, compute strictly trailing realized vol:
       σ_realized_t = std(returns over [t-12, t-1]) × sqrt(52)
     STRICTLY trailing: window does NOT include t (Cederburg 2020 critique-free)
  3. Compute gross scale:
       raw_scale_t  = 0.10 / σ_realized_t                         (target σ = 10%)
       gross_scale_t = clamp(raw_scale_t, 0.0, 2.0)               (cap blow-up, floor at 0)
  4. Apply overlay:
       overlay_return_t = gross_scale_t × baseline_return_t       (already-OOS scale)
  5. TC drag at each rebalance bar:
       turnover_t = |gross_scale_t − gross_scale_{t-1}|
       tc_t       = turnover_t × 5bp/side × 2 sides = turnover_t × 0.001
  6. Net overlay return:
       overlay_net_t = overlay_return_t − tc_t

Doctrine: no LLM, no spec parameter changes post-impl, strict OOS scale
application (scale at t uses ONLY data up to t-1).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants (per spec — DO NOT mutate without amendment_log entry)
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_PATH:       str   = "data/portfolio_replay/v1_combined_returns_weekly.parquet"
LOOKBACK_WEEKS:      int   = 12       # strictly trailing
TARGET_SIGMA_ANN:    float = 0.10     # 10% annualized portfolio vol target
SCALE_FLOOR:         float = 0.0
SCALE_CAP:           float = 2.0
TC_BPS_PER_SIDE:     float = 5.0      # SPY/futures-level execution
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0


@dataclass(frozen=True)
class VolTargetResult:
    """Output of run_vol_target_overlay()."""
    baseline_returns:     pd.Series   # original combined weekly
    overlay_returns_gross: pd.Series   # scaled × baseline (no TC)
    overlay_returns_net:   pd.Series   # gross − tc
    overlay_tc_drag:       pd.Series   # per-week TC
    gross_scales:          pd.Series   # gross_scale_t
    realized_vol_ann:      pd.Series   # σ_realized_t annualized
    n_weeks_evaluated:     int         # weeks where overlay applied (after warmup)
    notes:                 list[str] = field(default_factory=list)


def run_vol_target_overlay(
    baseline_path:    str = BASELINE_PATH,
    lookback_weeks:   int = LOOKBACK_WEEKS,
    target_sigma_ann: float = TARGET_SIGMA_ANN,
    scale_floor:      float = SCALE_FLOOR,
    scale_cap:        float = SCALE_CAP,
    tc_decimal_side:  float = TC_DECIMAL_PER_SIDE,
) -> VolTargetResult:
    """Apply vol-target overlay to baseline 4-sleeve combined returns."""
    df = pd.read_parquet(baseline_path)
    df = df.astype("float64").fillna(0.0)
    df.index = pd.to_datetime(df.index)
    baseline = df["combined_return"].copy().rename("baseline")

    n = len(baseline)
    overlay_gross = pd.Series(np.nan, index=baseline.index, name="overlay_gross")
    overlay_net   = pd.Series(np.nan, index=baseline.index, name="overlay_net")
    overlay_tc    = pd.Series(0.0,    index=baseline.index, name="overlay_tc")
    gross_scales  = pd.Series(np.nan, index=baseline.index, name="gross_scale")
    sigma_ann     = pd.Series(np.nan, index=baseline.index, name="sigma_realized")

    prev_scale = 0.0
    for t in range(n):
        # Need at least lookback_weeks of prior data
        if t < lookback_weeks:
            continue

        # Strictly trailing window [t-12, t-1] (does NOT include t)
        window = baseline.iloc[t - lookback_weeks : t]
        sigma_t = float(window.std() * math.sqrt(52))
        sigma_ann.iloc[t] = sigma_t

        if sigma_t <= 0:
            scale_t = scale_floor
        else:
            raw = target_sigma_ann / sigma_t
            scale_t = max(scale_floor, min(scale_cap, raw))

        gross_scales.iloc[t] = scale_t

        # Apply to week t's baseline return
        base_ret_t = float(baseline.iloc[t])
        gross_ret_t = scale_t * base_ret_t
        overlay_gross.iloc[t] = gross_ret_t

        # TC drag on scale change
        turnover = abs(scale_t - prev_scale)
        tc_t = turnover * tc_decimal_side * 2.0
        overlay_tc.iloc[t] = tc_t

        overlay_net.iloc[t] = gross_ret_t - tc_t
        prev_scale = scale_t

    # n_weeks_evaluated = weeks with non-NaN overlay
    n_eval = int(overlay_net.notna().sum())

    return VolTargetResult(
        baseline_returns      = baseline,
        overlay_returns_gross = overlay_gross,
        overlay_returns_net   = overlay_net,
        overlay_tc_drag       = overlay_tc,
        gross_scales          = gross_scales,
        realized_vol_ann      = sigma_ann,
        n_weeks_evaluated     = n_eval,
        notes                 = [],
    )


def save_overlay_parquet(
    result:    VolTargetResult,
    save_path: str = "data/portfolio_replay/v1_path_y_voltarget_weekly.parquet",
) -> Path:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "baseline":         result.baseline_returns,
        "overlay_gross":    result.overlay_returns_gross,
        "overlay_net":      result.overlay_returns_net,
        "overlay_tc":       result.overlay_tc_drag,
        "gross_scale":      result.gross_scales,
        "sigma_realized":   result.realized_vol_ann,
    })
    df.index.name = "week_end"
    df.to_parquet(p)
    return p
