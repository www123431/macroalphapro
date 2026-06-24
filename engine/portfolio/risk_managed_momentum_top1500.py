"""
engine/portfolio/risk_managed_momentum_top1500.py — Path AE impl.

Spec: docs/spec_path_ae_risk_managed_momentum_top1500_v3_v1.md
Spec id=79, hash 8353c298ba982585a7f6971b86aba46eca058eef.

Daniel-Moskowitz 2016 dynamic vol-scaling overlay on the same 12-1 cumulative
momentum top-1500 signal as Path AD. Weekly vol-scaling between monthly
position rebalances.

Locked parameters (per spec §2.3):
  SIGMA_TARGET_ANN  = 0.18    (18% annualized)
  SIGMA_WINDOW_W    = 22      (rolling weeks for vol estimation)
  SCALE_CLAMP       = (0.5, 2.0)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from engine.portfolio.cross_sectional_momentum_top1500 import (
    load_panel, compute_signal, select_top_decile, build_rebalance_dates,
    TC_DECIMAL_PER_SIDE, LOOKBACK_WEEKS,
)

logger = logging.getLogger(__name__)


# Locked per spec §2.3 (Daniel-Moskowitz 2016 paper §3 normalization)
SIGMA_TARGET_ANN: float = 0.18
SIGMA_WINDOW_W:   int   = 22
SCALE_CLAMP_LO:   float = 0.5
SCALE_CLAMP_HI:   float = 2.0


@dataclass(frozen=True)
class AEBacktestResult:
    weekly_returns_gross: pd.Series
    weekly_returns_net:   pd.Series
    weekly_tc_drag:       pd.Series
    weekly_scale:         pd.Series
    rebalance_dates:      list[pd.Timestamp]
    n_weeks:              int
    n_rebalances:         int
    avg_scale:            float
    pct_clamped_low:      float
    pct_clamped_high:     float
    notes:                list[str] = field(default_factory=list)


def run_ae_backtest() -> AEBacktestResult:
    weekly = load_panel()
    weekly_returns = weekly.pct_change()

    rebal_dates = build_rebalance_dates(weekly)
    if not rebal_dates:
        raise RuntimeError("No rebalance dates after warmup")
    rebal_set = set(rebal_dates)

    positions_history: dict[pd.Timestamp, pd.Series] = {}
    current_weights = pd.Series(dtype=float)
    base_gross_ret: dict[pd.Timestamp, float] = {}     # AD-style unscaled
    weekly_tc:      dict[pd.Timestamp, float] = {}
    weekly_scale:   dict[pd.Timestamp, float] = {}
    n_skipped = 0

    weeks = list(weekly.index)
    last_scale = 1.0

    # Pass 1: compute base (unscaled) weekly returns + monthly rebalance TC
    for i, week in enumerate(weeks):
        if i > 0 and not current_weights.empty:
            r_t = weekly_returns.iloc[i].reindex(current_weights.index).fillna(0.0)
            port_ret = float((current_weights * r_t).sum())
            base_gross_ret[week] = port_ret
        else:
            base_gross_ret[week] = 0.0
        weekly_tc[week] = 0.0

        if week in rebal_set:
            sig = compute_signal(weekly, week)
            new_w = select_top_decile(sig)
            if new_w.empty:
                n_skipped += 1
                continue
            all_tk = current_weights.index.union(new_w.index)
            w_old = current_weights.reindex(all_tk, fill_value=0.0)
            w_new = new_w.reindex(all_tk, fill_value=0.0)
            turnover = float((w_new - w_old).abs().sum())
            tc = turnover * TC_DECIMAL_PER_SIDE
            weekly_tc[week] = tc
            positions_history[week] = new_w.copy()
            current_weights = new_w

    base_gross = pd.Series(base_gross_ret, name="ae_base_gross")
    base_tc    = pd.Series(weekly_tc, name="ae_base_tc")

    # Pass 2: compute weekly scale from rolling vol of base gross returns
    # σ_realized(t) = std(base_gross[t-22:t]) × sqrt(52); scale(t) = clip(σ_target/σ_realized, 0.5, 2.0)
    scale_series = pd.Series(np.nan, index=base_gross.index)
    rolled_std = base_gross.rolling(SIGMA_WINDOW_W, min_periods=SIGMA_WINDOW_W).std()
    sigma_realized_ann = rolled_std * math.sqrt(52)
    raw_scale = SIGMA_TARGET_ANN / sigma_realized_ann.replace(0.0, np.nan)
    scale_series = raw_scale.clip(SCALE_CLAMP_LO, SCALE_CLAMP_HI).fillna(1.0)

    # Pass 3: apply scale(t-1) to base_gross(t); compute scale-delta TC
    ae_gross = pd.Series(0.0, index=base_gross.index)
    scale_delta_tc = pd.Series(0.0, index=base_gross.index)
    scale_lag = scale_series.shift(1).fillna(1.0)
    pct_clamped_lo_count = 0
    pct_clamped_hi_count = 0

    for i, week in enumerate(weeks):
        s_now = scale_lag.iloc[i]
        ae_gross.iloc[i] = s_now * base_gross.iloc[i]
        if i > 0:
            s_prev = scale_lag.iloc[i - 1]
            delta_scale = abs(s_now - s_prev)
            # Scale delta TC: gross-position change in scale × current weights × TC
            # Approximated: turnover = abs(delta_scale) × 1.0 (gross long-only)
            scale_delta_tc.iloc[i] = delta_scale * TC_DECIMAL_PER_SIDE
        if i >= SIGMA_WINDOW_W:
            if raw_scale.iloc[i] <= SCALE_CLAMP_LO:
                pct_clamped_lo_count += 1
            elif raw_scale.iloc[i] >= SCALE_CLAMP_HI:
                pct_clamped_hi_count += 1

    total_tc = base_tc + scale_delta_tc
    ae_net = ae_gross - total_tc

    n_post_warmup = len(weeks) - SIGMA_WINDOW_W
    pct_lo = pct_clamped_lo_count / max(1, n_post_warmup)
    pct_hi = pct_clamped_hi_count / max(1, n_post_warmup)
    avg_scale_post_warmup = float(scale_series.iloc[SIGMA_WINDOW_W:].mean())

    notes = []
    if n_skipped:
        notes.append(f"{n_skipped} rebalance(s) skipped (insufficient signals)")
    notes.append(f"σ_target={SIGMA_TARGET_ANN} σ_window={SIGMA_WINDOW_W}w clamp=[{SCALE_CLAMP_LO},{SCALE_CLAMP_HI}]")
    notes.append(f"avg scale post-warmup={avg_scale_post_warmup:.3f}; "
                 f"clamped LO {pct_lo*100:.1f}% / HI {pct_hi*100:.1f}%")

    return AEBacktestResult(
        weekly_returns_gross = ae_gross,
        weekly_returns_net   = ae_net,
        weekly_tc_drag       = total_tc,
        weekly_scale         = scale_series,
        rebalance_dates      = sorted(positions_history.keys()),
        n_weeks              = len(weeks),
        n_rebalances         = len(positions_history),
        avg_scale            = avg_scale_post_warmup,
        pct_clamped_low      = pct_lo,
        pct_clamped_high     = pct_hi,
        notes                = notes,
    )


def save_ae_parquet(
    result:    AEBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_ae_rm_momentum_top1500_weekly.parquet",
) -> Path:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "gross": result.weekly_returns_gross,
        "tc":    result.weekly_tc_drag,
        "net":   result.weekly_returns_net,
        "scale": result.weekly_scale,
    })
    df.index.name = "week_end"
    df.to_parquet(p)
    return p
