"""
engine/portfolio/vol_target_overlay_5sleeve.py — Path AJ impl.

Spec: docs/spec_path_aj_vol_target_overlay_5sleeve_v3_v1.md
Spec id=82, hash 568541f0e514acda9ea710244a88ca59da100af3 (v3 overlay class).

Moreira-Muir 2017 vol-target overlay applied to 5-sleeve baseline.
Conservative leverage clamp [0.5, 1.5]; σ_target=8% (matches 5-sleeve historical vol).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Locked per spec §2.2
SIGMA_TARGET_ANN: float = 0.08
SIGMA_WINDOW_W:   int   = 22
SCALE_CLAMP_LO:   float = 0.5
SCALE_CLAMP_HI:   float = 1.5

# 5-sleeve production weights (per PAPER_TRADE_SLEEVE_ALLOCATION post-AC)
SLEEVE_5_WEIGHTS = {
    "K1_BAB":              0.324,
    "D_PEAD":              0.243,
    "PATH_N":              0.243,
    "CTA_PQTIX":           0.090,
    "AC_proxy_AB_2014_23": 0.100,
}

TC_BPS_PER_SIDE:    float = 4.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0


@dataclass(frozen=True)
class AJBacktestResult:
    weekly_returns_base:    pd.Series   # 5-sleeve baseline
    weekly_returns_aj:      pd.Series   # post-overlay
    weekly_returns_aj_net:  pd.Series   # net of TC
    weekly_scale:           pd.Series
    weekly_tc_drag:         pd.Series
    n_weeks:                int
    avg_scale:              float
    pct_clamped_low:        float
    pct_clamped_high:       float
    notes:                  list[str] = field(default_factory=list)


def run_aj_backtest() -> AJBacktestResult:
    repo_root = Path(__file__).resolve().parent.parent.parent
    ex = pd.read_parquet(
        repo_root / "data/portfolio_replay/v2_per_strategy_returns_5sleeve_weekly.parquet"
    ).astype("float64").fillna(0.0)
    ex.index = pd.to_datetime(ex.index)

    # Compose 5-sleeve baseline
    base = pd.Series(0.0, index=ex.index, name="base_return")
    for col, w in SLEEVE_5_WEIGHTS.items():
        if col in ex.columns:
            base = base + w * ex[col]

    # Rolling realized vol (annualized)
    rolling_std = base.rolling(SIGMA_WINDOW_W, min_periods=SIGMA_WINDOW_W).std()
    sigma_realized_ann = rolling_std * math.sqrt(52)

    # Scale = sigma_target / sigma_realized, clipped
    raw_scale = SIGMA_TARGET_ANN / sigma_realized_ann.replace(0.0, np.nan)
    scale_series = raw_scale.clip(SCALE_CLAMP_LO, SCALE_CLAMP_HI).fillna(1.0)

    # AJ return = scale(t-1) × base(t)
    scale_lag = scale_series.shift(1).fillna(1.0)
    aj_gross = (scale_lag * base).rename("aj_gross")

    # TC = |scale(t) - scale(t-1)| × TC_decimal_per_side × 2 (round trip)
    scale_delta = scale_series.diff().abs().fillna(0.0)
    tc_drag = scale_delta * TC_DECIMAL_PER_SIDE * 2.0
    aj_net = (aj_gross - tc_drag).rename("aj_net")

    # Stats
    post_warmup = scale_series.iloc[SIGMA_WINDOW_W:]
    avg_scale = float(post_warmup.mean())
    pct_lo = float((raw_scale.iloc[SIGMA_WINDOW_W:] <= SCALE_CLAMP_LO).sum() / len(post_warmup))
    pct_hi = float((raw_scale.iloc[SIGMA_WINDOW_W:] >= SCALE_CLAMP_HI).sum() / len(post_warmup))

    notes = []
    notes.append(f"σ_target={SIGMA_TARGET_ANN} σ_window={SIGMA_WINDOW_W}w clamp=[{SCALE_CLAMP_LO}, {SCALE_CLAMP_HI}]")
    notes.append(f"avg scale post-warmup={avg_scale:.3f}; clamped LO {pct_lo*100:.1f}% / HI {pct_hi*100:.1f}%")

    return AJBacktestResult(
        weekly_returns_base   = base,
        weekly_returns_aj     = aj_gross,
        weekly_returns_aj_net = aj_net,
        weekly_scale          = scale_series,
        weekly_tc_drag        = tc_drag,
        n_weeks               = len(base),
        avg_scale             = avg_scale,
        pct_clamped_low       = pct_lo,
        pct_clamped_high      = pct_hi,
        notes                 = notes,
    )


def save_aj_parquet(
    result:    AJBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_aj_voltarget_5sleeve_weekly.parquet",
) -> Path:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "base_return": result.weekly_returns_base,
        "aj_gross":    result.weekly_returns_aj,
        "aj_net":      result.weekly_returns_aj_net,
        "scale":       result.weekly_scale,
        "tc":          result.weekly_tc_drag,
    })
    df.index.name = "week_end"
    df.to_parquet(p)
    return p
