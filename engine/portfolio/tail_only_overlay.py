"""
engine/portfolio/tail_only_overlay.py — Path AO impl.

Spec: docs/spec_path_ao_tail_only_overlay_v3_v1.md
Spec id=87, hash 4387b7bfb1e747f09cf47ec5e2706696e2f10689 (v3 overlay class).

Tail-only asymmetric crisis overlay. Stricter threshold (+1.5σ) than AN-1.
NO CALM regime. Bigger AC bump (25% vs 21%) when STRESS active.
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from engine.portfolio.vix_oas_regime_overlay import (
    load_fred_signals_vix_oas, compute_composite_vix_oas, build_rebalance_dates,
    Z_WINDOW_W,
)

WINDOW_START = "2014-09-12"
WINDOW_END   = "2023-12-29"

# Locked per AO spec §2.2
TAIL_THRESHOLD: float = 1.5

TC_BPS_PER_SIDE: float = 4.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0

# Locked per AO spec §2.3
STATIC_WEIGHTS = {"K1_BAB": 0.324, "D_PEAD": 0.243, "PATH_N": 0.243, "CTA_PQTIX": 0.090, "AC_proxy_AB_2014_23": 0.100}
STRESS_WEIGHTS = {"K1_BAB": 0.250, "D_PEAD": 0.180, "PATH_N": 0.180, "CTA_PQTIX": 0.140, "AC_proxy_AB_2014_23": 0.250}


@dataclass(frozen=True)
class AOBacktestResult:
    weekly_returns_static: pd.Series
    weekly_returns_ao:     pd.Series
    weekly_returns_ao_net: pd.Series
    weekly_regime:         pd.Series
    weekly_composite:      pd.Series
    weekly_tc_drag:        pd.Series
    n_weeks:               int
    n_rebalances:          int
    pct_stress:            float
    pct_static:            float


def classify_regime_tail_only(composite: pd.Series) -> pd.Series:
    """STRESS only if composite > +1.5σ; otherwise STATIC. NO CALM."""
    def _c(z):
        if pd.isna(z): return "STATIC"
        if z > TAIL_THRESHOLD: return "STRESS"
        return "STATIC"
    return composite.apply(_c)


def run_ao_backtest() -> AOBacktestResult:
    repo_root = Path(__file__).resolve().parent.parent.parent
    ex = pd.read_parquet(repo_root / "data/portfolio_replay/v2_per_strategy_returns_5sleeve_weekly.parquet")
    ex = ex.astype("float64").fillna(0.0)
    ex.index = pd.to_datetime(ex.index)

    # Static baseline (default 5-sleeve weights)
    static_ret = pd.Series(0.0, index=ex.index)
    for col, w in STATIC_WEIGHTS.items():
        if col in ex.columns: static_ret += w * ex[col]

    # Load signals + compute composite (reuse AN-1 helpers)
    signals = load_fred_signals_vix_oas()
    composite = compute_composite_vix_oas(signals)
    regime = classify_regime_tail_only(composite)

    common = ex.index.intersection(regime.index)
    static_ret_a = static_ret.loc[common]
    regime_a = regime.loc[common]
    composite_a = composite.loc[common]

    spec_mask = (common >= pd.Timestamp(WINDOW_START)) & (common <= pd.Timestamp(WINDOW_END))
    common_window = common[spec_mask]

    rebal_dates = build_rebalance_dates(ex.loc[common_window])
    rebal_set = set(rebal_dates)

    current_regime = "STATIC"
    last_weights = pd.Series(STATIC_WEIGHTS, dtype=float)
    weekly_ao_ret = {}
    weekly_tc = {}
    weekly_regime_used = {}
    n_rebals = 0

    for i, week in enumerate(common_window):
        week_ret = 0.0
        for col, w in last_weights.items():
            if col in ex.columns:
                week_ret += float(w) * float(ex.loc[week, col])
        weekly_ao_ret[week] = week_ret
        weekly_tc[week] = 0.0
        weekly_regime_used[week] = current_regime

        if week in rebal_set:
            new_regime = regime_a.loc[week] if week in regime_a.index else "STATIC"
            if new_regime != current_regime:
                if new_regime == "STRESS":
                    new_weights = pd.Series(STRESS_WEIGHTS, dtype=float)
                else:
                    new_weights = pd.Series(STATIC_WEIGHTS, dtype=float)
                all_idx = last_weights.index.union(new_weights.index)
                old_w = last_weights.reindex(all_idx, fill_value=0.0)
                new_w = new_weights.reindex(all_idx, fill_value=0.0)
                turnover = float((new_w - old_w).abs().sum())
                tc = turnover * TC_DECIMAL_PER_SIDE * 2.0
                weekly_tc[week] = tc
                last_weights = new_weights
                current_regime = new_regime
                n_rebals += 1

    ao_gross = pd.Series(weekly_ao_ret, name="ao_gross")
    tcs = pd.Series(weekly_tc, name="ao_tc")
    ao_net = (ao_gross - tcs).rename("ao_net")
    regime_used = pd.Series(weekly_regime_used, name="regime")
    static_window = static_ret_a.loc[common_window]

    pct_s = float((regime_used == "STRESS").sum() / len(regime_used) * 100)
    pct_static = float((regime_used == "STATIC").sum() / len(regime_used) * 100)

    return AOBacktestResult(
        weekly_returns_static=static_window, weekly_returns_ao=ao_gross,
        weekly_returns_ao_net=ao_net, weekly_regime=regime_used,
        weekly_composite=composite_a.loc[common_window], weekly_tc_drag=tcs,
        n_weeks=len(common_window), n_rebalances=n_rebals,
        pct_stress=pct_s, pct_static=pct_static,
    )


def save_ao_parquet(result: AOBacktestResult,
                    save_path: str = "data/portfolio_replay/v1_path_ao_tail_only_weekly.parquet") -> Path:
    p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "static_return": result.weekly_returns_static,
        "ao_gross": result.weekly_returns_ao,
        "ao_net": result.weekly_returns_ao_net,
        "regime": result.weekly_regime,
        "composite": result.weekly_composite,
        "tc": result.weekly_tc_drag,
    })
    df.index.name = "week_end"; df.to_parquet(p); return p
