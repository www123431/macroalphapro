"""
engine/portfolio/multisig_regime_overlay.py — Path AM impl.

Spec: docs/spec_path_am_multisig_regime_overlay_v3_v1.md
Spec id=85, hash 13272afc5b36be9e663283f201813cc2df67af82 (v3 overlay class).

Multi-signal composite (VIX + Credit OAS + 2s10s) regime classifier driving
dynamic allocation across 3 grids (CALM / NORMAL / STRESS).
"""
from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WINDOW_START: str = "2014-09-12"
WINDOW_END:   str = "2023-12-29"

Z_WINDOW_W: int = 52    # 12-month rolling for z-score
REGIME_THRESHOLD: float = 1.0   # ±1σ

TC_BPS_PER_SIDE:    float = 4.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0

# Locked per spec §2.4
ALLOCATION_GRID = {
    "CALM":   {"K1_BAB": 0.380, "D_PEAD": 0.270, "PATH_N": 0.230, "CTA_PQTIX": 0.070, "AC_proxy_AB_2014_23": 0.050},
    "NORMAL": {"K1_BAB": 0.324, "D_PEAD": 0.243, "PATH_N": 0.243, "CTA_PQTIX": 0.090, "AC_proxy_AB_2014_23": 0.100},
    "STRESS": {"K1_BAB": 0.260, "D_PEAD": 0.195, "PATH_N": 0.195, "CTA_PQTIX": 0.140, "AC_proxy_AB_2014_23": 0.210},
}


@dataclass(frozen=True)
class AMBacktestResult:
    weekly_returns_static: pd.Series       # static 5-sleeve baseline
    weekly_returns_am:     pd.Series       # post-overlay
    weekly_returns_am_net: pd.Series       # net of TC
    weekly_regime:         pd.Series       # CALM/NORMAL/STRESS per week
    weekly_composite:      pd.Series       # composite z-score
    weekly_tc_drag:        pd.Series
    n_weeks:               int
    n_rebalances:          int
    pct_calm:              float
    pct_normal:            float
    pct_stress:            float
    notes:                 list[str] = field(default_factory=list)


def load_fred_signals() -> pd.DataFrame:
    """Pull VIX / BAA10Y / DGS2 / DGS10 from FRED, resample W-FRI."""
    import pandas_datareader.data as web
    df = web.DataReader(
        ["VIXCLS", "BAA10Y", "DGS2", "DGS10"], "fred",
        datetime.date(2013, 1, 1),
        datetime.date(2024, 1, 15),
    )
    df.index = pd.to_datetime(df.index)
    df["spread_2s10s"] = df["DGS10"] - df["DGS2"]
    weekly = df.resample("W-FRI").last()
    return weekly[["VIXCLS", "BAA10Y", "spread_2s10s"]].dropna()


def compute_composite(signals: pd.DataFrame) -> pd.Series:
    """Standardize each signal via 12-mo rolling z-score, equal-weight average."""
    def _rolling_z(s: pd.Series) -> pd.Series:
        med = s.rolling(Z_WINDOW_W, min_periods=Z_WINDOW_W).median()
        std_ = s.rolling(Z_WINDOW_W, min_periods=Z_WINDOW_W).std()
        return (s - med) / std_

    vix_z = _rolling_z(signals["VIXCLS"])
    oas_z = _rolling_z(signals["BAA10Y"])
    # 2s10s: negative spread = bad → negate so high = stress
    curve_z = _rolling_z(-signals["spread_2s10s"])

    composite = (vix_z + oas_z + curve_z) / 3.0
    return composite


def classify_regime(composite: pd.Series) -> pd.Series:
    """Map composite z-score to regime label."""
    def _classify(z: float) -> str:
        if pd.isna(z): return "NORMAL"
        if z > REGIME_THRESHOLD: return "STRESS"
        if z < -REGIME_THRESHOLD: return "CALM"
        return "NORMAL"
    return composite.apply(_classify)


def build_rebalance_dates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    last_month = None
    for d in weekly.index:
        ym = (d.year, d.month)
        if ym != last_month:
            dates.append(d); last_month = ym
    return dates


def run_am_backtest() -> AMBacktestResult:
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Load 5-sleeve panel
    ex = pd.read_parquet(repo_root / "data/portfolio_replay/v2_per_strategy_returns_5sleeve_weekly.parquet")
    ex = ex.astype("float64").fillna(0.0)
    ex.index = pd.to_datetime(ex.index)

    # Static 5-sleeve baseline
    static_w = pd.Series(ALLOCATION_GRID["NORMAL"], dtype=float)
    static_ret = pd.Series(0.0, index=ex.index)
    for col, w in static_w.items():
        if col in ex.columns:
            static_ret += w * ex[col]

    # Load FRED signals
    signals = load_fred_signals()
    composite = compute_composite(signals)
    regime = classify_regime(composite)

    # Align indices
    common = ex.index.intersection(regime.index)
    static_ret_a = static_ret.loc[common]
    regime_a = regime.loc[common]
    composite_a = composite.loc[common]

    # Filter to spec window
    spec_mask = (common >= pd.Timestamp(WINDOW_START)) & (common <= pd.Timestamp(WINDOW_END))
    common_window = common[spec_mask]

    # Determine regime AT each monthly rebalance date; apply to following month
    rebal_dates = build_rebalance_dates(ex.loc[common_window])
    rebal_set = set(rebal_dates)

    # Build dynamic weights series (apply per regime at each rebalance)
    current_regime = "NORMAL"
    weekly_am_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:    dict[pd.Timestamp, float] = {}
    weekly_regime_used: dict[pd.Timestamp, str] = {}
    last_weights = pd.Series(ALLOCATION_GRID["NORMAL"], dtype=float)
    n_rebals_executed = 0

    weeks = list(common_window)
    for i, week in enumerate(weeks):
        # Apply current weights × this week's per-sleeve returns
        week_ret = 0.0
        for col, w in last_weights.items():
            if col in ex.columns:
                week_ret += float(w) * float(ex.loc[week, col])
        weekly_am_ret[week] = week_ret
        weekly_tc[week] = 0.0
        weekly_regime_used[week] = current_regime

        # Rebalance on first weekly bar of each month
        if week in rebal_set:
            new_regime = regime_a.loc[week] if week in regime_a.index else "NORMAL"
            if new_regime != current_regime:
                # Compute turnover from old to new allocation
                new_weights = pd.Series(ALLOCATION_GRID[new_regime], dtype=float)
                all_idx = last_weights.index.union(new_weights.index)
                old_w = last_weights.reindex(all_idx, fill_value=0.0)
                new_w = new_weights.reindex(all_idx, fill_value=0.0)
                turnover = float((new_w - old_w).abs().sum())
                tc = turnover * TC_DECIMAL_PER_SIDE * 2.0
                weekly_tc[week] = tc
                last_weights = new_weights
                current_regime = new_regime
                n_rebals_executed += 1

    am_gross = pd.Series(weekly_am_ret, name="am_gross")
    tcs = pd.Series(weekly_tc, name="am_tc")
    am_net = (am_gross - tcs).rename("am_net")
    regime_used = pd.Series(weekly_regime_used, name="regime")

    # Compute static baseline returns in same window for comparison
    static_window = static_ret_a.loc[common_window]

    pct_c = float((regime_used == "CALM").sum() / len(regime_used) * 100)
    pct_n = float((regime_used == "NORMAL").sum() / len(regime_used) * 100)
    pct_s = float((regime_used == "STRESS").sum() / len(regime_used) * 100)

    notes = [
        f"CALM: {pct_c:.1f}% · NORMAL: {pct_n:.1f}% · STRESS: {pct_s:.1f}%",
        f"Z-window={Z_WINDOW_W}w · regime threshold=±{REGIME_THRESHOLD}σ",
    ]

    return AMBacktestResult(
        weekly_returns_static = static_window,
        weekly_returns_am     = am_gross,
        weekly_returns_am_net = am_net,
        weekly_regime         = regime_used,
        weekly_composite      = composite_a.loc[common_window],
        weekly_tc_drag        = tcs,
        n_weeks               = len(weeks),
        n_rebalances          = n_rebals_executed,
        pct_calm              = pct_c,
        pct_normal            = pct_n,
        pct_stress            = pct_s,
        notes                 = notes,
    )


def save_am_parquet(
    result: AMBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_am_multisig_overlay_weekly.parquet",
) -> Path:
    p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "static_return": result.weekly_returns_static,
        "am_gross":      result.weekly_returns_am,
        "am_net":        result.weekly_returns_am_net,
        "regime":        result.weekly_regime,
        "composite":     result.weekly_composite,
        "tc":            result.weekly_tc_drag,
    })
    df.index.name = "week_end"; df.to_parquet(p); return p
