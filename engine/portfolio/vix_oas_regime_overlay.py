"""
engine/portfolio/vix_oas_regime_overlay.py — Path AN-1 impl.

Spec: docs/spec_path_an1_vix_oas_regime_overlay_v3_v1.md
Spec id=86, hash 626db10f89f4d642c70d62d9e679011019c138d5.

Refinement of Path AM with curve signal DROPPED. VIX + OAS composite only.
All other parameters (Z window 52w, threshold ±1σ, allocation grids, monthly
rebalance) identical to AM per spec §2.1-§2.5.
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
Z_WINDOW_W: int = 52
REGIME_THRESHOLD: float = 1.0

TC_BPS_PER_SIDE:    float = 4.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0

# Same allocation grid as AM (locked in AM §2.4)
ALLOCATION_GRID = {
    "CALM":   {"K1_BAB": 0.380, "D_PEAD": 0.270, "PATH_N": 0.230, "CTA_PQTIX": 0.070, "AC_proxy_AB_2014_23": 0.050},
    "NORMAL": {"K1_BAB": 0.324, "D_PEAD": 0.243, "PATH_N": 0.243, "CTA_PQTIX": 0.090, "AC_proxy_AB_2014_23": 0.100},
    "STRESS": {"K1_BAB": 0.260, "D_PEAD": 0.195, "PATH_N": 0.195, "CTA_PQTIX": 0.140, "AC_proxy_AB_2014_23": 0.210},
}


@dataclass(frozen=True)
class AN1BacktestResult:
    weekly_returns_static: pd.Series
    weekly_returns_an:     pd.Series
    weekly_returns_an_net: pd.Series
    weekly_regime:         pd.Series
    weekly_composite:      pd.Series
    weekly_tc_drag:        pd.Series
    n_weeks:               int
    n_rebalances:          int
    pct_calm:              float
    pct_normal:            float
    pct_stress:            float


def load_fred_signals_vix_oas() -> pd.DataFrame:
    """Pull VIX + BAA10Y only (NO curve) from FRED."""
    import pandas_datareader.data as web
    df = web.DataReader(
        ["VIXCLS", "BAA10Y"], "fred",
        datetime.date(2013, 1, 1),
        datetime.date(2024, 1, 15),
    )
    df.index = pd.to_datetime(df.index)
    weekly = df.resample("W-FRI").last()
    return weekly[["VIXCLS", "BAA10Y"]].dropna()


def compute_composite_vix_oas(signals: pd.DataFrame) -> pd.Series:
    """Equal-weight z-score composite of VIX + OAS (2 signals not 3)."""
    def _rolling_z(s):
        med = s.rolling(Z_WINDOW_W, min_periods=Z_WINDOW_W).median()
        std_ = s.rolling(Z_WINDOW_W, min_periods=Z_WINDOW_W).std()
        return (s - med) / std_

    vix_z = _rolling_z(signals["VIXCLS"])
    oas_z = _rolling_z(signals["BAA10Y"])
    return (vix_z + oas_z) / 2.0


def classify_regime(composite: pd.Series) -> pd.Series:
    def _c(z):
        if pd.isna(z): return "NORMAL"
        if z > REGIME_THRESHOLD: return "STRESS"
        if z < -REGIME_THRESHOLD: return "CALM"
        return "NORMAL"
    return composite.apply(_c)


def build_rebalance_dates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    dates = []
    last_month = None
    for d in weekly.index:
        ym = (d.year, d.month)
        if ym != last_month:
            dates.append(d); last_month = ym
    return dates


def run_an1_backtest() -> AN1BacktestResult:
    repo_root = Path(__file__).resolve().parent.parent.parent
    ex = pd.read_parquet(repo_root / "data/portfolio_replay/v2_per_strategy_returns_5sleeve_weekly.parquet")
    ex = ex.astype("float64").fillna(0.0)
    ex.index = pd.to_datetime(ex.index)

    static_w = pd.Series(ALLOCATION_GRID["NORMAL"], dtype=float)
    static_ret = pd.Series(0.0, index=ex.index)
    for col, w in static_w.items():
        if col in ex.columns: static_ret += w * ex[col]

    signals = load_fred_signals_vix_oas()
    composite = compute_composite_vix_oas(signals)
    regime = classify_regime(composite)

    common = ex.index.intersection(regime.index)
    static_ret_a = static_ret.loc[common]
    regime_a = regime.loc[common]
    composite_a = composite.loc[common]

    spec_mask = (common >= pd.Timestamp(WINDOW_START)) & (common <= pd.Timestamp(WINDOW_END))
    common_window = common[spec_mask]

    rebal_dates = build_rebalance_dates(ex.loc[common_window])
    rebal_set = set(rebal_dates)

    current_regime = "NORMAL"
    last_weights = pd.Series(ALLOCATION_GRID["NORMAL"], dtype=float)
    weekly_an_ret, weekly_tc, weekly_regime_used = {}, {}, {}
    n_rebals = 0

    for i, week in enumerate(common_window):
        week_ret = 0.0
        for col, w in last_weights.items():
            if col in ex.columns:
                week_ret += float(w) * float(ex.loc[week, col])
        weekly_an_ret[week] = week_ret
        weekly_tc[week] = 0.0
        weekly_regime_used[week] = current_regime

        if week in rebal_set:
            new_regime = regime_a.loc[week] if week in regime_a.index else "NORMAL"
            if new_regime != current_regime:
                new_weights = pd.Series(ALLOCATION_GRID[new_regime], dtype=float)
                all_idx = last_weights.index.union(new_weights.index)
                old_w = last_weights.reindex(all_idx, fill_value=0.0)
                new_w = new_weights.reindex(all_idx, fill_value=0.0)
                turnover = float((new_w - old_w).abs().sum())
                tc = turnover * TC_DECIMAL_PER_SIDE * 2.0
                weekly_tc[week] = tc
                last_weights = new_weights
                current_regime = new_regime
                n_rebals += 1

    an_gross = pd.Series(weekly_an_ret, name="an_gross")
    tcs = pd.Series(weekly_tc, name="an_tc")
    an_net = (an_gross - tcs).rename("an_net")
    regime_used = pd.Series(weekly_regime_used, name="regime")
    static_window = static_ret_a.loc[common_window]

    pct_c = float((regime_used == "CALM").sum() / len(regime_used) * 100)
    pct_n = float((regime_used == "NORMAL").sum() / len(regime_used) * 100)
    pct_s = float((regime_used == "STRESS").sum() / len(regime_used) * 100)

    return AN1BacktestResult(
        weekly_returns_static=static_window, weekly_returns_an=an_gross,
        weekly_returns_an_net=an_net, weekly_regime=regime_used,
        weekly_composite=composite_a.loc[common_window], weekly_tc_drag=tcs,
        n_weeks=len(common_window), n_rebalances=n_rebals,
        pct_calm=pct_c, pct_normal=pct_n, pct_stress=pct_s,
    )


def save_an1_parquet(result: AN1BacktestResult,
                     save_path: str = "data/portfolio_replay/v1_path_an1_vix_oas_overlay_weekly.parquet") -> Path:
    p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "static_return": result.weekly_returns_static,
        "an_gross": result.weekly_returns_an,
        "an_net": result.weekly_returns_an_net,
        "regime": result.weekly_regime,
        "composite": result.weekly_composite,
        "tc": result.weekly_tc_drag,
    })
    df.index.name = "week_end"; df.to_parquet(p); return p
