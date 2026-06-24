"""
engine/portfolio/commodity_momentum.py — Path AK impl.

Spec: docs/spec_path_ak_commodity_momentum_v3_v1.md
Spec id=83, hash daf5f9a62d1562b744575ca224622ac71e6adde2 (v3 alpha class).

Cross-sectional 12-1 momentum on 5 commodity ETFs. Long-only top-2 by rank.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WINDOW_START: str = "2013-08-01"   # buffer for 52w lookback
WINDOW_END:   str = "2024-01-15"

UNIVERSE: tuple[str, ...] = ("DBA", "DBB", "DBE", "DBO", "SLV")

LOOKBACK_WEEKS: int = 52
SKIP_WEEKS:     int = 4
N_LONG:         int = 2   # top-2 long-only

TC_BPS_PER_SIDE:    float = 4.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0


@dataclass(frozen=True)
class AKBacktestResult:
    weekly_returns_gross: pd.Series
    weekly_returns_net:   pd.Series
    weekly_tc_drag:       pd.Series
    rebalance_dates:      list[pd.Timestamp]
    n_weeks:              int
    n_rebalances:         int
    avg_turnover:         float
    notes:                list[str] = field(default_factory=list)


def load_panel() -> pd.DataFrame:
    import yfinance as _yf
    daily = _yf.download(
        list(UNIVERSE), start=WINDOW_START, end=WINDOW_END,
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    if "Close" in daily.columns:
        daily = daily["Close"]
    daily.index = pd.to_datetime(daily.index)
    weekly = daily.resample("W-FRI").last()
    return weekly[list(UNIVERSE)].dropna()


def compute_signal(weekly: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """12-1 cumulative return (skip 4w)."""
    if t not in weekly.index:
        prior = weekly.index[weekly.index <= t]
        if len(prior) == 0:
            return pd.Series(dtype=float)
        t = prior[-1]
    idx_t = weekly.index.get_loc(t)
    if idx_t < LOOKBACK_WEEKS:
        return pd.Series(dtype=float)
    p_now = weekly.iloc[idx_t - SKIP_WEEKS]
    p_then = weekly.iloc[idx_t - LOOKBACK_WEEKS]
    sig = p_now / p_then - 1.0
    return sig.where(p_now.notna() & p_then.notna() & (p_now > 0) & (p_then > 0))


def select_top2(signal: pd.Series) -> pd.Series:
    valid = signal.dropna()
    if len(valid) < N_LONG:
        return pd.Series(dtype=float)
    top = valid.sort_values(ascending=False).head(N_LONG).index
    return pd.Series(1.0 / N_LONG, index=top)


def build_rebalance_dates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    last_month = None
    for i, d in enumerate(weekly.index):
        ym = (d.year, d.month)
        if ym != last_month and i >= LOOKBACK_WEEKS:
            dates.append(d)
            last_month = ym
        elif ym != last_month:
            last_month = ym
    return dates


def run_ak_backtest() -> AKBacktestResult:
    weekly = load_panel()
    weekly_returns = weekly.pct_change()

    rebal_dates = build_rebalance_dates(weekly)
    rebal_set = set(rebal_dates)

    positions_history: dict[pd.Timestamp, pd.Series] = {}
    current_weights = pd.Series(dtype=float)
    weekly_gross_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:        dict[pd.Timestamp, float] = {}

    weeks = list(weekly.index)
    for i, week in enumerate(weeks):
        if i > 0 and not current_weights.empty:
            r_t = weekly_returns.iloc[i].reindex(current_weights.index).fillna(0.0)
            port_ret = float((current_weights * r_t).sum())
            weekly_gross_ret[week] = port_ret
        else:
            weekly_gross_ret[week] = 0.0
        weekly_tc[week] = 0.0

        if week in rebal_set:
            sig = compute_signal(weekly, week)
            new_w = select_top2(sig)
            if new_w.empty:
                continue
            all_tk = current_weights.index.union(new_w.index)
            w_old = current_weights.reindex(all_tk, fill_value=0.0)
            w_new = new_w.reindex(all_tk, fill_value=0.0)
            turnover = float((w_new - w_old).abs().sum())
            tc = turnover * TC_DECIMAL_PER_SIDE
            weekly_tc[week] = tc
            positions_history[week] = new_w.copy()
            current_weights = new_w

    gross = pd.Series(weekly_gross_ret, name="ak_gross")
    tcs   = pd.Series(weekly_tc, name="ak_tc")
    net   = (gross - tcs).rename("ak_net")

    rebal_list = sorted(positions_history.keys())
    turnovers = []
    for i, d in enumerate(rebal_list):
        if i == 0: continue
        prev = positions_history[rebal_list[i-1]]; curr = positions_history[d]
        tk = prev.index.union(curr.index)
        turnovers.append(float((curr.reindex(tk, fill_value=0) - prev.reindex(tk, fill_value=0)).abs().sum()))
    avg_to = float(np.mean(turnovers)) if turnovers else 0.0

    return AKBacktestResult(
        weekly_returns_gross=gross, weekly_returns_net=net, weekly_tc_drag=tcs,
        rebalance_dates=rebal_list, n_weeks=len(weeks), n_rebalances=len(rebal_list),
        avg_turnover=avg_to,
    )


def save_ak_parquet(
    result: AKBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_ak_commodity_mom_weekly.parquet",
) -> Path:
    p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"gross": result.weekly_returns_gross, "tc": result.weekly_tc_drag, "net": result.weekly_returns_net})
    df.index.name = "week_end"; df.to_parquet(p); return p
