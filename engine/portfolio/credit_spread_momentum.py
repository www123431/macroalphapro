"""
engine/portfolio/credit_spread_momentum.py — Path AL impl.

Spec: docs/spec_path_al_credit_spread_v3_v1.md
Spec id=84, hash 0af2db7fd922ff92f45243ca22dcbf7c75ab8714.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WINDOW_START: str = "2013-08-01"
WINDOW_END:   str = "2024-01-15"
UNIVERSE: tuple[str, ...] = ("HYG", "LQD")
SIGNAL_WINDOW_W: int = 26   # 6 months
TC_BPS_PER_SIDE: float = 4.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0


@dataclass(frozen=True)
class ALBacktestResult:
    weekly_returns_gross: pd.Series
    weekly_returns_net:   pd.Series
    weekly_tc_drag:       pd.Series
    weekly_position:      pd.Series
    n_weeks:              int
    n_rebalances:         int
    pct_in_hyg:           float
    pct_in_lqd:           float
    notes:                list[str] = field(default_factory=list)


def load_panel() -> pd.DataFrame:
    import yfinance as _yf
    daily = _yf.download(list(UNIVERSE), start=WINDOW_START, end=WINDOW_END,
                          auto_adjust=True, progress=False, multi_level_index=False)
    if "Close" in daily.columns: daily = daily["Close"]
    daily.index = pd.to_datetime(daily.index)
    weekly = daily.resample("W-FRI").last()
    return weekly[list(UNIVERSE)].dropna()


def build_rebalance_dates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    last_month = None
    for d in weekly.index:
        ym = (d.year, d.month)
        if ym != last_month:
            dates.append(d); last_month = ym
    return dates


def run_al_backtest() -> ALBacktestResult:
    weekly = load_panel()
    weekly_returns = weekly.pct_change()

    # Signal: HYG 6m return - LQD 6m return
    hyg_6m = weekly["HYG"] / weekly["HYG"].shift(SIGNAL_WINDOW_W) - 1.0
    lqd_6m = weekly["LQD"] / weekly["LQD"].shift(SIGNAL_WINDOW_W) - 1.0
    signal = hyg_6m - lqd_6m

    rebal_dates = build_rebalance_dates(weekly)
    rebal_set = set(rebal_dates)

    weekly_gross_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:        dict[pd.Timestamp, float] = {}
    weekly_position:  dict[pd.Timestamp, str] = {}
    current_holding: str | None = None
    n_rebals = 0

    weeks = list(weekly.index)
    for i, week in enumerate(weeks):
        if i > 0 and current_holding is not None:
            r_t = float(weekly_returns.iloc[i].get(current_holding, 0.0))
            if np.isnan(r_t): r_t = 0.0
            weekly_gross_ret[week] = r_t
        else:
            weekly_gross_ret[week] = 0.0
        weekly_tc[week] = 0.0
        weekly_position[week] = current_holding or "NONE"

        if week in rebal_set:
            sig = signal.loc[week]
            if pd.isna(sig): continue
            new_holding = "HYG" if sig > 0 else "LQD"
            if new_holding != current_holding:
                tc = 1.0 * TC_DECIMAL_PER_SIDE * 2.0
                weekly_tc[week] = tc
            current_holding = new_holding
            weekly_position[week] = current_holding
            n_rebals += 1

    gross = pd.Series(weekly_gross_ret, name="al_gross")
    tcs = pd.Series(weekly_tc, name="al_tc")
    net = (gross - tcs).rename("al_net")
    position = pd.Series(weekly_position, name="position")

    pct_h = float((position == "HYG").sum() / len(position) * 100)
    pct_l = float((position == "LQD").sum() / len(position) * 100)

    return ALBacktestResult(
        weekly_returns_gross=gross, weekly_returns_net=net, weekly_tc_drag=tcs,
        weekly_position=position, n_weeks=len(weeks), n_rebalances=n_rebals,
        pct_in_hyg=pct_h, pct_in_lqd=pct_l,
        notes=[f"HYG: {pct_h:.1f}% · LQD: {pct_l:.1f}%"],
    )


def save_al_parquet(
    result: ALBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_al_credit_spread_weekly.parquet",
) -> Path:
    p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"gross": result.weekly_returns_gross, "tc": result.weekly_tc_drag,
                       "net": result.weekly_returns_net, "position": result.weekly_position})
    df.index.name = "week_end"; df.to_parquet(p); return p
