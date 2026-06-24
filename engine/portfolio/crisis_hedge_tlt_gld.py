"""
engine/portfolio/crisis_hedge_tlt_gld.py — Path AB TLT/GLD Crisis Hedge Sleeve.

Spec: docs/spec_path_ab_tlt_gld_crisis_hedge_v1.md
Spec id=75, hash d79261179add3fcfd87f3156d6044eafd0c13810 (active, v2 gate framework).

Implements Brunnermeier-Pedersen 2009 flight-to-quality + Baur-Lucey 2010 gold
safe-haven mechanism via a fixed 50/50 long-only TLT + GLD passive sleeve.
NO signal layer per spec ss.2.2; monthly rebalance to exact 50/50 target.

Algorithm (locked):
  1. Universe: 2 ETFs (TLT, GLD)
  2. Weekly resample to W-FRI close
  3. Monthly rebalance (first weekly bar of each month)
  4. Target weights: w_TLT=0.5, w_GLD=0.5 (fixed)
  5. Drift between rebalances (allow weights to diverge based on returns)
  6. TC: 4bp per side per rebalance (ETF Tier-1 baseline)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WINDOW_START: str = "2014-09-12"
WINDOW_END:   str = "2023-12-29"

UNIVERSE: tuple[str, ...] = ("TLT", "GLD")

TARGET_WEIGHTS: dict[str, float] = {"TLT": 0.5, "GLD": 0.5}

TC_BPS_PER_SIDE:     float = 4.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0


@dataclass(frozen=True)
class ABBacktestResult:
    weekly_returns_gross: pd.Series
    weekly_returns_net:   pd.Series
    weekly_tc_drag:       pd.Series
    rebalance_dates:      list[pd.Timestamp]
    n_weeks:              int
    n_rebalances:         int
    avg_turnover:         float
    notes:                list[str] = field(default_factory=list)


def load_panel(
    window_start: str = WINDOW_START,
    window_end:   str = WINDOW_END,
) -> pd.DataFrame:
    """Load TLT + GLD weekly W-FRI panel via yfinance."""
    import yfinance as _yf
    daily = _yf.download(
        list(UNIVERSE),
        start=window_start, end=window_end,
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    if "Close" in daily.columns:
        daily = daily["Close"]
    daily.index = pd.to_datetime(daily.index)
    weekly = daily.resample("W-FRI").last()
    out = weekly[list(UNIVERSE)].dropna(how="all")
    return out


def build_rebalance_dates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    """First weekly bar of each month."""
    dates: list[pd.Timestamp] = []
    last_month: tuple[int, int] | None = None
    for d in weekly.index:
        ym = (d.year, d.month)
        if ym != last_month:
            dates.append(d)
            last_month = ym
    return dates


def run_ab_backtest() -> ABBacktestResult:
    weekly = load_panel()
    weekly_returns = weekly.pct_change()

    rebal_dates = build_rebalance_dates(weekly)
    if not rebal_dates:
        raise RuntimeError("No rebalance dates after warmup")
    rebal_set = set(rebal_dates)

    target = pd.Series(TARGET_WEIGHTS, dtype=float)

    current_weights = pd.Series(dtype=float)
    positions_history: dict[pd.Timestamp, pd.Series] = {}
    weekly_gross_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:        dict[pd.Timestamp, float] = {}

    weeks = list(weekly.index)
    for i, week in enumerate(weeks):
        # Apply weekly returns to prior weights (drift)
        if i > 0 and not current_weights.empty:
            r_t = weekly_returns.iloc[i].reindex(current_weights.index).fillna(0.0)
            port_ret = float((current_weights * r_t).sum())
            weekly_gross_ret[week] = port_ret
            # Drift weights: w_t = w_{t-1} * (1 + r_t) / (1 + port_ret)
            new_unscaled = current_weights * (1.0 + r_t)
            denom = float(new_unscaled.sum())
            if denom > 0:
                current_weights = new_unscaled / denom
        else:
            weekly_gross_ret[week] = 0.0
        weekly_tc[week] = 0.0

        # Rebalance at first weekly bar of each month
        if week in rebal_set:
            new_w = target.copy()
            all_tk = current_weights.index.union(new_w.index)
            w_old = current_weights.reindex(all_tk, fill_value=0.0)
            w_new = new_w.reindex(all_tk, fill_value=0.0)
            turnover = float((w_new - w_old).abs().sum())
            tc = turnover * TC_DECIMAL_PER_SIDE * 2.0
            weekly_tc[week] = tc
            positions_history[week] = new_w.copy()
            current_weights = new_w

    gross = pd.Series(weekly_gross_ret, name="ab_gross")
    tcs   = pd.Series(weekly_tc,        name="ab_tc")
    net   = (gross - tcs).rename("ab_net")

    rebal_list = sorted(positions_history.keys())
    turnovers = []
    for i, d in enumerate(rebal_list):
        if i == 0:
            continue
        prev = positions_history[rebal_list[i - 1]]
        curr = positions_history[d]
        tk = prev.index.union(curr.index)
        turnovers.append(float(
            (curr.reindex(tk, fill_value=0.0) - prev.reindex(tk, fill_value=0.0)).abs().sum()
        ))
    avg_to = float(np.mean(turnovers)) if turnovers else 0.0

    return ABBacktestResult(
        weekly_returns_gross = gross,
        weekly_returns_net   = net,
        weekly_tc_drag       = tcs,
        rebalance_dates      = rebal_list,
        n_weeks              = len(weeks),
        n_rebalances         = len(rebal_list),
        avg_turnover         = avg_to,
        notes                = [],
    )


def save_ab_parquet(
    result:    ABBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_ab_tlt_gld_crisis_hedge_weekly.parquet",
) -> Path:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "gross": result.weekly_returns_gross,
        "tc":    result.weekly_tc_drag,
        "net":   result.weekly_returns_net,
    })
    df.index.name = "week_end"
    df.to_parquet(p)
    return p
