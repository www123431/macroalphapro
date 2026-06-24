"""
engine/portfolio/yield_curve_momentum.py — Path AI impl.

Spec: docs/spec_path_ai_yield_curve_momentum_v3_v1.md
Spec id=81, hash 39999bb0bf7e9d2888387a9303d566547fa22950.

Yield curve momentum on IEF (7-10y Treasury) / SHY (1-3y Treasury).
Signal: 2s10s spread 6-month rolling change from FRED DGS2/DGS10.
Position: long IEF when signal > 0 (steepening), long SHY otherwise.
Monthly rebalance, 4bp/side TC.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WINDOW_START: str = "2014-09-12"
WINDOW_END:   str = "2024-01-15"   # buffer

UNIVERSE: tuple[str, ...] = ("IEF", "SHY")
SIGNAL_WINDOW_W: int = 26   # 6 months = 26 weekly bars

TC_BPS_PER_SIDE:    float = 4.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0


@dataclass(frozen=True)
class AIBacktestResult:
    weekly_returns_gross: pd.Series
    weekly_returns_net:   pd.Series
    weekly_tc_drag:       pd.Series
    weekly_position:      pd.Series   # "IEF" or "SHY" each week
    signal_series:        pd.Series   # 2s10s 6m change
    rebalance_dates:      list[pd.Timestamp]
    n_weeks:              int
    n_rebalances:         int
    pct_in_ief:           float
    pct_in_shy:           float
    notes:                list[str] = field(default_factory=list)


def load_fred_2s10s_weekly() -> pd.Series:
    """Pull DGS2/DGS10 from FRED, resample to W-FRI, return spread."""
    import pandas_datareader.data as web
    df = web.DataReader(
        ["DGS10", "DGS2"], "fred",
        datetime.date(2013, 1, 1),   # buffer for 6m lookback
        datetime.date(2024, 1, 15),
    )
    df.index = pd.to_datetime(df.index)
    df["spread"] = df["DGS10"] - df["DGS2"]
    # Resample to W-FRI
    weekly = df["spread"].resample("W-FRI").last()
    return weekly.dropna()


def load_etf_panel() -> pd.DataFrame:
    """Pull IEF + SHY weekly W-FRI."""
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


def build_rebalance_dates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    last_month = None
    for d in weekly.index:
        ym = (d.year, d.month)
        if ym != last_month:
            dates.append(d)
            last_month = ym
    return dates


def run_ai_backtest() -> AIBacktestResult:
    # Load data
    spread_weekly = load_fred_2s10s_weekly()
    etf_weekly = load_etf_panel()

    # Align indices
    common_idx = etf_weekly.index.intersection(spread_weekly.index)
    spread_aligned = spread_weekly.loc[common_idx]
    etf_aligned = etf_weekly.loc[common_idx]

    # Compute 6m rolling change in 2s10s spread
    signal = spread_aligned - spread_aligned.shift(SIGNAL_WINDOW_W)

    # Weekly returns per ETF
    etf_returns = etf_aligned.pct_change()

    rebal_dates = build_rebalance_dates(etf_aligned)
    rebal_set = set(rebal_dates)

    weekly_gross_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:        dict[pd.Timestamp, float] = {}
    weekly_position:  dict[pd.Timestamp, str]   = {}
    current_holding: str | None = None   # "IEF" or "SHY"
    n_rebals_executed = 0

    weeks = list(etf_aligned.index)
    for i, week in enumerate(weeks):
        # Apply current position to this week's return
        if i > 0 and current_holding is not None:
            r_t = float(etf_returns.iloc[i].get(current_holding, 0.0))
            if np.isnan(r_t):
                r_t = 0.0
            weekly_gross_ret[week] = r_t
        else:
            weekly_gross_ret[week] = 0.0
        weekly_tc[week] = 0.0
        weekly_position[week] = current_holding or "NONE"

        # Rebalance check
        if week in rebal_set:
            sig = signal.loc[week]
            if pd.isna(sig):
                # Insufficient history for signal — stay or skip
                continue
            new_holding = "IEF" if sig > 0 else "SHY"
            if new_holding != current_holding:
                # Turnover = 100% (full switch)
                tc = 1.0 * TC_DECIMAL_PER_SIDE * 2.0   # 2-sided TC
                weekly_tc[week] = tc
            current_holding = new_holding
            weekly_position[week] = current_holding
            n_rebals_executed += 1

    gross = pd.Series(weekly_gross_ret, name="ai_gross")
    tcs   = pd.Series(weekly_tc, name="ai_tc")
    net   = (gross - tcs).rename("ai_net")
    position = pd.Series(weekly_position, name="position")

    pct_ief = float((position == "IEF").sum() / len(position) * 100)
    pct_shy = float((position == "SHY").sum() / len(position) * 100)

    notes = []
    notes.append(f"IEF holding: {pct_ief:.1f}% of weeks · SHY: {pct_shy:.1f}%")
    notes.append(f"Signal 2s10s 6m change: window={SIGNAL_WINDOW_W} weeks")

    return AIBacktestResult(
        weekly_returns_gross = gross,
        weekly_returns_net   = net,
        weekly_tc_drag       = tcs,
        weekly_position      = position,
        signal_series        = signal,
        rebalance_dates      = sorted([d for d in rebal_dates if d in rebal_set]),
        n_weeks              = len(weeks),
        n_rebalances         = n_rebals_executed,
        pct_in_ief           = pct_ief,
        pct_in_shy           = pct_shy,
        notes                = notes,
    )


def save_ai_parquet(
    result:    AIBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_ai_yield_curve_mom_weekly.parquet",
) -> Path:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "gross":    result.weekly_returns_gross,
        "tc":       result.weekly_tc_drag,
        "net":      result.weekly_returns_net,
        "position": result.weekly_position,
        "signal":   result.signal_series.reindex(result.weekly_returns_net.index),
    })
    df.index.name = "week_end"
    df.to_parquet(p)
    return p
