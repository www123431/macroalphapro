"""
engine/portfolio/crisis_hedge_tlt_gld_extended.py — Path AC TLT/GLD on extended window 2005-2023.

Spec: docs/spec_path_ac_tlt_gld_extended_v3_v1.md
Spec id=77, hash 4db40176056a882d0e365d45fea335599bed5182 (active, v3 insurance class).

Same instruments + weights as Path AB (TLT + GLD 50/50 monthly rebalance, 4bp/side TC).
Extended window 2005-01-07 -> 2023-12-29 (993 weeks).
60/40 SPY/AGG institutional baseline for G3/G7 (Asness-Israelov 2017 RMS standard).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WINDOW_START: str = "2005-01-07"
WINDOW_END:   str = "2024-01-15"  # buffer for W-FRI alignment; spec ends 2023-12-29

UNIVERSE: tuple[str, ...] = ("TLT", "GLD")
BASELINE_UNIVERSE: tuple[str, ...] = ("SPY", "AGG")
BASELINE_WEIGHTS: dict[str, float] = {"SPY": 0.60, "AGG": 0.40}

TARGET_WEIGHTS: dict[str, float] = {"TLT": 0.5, "GLD": 0.5}

TC_BPS_PER_SIDE:     float = 4.0
TC_DECIMAL_PER_SIDE: float = TC_BPS_PER_SIDE / 10_000.0


@dataclass(frozen=True)
class ACBacktestResult:
    weekly_returns_gross: pd.Series
    weekly_returns_net:   pd.Series
    weekly_tc_drag:       pd.Series
    baseline_60_40:       pd.Series
    rebalance_dates:      list[pd.Timestamp]
    n_weeks:              int
    n_rebalances:         int


def load_extended_panel() -> pd.DataFrame:
    """Load TLT, GLD, SPY, AGG weekly W-FRI 2005-2023."""
    import yfinance as _yf
    daily = _yf.download(
        list(UNIVERSE) + list(BASELINE_UNIVERSE),
        start=WINDOW_START, end=WINDOW_END,
        auto_adjust=True, progress=False, multi_level_index=False,
    )
    if "Close" in daily.columns:
        daily = daily["Close"]
    daily.index = pd.to_datetime(daily.index)
    weekly = daily.resample("W-FRI").last()
    return weekly.dropna(how="all")


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


def run_ac_backtest() -> ACBacktestResult:
    panel = load_extended_panel()
    tlt_gld = panel[list(UNIVERSE)].dropna()
    spy_agg = panel[list(BASELINE_UNIVERSE)].dropna()

    common_idx = tlt_gld.index.intersection(spy_agg.index)
    tlt_gld = tlt_gld.loc[common_idx]
    spy_agg = spy_agg.loc[common_idx]

    weekly_returns_tg = tlt_gld.pct_change()
    weekly_returns_sa = spy_agg.pct_change()

    rebal_dates = build_rebalance_dates(tlt_gld)
    if not rebal_dates:
        raise RuntimeError("No rebalance dates after warmup")
    rebal_set = set(rebal_dates)

    target = pd.Series(TARGET_WEIGHTS, dtype=float)
    baseline_target = pd.Series(BASELINE_WEIGHTS, dtype=float)

    current_weights = pd.Series(dtype=float)
    current_baseline_weights = pd.Series(dtype=float)
    weekly_gross_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:        dict[pd.Timestamp, float] = {}
    weekly_baseline:  dict[pd.Timestamp, float] = {}

    weeks = list(tlt_gld.index)
    n_rebals = 0

    for i, week in enumerate(weeks):
        # TLT/GLD strategy returns
        if i > 0 and not current_weights.empty:
            r_t = weekly_returns_tg.iloc[i].reindex(current_weights.index).fillna(0.0)
            port_ret = float((current_weights * r_t).sum())
            weekly_gross_ret[week] = port_ret
            new_unscaled = current_weights * (1.0 + r_t)
            denom = float(new_unscaled.sum())
            if denom > 0:
                current_weights = new_unscaled / denom
        else:
            weekly_gross_ret[week] = 0.0
        weekly_tc[week] = 0.0

        # 60/40 baseline returns
        if i > 0 and not current_baseline_weights.empty:
            r_b = weekly_returns_sa.iloc[i].reindex(current_baseline_weights.index).fillna(0.0)
            baseline_ret = float((current_baseline_weights * r_b).sum())
            weekly_baseline[week] = baseline_ret
            new_baseline_unscaled = current_baseline_weights * (1.0 + r_b)
            denom_b = float(new_baseline_unscaled.sum())
            if denom_b > 0:
                current_baseline_weights = new_baseline_unscaled / denom_b
        else:
            weekly_baseline[week] = 0.0

        # Monthly rebalance
        if week in rebal_set:
            n_rebals += 1
            new_w = target.copy()
            all_tk = current_weights.index.union(new_w.index)
            w_old = current_weights.reindex(all_tk, fill_value=0.0)
            w_new = new_w.reindex(all_tk, fill_value=0.0)
            turnover = float((w_new - w_old).abs().sum())
            tc = turnover * TC_DECIMAL_PER_SIDE * 2.0
            weekly_tc[week] = tc
            current_weights = new_w

            # Baseline rebalance too
            current_baseline_weights = baseline_target.copy()

    gross = pd.Series(weekly_gross_ret, name="ac_gross")
    tcs   = pd.Series(weekly_tc,        name="ac_tc")
    net   = (gross - tcs).rename("ac_net")
    baseline = pd.Series(weekly_baseline, name="baseline_60_40")

    return ACBacktestResult(
        weekly_returns_gross = gross,
        weekly_returns_net   = net,
        weekly_tc_drag       = tcs,
        baseline_60_40       = baseline,
        rebalance_dates      = sorted([d for d in rebal_dates]),
        n_weeks              = len(weeks),
        n_rebalances         = n_rebals,
    )


def save_ac_parquet(
    result:    ACBacktestResult,
    save_path: str = "data/portfolio_replay/v1_path_ac_tlt_gld_extended_weekly.parquet",
) -> Path:
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "gross":           result.weekly_returns_gross,
        "tc":              result.weekly_tc_drag,
        "net":             result.weekly_returns_net,
        "baseline_60_40":  result.baseline_60_40,
    })
    df.index.name = "week_end"
    df.to_parquet(p)
    return p
