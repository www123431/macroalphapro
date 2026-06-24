"""
engine/path_h/nearness_strategy.py — 52-Week-High Momentum signal + backtest.

Pre-registration: docs/spec_path_h_52wh_v1.md (id=67) §2.3-2.6.

Self-contained per spec §4.1 (signal + monthly cohort backtest in one module
for sprint cost reasons; spec §4.1 lists 2 modules but parameters unchanged
so this is not a HARKing concern).

Reuses engine.path_c verdict utilities + Path F incremental_alpha_vs_baseline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spec §六 LOCKED — do not modify
LOOKBACK_DAYS_LOCKED       = 252         # 52 weeks
SKIP_DAYS_LOCKED           = 21          # 1 month, George-Hwang canonical
HOLD_DAYS_LOCKED           = 126         # 6 months, George-Hwang canonical
DECILE_TOP_LOCKED          = 0.90        # top 10%
DECILE_BOT_LOCKED          = 0.10        # bottom 10%
TC_BPS_ROUNDTRIP_LOCKED    = 30.0        # single-stock standing rule
UNIVERSE_RANK_MAX_LOCKED   = 1500        # top-1500 CRSP
NW_LAG_LOCKED              = 126         # matches hold horizon


def compute_nearness_panel(
    daily_prices: pd.DataFrame,
    lookback: int = LOOKBACK_DAYS_LOCKED,
) -> pd.DataFrame:
    """Compute nearness ratio NR_id = abs(price_id) / max(abs(price) over trailing lookback).

    Inputs:
        daily_prices: DataFrame indexed by trading date, columns = ticker, values = adjusted price
                      (CRSP convention: negative prc encodes bid-ask midpoint — we use abs())

    Returns:
        DataFrame same shape; NaN where insufficient lookback or price missing.
    """
    prices_abs = daily_prices.abs()
    rolling_max = prices_abs.rolling(window=lookback, min_periods=lookback).max()
    with np.errstate(divide='ignore', invalid='ignore'):
        nearness = prices_abs / rolling_max
    # Clip to (0, 1] — should be by construction but guards against precision artifacts
    nearness = nearness.where(nearness > 0)
    nearness = nearness.where(nearness <= 1.0001, np.nan)
    return nearness


def form_monthly_cohorts(
    nearness: pd.DataFrame,
    universe_at_month_end: dict,  # pd.Timestamp -> set[str] tickers in top-N at m_end
    top_pct: float = DECILE_TOP_LOCKED,
    bot_pct: float = DECILE_BOT_LOCKED,
    min_cohort_size: int = 100,
) -> dict:
    """For each month-end with a universe snapshot, form long/short decile cohorts.

    Returns:
        dict[pd.Timestamp -> (longs:set, shorts:set, nr_panel_at_m:Series)]
    """
    cohorts = {}
    for m_end, tickers in sorted(universe_at_month_end.items()):
        # Snap to actual trading-day index
        if m_end not in nearness.index:
            # Find nearest trading day ≤ m_end
            valid_dates = nearness.index[nearness.index <= m_end]
            if len(valid_dates) == 0:
                continue
            m_end_snap = valid_dates[-1]
        else:
            m_end_snap = m_end

        tickers_in_panel = [t for t in tickers if t in nearness.columns]
        if len(tickers_in_panel) == 0:
            continue
        nr_at_m = nearness.loc[m_end_snap, tickers_in_panel].dropna()

        if len(nr_at_m) < min_cohort_size:
            logger.debug("Month %s: only %d firms with NR — skip", m_end_snap, len(nr_at_m))
            continue

        # Use rank-based percentile to guarantee ~equal leg sizes even with ties
        # (52WH often clusters at 1.0 in bull markets — naive quantile inflates long leg)
        ranks_pct = nr_at_m.rank(method='average', pct=True)
        longs  = set(ranks_pct[ranks_pct >= top_pct].index)
        shorts = set(ranks_pct[ranks_pct <= bot_pct].index)

        cohorts[m_end_snap] = (longs, shorts, nr_at_m)

    return cohorts


@dataclass
class StrategyResult:
    daily_returns: pd.Series          # net daily L-S returns
    daily_gross:   pd.Series          # gross (pre-TC)
    daily_long_size:  pd.Series
    daily_short_size: pd.Series
    daily_turnover_one_way: pd.Series
    mean_long_size:   float
    mean_short_size:  float
    mean_one_way_turnover_daily: float
    annual_turnover_one_way_pct: float
    tc_drag_annual_pct: float
    n_cohorts: int


def compute_strategy_returns(
    cohorts: dict,
    daily_returns_panel: pd.DataFrame,  # rows date, cols ticker
    skip: int = SKIP_DAYS_LOCKED,
    hold: int = HOLD_DAYS_LOCKED,
    tc_bps_roundtrip: float = TC_BPS_ROUNDTRIP_LOCKED,
) -> StrategyResult:
    """Compute daily L-S returns from monthly overlapping cohorts.

    Mechanics:
      - For each cohort (m_end, longs, shorts), define entry = m_end + skip business days,
        exit = entry + hold business days
      - At each daily d, active book = union of all cohorts c with entry_c ≤ d ≤ exit_c
      - Same-firm dedup: if firm i appears in both active_longs and active_shorts → drop from both
      - Daily L-S = mean(daily_returns[active_longs at d]) - mean(daily_returns[active_shorts at d])
      - TC: 30bp × daily one-way turnover (added + removed) / (2 × book size)
    """
    all_dates = daily_returns_panel.index
    cohort_dates = sorted(cohorts.keys())

    # Precompute cohort entry/exit by index position
    cohort_intervals = []
    for m_end in cohort_dates:
        try:
            idx_m = all_dates.searchsorted(m_end)
        except Exception:
            continue
        if idx_m >= len(all_dates):
            continue
        entry_idx = idx_m + skip + 1
        exit_idx  = entry_idx + hold
        if entry_idx >= len(all_dates):
            continue
        exit_idx = min(exit_idx, len(all_dates) - 1)
        longs, shorts, _ = cohorts[m_end]
        cohort_intervals.append({
            'm_end':    m_end,
            'entry_d':  all_dates[entry_idx],
            'exit_d':   all_dates[exit_idx],
            'longs':    longs,
            'shorts':   shorts,
        })

    if not cohort_intervals:
        raise ValueError("No cohorts produced active intervals; check skip+hold vs window")

    gross_returns = pd.Series(0.0, index=all_dates)
    long_size  = pd.Series(0, index=all_dates)
    short_size = pd.Series(0, index=all_dates)
    turnover_one_way = pd.Series(0.0, index=all_dates)

    prev_active_longs:  set = set()
    prev_active_shorts: set = set()

    for d in all_dates:
        active_longs:  set = set()
        active_shorts: set = set()
        for c in cohort_intervals:
            if c['entry_d'] <= d <= c['exit_d']:
                active_longs.update(c['longs'])
                active_shorts.update(c['shorts'])

        # Dedup cross-cohort: name in both → drop both
        both = active_longs & active_shorts
        if both:
            active_longs  -= both
            active_shorts -= both

        if not active_longs or not active_shorts:
            prev_active_longs  = active_longs
            prev_active_shorts = active_shorts
            continue

        # Daily L-S return
        long_tickers  = [t for t in active_longs  if t in daily_returns_panel.columns]
        short_tickers = [t for t in active_shorts if t in daily_returns_panel.columns]
        if not long_tickers or not short_tickers:
            prev_active_longs  = active_longs
            prev_active_shorts = active_shorts
            continue

        long_ret  = daily_returns_panel.loc[d, long_tickers].dropna()
        short_ret = daily_returns_panel.loc[d, short_tickers].dropna()
        if len(long_ret) == 0 or len(short_ret) == 0:
            prev_active_longs  = active_longs
            prev_active_shorts = active_shorts
            continue

        ls = float(long_ret.mean() - short_ret.mean())
        gross_returns.loc[d] = ls
        long_size.loc[d]  = len(active_longs)
        short_size.loc[d] = len(active_shorts)

        # Turnover = (names added + names removed) / (2 × book size) — one-way fraction
        added_longs   = active_longs  - prev_active_longs
        removed_longs = prev_active_longs - active_longs
        added_shorts  = active_shorts - prev_active_shorts
        removed_shorts = prev_active_shorts - active_shorts
        total_book = len(active_longs) + len(active_shorts)
        total_churn = len(added_longs) + len(removed_longs) + len(added_shorts) + len(removed_shorts)
        if total_book > 0:
            turnover_one_way.loc[d] = total_churn / (2 * total_book)

        prev_active_longs  = active_longs
        prev_active_shorts = active_shorts

    tc_drag_daily = turnover_one_way * (tc_bps_roundtrip / 10000.0)
    net_returns   = gross_returns - tc_drag_daily

    mean_long_size = float(long_size[long_size > 0].mean()) if (long_size > 0).any() else 0.0
    mean_short_size = float(short_size[short_size > 0].mean()) if (short_size > 0).any() else 0.0
    mean_one_way   = float(turnover_one_way.mean())
    annual_turn    = mean_one_way * 252
    tc_drag_ann    = float(tc_drag_daily.mean() * 252)

    return StrategyResult(
        daily_returns=net_returns,
        daily_gross=gross_returns,
        daily_long_size=long_size,
        daily_short_size=short_size,
        daily_turnover_one_way=turnover_one_way,
        mean_long_size=mean_long_size,
        mean_short_size=mean_short_size,
        mean_one_way_turnover_daily=mean_one_way,
        annual_turnover_one_way_pct=annual_turn * 100,
        tc_drag_annual_pct=tc_drag_ann * 100,
        n_cohorts=len(cohort_intervals),
    )
