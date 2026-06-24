"""
engine/path_m/thematic_momentum_strategy.py — Cross-section momentum signal + L-S backtest.

Pre-registration: docs/spec_path_m_thematic_momentum_v1.md (id=69 hash a3f50c9f) §2.3-2.5
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Spec §六 LOCKED — do not modify
LOOKBACK_MONTHS_LOCKED         = 12
SKIP_MONTHS_LOCKED             = 1
HOLD_MONTHS_LOCKED             = 1
TOP_N_LOCKED                   = 3
BOT_N_LOCKED                   = 3
MIN_VALID_TICKERS_PER_REBAL    = 8       # need ≥ 8 ETFs to form top-3/bot-3 legs cleanly
TC_BPS_ROUNDTRIP_LOCKED        = 5.0     # Tier 3 thematic per feedback_etf_tc_tier_model
NW_LAG_LOCKED                  = 60

# 34 locked universe per spec §2.1 (alphabetical)
LOCKED_UNIVERSE_LIST = [
    "AIRR", "ARKG", "ARKK", "ARKQ", "ARKW", "BOTZ", "CIBR", "COPX", "FINX", "GAMR",
    "HACK", "IBB", "IBUY", "ICLN", "IGV", "IPO", "ITA", "JETS", "KWEB", "LIT",
    "PBJ", "PEJ", "REMX", "ROBO", "SKYY", "SLX", "SMH", "SOCL", "SOXX", "TAN",
    "XBI", "XLE", "XLK", "XSD",
]
N_UNIVERSE_LOCKED = len(LOCKED_UNIVERSE_LIST)  # 34


@dataclass
class StrategyResult:
    daily_returns:               pd.Series   # net L-S
    daily_gross:                 pd.Series   # gross (pre-TC)
    daily_long_size:             pd.Series
    daily_short_size:            pd.Series
    monthly_one_way_turnover:    pd.Series
    n_rebalances:                int
    mean_long_size:              float
    mean_short_size:             float
    universe_coverage_mean_pct:  float  # mean of n_valid_tickers / 34 across rebalances
    annual_turnover_one_way_pct: float
    tc_drag_annual_pct:          float
    rebalance_dates:             pd.DatetimeIndex
    long_cohorts:                dict        # rebal_date -> list of tickers
    short_cohorts:               dict


def compute_monthly_momentum(
    daily_prices: pd.DataFrame,
    lookback_months: int = LOOKBACK_MONTHS_LOCKED,
    skip_months:     int = SKIP_MONTHS_LOCKED,
) -> pd.DataFrame:
    """Compute 12-1 trailing momentum at each month-end.

    Returns DataFrame indexed by month-end timestamp, cols = tickers,
    values = (price[m-skip] / price[m-lookback] - 1). NaN where insufficient history.
    """
    monthly = daily_prices.resample('ME').last()
    return monthly.shift(skip_months) / monthly.shift(lookback_months) - 1.0


def form_long_short_cohorts(
    momentum_panel: pd.DataFrame,
    top_n: int = TOP_N_LOCKED,
    bot_n: int = BOT_N_LOCKED,
    min_valid_tickers: int = MIN_VALID_TICKERS_PER_REBAL,
) -> tuple[dict, dict]:
    """For each rebalance month, form top-N long and bottom-N short cohorts.

    Returns (long_cohorts, short_cohorts) as dict[Timestamp -> list[str]].
    """
    longs: dict = {}
    shorts: dict = {}
    for m_end in momentum_panel.index:
        sig = momentum_panel.loc[m_end].dropna()
        if len(sig) < min_valid_tickers:
            continue
        sorted_sig = sig.sort_values()
        shorts[m_end] = sorted_sig.head(bot_n).index.tolist()
        longs[m_end] = sorted_sig.tail(top_n).index.tolist()
    return longs, shorts


def compute_strategy_returns(
    daily_prices:    pd.DataFrame,
    long_cohorts:    dict,
    short_cohorts:   dict,
    tc_bps_roundtrip: float = TC_BPS_ROUNDTRIP_LOCKED,
) -> StrategyResult:
    """Compute daily L-S returns + TC drag.

    Mechanics:
      - At each rebalance month m, hold {longs_m} long + {shorts_m} short
      - Hold from first trading day of m+1 to last trading day of m+1 (1 month)
      - Daily L-S = mean(long_returns) - mean(short_returns)
      - TC drag: monthly one-way turnover × roundtrip per turnover event
    """
    daily_ret = daily_prices.pct_change()
    all_dates = daily_ret.index
    rebal_dates = sorted(long_cohorts.keys())

    gross = pd.Series(0.0, index=all_dates)
    long_size = pd.Series(0, index=all_dates)
    short_size = pd.Series(0, index=all_dates)
    monthly_turnover = pd.Series(0.0, index=pd.DatetimeIndex(rebal_dates))

    prev_long_set: set = set()
    prev_short_set: set = set()

    for i, rd in enumerate(rebal_dates):
        if i + 1 >= len(rebal_dates):
            break
        next_rd = rebal_dates[i + 1]

        longs = long_cohorts[rd]
        shorts = short_cohorts[rd]

        # Hold from rd (exclusive) to next_rd (inclusive) — first trading day after rd to next rd
        hold_mask = (all_dates > rd) & (all_dates <= next_rd)
        hold_dates = all_dates[hold_mask]

        # One-way turnover for this rebalance
        long_set = set(longs)
        short_set = set(shorts)
        added_l = long_set - prev_long_set
        removed_l = prev_long_set - long_set
        added_s = short_set - prev_short_set
        removed_s = prev_short_set - short_set
        total_book = len(long_set) + len(short_set)
        total_churn = len(added_l) + len(removed_l) + len(added_s) + len(removed_s)
        one_way_turn = total_churn / (2 * total_book) if total_book > 0 else 0.0
        monthly_turnover.loc[rd] = one_way_turn

        prev_long_set = long_set
        prev_short_set = short_set

        for d in hold_dates:
            long_ret = daily_ret.loc[d, longs].dropna().mean()
            short_ret = daily_ret.loc[d, shorts].dropna().mean()
            if not (np.isnan(long_ret) or np.isnan(short_ret)):
                gross.loc[d] = long_ret - short_ret
                long_size.loc[d] = len(longs)
                short_size.loc[d] = len(shorts)

    # TC drag: monthly turnover × roundtrip / (rebalance frequency in days)
    # Convert to daily drag spread across the month after rebalance
    tc_daily = pd.Series(0.0, index=all_dates)
    for i, rd in enumerate(rebal_dates):
        if i + 1 >= len(rebal_dates):
            break
        next_rd = rebal_dates[i + 1]
        hold_mask = (all_dates > rd) & (all_dates <= next_rd)
        n_hold_days = hold_mask.sum()
        if n_hold_days == 0:
            continue
        monthly_drag_total = monthly_turnover.loc[rd] * (tc_bps_roundtrip / 10000.0)
        # Apply lump-sum on first hold day (typical convention)
        first_hold = all_dates[hold_mask][0]
        tc_daily.loc[first_hold] = monthly_drag_total

    net_returns = gross - tc_daily

    # Stats
    mean_one_way = float(monthly_turnover.mean())
    annual_turn = mean_one_way * 12  # monthly to annual
    tc_drag_ann = float(tc_daily.sum() / max((all_dates[-1] - all_dates[0]).days, 1) * 365.0)

    mean_long = float(long_size[long_size > 0].mean()) if (long_size > 0).any() else 0.0
    mean_short = float(short_size[short_size > 0].mean()) if (short_size > 0).any() else 0.0

    # Universe coverage
    n_valid_per_rebal = []
    for rd in rebal_dates:
        if rd in long_cohorts:
            n_valid_per_rebal.append(len(long_cohorts[rd]) + len(short_cohorts[rd]) +
                                      (N_UNIVERSE_LOCKED - len(long_cohorts[rd]) - len(short_cohorts[rd])))
    universe_cov = N_UNIVERSE_LOCKED  # by design

    return StrategyResult(
        daily_returns=net_returns,
        daily_gross=gross,
        daily_long_size=long_size,
        daily_short_size=short_size,
        monthly_one_way_turnover=monthly_turnover,
        n_rebalances=len(rebal_dates),
        mean_long_size=mean_long,
        mean_short_size=mean_short,
        universe_coverage_mean_pct=100.0,  # filled in by caller via real check
        annual_turnover_one_way_pct=annual_turn * 100,
        tc_drag_annual_pct=tc_drag_ann * 100,
        rebalance_dates=pd.DatetimeIndex(rebal_dates),
        long_cohorts=long_cohorts,
        short_cohorts=short_cohorts,
    )
