"""
engine/path_e/pre_fomc_signal.py — Pre-FOMC event return computation.

Pre-registration: docs/spec_path_e_pre_fomc_drift_v1.md (id=64) §2.3

For each FOMC event:
  - t_open = max trading day ≤ (statement_release_date - 1 calendar day)
  - t_close = statement_release_date (or next trading day if statement on weekend; rare)
  - event_return = mean equity-ETF basket return from t_open close to t_close close
  - tc_drag = 16 bp (8bp roundtrip × 2 sides)
  - net_event_return = event_return - tc_drag/10000
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spec §六 locked (amendment 1 2026-05-12: 16bp/event → 4bp/event ETF tier-1 industry std)
TC_BPS_ROUNDTRIP_LOCKED  = 2.0     # ETF tier-1 roundtrip (SPY 0.5-1bp, sector 1-2bp, blended 2bp)
TC_SIDES_LOCKED          = 2       # enter + exit
TC_BPS_PER_EVENT_LOCKED  = TC_BPS_ROUNDTRIP_LOCKED * TC_SIDES_LOCKED  # 4 bp per event


def find_prior_trading_day(
    target_date:      datetime.date,
    trading_calendar: pd.DatetimeIndex,
) -> Optional[datetime.date]:
    """Return last trading day strictly BEFORE target_date in trading_calendar."""
    cal_dates = [d.date() if hasattr(d, 'date') else d for d in trading_calendar]
    prior = [d for d in cal_dates if d < target_date]
    if not prior:
        return None
    return max(prior)


def find_next_trading_day_at_or_after(
    target_date:      datetime.date,
    trading_calendar: pd.DatetimeIndex,
) -> Optional[datetime.date]:
    """Return first trading day ≥ target_date in trading_calendar.

    Used when FOMC statement falls on a non-trading day (rare; e.g., if 2-day
    meeting ends on a Friday and statement issued late; or holiday).
    """
    cal_dates = [d.date() if hasattr(d, 'date') else d for d in trading_calendar]
    candidates = [d for d in cal_dates if d >= target_date]
    if not candidates:
        return None
    return min(candidates)


def compute_event_returns(
    fomc_events:      list,           # list[FomcEvent]
    price_panel:      pd.DataFrame,   # columns = ETF tickers; index = trading dates
) -> pd.DataFrame:
    """For each FOMC event, compute:
       - t_open (last trading day before statement_release_date)
       - t_close (statement_release_date, or next trading day if non-trading)
       - per-ticker close-to-close return
       - basket equal-weighted mean return
       - n_active_tickers (tickers with non-NaN prices on both t_open + t_close)

    Returns DataFrame: rows = events, columns = [
       event_index, statement_release_date, t_open, t_close, basket_return,
       basket_return_net, n_active_tickers, ticker_returns_dict
    ]
    """
    if price_panel.empty:
        raise ValueError("compute_event_returns: empty price panel")

    trading_calendar = price_panel.index
    rows = []
    for idx, event in enumerate(fomc_events):
        statement_date = event.statement_release_date

        t_open  = find_prior_trading_day(statement_date, trading_calendar)
        t_close = find_next_trading_day_at_or_after(statement_date, trading_calendar)

        if t_open is None or t_close is None:
            logger.warning(f"Event {idx} {statement_date}: missing trading days "
                           f"(open={t_open}, close={t_close})")
            continue

        # Per-ticker close-to-close return
        try:
            row_open  = price_panel.loc[pd.Timestamp(t_open)]
            row_close = price_panel.loc[pd.Timestamp(t_close)]
        except KeyError as exc:
            logger.warning(f"Event {idx} {statement_date}: price panel lookup error {exc}")
            continue

        ticker_returns = {}
        for ticker in price_panel.columns:
            p_open  = row_open[ticker]
            p_close = row_close[ticker]
            if pd.isna(p_open) or pd.isna(p_close) or p_open <= 0:
                continue
            ticker_returns[ticker] = float(p_close / p_open - 1.0)

        n_active = len(ticker_returns)
        if n_active == 0:
            logger.warning(f"Event {idx} {statement_date}: 0 active tickers")
            continue

        basket_return = float(np.mean(list(ticker_returns.values())))
        basket_return_net = basket_return - TC_BPS_PER_EVENT_LOCKED / 10000.0

        rows.append({
            "event_index":               idx,
            "statement_release_date":    statement_date,
            "t_open":                    t_open,
            "t_close":                   t_close,
            "basket_return_gross":       basket_return,
            "basket_return_net":         basket_return_net,
            "n_active_tickers":          n_active,
            "ticker_returns_json":       str(ticker_returns),  # compact str repr
        })

    return pd.DataFrame(rows)


def build_daily_strategy_returns(
    event_returns:    pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build daily TS strategy return series.

    On t_close of each event: return = basket_return_net
    All other days: return = 0
    Index: trading_calendar dates.
    """
    daily = pd.DataFrame({
        "strategy_return": [0.0] * len(trading_calendar)
    }, index=trading_calendar)

    for _, row in event_returns.iterrows():
        t_close = pd.Timestamp(row["t_close"])
        if t_close in daily.index:
            daily.loc[t_close, "strategy_return"] = float(row["basket_return_net"])
        else:
            logger.warning(f"Event t_close {t_close} not in trading_calendar (skipped)")

    return daily
