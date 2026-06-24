"""
engine/factor_ensemble_crypto/signal.py — TSMOM 12-1 per Moskowitz 2012.

Pre-registration: spec id=71 hash 48db143d §2.2 (LOCKED).

Signal at month-end UTC date t for each asset s independently:
    trailing_return[s, t] = price[s, t-1m] / price[s, t-12m] - 1
    signal[s, t] = +1 if trailing_return > 0 else -1

skip-1: per Moskowitz 2012 §2 to avoid Jegadeesh 1990 short-term reversal.
lookback-12: canonical TSMOM horizon.

NOT included (would require spec_amend per §六):
  - volatility scaling
  - alternative lookback (6 / 24 month)
  - smoothing (e.g., 3-month signal moving avg)
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


# Spec §2.2 — LOCKED
LOOKBACK_MONTHS_LOCKED: int = 12
SKIP_MONTHS_LOCKED:     int = 1


def compute_tsmom_signal_panel(
    daily_prices:    pd.DataFrame,
    rebalance_dates: list,
    lookback_months: int = LOOKBACK_MONTHS_LOCKED,
    skip_months:     int = SKIP_MONTHS_LOCKED,
) -> pd.DataFrame:
    """
    Compute TSMOM 12-1 signal at each rebalance date for each asset.

    Args:
        daily_prices: wide DataFrame from load_crypto_panel (date index,
                      ticker columns, close values)
        rebalance_dates: list of datetime.date — month-end UTC rebalance points
        lookback_months: trailing window in months (LOCKED 12)
        skip_months: months to skip (LOCKED 1)

    Returns:
        DataFrame indexed by rebalance date, columns = tickers, values in {-1, +1, None}.
        None when insufficient lookback data (first ~12 months of window).
    """
    import datetime
    assets = list(daily_prices.columns)
    out = pd.DataFrame(index=pd.DatetimeIndex(rebalance_dates), columns=assets, dtype=object)
    out.index.name = "rebalance_date"

    # Normalize daily_prices index to date for lookup
    px_date_indexed = daily_prices.copy()
    px_date_indexed.index = pd.to_datetime(px_date_indexed.index).date

    for t in rebalance_dates:
        # t-skip_months endpoint (most recent allowed close)
        end_dt = _shift_months(t, -skip_months)
        start_dt = _shift_months(t, -(lookback_months + skip_months))

        # Find closest available trading dates ≤ each endpoint
        end_close_date = _last_date_at_or_before(px_date_indexed.index, end_dt)
        start_close_date = _last_date_at_or_before(px_date_indexed.index, start_dt)

        if end_close_date is None or start_close_date is None:
            for asset in assets:
                out.loc[pd.Timestamp(t), asset] = None
            continue

        for asset in assets:
            try:
                p_end = float(px_date_indexed.loc[end_close_date, asset])
                p_start = float(px_date_indexed.loc[start_close_date, asset])
            except Exception:
                out.loc[pd.Timestamp(t), asset] = None
                continue
            if not (p_start > 0 and p_end > 0):
                out.loc[pd.Timestamp(t), asset] = None
                continue
            trailing_ret = p_end / p_start - 1.0
            out.loc[pd.Timestamp(t), asset] = (
                +1 if trailing_ret > 0 else (-1 if trailing_ret < 0 else 0)
            )
    return out


def _shift_months(d, months: int):
    """Shift a date by N months (positive = forward, negative = backward)."""
    import datetime
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    # Clamp day to month-end
    try:
        return datetime.date(year, month, d.day)
    except ValueError:
        # day overflow (e.g., shifting Mar 31 → Feb 28)
        if month == 12:
            next_month_start = datetime.date(year + 1, 1, 1)
        else:
            next_month_start = datetime.date(year, month + 1, 1)
        return next_month_start - datetime.timedelta(days=1)


def _last_date_at_or_before(index, target):
    """Find last date in index ≤ target."""
    candidates = [d for d in index if d <= target]
    return max(candidates) if candidates else None
