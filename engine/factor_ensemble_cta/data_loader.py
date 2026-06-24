"""
engine/factor_ensemble_cta/data_loader.py — Path O data loading (spec id=73 §2.1, §2.5).

LOCKED universe = {PQTIX} (PIMCO TRENDS Managed Futures Strategy I-class).
LOCKED window = 2014-09-03 (PQTIX inception) → 2025-12-31.

Data source: yfinance daily adjusted close (net of 1.30% expense ratio).
Equity proxy for SAA combination: SPY (broad market index).
"""
from __future__ import annotations

import datetime
import logging
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

UNIVERSE_LOCKED:      tuple[str, ...] = ("PQTIX",)
EQUITY_PROXY_TICKER:  str             = "SPY"
WINDOW_START_LOCKED:  datetime.date   = datetime.date(2014, 9, 3)
WINDOW_END_LOCKED:    datetime.date   = datetime.date(2025, 12, 31)


def load_cta_panel(
    universe:     Iterable[str]   = UNIVERSE_LOCKED,
    window_start: datetime.date   = WINDOW_START_LOCKED,
    window_end:   datetime.date   = WINDOW_END_LOCKED,
    use_cache:    bool            = True,
) -> pd.DataFrame:
    """Load CTA universe daily adjusted close prices.

    Returns DataFrame indexed by date, columns = tickers (single column for v1).
    Adjusted close is already net of expense ratio per yfinance convention.
    """
    tickers = list(universe)
    end_exclusive = window_end + datetime.timedelta(days=1)
    data = yf.download(
        tickers, start=str(window_start),
        end=str(end_exclusive),
        progress=False, auto_adjust=True,
    )
    if data is None or data.empty:
        raise RuntimeError(
            f"yfinance returned empty for CTA universe {tickers} "
            f"({window_start} to {window_end})"
        )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        # single-ticker case: data has columns Open/High/Low/Close/Volume
        prices = data[["Close"]].copy()
        prices.columns = tickers
    prices = prices.dropna(how="all")
    if prices.empty:
        raise RuntimeError(f"After dropna, CTA panel is empty for {tickers}")
    return prices


def load_equity_proxy(
    window_start: datetime.date = WINDOW_START_LOCKED,
    window_end:   datetime.date = WINDOW_END_LOCKED,
) -> pd.DataFrame:
    """Load SPY daily adjusted close as equity proxy for SAA combination."""
    end_exclusive = window_end + datetime.timedelta(days=1)
    data = yf.download(
        EQUITY_PROXY_TICKER, start=str(window_start),
        end=str(end_exclusive),
        progress=False, auto_adjust=True,
    )
    if data is None or data.empty:
        raise RuntimeError(
            f"yfinance returned empty for {EQUITY_PROXY_TICKER} "
            f"({window_start} to {window_end})"
        )
    spy = data[["Close"]].copy()
    spy.columns = [EQUITY_PROXY_TICKER]
    return spy
