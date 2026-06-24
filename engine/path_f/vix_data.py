"""
engine/path_f/vix_data.py — yfinance VIX + VIX3M + SVXY daily data fetch + alignment.

Pre-registration: docs/spec_path_f_vix_term_structure_v1.md (id=65) §2.1
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_vix_panel(
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Fetch yfinance ^VIX, ^VIX3M, SVXY daily closes; align + return single DataFrame.

    Columns:
      - VIX (30-day implied vol)
      - VIX3M (3-month implied vol)
      - SVXY (ProShares short-VIX ETF)

    Drops rows where any series missing (left-join + dropna).
    """
    import yfinance as yf

    end_extended = end_date + datetime.timedelta(days=14)

    out = {}
    for ticker, label in [("^VIX", "VIX"), ("^VIX3M", "VIX3M"), ("SVXY", "SVXY")]:
        logger.info("Fetching %s [%s, %s]", ticker, start_date, end_extended)
        df = yf.download(
            tickers=ticker,
            start=start_date.isoformat(),
            end=end_extended.isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise RuntimeError(f"yfinance returned empty for {ticker}")
        # extract Close
        if isinstance(df.columns, pd.MultiIndex):
            close = df[("Close", ticker)] if ("Close", ticker) in df.columns else df["Close"].iloc[:, 0]
        else:
            close = df["Close"]
        close.index = pd.DatetimeIndex([d.normalize() if hasattr(d, "normalize") else pd.Timestamp(d).normalize() for d in close.index])
        out[label] = close

    panel = pd.DataFrame(out).sort_index()
    n_before = len(panel)
    panel = panel.dropna()
    n_after = len(panel)
    logger.info("VIX panel aligned: %d → %d daily obs after dropna", n_before, n_after)

    # Filter to requested window
    panel = panel[(panel.index.date >= start_date) & (panel.index.date <= end_date)]
    return panel
