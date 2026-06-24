"""
engine/factor_ensemble_crypto/data_loader.py — yfinance crypto pull + UTC alignment.

Pre-spec gate G1 verified yfinance BTC-USD/ETH-USD against Binance BTCUSDT/
ETHUSDT on 60-day sample: mean diff -0.017%, std 0.06%, 0/59 days exceeded
0.5% threshold. Data quality institutional-grade.

Universe + window LOCKED per spec id=71 hash 48db143d §2.1 + §2.5.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ── Spec §2.1 + §2.5 — LOCKED ────────────────────────────────────────────
UNIVERSE_LOCKED: tuple[str, ...] = ("BTC-USD", "ETH-USD")
WINDOW_START_LOCKED: datetime.date = datetime.date(2018, 1, 1)
WINDOW_END_LOCKED:   datetime.date = datetime.date(2026, 5, 13)

# Cache for repeat walk_forward runs (yfinance has rate limits + transient errors)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "path_n_crypto" / "_cache"


def _cache_path(ticker: str) -> Path:
    return _CACHE_DIR / f"{ticker.replace('-', '_').lower()}_daily.parquet"


def load_crypto_panel(
    universe:    tuple[str, ...] = UNIVERSE_LOCKED,
    window_start: datetime.date  = WINDOW_START_LOCKED,
    window_end:   datetime.date  = WINDOW_END_LOCKED,
    use_cache:   bool = True,
) -> pd.DataFrame:
    """
    Fetch daily close prices for the universe over the window. Returns wide
    DataFrame indexed by date with columns = universe tickers.

    Args:
        universe:    tuple of yfinance tickers (default LOCKED BTC-USD + ETH-USD)
        window_start: inclusive start
        window_end:   inclusive end
        use_cache:    if True (default), load from local parquet cache when
                      available (refresh via use_cache=False or delete cache file)

    Returns:
        DataFrame with shape (n_days, n_tickers), values = close in USD,
        index = date (datetime.date), columns = tickers.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panels: list[pd.DataFrame] = []

    for ticker in universe:
        cache_file = _cache_path(ticker)
        df: Optional[pd.DataFrame] = None

        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                # Validate range coverage
                if df.index.min().date() > window_start or df.index.max().date() < window_end:
                    logger.info("cache stale for %s, refetching", ticker)
                    df = None
            except Exception as exc:
                logger.warning("cache read failed for %s: %s; refetching", ticker, exc)
                df = None

        if df is None:
            logger.info("yfinance fetching %s %s -> %s", ticker, window_start, window_end)
            yf_data = yf.download(
                ticker,
                start=str(window_start),
                end=str(window_end + datetime.timedelta(days=1)),
                progress=False,
                auto_adjust=True,
            )
            if yf_data is None or yf_data.empty:
                raise RuntimeError(f"yfinance returned empty for {ticker}")
            df = yf_data[["Close"]].copy()
            df.columns = [ticker]
            try:
                df.to_parquet(cache_file)
            except Exception as exc:
                logger.warning("cache write failed for %s: %s", ticker, exc)

        # Single-column rename guard (in case cache was multi-col)
        if ticker not in df.columns:
            # Try common multiindex/multicol pattern
            for col in df.columns:
                if str(col).endswith(ticker) or col == "Close":
                    df = df[[col]].copy()
                    df.columns = [ticker]
                    break

        panels.append(df)

    out = pd.concat(panels, axis=1).sort_index()
    # Restrict to window
    out = out.loc[(out.index.date >= window_start) & (out.index.date <= window_end)]
    return out


def get_month_end_utc_dates(
    window_start: datetime.date = WINDOW_START_LOCKED,
    window_end:   datetime.date = WINDOW_END_LOCKED,
) -> list[datetime.date]:
    """Generate month-end UTC rebalance dates per spec §2.3."""
    dates: list[datetime.date] = []
    cur = datetime.date(window_start.year, window_start.month, 1)
    while cur <= window_end:
        # advance to next month, then back to last day of current
        if cur.month == 12:
            nxt = datetime.date(cur.year + 1, 1, 1)
        else:
            nxt = datetime.date(cur.year, cur.month + 1, 1)
        month_end = nxt - datetime.timedelta(days=1)
        if month_end >= window_start and month_end <= window_end:
            dates.append(month_end)
        cur = nxt
    return dates
