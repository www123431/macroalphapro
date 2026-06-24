"""engine/data/fetchers/api_yfinance.py — Yahoo Finance equity/index/ETF.

Free public source. No auth required. Adjusted prices (good for backtest).

Known senior-quant pitfalls handled:
- Adjusted prices via auto_adjust=True (handles splits + dividends)
- 404 on delisted tickers → graceful skip (logged)
- NaN holes in raw data → preserved (caller decides how to handle)
- TZ → normalized to UTC via _common.to_utc_dates
- Sentinel values: yfinance rarely uses sentinels; relies on NaN

Known limitations:
- yfinance ≠ CRSP delisting return (DLRET) — silently drops delisted tickers
- Rate limiting: ~2 req/sec sustainable
- Schema can break with yfinance package upgrades
"""
from __future__ import annotations

import logging

import pandas as pd

from engine.data.orchestrator import ProbeResult
from engine.data.fetchers._common import to_utc_dates, replace_sentinel_values

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def probe(start: str, end: str, *, target_function: str | None = None,
            **kw) -> ProbeResult:
    """Probe yfinance availability via a 1-day SPY fetch.

    Light query that detects: network unreachable / yfinance import broken /
    Yahoo API rejection (rare but possible)."""
    import time
    t0 = time.time()
    try:
        import yfinance as yf
    except ImportError as exc:
        return ProbeResult(
            available=False, error=f"yfinance not installed: {exc}",
            error_class="schema_unknown", elapsed_secs=time.time() - t0,
        )
    try:
        df = yf.Ticker("SPY").history(period="1d")
        if df is None or df.empty:
            return ProbeResult(
                available=False, error="SPY probe returned empty",
                error_class="network", elapsed_secs=time.time() - t0,
            )
    except Exception as exc:
        return ProbeResult(
            available=False, error=f"yfinance probe failed: {exc}",
            error_class="network", elapsed_secs=time.time() - t0,
        )
    return ProbeResult(
        available=True, error=None, error_class=None,
        elapsed_secs=time.time() - t0,
    )


def fetch_equity_daily(start: str, end: str, *,
                        tickers: list[str] | None = None,
                        **kw) -> pd.DataFrame:
    """Daily adjusted prices + returns for a list of tickers.

    Args:
      tickers: list of Yahoo ticker symbols. Defaults to SPY (1-ticker smoke)
               if not provided. Real callers always specify tickers.

    Returns long-format DataFrame with columns:
      date, ticker, prc (adjusted close), ret (log return)
    """
    import yfinance as yf
    tickers = tickers or ["SPY"]
    frames = []
    for tk in tickers:
        try:
            df = yf.download(
                tk, start=start, end=end, progress=False,
                auto_adjust=True, threads=False,
            )
        except Exception as exc:
            logger.warning("yfinance failed for %s: %s", tk, exc)
            continue
        if df is None or df.empty:
            continue
        # Reset to long format
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                       for c in df.columns]
        # yfinance multi-ticker returns multi-index columns; single-ticker is flat.
        df["ticker"] = tk
        df = df.rename(columns={"close": "prc"})
        df["date"] = to_utc_dates(df["date"])
        df = df[["date", "ticker", "prc"]].copy()
        df["ret"] = (df.groupby("ticker")["prc"]
                       .transform(lambda x: x.pct_change()))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "prc", "ret"])
    out = pd.concat(frames, ignore_index=True)
    out = replace_sentinel_values(out)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_equity_monthly(start: str, end: str, **kw) -> pd.DataFrame:
    """Monthly compounded returns from daily."""
    daily = fetch_equity_daily(start, end, **kw)
    if daily.empty:
        return daily
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = (daily.groupby(["ticker", "month"], as_index=False)
                 .agg(ret=("ret", lambda r: (1 + r).prod() - 1),
                       prc=("prc", "last")))
    monthly = monthly.rename(columns={"month": "date"})
    return monthly[["date", "ticker", "ret", "prc"]]


def fetch_index(start: str, end: str, *,
                  symbol: str = "^VIX", **kw) -> pd.DataFrame:
    """Index series (default: VIX). Returns date + value."""
    import yfinance as yf
    try:
        df = yf.download(symbol, start=start, end=end, progress=False,
                          auto_adjust=False, threads=False)
    except Exception as exc:
        logger.warning("yfinance index %s failed: %s", symbol, exc)
        return pd.DataFrame(columns=["date", "value"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df = df.reset_index()
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                   for c in df.columns]
    df = df.rename(columns={"close": "value"})
    df["date"] = to_utc_dates(df["date"])
    return df[["date", "value"]].copy()
