"""
engine/factors_singlename/dividend_yield.py — single-stock dividend yield factor.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.2 Wave A
Literature: simplified KMPV 2018 carry (mislabel — actually closer to FF 1993 value)

Wave A placeholder for Wave B P/E ratio Value factor (which requires vintage
fundamentals from Compustat, unavailable until WRDS approval).

Mechanism:
  Trailing 12mo dividend yield = sum(dividends in last 365 days) / price[t]

Reads from:
  - panel:      daily prices (yfinance; for price_at_t)
  - dividends:  yfinance.Ticker(t).dividends (full history, cached per-ticker)

Caches yfinance dividend history per-ticker (~500 calls one-time, then disk hit).
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Locked per spec §2.2 — KMPV 2018 simplified
DIVIDEND_LOOKBACK_DAYS_LOCKED: int = 365

# Disk cache for full dividend history per ticker (mirrors carry_equity pattern)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "factors_singlename"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_DIV_CACHE_PATH: Path = _CACHE_DIR / "_dividends_full_history.parquet"

_DIV_CACHE_DF: Optional[pd.DataFrame] = None


def _load_dividend_cache() -> Optional[pd.DataFrame]:
    """Lazy-load disk cache."""
    global _DIV_CACHE_DF
    if _DIV_CACHE_DF is None and _DIV_CACHE_PATH.exists():
        try:
            _DIV_CACHE_DF = pd.read_parquet(_DIV_CACHE_PATH)
        except Exception as exc:
            logger.warning("dividend cache load failed: %s", exc)
    return _DIV_CACHE_DF


def _persist_dividend_cache() -> None:
    global _DIV_CACHE_DF
    if _DIV_CACHE_DF is not None and not _DIV_CACHE_DF.empty:
        try:
            _DIV_CACHE_DF.to_parquet(_DIV_CACHE_PATH)
        except Exception as exc:
            logger.warning("dividend cache persist failed: %s", exc)


def _ensure_ticker_dividends(ticker: str) -> bool:
    """Ensure full dividend history for ticker is in module cache.

    Fetches via yfinance once per ticker per process; persisted to disk.
    Returns True if usable, False on fetch failure.
    """
    global _DIV_CACHE_DF
    cache = _load_dividend_cache()
    if cache is not None and ticker in cache.columns:
        return True

    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        div_series = t.dividends
        if div_series is None:
            div_series = pd.Series(dtype=float)
        if not div_series.empty and hasattr(div_series.index, "tz") and div_series.index.tz is not None:
            div_series = div_series.copy()
            div_series.index = div_series.index.tz_localize(None)
        new_div = div_series.to_frame(name=ticker)
        if _DIV_CACHE_DF is None or _DIV_CACHE_DF.empty:
            _DIV_CACHE_DF = new_div
        else:
            _DIV_CACHE_DF = _DIV_CACHE_DF.combine_first(new_div)
    except Exception as exc:
        logger.debug("dividend_yield: yfinance fetch failed for %s: %s", ticker, exc)
        return False

    _persist_dividend_cache()
    return True


def compute_dividend_yield_singlestock_signal(
    as_of:         datetime.date,
    universe:      list[str],
    asset_classes: Optional[dict[str, str]] = None,
    panel:         Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Trailing 12mo dividend yield z-score signal.

    Mechanism:
      1. For each ticker: trailing 12mo dividend sum / price at as_of
      2. Cross-section z-score (high yield → positive z = "value-like")
      3. Returns continuous z-score (NOT signed ±1; consistent with v1/v2 carry-eq)

    Args:
        as_of:          decision date
        universe:       list of tickers
        asset_classes:  ignored
        panel:          pre-fetched price panel for price_at_t lookup

    Returns:
        pd.Series indexed by ticker, continuous z-score; NaN for missing data.
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)
    if panel is None or panel.empty:
        logger.warning("compute_dividend_yield_singlestock_signal: panel required → all-NaN")
        return pd.Series(np.nan, index=universe, dtype=float)

    end = as_of
    start = end - datetime.timedelta(days=DIVIDEND_LOOKBACK_DAYS_LOCKED)

    raw_yields: dict[str, float] = {}
    for ticker in universe:
        if ticker not in panel.columns:
            raw_yields[ticker] = np.nan
            continue

        # Price at-or-before as_of
        ts = panel[ticker].dropna()
        before_end = ts[ts.index <= pd.Timestamp(end)]
        if before_end.empty:
            raw_yields[ticker] = np.nan
            continue
        price = float(before_end.iloc[-1])
        if price <= 0:
            raw_yields[ticker] = np.nan
            continue

        # Dividend sum from cache
        if not _ensure_ticker_dividends(ticker):
            raw_yields[ticker] = np.nan
            continue
        cache = _load_dividend_cache()
        if cache is None or ticker not in cache.columns:
            raw_yields[ticker] = np.nan
            continue
        div_col = cache[ticker].dropna()
        if div_col.empty:
            div_amount = 0.0
        else:
            mask = (div_col.index >= pd.Timestamp(start)) & (div_col.index <= pd.Timestamp(end))
            div_in_window = div_col[mask]
            div_amount = float(div_in_window.sum()) if not div_in_window.empty else 0.0
        raw_yields[ticker] = div_amount / price

    raw_series = pd.Series(raw_yields, dtype=float)

    # Cross-section z-score within universe
    valid = raw_series.dropna()
    if len(valid) < 5:  # need minimum cross-section for meaningful z-score
        return pd.Series(np.nan, index=universe, dtype=float)
    mean = float(valid.mean())
    std = float(valid.std(ddof=1))
    if std <= 1e-9:
        return pd.Series(np.nan, index=universe, dtype=float)

    out: dict[str, float] = {}
    for ticker in universe:
        v = raw_series.get(ticker, np.nan)
        out[ticker] = (v - mean) / std if np.isfinite(v) else np.nan
    return pd.Series(out, dtype=float)
