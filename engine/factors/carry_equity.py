"""
engine/factors/carry_equity.py — Carry factor (equity-only scope per Framework E v1).

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50) §2.2.2
Spec lock:
  - Equity-only scope: equity_sector + equity_factor (24 ETFs)
  - Definition: trailing 12-month dividend yield = sum(div ∈ [t-365d, t]) / price[t]
  - Non-equity ETFs (commodity / fixed_income / volatility / fx) → NaN
  - Reason for restriction: free data on commodity/FI/FX carry degenerates to
    momentum (1mo return proxy = TSMOM); Carry-equity-only preserves real
    diversification per §rule-9 N3

Literature: Koijen-Moskowitz-Pedersen-Vrugt 2018 *JFE* "Carry"; AMP 2013 *FAJ*
dividend-yield as quality-tilted carry signal.

NaN protocol (per spec §2.3):
  - Non-equity asset class → NaN (excluded by design)
  - ETF without 12mo dividend history → NaN
  - ETF inception too recent (< 365 days before as_of) → NaN
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# Locked scope (per spec §2.2.2)
EQUITY_ASSET_CLASSES: frozenset[str] = frozenset({"equity_sector", "equity_factor"})

# Lookback for trailing dividend yield
DIVIDEND_LOOKBACK_DAYS: int = 365


def compute_carry_equity_signal(
    as_of:          datetime.date,
    universe:       list[str],
    asset_classes:  dict[str, str],
    use_cache:      bool = False,
) -> pd.Series:
    """
    Compute per-ETF Carry-equity signal at as_of.

    For each equity ETF: trailing 12mo dividend yield = sum(dividends in last 365d) / price_at_t.
    Non-equity ETFs return NaN per spec equity-only scope.

    Args:
        as_of:         signal computation date
        universe:      list of ETF tickers
        asset_classes: {ticker: asset_class} REQUIRED (Carry-equity is class-aware)
        use_cache:     reserved for future yfinance caching layer

    Returns:
        pd.Series indexed by ticker with carry signal values or NaN.
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)
    if asset_classes is None:
        raise ValueError("Carry-equity requires asset_classes to enforce equity-only scope")

    out: dict[str, float] = {}
    for ticker in universe:
        ac = asset_classes.get(ticker)
        if ac not in EQUITY_ASSET_CLASSES:
            out[ticker] = np.nan  # non-equity excluded per spec §2.1
            continue
        try:
            yield_value = _compute_etf_dividend_yield(ticker, as_of)
            out[ticker] = yield_value if yield_value is not None else np.nan
        except Exception as exc:
            logger.debug(
                "carry_equity: dividend yield fetch failed for %s: %s — NaN",
                ticker, exc,
            )
            out[ticker] = np.nan

    return pd.Series(out, dtype=float)


# Per-process + on-disk cache for full dividend / price history per ticker.
# Spec id=50 amendment 2026-05-09 (clarification, post-Gate-0 speedup).
# Replaces ~8000 per-period yfinance calls (24 ETFs × 168 periods × 2 sites)
# with ~24 full-history calls (one per ticker, persisted across runs).
_CARRY_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "factor_carry_equity"
_CARRY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_DIV_CACHE_PATH:   Path = _CARRY_CACHE_DIR / "_dividends_full_history.parquet"
_PRICE_CACHE_PATH: Path = _CARRY_CACHE_DIR / "_prices_full_history.parquet"

# Module-level in-memory cache (loaded lazily on first access)
_DIV_CACHE_DF:   Optional[pd.DataFrame] = None
_PRICE_CACHE_DF: Optional[pd.DataFrame] = None


def _load_full_history_cache() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Lazy-load the on-disk caches into module-level dicts."""
    global _DIV_CACHE_DF, _PRICE_CACHE_DF
    if _DIV_CACHE_DF is None and _DIV_CACHE_PATH.exists():
        try:
            _DIV_CACHE_DF = pd.read_parquet(_DIV_CACHE_PATH)
        except Exception as exc:
            logger.warning("carry_equity div cache load failed: %s", exc)
    if _PRICE_CACHE_DF is None and _PRICE_CACHE_PATH.exists():
        try:
            _PRICE_CACHE_DF = pd.read_parquet(_PRICE_CACHE_PATH)
        except Exception as exc:
            logger.warning("carry_equity price cache load failed: %s", exc)
    return _DIV_CACHE_DF, _PRICE_CACHE_DF


def _persist_caches() -> None:
    global _DIV_CACHE_DF, _PRICE_CACHE_DF
    try:
        if _DIV_CACHE_DF is not None and not _DIV_CACHE_DF.empty:
            _DIV_CACHE_DF.to_parquet(_DIV_CACHE_PATH)
    except Exception as exc:
        logger.warning("carry_equity div cache persist failed: %s", exc)
    try:
        if _PRICE_CACHE_DF is not None and not _PRICE_CACHE_DF.empty:
            _PRICE_CACHE_DF.to_parquet(_PRICE_CACHE_PATH)
    except Exception as exc:
        logger.warning("carry_equity price cache persist failed: %s", exc)


def _ensure_ticker_full_history(ticker: str) -> bool:
    """Ensure dividends + prices full history for `ticker` is in module cache.
    Fetches via yfinance once per ticker per process. Returns True if usable."""
    global _DIV_CACHE_DF, _PRICE_CACHE_DF

    div_df, price_df = _load_full_history_cache()
    div_present = div_df is not None and ticker in div_df.columns
    price_present = price_df is not None and ticker in price_df.columns
    if div_present and price_present:
        return True

    try:
        t = yf.Ticker(ticker)
    except Exception as exc:
        logger.debug("carry_equity: yf.Ticker(%s) failed: %s", ticker, exc)
        return False

    # Dividends — full history (1 yfinance call)
    if not div_present:
        try:
            div_series = t.dividends
            if div_series is None or div_series.empty:
                div_series = pd.Series(dtype=float)
            else:
                if hasattr(div_series.index, "tz") and div_series.index.tz is not None:
                    div_series = div_series.copy()
                    div_series.index = div_series.index.tz_localize(None)
            new_div = div_series.to_frame(name=ticker)
            if _DIV_CACHE_DF is None or _DIV_CACHE_DF.empty:
                _DIV_CACHE_DF = new_div
            else:
                _DIV_CACHE_DF = _DIV_CACHE_DF.combine_first(new_div)
        except Exception as exc:
            logger.debug("carry_equity: dividend fetch failed for %s: %s", ticker, exc)
            return False

    # Prices — full available history (1 yfinance call via period='max')
    if not price_present:
        try:
            hist = t.history(period="max")
            if hist is None or hist.empty or "Close" not in hist.columns:
                return False
            close = hist["Close"].copy()
            if hasattr(close.index, "tz") and close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            close.name = ticker
            new_price = close.to_frame()
            if _PRICE_CACHE_DF is None or _PRICE_CACHE_DF.empty:
                _PRICE_CACHE_DF = new_price
            else:
                _PRICE_CACHE_DF = _PRICE_CACHE_DF.combine_first(new_price)
        except Exception as exc:
            logger.debug("carry_equity: price fetch failed for %s: %s", ticker, exc)
            return False

    _persist_caches()
    return True


def _compute_etf_dividend_yield(ticker: str, as_of: datetime.date) -> Optional[float]:
    """
    Trailing 12mo dividend yield = sum(dividends in [t-365d, t]) / price[t].
    Returns None on data fetch failure (caller maps to NaN).

    Per spec amendment 2026-05-09: uses module-level full-history cache,
    1 yfinance call per ticker per process (not per period).
    """
    if not _ensure_ticker_full_history(ticker):
        return None

    end = as_of
    start = end - datetime.timedelta(days=DIVIDEND_LOOKBACK_DAYS)

    # Dividends in [start, end] from cache
    if _DIV_CACHE_DF is None or ticker not in _DIV_CACHE_DF.columns:
        return None
    div_col = _DIV_CACHE_DF[ticker].dropna()
    if div_col.empty:
        div_amount = 0.0
    else:
        mask = (div_col.index >= pd.Timestamp(start)) & (div_col.index <= pd.Timestamp(end))
        div_in_window = div_col[mask]
        div_amount = float(div_in_window.sum()) if not div_in_window.empty else 0.0

    # Price at-or-before as_of from cache
    if _PRICE_CACHE_DF is None or ticker not in _PRICE_CACHE_DF.columns:
        return None
    price_col = _PRICE_CACHE_DF[ticker].dropna()
    if price_col.empty:
        return None
    price_in_window = price_col[price_col.index <= pd.Timestamp(end)]
    if price_in_window.empty:
        return None
    price = float(price_in_window.iloc[-1])
    if price <= 0:
        return None

    return div_amount / price
