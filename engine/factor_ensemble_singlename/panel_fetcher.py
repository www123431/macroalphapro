"""
engine/factor_ensemble_singlename/panel_fetcher.py — chunked bulk price panel.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.6

Per pre-Wave-A audit Issue M1 (yfinance rate-limit at 500 ticker batch):
  - chunked sub-fetch: 100 tickers × N batches
  - retry with exponential backoff
  - missing-ticker tolerance (fail-safe NaN)
  - disk cache (parquet, mirrors v2 panel cache)
"""
from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Locked per spec §2.6 + audit Issue M1
CHUNK_SIZE_LOCKED:        int = 100   # tickers per yfinance batch
MAX_RETRIES_LOCKED:       int = 3
RETRY_BASE_SLEEP_SEC:     float = 5.0
PANEL_BUFFER_DAYS_BEFORE: int = 400   # buffer for 12-month TSMOM window
PANEL_BUFFER_DAYS_AFTER:  int = 45    # buffer for next-month realized return

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "factor_ensemble_singlename"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PANEL_CACHE_PATH: Path = _CACHE_DIR / "_yf_singlestock_panel.parquet"


def _chunked(tickers: list[str], chunk_size: int = CHUNK_SIZE_LOCKED):
    """Yield successive chunks of `chunk_size` tickers."""
    for i in range(0, len(tickers), chunk_size):
        yield tickers[i:i + chunk_size]


def _fetch_chunk_with_retry(
    tickers: list[str],
    start:   datetime.date,
    end:     datetime.date,
) -> pd.DataFrame:
    """Fetch a chunk with exponential backoff retry on failure."""
    import yfinance as yf
    last_exc = None
    for attempt in range(MAX_RETRIES_LOCKED):
        try:
            raw = yf.download(
                tickers,
                start=str(start),
                end=str(end + datetime.timedelta(days=1)),
                progress=False,
                auto_adjust=True,
                group_by="column",
                threads=True,
            )
            if raw is None or raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]].rename(columns={"Close": tickers[0]})
            return close.dropna(how="all")
        except Exception as exc:
            last_exc = exc
            sleep_sec = RETRY_BASE_SLEEP_SEC * (2 ** attempt)
            logger.warning(
                "yfinance fetch attempt %d/%d failed for %d tickers: %s; sleeping %.0fs",
                attempt + 1, MAX_RETRIES_LOCKED, len(tickers), exc, sleep_sec,
            )
            if attempt < MAX_RETRIES_LOCKED - 1:
                time.sleep(sleep_sec)
    logger.error("yfinance fetch exhausted retries for %d tickers: %s", len(tickers), last_exc)
    return pd.DataFrame()


def bulk_fetch_singlestock_panel(
    tickers:       list[str],
    start_date:    datetime.date,
    end_date:      datetime.date,
    use_cache:     bool = True,
) -> pd.DataFrame:
    """Bulk-fetch single-stock daily price panel with chunked + retry + cache.

    Args:
        tickers:    list of S&P 500 ticker symbols (typically ~500 names)
        start_date: panel start (will be padded by PANEL_BUFFER_DAYS_BEFORE)
        end_date:   panel end (will be padded by PANEL_BUFFER_DAYS_AFTER)
        use_cache:  load existing cache if covers requested range × tickers

    Returns:
        pd.DataFrame indexed by date, columns = tickers (NaN for missing/delisted).
    """
    needed_start_ts = pd.Timestamp(start_date - datetime.timedelta(days=PANEL_BUFFER_DAYS_BEFORE))
    needed_end_ts = pd.Timestamp(end_date + datetime.timedelta(days=PANEL_BUFFER_DAYS_AFTER))
    needed_tickers = sorted(set(tickers))

    # Cache load
    cache_df: Optional[pd.DataFrame] = None
    if use_cache and _PANEL_CACHE_PATH.exists():
        try:
            cache_df = pd.read_parquet(_PANEL_CACHE_PATH)
        except Exception as exc:
            logger.warning("panel cache load failed: %s — refetching", exc)
            cache_df = None

    cache_ok = (
        cache_df is not None
        and not cache_df.empty
        and cache_df.index.min() <= needed_start_ts
        and cache_df.index.max() >= needed_end_ts
        and all(t in cache_df.columns for t in needed_tickers)
    )
    if cache_ok:
        logger.info("singlestock panel cache HIT: %d tickers, [%s, %s]",
                    len(needed_tickers), needed_start_ts.date(), needed_end_ts.date())
        return cache_df

    # Cache miss → chunked fetch
    logger.info("singlestock panel cache MISS — chunked-fetching %d tickers in %d-batch sizes",
                len(needed_tickers), CHUNK_SIZE_LOCKED)
    all_chunks: list[pd.DataFrame] = []
    for chunk_idx, chunk in enumerate(_chunked(needed_tickers)):
        logger.info("  chunk %d/%d (%d tickers)",
                    chunk_idx + 1, (len(needed_tickers) + CHUNK_SIZE_LOCKED - 1) // CHUNK_SIZE_LOCKED,
                    len(chunk))
        chunk_df = _fetch_chunk_with_retry(chunk, needed_start_ts.date(), needed_end_ts.date())
        if not chunk_df.empty:
            all_chunks.append(chunk_df)

    if not all_chunks:
        logger.error("singlestock panel: ALL chunks failed")
        return cache_df if cache_df is not None else pd.DataFrame()

    new_panel = pd.concat(all_chunks, axis=1)
    # De-dup columns if any
    new_panel = new_panel.loc[:, ~new_panel.columns.duplicated()]

    # Merge with existing cache (prefer new on overlap)
    if cache_df is not None and not cache_df.empty:
        combined = new_panel.combine_first(cache_df)
    else:
        combined = new_panel

    # Persist
    try:
        combined.to_parquet(_PANEL_CACHE_PATH)
        logger.info("singlestock panel cache persisted: %d tickers × %d dates",
                    combined.shape[1], combined.shape[0])
    except Exception as exc:
        logger.warning("singlestock panel persist failed: %s", exc)

    return combined
