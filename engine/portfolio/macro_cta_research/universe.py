"""
Universe data loader for self-built CTA horse race specs (P/Q/S/T/U).

Locked universe (per spec §2.1 all 5 active specs): TLT/HYG/DBC/GLD + PQTIX
(baseline) + SPY (regime overlay for Path T) + ^VIX (overlay for Path U).

Design constraints:
  - Single source of truth: one cache parquet per (universe, window) tuple
  - Dataset hash registered as part of reproducibility trinity
    (code_commit · spec_hash · dataset_hash)
  - Force-refresh capability for forward extension when paper trade rolls in
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Per spec §2.1 — 4-ETF locked universe (no VXX due to 2018 issuer continuity break)
HORSE_RACE_UNIVERSE: tuple[str, ...] = ("TLT", "HYG", "DBC", "GLD")

# Baseline (head-to-head gate G1)
PQTIX_TICKER: str = "PQTIX"

# Regime overlay for Path T (Antonacci Dual Momentum)
REGIME_INDICATOR: str = "SPY"

# Vol overlay for Path U (Moreira-Muir Vol-Scaled Risk Parity)
VOL_INDICATOR: str = "^VIX"

# Spec window per all 5 active specs (locked)
WINDOW_START: datetime.date = datetime.date(2014, 9, 1)
WINDOW_END:   datetime.date = datetime.date(2023, 12, 31)

# Storage paths
_CACHE_DIR = Path("data/macro_cta_research")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(tickers: tuple[str, ...], start: datetime.date,
                end: datetime.date) -> Path:
    key = f"{'_'.join(sorted(tickers))}_{start.isoformat()}_{end.isoformat()}"
    return _CACHE_DIR / f"prices_{key}.parquet"


def _meta_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".meta.json")


def _compute_dataset_hash(df: pd.DataFrame) -> str:
    """SHA-256 over normalized parquet bytes (used in reproducibility trinity)."""
    payload = df.to_csv(float_format="%.10f", date_format="%Y-%m-%d").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_universe_weekly(
    tickers: tuple[str, ...] = HORSE_RACE_UNIVERSE + (PQTIX_TICKER, REGIME_INDICATOR),
    start: datetime.date = WINDOW_START,
    end: datetime.date = WINDOW_END,
    *,
    force_refresh: bool = False,
    include_vix: bool = True,
) -> dict:
    """Load weekly Friday-close prices for given tickers over [start, end].

    Returns dict:
      - prices: DataFrame index=W-FRI dates · columns=tickers
      - returns: DataFrame weekly simple returns (prices.pct_change.dropna)
      - vix: Series weekly ^VIX close (if include_vix=True, else None)
      - meta: {"dataset_hash": str, "fetched_at": iso, "n_weeks": int}
    """
    full_tickers = tuple(sorted(set(tickers) | ({VOL_INDICATOR} if include_vix else set())))
    cache_p = _cache_path(full_tickers, start, end)
    meta_p  = _meta_path(cache_p)

    if cache_p.exists() and not force_refresh:
        try:
            prices = pd.read_parquet(cache_p)
            meta   = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
            logger.info("universe loaded from cache %s · n_weeks=%d", cache_p.name, len(prices))
        except Exception as exc:
            logger.warning("cache read failed (%s) · refetching", exc)
            return load_universe_weekly(tickers, start, end,
                                         force_refresh=True, include_vix=include_vix)
    else:
        import yfinance as yf
        logger.info("yfinance fetching %d tickers %s → %s", len(full_tickers), start, end)
        df_raw = yf.download(list(full_tickers), start=start, end=end + datetime.timedelta(days=1),
                              auto_adjust=False, progress=False)
        # Adj Close handles dividends + splits (total-return-equivalent, fair vs PQTIX NAV)
        if isinstance(df_raw.columns, pd.MultiIndex):
            prices_daily = df_raw["Adj Close"]
        else:
            prices_daily = df_raw[["Adj Close"]]
            prices_daily.columns = [full_tickers[0]]
        # Resample to weekly Friday close (matches Sprint B replay convention)
        prices = prices_daily.resample("W-FRI").last().dropna(how="all")
        # Ensure column order
        prices = prices[list(full_tickers)]

        # Persist
        prices.to_parquet(cache_p)
        meta = {
            "tickers":        list(full_tickers),
            "start":          start.isoformat(),
            "end":            end.isoformat(),
            "n_weeks":        len(prices),
            "dataset_hash":   _compute_dataset_hash(prices),
            "fetched_at":     datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source":         "yfinance",
        }
        meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("universe cached %s · hash=%s · n_weeks=%d",
                     cache_p.name, meta["dataset_hash"][:12], meta["n_weeks"])

    # Compute returns
    returns = prices.pct_change().dropna(how="all")

    # Separate VIX if present
    vix = None
    if include_vix and VOL_INDICATOR in prices.columns:
        vix = prices[VOL_INDICATOR].copy()

    return {"prices": prices, "returns": returns, "vix": vix, "meta": meta}


def get_dataset_hash(start: datetime.date = WINDOW_START,
                      end: datetime.date = WINDOW_END) -> str:
    """Return dataset hash for reproducibility trinity (binds spec hash to data)."""
    res = load_universe_weekly(start=start, end=end, force_refresh=False)
    return res["meta"].get("dataset_hash", "unknown")
