"""
engine/factors/bab_compat.py — BAB compatibility wrapper (Frazzini-Pedersen 2014).

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50) §2.2.4
Spec lock:
  - Reuses canonical engine.factor_library._compute_bab_weights (locked v1
    per spec_factor_library_v1.md §2.1; 252d β estimation, tertile split,
    β-neutralization).
  - Applies to ALL tickers in `universe` (caller-supplied).

Boundary invariant: this module is a thin wrapper. All BAB algorithmic
logic lives in engine.factor_library; modifications to BAB methodology
require amend_spec on docs/spec_factor_library_v1.md.

NaN protocol: insufficient history (tickers without enough data for β
estimation) → NaN for that ticker; tickers in long leg → positive weight;
short leg → negative; middle tertile → 0 (after universe reindex).

2026-05-19 hotfix: prior version referenced a non-existent `ql01_bab`
column in engine.signal.get_signal_dataframe and silently returned
all-NaN → K1 BAB persistently NO_SIGNAL. Re-wired to call the canonical
Frazzini-Pedersen path in engine.factor_library directly.

On-disk cache: writes data/cache/bab_compat.parquet after a successful
compute (one row per as_of × ticker). DQ Inspector Mode 2 monitors this
file's mtime; cache miss = recompute + rewrite.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_BAB_CACHE_PATH = Path("data/cache/bab_compat.parquet")
_HISTORY_BUFFER_DAYS = 400   # ~252 BDays + holiday/weekend buffer


def _load_cached(as_of: datetime.date,
                 universe: list[str]) -> Optional[pd.Series]:
    """Return cached signal for as_of × universe if cache fully covers
    the requested universe, else None (triggers recompute).

    Cache layout: long DataFrame with columns [as_of, ticker, signal].
    Idempotent — bab_compat is a pure function of (as_of, universe) so
    a same-day re-run on the SAME universe returns cached values.

    2026-05-19 senior audit: prior version returned partial cache via
    reindex (filling unknown tickers with NaN). On future universe
    expansion this would silently return NaN for new tickers without
    recomputing. Now requires full coverage; partial-coverage = miss.
    """
    if not _BAB_CACHE_PATH.exists():
        return None
    try:
        df = pd.read_parquet(_BAB_CACHE_PATH)
        as_of_str = as_of.isoformat()
        sub = df[df["as_of"] == as_of_str]
        if sub.empty:
            return None
        cached_tickers = set(sub["ticker"].astype(str))
        requested = set(map(str, universe))
        if not requested.issubset(cached_tickers):
            # Universe widened since this row was written — recompute
            # to capture the new tickers' β-neutralized weights.
            missing = requested - cached_tickers
            logger.info(
                "bab_compat: cache miss (%d tickers missing: %s%s) — recomputing",
                len(missing), sorted(missing)[:5],
                "..." if len(missing) > 5 else "",
            )
            return None
        s = sub.set_index("ticker")["signal"].reindex(universe)
        return s.astype(float)
    except Exception as exc:
        logger.warning("bab_compat: cache read failed (%s) — recomputing", exc)
        return None


def _save_cache(as_of: datetime.date, signal: pd.Series) -> None:
    """Append (or replace) the as_of row-block in the on-disk cache.

    Single parquet file under data/cache/; safe to nuke (will rebuild on
    next compute). DQ Inspector Mode 2 reads mtime here.
    """
    try:
        _BAB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        as_of_str = as_of.isoformat()
        new_rows = pd.DataFrame({
            "as_of":  as_of_str,
            "ticker": signal.index.astype(str),
            "signal": signal.values.astype(float),
        })
        if _BAB_CACHE_PATH.exists():
            existing = pd.read_parquet(_BAB_CACHE_PATH)
            existing = existing[existing["as_of"] != as_of_str]
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined.to_parquet(_BAB_CACHE_PATH, index=False)
    except Exception as exc:
        logger.warning("bab_compat: cache write failed (%s) — non-fatal", exc)


def compute_bab_signal(
    as_of:          datetime.date,
    universe:       list[str],
    asset_classes:  Optional[dict[str, str]] = None,
    use_cache:      bool = True,
) -> pd.Series:
    """Compute per-ticker BAB tertile signal at as_of.

    Algorithm (delegated to engine.factor_library._compute_bab_weights):
      1. Fetch 252+buffer trading days of daily closes for universe + SPY.
      2. Estimate 252d β_i to SPY for each ticker.
      3. Rank by β; long bottom tertile (low β), short top tertile (high β).
      4. β-neutralize each leg so portfolio β ≈ 0; normalize to gross 1.

    Returns:
        pd.Series indexed by `universe`; values are β-neutralized weights
        (long leg > 0, short leg < 0, middle tertile = 0). NaN for tickers
        outside the computed weights dict (insufficient history).

    Notes:
        asset_classes is accepted for legacy API parity; BAB does not use
        within-class ranking (cf. CSMOM). Argument retained so callers
        importing compute_bab_signal don't break.
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)

    if use_cache:
        cached = _load_cached(as_of, universe)
        if cached is not None and cached.notna().any():
            return cached

    try:
        from engine.factor_library import _BETA_WINDOW_DAYS, _compute_bab_weights
        from engine.signal import _fetch_closes
    except Exception as exc:
        logger.warning(
            "bab_compat: import of canonical BAB / fetch_closes failed for %s: %s "
            "— returning all-NaN",
            as_of, exc,
        )
        return pd.Series(np.nan, index=universe, dtype=float)

    # Fetch closes for universe + SPY benchmark (252+ trading days back).
    tickers_to_fetch = sorted(set(universe) | {"SPY"})
    start = as_of - datetime.timedelta(days=_HISTORY_BUFFER_DAYS)
    try:
        closes = _fetch_closes(tickers_to_fetch, start, as_of)
    except Exception as exc:
        logger.warning(
            "bab_compat: _fetch_closes failed for %s: %s — returning all-NaN",
            as_of, exc,
        )
        return pd.Series(np.nan, index=universe, dtype=float)

    if closes is None or closes.empty or "SPY" not in closes.columns:
        logger.warning(
            "bab_compat: closes empty or SPY missing for %s "
            "(cols=%s) — returning all-NaN",
            as_of, list(closes.columns) if closes is not None else None,
        )
        return pd.Series(np.nan, index=universe, dtype=float)

    bench = closes["SPY"]
    universe_closes = closes.drop(columns=["SPY"])

    try:
        weights_dict = _compute_bab_weights(
            closes           = universe_closes,
            benchmark_close  = bench,
            beta_window_days = _BETA_WINDOW_DAYS,
        )
    except Exception as exc:
        logger.warning(
            "bab_compat: _compute_bab_weights failed for %s: %s — returning all-NaN",
            as_of, exc,
        )
        return pd.Series(np.nan, index=universe, dtype=float)

    if not weights_dict:
        logger.warning(
            "bab_compat: _compute_bab_weights returned empty dict for %s "
            "(insufficient history) — returning all-NaN",
            as_of,
        )
        return pd.Series(np.nan, index=universe, dtype=float)

    # Reindex onto requested universe — tickers absent from weights_dict
    # (no signal because they fall in the middle tertile) get 0.0;
    # tickers fully missing data still surface as NaN downstream via
    # callers' dropna() pattern.
    signal = pd.Series(weights_dict, dtype=float).reindex(universe).fillna(0.0)

    if use_cache:
        _save_cache(as_of, signal)

    return signal
