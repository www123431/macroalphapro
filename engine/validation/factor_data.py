"""engine/validation/factor_data.py — Ken French FF5 + Momentum, weekly.

Pulls the Fama-French 5-factor daily series + the Momentum (UMD) daily
factor from the Ken French Data Library via pandas_datareader, resamples
to weekly (Friday week-end to match the strategy return parquet), and
caches to data/cache/ff_factors_weekly.parquet.

Factor set: Mkt-RF, SMB, HML, RMW, CMA, UMD, RF. All in DECIMAL
(Ken French publishes in percent; we divide by 100).

NOTE on completeness: AQR's BAB + QMJ factors would be ideal for
decomposing K1 BAB specifically (it literally IS betting-against-beta),
but AQR distributes them as Excel downloads that are fragile to
automate. FF5 + UMD is the standard academic first-pass and is what we
regress against here. If a strategy shows high residual alpha after
FF5 + UMD, that is the signal to go pull the actual BAB / QMJ factor
and check whether the "alpha" is just BAB exposure.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE = Path("data/cache/ff_factors_weekly.parquet")
_FF5_DAILY = "F-F_Research_Data_5_Factors_2x3_daily"
_MOM_DAILY = "F-F_Momentum_Factor_daily"


def _fetch_daily(start: str, end: str) -> pd.DataFrame:
    """Fetch FF5 + Momentum daily from Ken French, merged, in decimal."""
    import pandas_datareader.data as web

    ff5 = web.DataReader(_FF5_DAILY, "famafrench", start=start, end=end)[0]
    mom = web.DataReader(_MOM_DAILY, "famafrench", start=start, end=end)[0]
    # Ken French publishes in percent.
    ff5 = ff5 / 100.0
    mom = mom / 100.0
    mom.columns = [c.strip() for c in mom.columns]   # 'Mom   ' → 'Mom'
    df = ff5.join(mom, how="inner")
    # Standardize the momentum column name to UMD.
    rename = {}
    for c in df.columns:
        if c.lower().startswith("mom"):
            rename[c] = "UMD"
    df = df.rename(columns=rename)
    df.index = pd.to_datetime(df.index)
    return df


def _resample_weekly(daily: pd.DataFrame, week_anchor: str = "W-FRI") -> pd.DataFrame:
    """Compound daily factor returns into weekly (Friday week-end).

    Factor returns compound multiplicatively within the week:
      weekly = prod(1 + daily) - 1
    RF compounds the same way.
    """
    return daily.resample(week_anchor).apply(lambda x: (1.0 + x).prod() - 1.0)


def load_factors_weekly(
    start:        str = "2014-01-01",
    end:          str = "2024-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return weekly FF5 + UMD + RF, cached.

    Columns: Mkt-RF, SMB, HML, RMW, CMA, UMD, RF (all decimal weekly).
    Index: weekly Friday timestamps.

    Caches to data/cache/ff_factors_weekly.parquet. Pass force_refresh
    to re-fetch from Ken French.
    """
    if _CACHE.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(_CACHE)
            # Use cache only if it spans the requested window.
            if (cached.index.min() <= pd.Timestamp(start)
                    and cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=14)):
                return cached
        except Exception as exc:
            logger.warning("factor_data: cache read failed, refetching: %s", exc)

    daily  = _fetch_daily(start, end)
    weekly = _resample_weekly(daily)
    weekly = weekly.dropna(how="all")

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        weekly.to_parquet(_CACHE)
    except Exception as exc:
        logger.warning("factor_data: cache write failed: %s", exc)
    return weekly


def align_returns_to_factors(
    strat_returns: pd.DataFrame,
    factors:       pd.DataFrame,
    tolerance_days: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align a weekly strategy-returns frame (index = week_end) with the
    weekly factor frame. Both are weekly but week-end anchors may differ
    by a few days; we reindex factors onto the strategy index using a
    nearest-date merge within ``tolerance_days``.

    Returns (aligned_strat, aligned_factors) on the SAME index, rows
    with any missing factor dropped.
    """
    s = strat_returns.copy()
    f = factors.copy()
    s.index = pd.to_datetime(s.index)
    f.index = pd.to_datetime(f.index)

    # Nearest-date alignment: for each strategy week_end, grab the closest
    # factor week within tolerance.
    merged = pd.merge_asof(
        s.sort_index().reset_index().rename(columns={s.index.name or "index": "date"}),
        f.sort_index().reset_index().rename(columns={f.index.name or "index": "fdate"}),
        left_on="date", right_on="fdate",
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    merged = merged.dropna(subset=list(f.columns))
    aligned_strat   = merged.set_index("date")[list(s.columns)]
    aligned_factors = merged.set_index("date")[list(f.columns)]
    return aligned_strat, aligned_factors
