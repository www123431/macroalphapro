"""scripts/fetch_cross_asset_macro_anchors.py — Tier C cross-asset
macro regime anchor library (lite).

Pulls 5 monthly cross-asset macro regime indicators from FRED for
use as anchor variables in carry / TSMOM / cross-asset factor
attribution. Per [[feedback-anchor-panel-sequential-residual-doctrine]]
amendment 2026-06-09: these are designed to be used as ADDITIONAL
regressors in a JOINT-model framework alongside FF5+MOM, not as
sequential residual-on-residual partialing.

Scope: this is the LITE version. True academic-standard cross-asset
anchors would include:
  - Lustig-Roussanov-Verdelhan 2011 HML_FX + DOL (from ~30 currency
    FX spot rates — self-construct, 2-3 days work)
  - Koijen-Moskowitz-Pedersen-Vrugt 2013 carry index across 5 asset
    classes (gated behind AQR registration)
  - He-Kelly-Manela 2017 intermediary capital ratio (from Manela
    website, manageable but separate fetcher)

We use FRED-available macro REGIME PROXIES that capture the spirit
of these academic factors:

  | FRED series  | Concept proxy                | Why for carry/cross-asset |
  |---|---|---|
  | VIXCLS       | Global volatility regime     | Carry crashes during high vol (Brunnermeier-Nagel) |
  | DTWEXBGS     | Broad USD index              | Dollar premium / DOL factor proxy |
  | BAA10Y       | Moody's Baa - 10Y spread     | Credit risk / funding liquidity (1990+) |
  | T10Y3M       | Term spread (10Y - 3M)       | Monetary regime / funding rates |
  | T10YIE       | 10Y breakeven inflation      | Inflation regime / real rates |

Output schema (data/anchor_library/cross_asset_macro_monthly.parquet):
  date                month-end DatetimeIndex (as column)
  VIX_level           CBOE VIX month-end level (LEVEL, not return)
  VIX_change          Δ VIX_level month-over-month
  DXY_return          DTWEXBGS monthly return (pct change)
  BAA_spread_change   Δ Moody's Baa - 10Y spread month-over-month (in pct)
  T10Y3M_change       Δ term spread month-over-month
  T10YIE_change       Δ breakeven inflation month-over-month

NOTE on regressor type: for regression on factor returns, CHANGES
(deltas) are the appropriate regressors (returns load on shocks,
not levels). VIX_level is kept as additional regime indicator for
non-linear regime tests.

NOTE on history: all FRED series have different start dates:
  - VIX: 1990-01
  - DTWEXBGS: 2006-01 (newer; older DTWEXM goes back to 1973)
  - BAMLH0A0HYM2: 1997-01
  - T10Y3M: 1981-12 (10Y) / 1981 (3M) — restricted by 3M start
  - T10YIE: 2003-01
The merged parquet uses INNER JOIN on date → effective history
starts at the MAX of all series start dates (2006-01).

Idempotent: --force re-downloads; default skips if cached.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "data" / "anchor_library"
OUT_PATH  = OUT_DIR / "cross_asset_macro_monthly.parquet"

sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


FRED_SERIES = {
    "VIXCLS":       "VIX_level",            # CBOE VIX daily close
    "DTWEXBGS":     "_dxy_level",           # Broad Dollar Index daily
    "BAA10Y":       "_baa_spread_level",    # Moody's Baa - 10Y spread
                                            # (1990+, replaces BAMLH0A0HYM2
                                            # which on FRED truncates to 2023+)
    "T10Y3M":       "_term_spread_level",   # 10Y-3M Treasury spread daily
    "T10YIE":       "_breakeven_level",     # 10Y breakeven daily (2003+)
}


def _to_month_end_close(daily_df: pd.DataFrame, series_id: str) -> pd.Series:
    """Convert FRED-fetch long-format daily DataFrame for ONE series
    to a monthly-end Series of last-observed values."""
    sub = daily_df[daily_df["series_id"] == series_id].copy()
    if sub.empty:
        return pd.Series(dtype=float)
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.set_index("date").sort_index()
    # FRED daily series sometimes have NaN (holidays); keep last
    # valid observation in each month
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    monthly = sub["value"].resample("ME").last().dropna()
    return monthly


def _ensure_fred_api_key_in_env() -> None:
    """get_secret() relies on streamlit context which is absent in
    standalone scripts. Read .streamlit/secrets.toml directly and
    inject into os.environ so engine.data.fetchers.api_fred works."""
    import os
    if os.environ.get("FRED_API_KEY"):
        return
    secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib   # type: ignore
    with secrets_path.open("rb") as f:
        data = tomllib.load(f)
    key = data.get("FRED_API_KEY") or data.get("fred_api_key")
    if key:
        os.environ["FRED_API_KEY"] = str(key)
        logger.info("FRED_API_KEY loaded from .streamlit/secrets.toml")


def fetch_cross_asset_macro_monthly(
    start: str = "1990-01-01",
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """Pull 5 FRED series, convert to monthly-end, derive change /
    return columns, inner-join on date. Returns the schema documented
    in module docstring."""
    _ensure_fred_api_key_in_env()
    from engine.data.fetchers.api_fred import fetch_series_batch
    if end is None:
        end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    series_ids = list(FRED_SERIES.keys())
    logger.info("fetching %d FRED series from %s to %s", len(series_ids),
                  start, end)
    raw = fetch_series_batch(start, end, series_ids=series_ids)
    if raw.empty:
        raise RuntimeError("FRED batch returned empty — check API key + connectivity")

    pieces = {}
    for sid, alias in FRED_SERIES.items():
        m = _to_month_end_close(raw, sid)
        if m.empty:
            logger.warning("FRED %s returned empty monthly series", sid)
            continue
        pieces[alias] = m
        logger.info("  %-12s %d months %s → %s",
                      sid, len(m), m.index.min().date(),
                      m.index.max().date())

    if not pieces:
        raise RuntimeError("all FRED series empty")

    df = pd.concat(pieces, axis=1)
    df.index.name = "date"

    # Derive change / return columns from levels
    out = pd.DataFrame(index=df.index)
    out["VIX_level"]        = df["VIX_level"]
    out["VIX_change"]       = df["VIX_level"].diff()
    out["DXY_return"]       = df["_dxy_level"].pct_change()
    out["BAA_spread_change"] = df["_baa_spread_level"].diff()
    out["T10Y3M_change"]    = df["_term_spread_level"].diff()
    out["T10YIE_change"]    = df["_breakeven_level"].diff()

    # Inner-join on date: drop first month (diff/pct_change introduce
    # NaN) AND drop any rows where ANY column is NaN (T10YIE doesn't
    # exist before 2003-01, etc.)
    out = out.dropna(how="any")
    out = out.reset_index()

    logger.info("merged: %d months (%s → %s), %d columns",
                  len(out), out["date"].min().date(),
                  out["date"].max().date(), len(out.columns))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                          help="re-download even if parquet exists")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        existing = pd.read_parquet(OUT_PATH)
        logger.info("cross-asset macro library cached at %s "
                      "(%d months %s → %s); use --force to refresh",
                      OUT_PATH, len(existing),
                      existing["date"].min().date(),
                      existing["date"].max().date())
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_cross_asset_macro_monthly()
    df.to_parquet(OUT_PATH, index=False)
    size_kb = OUT_PATH.stat().st_size / 1024
    logger.info("wrote %s (%.1f KB)", OUT_PATH, size_kb)

    # Sanity-print
    print()
    print("=== Cross-asset macro anchors (FRED, monthly) ===")
    print(f"Path:    {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Range:   {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Months:  {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    print("Last 3 rows:")
    print(df.tail(3).to_string(index=False))
    print()
    print("Summary stats (NB: VIX_level is LEVEL ~10-80; others are deltas/returns):")
    print(df.drop(columns=["date"]).describe().round(4).to_string())


if __name__ == "__main__":
    sys.exit(main())
