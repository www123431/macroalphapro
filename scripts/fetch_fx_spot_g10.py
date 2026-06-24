"""scripts/fetch_fx_spot_g10.py — G10 FX spot monthly fetcher.

Foundation commit for Lustig-Roussanov-Verdelhan 2011 HML_FX
construction. Per docs/spec_role_aware_test_routing.md §15.A1
cross-asset acknowledgments + cross_asset_attribution.py module
docstring TODO: the macro-lite proxy (5 FRED regime variables)
is mis-specified for FX carry; only LRV HML_FX + DOL constructed
from actual G10 spot data is academically defensible.

This commit ships the DATA LAYER ONLY. Subsequent commits will:
  - fetch G10 short-rate differentials (forward discounts)
  - construct monthly carry-sorted HML_FX + DOL portfolios per
    LRV 2011 / Menkhoff-Sarno-Schmeling-Schrimpf 2012
  - register as additional cross-asset anchors in
    cross_asset_attribution

G10 currency set (vs USD): JPY EUR GBP CHF CAD AUD NZD SEK NOK DKK
  - 9 series (USD is base; no own series)
  - EUR start 1999-01 (post-euro launch); restricts merged window
  - Other 9 go back to 1971

FRED quote convention mix
-------------------------
FRED uses INDIRECT quotes (foreign per USD) for some pairs and
DIRECT quotes (USD per foreign) for others. We normalize ALL to
USD-per-FCY (direct from US perspective) so log returns have
consistent sign: rises = foreign currency strengthening vs USD.

Indirect (need to INVERT):
  DEXJPUS = JPY per USD     → invert to USD per JPY
  DEXSZUS = CHF per USD     → invert
  DEXCAUS = CAD per USD     → invert
  DEXSDUS = SEK per USD     → invert
  DEXNOUS = NOK per USD     → invert
  DEXDNUS = DKK per USD     → invert

Direct (already USD-per-FCY):
  DEXUSEU = USD per EUR     → keep
  DEXUSUK = USD per GBP     → keep
  DEXUSAL = USD per AUD     → keep
  DEXUSNZ = USD per NZD     → keep

Output schema (data/anchor_library/fx_spot_g10_monthly.parquet):
  date                  month-end DatetimeIndex (as column)
  spot_<CCY>_per_USD    USD-per-FCY direct quote, month-end close
  logret_<CCY>          log monthly return of the FCY vs USD
                         (positive = FCY appreciated vs USD)

Idempotent: --force re-downloads + overwrites; default skips cache.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "data" / "anchor_library"
OUT_PATH  = OUT_DIR / "fx_spot_g10_monthly.parquet"

sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# (FRED_series, currency_code, is_direct_quote)
# is_direct_quote=True means series is already USD per FCY (no invert)
# is_direct_quote=False means series is FCY per USD (must invert)
FRED_FX_SERIES = (
    ("DEXJPUS", "JPY", False),
    ("DEXUSEU", "EUR", True),
    ("DEXUSUK", "GBP", True),
    ("DEXSZUS", "CHF", False),
    ("DEXCAUS", "CAD", False),
    ("DEXUSAL", "AUD", True),
    ("DEXUSNZ", "NZD", True),
    ("DEXSDUS", "SEK", False),
    ("DEXNOUS", "NOK", False),
    ("DEXDNUS", "DKK", False),
)


def _ensure_fred_api_key_in_env() -> None:
    """get_secret() relies on streamlit; standalone scripts need
    env var. Read .streamlit/secrets.toml directly and inject."""
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


def _to_month_end_close(daily_series: pd.Series) -> pd.Series:
    """Take last valid daily observation in each month."""
    s = daily_series.dropna().sort_index()
    s.index = pd.to_datetime(s.index)
    monthly = s.resample("ME").last().dropna()
    return monthly


def fetch_g10_fx_spot_monthly(
    start: str = "1999-01-01",
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """Pull 10 G10 FX series from FRED, normalize to USD-per-FCY,
    resample monthly-end, compute log returns. Inner-join on date.

    Default start 1999-01-01 chosen because EUR is the binding
    constraint (post-euro launch). For pre-1999 work, pass an
    earlier start AND drop EUR (or use synthetic DEM-basket EUR
    — out of scope for this commit).
    """
    _ensure_fred_api_key_in_env()
    from fredapi import Fred
    import os
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    if end is None:
        end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    pieces: dict[str, pd.Series] = {}
    for sid, ccy, is_direct in FRED_FX_SERIES:
        try:
            daily = fred.get_series(sid, observation_start=start,
                                       observation_end=end)
        except Exception as exc:
            logger.warning("%s fetch failed: %s", sid, exc)
            continue
        monthly = _to_month_end_close(daily)
        if monthly.empty:
            continue
        # Normalize to USD per FCY (direct quote)
        if not is_direct:
            monthly = 1.0 / monthly
        pieces[f"spot_{ccy}_per_USD"] = monthly
        logger.info("  %-10s %s %d months %s → %s",
                      sid, ccy, len(monthly),
                      monthly.index.min().date(),
                      monthly.index.max().date())

    if not pieces:
        raise RuntimeError("all FX series fetch failed")

    df = pd.concat(pieces, axis=1)
    df.index.name = "date"

    # Compute log monthly returns
    log_levels = np.log(df)
    log_returns = log_levels.diff()
    # Rename columns: spot_JPY_per_USD → logret_JPY
    log_returns.columns = [
        f"logret_{c.split('_')[1]}" for c in log_returns.columns
    ]

    # Concatenate spot levels + log returns
    out = pd.concat([df, log_returns], axis=1)
    # Drop first row (log return NaN by diff) + any row with any NaN
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
        logger.info("G10 FX library cached at %s "
                      "(%d months %s → %s); use --force to refresh",
                      OUT_PATH, len(existing),
                      existing["date"].min().date(),
                      existing["date"].max().date())
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_g10_fx_spot_monthly()
    df.to_parquet(OUT_PATH, index=False)
    size_kb = OUT_PATH.stat().st_size / 1024
    logger.info("wrote %s (%.1f KB)", OUT_PATH, size_kb)

    print()
    print("=== G10 FX spot monthly (FRED) ===")
    print(f"Path:    {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Range:   {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Months:  {len(df)}")
    print(f"Currencies: 10 (vs USD; all normalized to direct quote)")
    print()
    print("Last 3 rows (log returns only):")
    logret_cols = ["date"] + [c for c in df.columns if c.startswith("logret_")]
    print(df[logret_cols].tail(3).to_string(index=False))
    print()
    print("Summary stats (log monthly returns):")
    print(df[[c for c in df.columns if c.startswith("logret_")]]
            .describe().round(4).to_string())


if __name__ == "__main__":
    sys.exit(main())
