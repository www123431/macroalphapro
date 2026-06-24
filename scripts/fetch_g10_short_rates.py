"""scripts/fetch_g10_short_rates.py — G10 3-month interbank rates.

Second commit in the Lustig-Roussanov-Verdelhan 2011 HML_FX
construction chain. Per Covered Interest Parity:
  forward_discount ≈ i_foreign - i_USD
So short-rate differentials are the carry-sort key for HML_FX
portfolio construction (Menkhoff-Sarno-Schmeling-Schrimpf 2012 §2).

Data source
-----------
FRED IR3TIB01<CC>M156N family: 3-month interbank lending rates,
monthly observations, annualized %.
  US: IR3TIB01USM156N  (1999-01+; mean 2.33%)
  EU: IR3TIB01EZM156N  (1999-01+; mean 1.68%)
  JP: IR3TIB01JPM156N  (2002-04+; mean 0.25%)  ← binding constraint
  UK: IR3TIB01GBM156N  (1999-01+; mean 2.76%)
  CH: IR3TIB01CHM156N  (1999-07+; mean 0.44%)
  CA: IR3TIB01CAM156N  (1999-01+; mean 2.18%)
  AU: IR3TIB01AUM156N  (1999-01+; mean 3.78%)
  NZ: IR3TIB01NZM156N  (1999-01+; mean 4.11%)  ← highest carry destination
  SE: IR3TIB01SEM156N  (1999-01+; mean 1.63%)
  NO: IR3TIB01NOM156N  (1999-01+; mean 3.20%)
  DK: IR3TIB01DKM156N  (1999-01+; mean 1.88%)

After inner-join: 2002-04 → present (binding on JP series).

Output schema (data/anchor_library/g10_short_rates_monthly.parquet):
  date              month-end DatetimeIndex (as column)
  rate_<CCY>_pct    11 cols (annualized % short rate per currency)
  rdiff_<CCY>_pct   10 cols (rate_<CCY>_pct - rate_USD_pct;
                              USD itself omitted = 0 by construction)
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
OUT_PATH  = OUT_DIR / "g10_short_rates_monthly.parquet"

sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# (FRED_series, currency_code)
FRED_RATE_SERIES = (
    ("IR3TIB01USM156N", "USD"),
    ("IR3TIB01EZM156N", "EUR"),
    ("IR3TIB01JPM156N", "JPY"),
    ("IR3TIB01GBM156N", "GBP"),
    ("IR3TIB01CHM156N", "CHF"),
    ("IR3TIB01CAM156N", "CAD"),
    ("IR3TIB01AUM156N", "AUD"),
    ("IR3TIB01NZM156N", "NZD"),
    ("IR3TIB01SEM156N", "SEK"),
    ("IR3TIB01NOM156N", "NOK"),
    ("IR3TIB01DKM156N", "DKK"),
)


def _ensure_fred_api_key_in_env() -> None:
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


def _to_month_end(monthly_series: pd.Series) -> pd.Series:
    """FRED IR3TIB series have month-START observation dates.
    Shift to month-end for consistency with FX spot fetcher (which
    resamples daily to month-end last-close)."""
    s = monthly_series.dropna().sort_index()
    s.index = pd.to_datetime(s.index)
    # Map to month-end
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s


def fetch_g10_short_rates_monthly(
    start: str = "1999-01-01",
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """Pull G10 3-month interbank rates, normalize to month-end,
    compute rate differentials vs USD. Inner-join on date."""
    _ensure_fred_api_key_in_env()
    from fredapi import Fred
    import os
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    if end is None:
        end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    pieces: dict[str, pd.Series] = {}
    for sid, ccy in FRED_RATE_SERIES:
        try:
            raw = fred.get_series(sid, observation_start=start,
                                       observation_end=end)
        except Exception as exc:
            logger.warning("%s fetch failed: %s", sid, exc)
            continue
        monthly = _to_month_end(raw)
        if monthly.empty:
            continue
        pieces[f"rate_{ccy}_pct"] = monthly
        logger.info("  %-20s %s %3d months %s → %s  mean=%.3f%%",
                      sid, ccy, len(monthly),
                      monthly.index.min().date(),
                      monthly.index.max().date(),
                      monthly.mean())

    if not pieces:
        raise RuntimeError("all rate series fetch failed")

    df = pd.concat(pieces, axis=1)
    df.index.name = "date"

    # Inner-join (drop any row with any NaN) — binding constraint is JPY 2002-04
    df = df.dropna(how="any")

    # Compute rate differentials vs USD: i_FCY - i_USD
    # USD itself has rdiff = 0 by definition (omitted from output)
    usd_rate = df["rate_USD_pct"]
    for col in df.columns:
        if col == "rate_USD_pct":
            continue
        ccy = col.replace("rate_", "").replace("_pct", "")
        df[f"rdiff_{ccy}_pct"] = df[col] - usd_rate

    out = df.reset_index()
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
        logger.info("G10 rates cached at %s (%d months %s → %s); "
                      "use --force to refresh",
                      OUT_PATH, len(existing),
                      existing["date"].min().date(),
                      existing["date"].max().date())
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_g10_short_rates_monthly()
    df.to_parquet(OUT_PATH, index=False)
    size_kb = OUT_PATH.stat().st_size / 1024
    logger.info("wrote %s (%.1f KB)", OUT_PATH, size_kb)

    print()
    print("=== G10 3-month interbank rates (FRED) ===")
    print(f"Path:    {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Range:   {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Months:  {len(df)}")
    print()
    print("Mean rate differential vs USD (carry sort key):")
    rdiff_cols = sorted([c for c in df.columns if c.startswith("rdiff_")],
                            key=lambda c: -df[c].mean())
    for c in rdiff_cols:
        ccy = c.replace("rdiff_", "").replace("_pct", "")
        print(f"  {ccy:6s} mean_rdiff = {df[c].mean():+.3f}%  "
                f"(carry sort: {'HIGH' if df[c].mean()>0.5 else 'MID' if df[c].mean()>-0.5 else 'LOW'})")


if __name__ == "__main__":
    sys.exit(main())
