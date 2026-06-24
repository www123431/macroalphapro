"""scripts/fetch_gfx_vol_msss.py — MSSS 2012 global FX volatility factor.

B.3 (2026-06-10, deferred from senior施工建议 until A.1 shipped).
Constructs the Menkhoff-Sarno-Schmeling-Schrimpf 2012 "global FX
volatility" factor σ_FX from DAILY G10 spot data:

  σ_FX_t = (1/T_t) Σ_{τ∈t} [ (1/K_τ) Σ_k |r_{k,τ}| ]

  For each trading day τ in month t, average the ABSOLUTE daily log
  returns across the K available G10 currencies; then average those
  daily cross-sectional means over the days of the month. The
  INNOVATION (month-over-month change) is the priced risk factor —
  MSSS 2012 Table 3 shows carry's negative loading on Δσ_FX explains
  ~90%+ of the HML_FX cross-sectional spread. This is THE academic
  answer to "carry earns a premium for crash/volatility exposure".

Why this matters for our stack:
  cross_asset_attribution currently proxies vol exposure with
  VIX_change (US equity vol). MSSS's point is that FX-SPECIFIC vol
  is the priced factor for FX strategies — VIX correlates but is
  mis-specified for carry attribution. With GFX_VOL joined into the
  cross-asset extension, a carry sleeve's "crisis exposure" gets the
  textbook-correct regressor.

Data: same 10 FRED DEX series as fetch_fx_spot_g10.py but at DAILY
frequency (FRED's native granularity for these series). Quote
direction is IRRELEVANT here — |log return| is invariant to
inversion (|log(1/x)_t - log(1/x)_{t-1}| = |log(x)_t - log(x)_{t-1}|),
so no normalization needed.

Output (data/anchor_library/gfx_vol_monthly.parquet):
  date              month-end (as column for parquet round-trip)
  GFX_VOL_level     σ_FX in decimal daily-|return| units (~0.003-0.01)
  GFX_VOL_change    first difference of level — the priced innovation

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
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT_PATH = REPO_ROOT / "data" / "anchor_library" / "gfx_vol_monthly.parquet"

# Same series as fetch_fx_spot_g10.py. Quote direction irrelevant for
# absolute log returns (see module docstring).
FRED_FX_SERIES = (
    ("DEXJPUS", "JPY"), ("DEXUSEU", "EUR"), ("DEXUSUK", "GBP"),
    ("DEXSZUS", "CHF"), ("DEXCAUS", "CAD"), ("DEXUSAL", "AUD"),
    ("DEXUSNZ", "NZD"), ("DEXSDUS", "SEK"), ("DEXNOUS", "NOK"),
    ("DEXDNUS", "DKK"),
)

# Require at least this many currencies reporting on a day for the
# cross-sectional mean to be meaningful (holiday-calendar mismatches
# leave sparse days at panel edges).
MIN_CCYS_PER_DAY = 5
# Require at least this many trading days for a monthly σ_FX value.
MIN_DAYS_PER_MONTH = 10


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
        import os
        os.environ["FRED_API_KEY"] = str(key)


def compute_gfx_vol(daily_abs_returns: pd.DataFrame) -> pd.DataFrame:
    """Pure function: daily |log return| panel → monthly σ_FX.

    Args:
      daily_abs_returns: DatetimeIndex (daily), one column per currency,
        values = |daily log return| (decimal). NaN = no observation.

    Returns DataFrame indexed by month-end with columns
      GFX_VOL_level / GFX_VOL_change. Months with fewer than
      MIN_DAYS_PER_MONTH valid days are dropped.
    """
    # Daily cross-sectional mean over currencies with data that day
    n_ccys = daily_abs_returns.notna().sum(axis=1)
    daily_mean = daily_abs_returns.mean(axis=1, skipna=True)
    daily_mean = daily_mean[n_ccys >= MIN_CCYS_PER_DAY]

    # Monthly average of the daily means
    monthly = daily_mean.resample("ME").agg(["mean", "count"])
    level = monthly["mean"][monthly["count"] >= MIN_DAYS_PER_MONTH]
    out = pd.DataFrame({
        "GFX_VOL_level":  level,
        "GFX_VOL_change": level.diff(),
    })
    return out.dropna(how="all")


def fetch_gfx_vol_monthly(
    start: str = "1999-01-01",
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """Pull daily G10 FX from FRED, build MSSS σ_FX monthly."""
    _ensure_fred_api_key_in_env()
    import os
    from fredapi import Fred
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    if end is None:
        end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    abs_returns: dict[str, pd.Series] = {}
    for sid, ccy in FRED_FX_SERIES:
        try:
            daily = fred.get_series(sid, observation_start=start,
                                       observation_end=end)
        except Exception as exc:
            logger.warning("%s fetch failed: %s", sid, exc)
            continue
        s = daily.dropna().sort_index()
        s.index = pd.to_datetime(s.index)
        if len(s) < 100:
            logger.warning("%s too short (%d obs); skipping", sid, len(s))
            continue
        abs_ret = np.log(s).diff().abs()
        abs_returns[ccy] = abs_ret
        logger.info("  %-10s %s %d daily obs %s -> %s",
                      sid, ccy, len(abs_ret),
                      abs_ret.index.min().date(),
                      abs_ret.index.max().date())

    if len(abs_returns) < MIN_CCYS_PER_DAY:
        raise RuntimeError(
            f"only {len(abs_returns)} FX series fetched; "
            f"need >= {MIN_CCYS_PER_DAY}")

    panel = pd.concat(abs_returns, axis=1)
    out = compute_gfx_vol(panel)
    out.index.name = "date"
    logger.info("GFX_VOL: %d monthly obs %s -> %s "
                  "(level mean=%.5f, std=%.5f)",
                  len(out), out.index.min().date(),
                  out.index.max().date(),
                  out["GFX_VOL_level"].mean(),
                  out["GFX_VOL_level"].std())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                            help="Re-download + overwrite cache.")
    parser.add_argument("--start", default="1999-01-01")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"Cache exists at {OUT_PATH}; use --force to refresh.")
        return 0

    df = fetch_gfx_vol_monthly(start=args.start)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
