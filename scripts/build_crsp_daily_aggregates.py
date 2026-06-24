"""scripts/build_crsp_daily_aggregates.py — pre-compute monthly snapshots
of daily-derived signals from CRSP DSF.

Unlocks 3 SIGNAL_REGISTRY entries (crsp_panel kind) that can be used by
cross_sec_us_equities template without that template needing to load
the full 627MB DSF parquet at every dispatch:

  - MAX-effect (Bali-Cakici-Whitelaw 2011)
       MAX_5_t = mean of top-5 daily returns over last 21 trading days
       Direction: LOW MAX is the LONG side (lottery-effect: high-MAX
       stocks earn lower future returns)

  - Amihud illiquidity (Amihud 2002)
       ILLIQ_t = mean(|ret| / dollar_vol) over last 21 trading days
                where dollar_vol = price × volume
       Direction: HIGH ILLIQ is the LONG side (illiquidity premium)

  - idio_vol residual (Ang-Hodrick-Xing-Zhang 2006)
       Fit CAPM regression on daily returns over last 60 trading days,
       take std of residuals.
       Direction: LOW idio_vol is the LONG side (low-vol anomaly,
       sharper than monthly vol_12m signal)

Output: data/cache/_crsp_daily_aggregates_monthly.parquet
Columns: month_end, permno, max_5_d21, illiq_d21, idiov_d60
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import time
import pandas as pd
import numpy as np


DSF_PATH = REPO_ROOT / "data" / "cache" / "_crsp_dsf_top3000.parquet"
VW_PATH  = REPO_ROOT / "data" / "cache" / "_crsp_vwretd_monthly.parquet"
OUT_PATH = REPO_ROOT / "data" / "cache" / "_crsp_daily_aggregates_monthly.parquet"


def _load_market_daily() -> pd.Series:
    """Build market return series from SPX log returns."""
    spx_path = REPO_ROOT / "data" / "cache" / "_vix_spx_daily.parquet"
    if not spx_path.is_file():
        raise RuntimeError("need _vix_spx_daily.parquet for market return")
    df = pd.read_parquet(spx_path)
    spx = df["SPX"].dropna()
    return np.log(spx / spx.shift(1)).dropna()


def main() -> int:
    print(f"[in] loading DSF {DSF_PATH.name}…", flush=True)
    t0 = time.time()
    dsf = pd.read_parquet(DSF_PATH)
    dsf["date"] = pd.to_datetime(dsf["date"])
    print(f"[in] DSF loaded {len(dsf):,} rows in {time.time()-t0:.1f}s", flush=True)

    mkt = _load_market_daily()
    print(f"[mkt] daily market returns: {len(mkt):,} days "
            f"{mkt.index.min().date()} to {mkt.index.max().date()}", flush=True)

    dsf = dsf.sort_values(["permno", "date"]).reset_index(drop=True)
    dsf["month_end"] = dsf["date"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()

    # ── Vectorized MAX-5 over rolling 21d ──
    # Use per-permno groupby.rolling. For top-5 mean, compute via sort
    # within window — pandas doesn't have rolling.nlargest. Approximate
    # MAX-5 by max + 4 × 95th percentile (rough). For cleaner top-5,
    # use rolling apply with raw=True for speed.
    print(f"[max] computing rolling MAX-5 (21d)…", flush=True)
    t0 = time.time()
    def _top5_mean(arr):
        if len(arr) < 5:
            return np.nan
        return np.sort(arr)[-5:].mean()
    dsf["max_5_d21"] = (
        dsf.groupby("permno", sort=False)["ret"]
           .rolling(21, min_periods=15)
           .apply(_top5_mean, raw=True)
           .reset_index(level=0, drop=True)
    )
    print(f"[max] done in {time.time()-t0:.1f}s", flush=True)

    # ── Vectorized Amihud illiquidity over rolling 21d ──
    print(f"[illiq] computing rolling Amihud illiquidity (21d)…", flush=True)
    t0 = time.time()
    dsf["dollar_vol"] = dsf["prc"].abs() * dsf["vol"]
    dsf["illiq_daily"] = (dsf["ret"].abs() / dsf["dollar_vol"].replace(0, np.nan))
    dsf["illiq_daily"] = dsf["illiq_daily"].replace([np.inf, -np.inf], np.nan)
    dsf["illiq_d21"] = (
        dsf.groupby("permno", sort=False)["illiq_daily"]
           .rolling(21, min_periods=15)
           .mean()
           .reset_index(level=0, drop=True)
        * 1e6
    )
    dsf = dsf.drop(columns=["dollar_vol", "illiq_daily"])
    print(f"[illiq] done in {time.time()-t0:.1f}s", flush=True)

    # ── Vectorized idio_vol over rolling 60d (vs market) ──
    # Subtract market beta×market: we'll approximate by ret - mkt and
    # take the rolling std. This is "market-adjusted vol" — close to
    # idio_vol when beta ≈ 1 (cross-sectional approximation for speed;
    # exact CAPM-residual std would require per-stock rolling regression
    # which is 10x slower).
    print(f"[idiov] computing market-adjusted vol (60d)…", flush=True)
    t0 = time.time()
    dsf = dsf.merge(mkt.rename("mkt").to_frame(), left_on="date", right_index=True, how="left")
    dsf["resid_proxy"] = dsf["ret"] - dsf["mkt"]
    dsf["idiov_d60"] = (
        dsf.groupby("permno", sort=False)["resid_proxy"]
           .rolling(60, min_periods=40)
           .std(ddof=1)
           .reset_index(level=0, drop=True)
    )
    dsf = dsf.drop(columns=["mkt", "resid_proxy"])
    print(f"[idiov] done in {time.time()-t0:.1f}s", flush=True)

    # ── Take month-end snapshot per (permno, month_end) ──
    print(f"[snap] taking month-end snapshot…", flush=True)
    t0 = time.time()
    out = (dsf.dropna(subset=["max_5_d21", "illiq_d21", "idiov_d60"], how="all")
              .groupby(["permno", "month_end"], sort=False)
              [["max_5_d21", "illiq_d21", "idiov_d60"]]
              .last()
              .reset_index())
    print(f"[snap] {len(out):,} (permno, month) cells in "
            f"{time.time()-t0:.1f}s", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"[done] {OUT_PATH.name} {size_mb:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
