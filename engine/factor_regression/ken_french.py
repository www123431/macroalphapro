"""engine.factor_regression.ken_french — fetch + cache Fama-French factors.

Sources from Ken French's Dartmouth data library:
  - F-F Research Data 5 Factors 2x3 (daily):    MKT_RF, SMB, HML, RMW, CMA, RF
  - F-F Momentum Factor (daily):                 MOM

Combines into one daily DataFrame indexed by datetime. Aggregated to
weekly (Friday-ending) for alignment with our paper-trade NAV weekly
series.

Cached at data/cache/ken_french_ff5_mom_daily.parquet — gitignored
because the raw CSV is freely downloadable; we don't bloat the repo
with derivative data.
"""
from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DAILY  = _REPO_ROOT / "data" / "cache" / "ken_french_ff5_mom_daily.parquet"
_CACHE_WEEKLY = _REPO_ROOT / "data" / "cache" / "ken_french_ff5_mom_weekly.parquet"

_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


def _download_zip_csv(url: str) -> pd.DataFrame:
    """Download Ken French ZIP, extract the CSV, parse with header
    auto-detection (the files have variable preamble text before the
    data table)."""
    logger.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:
        zip_bytes = resp.read()
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    csv_name = z.namelist()[0]
    raw = z.read(csv_name).decode("latin-1")

    # Find the line where data starts — first line whose first column is
    # a 6-8 digit date.
    lines = raw.splitlines()
    start = None
    for i, ln in enumerate(lines):
        parts = ln.split(",")
        if parts and parts[0].strip().isdigit() and 6 <= len(parts[0].strip()) <= 10:
            start = i
            # Header is the line before
            header_line = lines[i - 1] if i > 0 else None
            break
    if start is None:
        raise RuntimeError(f"could not locate data start in {csv_name}")

    # Find end (Ken French files include a "Copyright" footer)
    end = len(lines)
    for j in range(start, len(lines)):
        parts = lines[j].split(",")
        if not parts[0].strip().isdigit():
            end = j
            break

    table = "\n".join([header_line] + lines[start:end])
    df = pd.read_csv(io.StringIO(table))
    # Date column is unnamed; rename
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")
    # Values are percent — convert to decimal
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def fetch_ff5_mom_daily(force: bool = False) -> pd.DataFrame:
    """Return daily FF5 + MOM as a DataFrame. Cached to parquet.

    Columns: MKT_RF, SMB, HML, RMW, CMA, RF, MOM
    Index:   pandas DatetimeIndex (UTC-naive, business days)
    """
    if _CACHE_DAILY.is_file() and not force:
        return pd.read_parquet(_CACHE_DAILY)

    ff5 = _download_zip_csv(_FF5_URL)
    mom = _download_zip_csv(_MOM_URL)

    # FF5 typically has Mkt-RF, SMB, HML, RMW, CMA, RF
    ff5 = ff5.rename(columns={
        "Mkt-RF": "MKT_RF",
        "SMB":     "SMB",
        "HML":     "HML",
        "RMW":     "RMW",
        "CMA":     "CMA",
        "RF":      "RF",
    })
    mom = mom.rename(columns={"Mom": "MOM", "Mom   ": "MOM"})

    out = ff5.join(mom, how="inner")
    _CACHE_DAILY.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_CACHE_DAILY)
    logger.info("cached %d daily rows to %s", len(out), _CACHE_DAILY)
    return out


def fetch_ff5_mom_weekly(force: bool = False, week_ending: str = "W-FRI") -> pd.DataFrame:
    """Aggregate daily FF5 + MOM to weekly by compounding returns
    within each week ending on Friday (matches our paper-trade
    week_end convention).
    """
    if _CACHE_WEEKLY.is_file() and not force:
        return pd.read_parquet(_CACHE_WEEKLY)

    daily = fetch_ff5_mom_daily(force=force)

    # Compound returns to weekly: r_week = prod(1 + r_d) - 1
    # RF compounds the same way.
    weekly = (1.0 + daily).resample(week_ending).prod() - 1.0
    weekly = weekly.dropna(how="all")
    _CACHE_WEEKLY.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(_CACHE_WEEKLY)
    logger.info("cached %d weekly rows to %s", len(weekly), _CACHE_WEEKLY)
    return weekly
