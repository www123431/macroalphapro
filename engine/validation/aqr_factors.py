"""engine/validation/aqr_factors.py — AQR Betting-Against-Beta factor.

K1 BAB literally IS a betting-against-beta strategy, so regressing it
against FF5+UMD (R²=0.01 — those are single-stock factors, wrong lens)
tells us almost nothing. The right benchmark is the published BAB
factor itself. This loader pulls AQR's "Betting Against Beta: Equity
Factors Monthly" workbook (USA column), so we can ask the sharp
question: is K1 BAB just harvesting the published BAB premium, or does
it add residual alpha on top?

Source: AQR Data Library, "Betting Against Beta: Equity Factors,
Monthly" (.xlsx). Fetched once and cached to
data/cache/aqr_bab_monthly.xlsx + parsed parquet.

Caveat: AQR BAB is WITHIN-US-EQUITY and MONTHLY. K1 trades 43 ETFs
ACROSS asset classes at WEEKLY rebalance. So even AQR BAB is an
imperfect lens for a cross-asset BAB — but it is the recognized
published benchmark and far better than FF5+UMD. A high loading on
AQR BAB + low residual alpha ⇒ K1 is mostly buyable BAB beta; a
surviving residual alpha ⇒ K1's cross-asset implementation adds
something the single-equity BAB factor does not capture.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_AQR_URL = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
            "Betting-Against-Beta-Equity-Factors-Monthly.xlsx")
_XLSX_CACHE    = Path("data/cache/aqr_bab_monthly.xlsx")
_PARQUET_CACHE = Path("data/cache/aqr_bab_usa_monthly.parquet")


def _download_xlsx() -> Path:
    import urllib.request
    _XLSX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(_AQR_URL, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=30).read()
    _XLSX_CACHE.write_bytes(data)
    return _XLSX_CACHE


def _parse_bab_usa(xlsx_path: Path) -> pd.Series:
    """Parse the BAB Factors sheet, return monthly USA BAB as a Series
    indexed by month-end date (decimal returns)."""
    raw = pd.read_excel(xlsx_path, sheet_name="BAB Factors", header=None)
    hdr = None
    for i in range(min(40, len(raw))):
        if any(str(x).strip().upper() == "DATE" for x in raw.iloc[i].tolist()):
            hdr = i
            break
    if hdr is None:
        raise ValueError("AQR BAB sheet: could not locate DATE header row")
    df = pd.read_excel(xlsx_path, sheet_name="BAB Factors", header=hdr)
    df = df.rename(columns={df.columns[0]: "DATE"})
    if "USA" not in df.columns:
        raise ValueError(f"AQR BAB sheet: USA column missing; got {list(df.columns)[:10]}")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE", "USA"])
    s = pd.Series(df["USA"].astype(float).values,
                  index=pd.DatetimeIndex(df["DATE"]), name="BAB")
    return s.sort_index()


def load_bab_usa_monthly(force_refresh: bool = False) -> pd.Series:
    """Return monthly USA BAB factor (decimal), cached. Index = month-end."""
    if _PARQUET_CACHE.exists() and not force_refresh:
        try:
            return pd.read_parquet(_PARQUET_CACHE)["BAB"]
        except Exception as exc:
            logger.warning("aqr_factors: parquet cache read failed: %s", exc)
    xlsx = _XLSX_CACHE if _XLSX_CACHE.exists() and not force_refresh else _download_xlsx()
    s = _parse_bab_usa(xlsx)
    try:
        s.to_frame().to_parquet(_PARQUET_CACHE)
    except Exception as exc:
        logger.warning("aqr_factors: parquet cache write failed: %s", exc)
    return s


def load_ff_monthly(start: str = "2014-01-01", end: str = "2026-12-31") -> pd.DataFrame:
    """Ken French FF5 + Momentum MONTHLY (decimal), to pair with AQR BAB."""
    import pandas_datareader.data as web
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench",
                         start=start, end=end)[0] / 100.0
    mom = web.DataReader("F-F_Momentum_Factor", "famafrench",
                         start=start, end=end)[0] / 100.0
    mom.columns = [c.strip() for c in mom.columns]
    mom = mom.rename(columns={c: "UMD" for c in mom.columns if c.lower().startswith("mom")})
    df = ff5.join(mom, how="inner")
    # Ken French monthly index is a Period; convert to month-end timestamp.
    df.index = df.index.to_timestamp(how="end").normalize()
    return df


def weekly_to_monthly(weekly_returns: pd.Series) -> pd.Series:
    """Compound a weekly return series into month-end returns."""
    s = weekly_returns.copy()
    s.index = pd.to_datetime(s.index)
    monthly = s.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    return monthly
