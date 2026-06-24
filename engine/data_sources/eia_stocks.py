"""engine.data_sources.eia_stocks — EIA weekly petroleum stocks fetcher.

Phase 1 Step 2 of Sharpe 1.5+ substrate inventory work. Fetches weekly
petroleum stock levels from EIA public archives — free, no API key
required.

Data series (covers 4 of our 22 deployed commodities):
  WCESTUS1  : U.S. Ending Stocks of Crude Oil (excluding SPR)
  WGTSTUS1  : U.S. Total Gasoline Ending Stocks
  WDISTUS1  : U.S. Distillate Fuel Oil Ending Stocks (heating oil proxy)
  NW2_EPG0_SAO_R48_MMcf : Working Gas in Underground Storage (nat gas)

Signal use case (Pindyck 2001 storage theory):
  storage_deficit_i,t = (seasonal_baseline_5yr - current_stock) / seasonal_baseline_5yr
  Long high-deficit (low storage relative to seasonal expectation) →
  high convenience yield → expected positive next-period return.
"""
from __future__ import annotations

import io
import logging
import time
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


_EIA_BASE = "https://www.eia.gov/dnav/pet/hist_xls"
_NG_BASE  = "https://www.eia.gov/dnav/ng/hist_xls"

# Petroleum series → (URL filename, deployed commodity symbol)
PETROLEUM_SERIES = {
    "crude_oil":  ("WCESTUS1w.xls",  "CL_WTI"),    # also covers BRN_Brent (shared storage signal)
    "gasoline":   ("WGTSTUS1w.xls",  "RB_Gasoline"),
    "distillate": ("WDISTUS1w.xls",  "HO_HeatOil"),
}
# Natural gas uses a different URL path
NATGAS_SERIES = {
    "natural_gas": ("NW2_EPG0_SAO_R48_MMcfw.xls", "NG_NatGas"),
}


def _download(url: str, timeout: int = 60) -> bytes:
    """Fetch URL, return raw bytes. Retry once on transient failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "macroalpha/1.0"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            if attempt == 0:
                logger.warning("EIA fetch %s failed (%s), retrying...",
                                 url, exc)
                time.sleep(2)
            else:
                raise


def parse_eia_xls(blob: bytes, value_col_keyword: str = "Ending Stocks"
                    ) -> pd.DataFrame:
    """Parse an EIA weekly XLS export.

    The structure: Sheet 1 has metadata; Sheet 2 ("Data 1") has weekly
    Date + value series. We always read the second sheet by index.
    """
    xl = pd.ExcelFile(io.BytesIO(blob))
    # Standard EIA pattern: sheet index 1 = "Data 1" with header on row 3
    df = pd.read_excel(xl, sheet_name=1, skiprows=2)
    df.columns = [str(c).strip() for c in df.columns]
    # Date column usually called "Date"; value column has long name
    date_col = next((c for c in df.columns if c.lower() == "date"),
                     df.columns[0])
    val_col = None
    for c in df.columns:
        if c != date_col:
            val_col = c
            break
    if val_col is None:
        raise ValueError(f"no value column found in EIA XLS, "
                          f"got columns: {df.columns.tolist()}")
    df = df[[date_col, val_col]].dropna()
    df = df.rename(columns={date_col: "date", val_col: "value"})
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def fetch_petroleum_stocks() -> dict[str, pd.Series]:
    """Returns dict[deployed_sym → weekly stock Series].

    Three petroleum commodities — crude oil, gasoline, distillate.
    """
    out: dict[str, pd.Series] = {}
    for kind, (fname, sym) in PETROLEUM_SERIES.items():
        url = f"{_EIA_BASE}/{fname}"
        logger.info("EIA %s: %s", kind, url)
        blob = _download(url)
        df = parse_eia_xls(blob)
        s = df.set_index("date")["value"].rename(sym).sort_index()
        out[sym] = s
        logger.info("  %s: %d rows, %s → %s",
                     sym, len(s), s.index.min().date(), s.index.max().date())
    return out


def fetch_natgas_storage() -> Optional[pd.Series]:
    """Returns NG_NatGas weekly working gas storage (MMcf).

    Returns None if EIA URL changes / not reachable.
    """
    fname, sym = NATGAS_SERIES["natural_gas"]
    url = f"{_NG_BASE}/{fname}"
    logger.info("EIA natural_gas: %s", url)
    try:
        blob = _download(url)
        df = parse_eia_xls(blob)
        s = df.set_index("date")["value"].rename(sym).sort_index()
        logger.info("  %s: %d rows, %s → %s",
                     sym, len(s), s.index.min().date(), s.index.max().date())
        return s
    except Exception as exc:
        logger.error("EIA natural_gas fetch failed: %s", exc)
        return None


def compute_storage_deficit_signal(stock_series: pd.Series,
                                      baseline_years: int = 5,
                                      ) -> pd.Series:
    """Storage deficit signal per Pindyck 2001 storage theory.

    For each week:
        seasonal_baseline = mean of same week-of-year across prior
                             `baseline_years` years
        deficit = (baseline - current) / baseline

    Positive deficit = stock below seasonal expectation = high
    convenience yield → expected positive return signal.

    Requires `baseline_years × 52` observations to start producing
    non-NaN signals.
    """
    s = stock_series.dropna().copy()
    s = s.sort_index()
    s.index = pd.to_datetime(s.index)
    # Week-of-year handling: use ISO calendar week
    woy = s.index.isocalendar().week
    s_df = pd.DataFrame({"value": s.values, "woy": woy.values}, index=s.index)
    # For each row, find prior 5-year same-week average
    baselines = pd.Series(index=s.index, dtype=float)
    for i, t in enumerate(s.index):
        w = s_df.loc[t, "woy"]
        prior_window_start = t - pd.DateOffset(years=baseline_years)
        prior_mask = (s_df.index >= prior_window_start) & (s_df.index < t)
        prior = s_df.loc[prior_mask]
        same_week = prior[prior["woy"] == w]["value"]
        if len(same_week) >= 3:
            baselines.iloc[i] = same_week.mean()
    deficit = (baselines - s) / baselines
    return deficit.dropna().rename(stock_series.name + "_deficit")
