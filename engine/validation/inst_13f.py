"""engine/validation/inst_13f.py — 2nd-alpha: 13F institutional holdings (free, EDGAR).

I/B/E/S/SDC/OptionMetrics are permission-denied on this WRDS account
([[reference-wrds-data-access-2026-05-21]]), so the realistic frontier for a
solo+AI operator is FREE alt-data. 13F = quarterly institutional-manager holdings
(>$100M AUM, filed 45 days after quarter-end). DIFFERENT trigger than D_PEAD
(institutional positioning, not earnings), and the holdings skew LARGE-CAP -> the
signal is in TRADEABLE names (unlike insider's micro-cap wall).

Signals (Chen-Hong-Stein 2002 'breadth of ownership'; Gompers-Metrick 2001):
  - breadth = # of institutions holding the stock; Δbreadth predicts returns
    (increasing breadth -> rising demand -> outperform).
  - aggregate institutional ownership change (net institutional buying).

Data: SEC DERA Form 13F structured data sets (free; ~72MB/quarter, ~2.8M holding
rows/quarter). Aggregated to (cusip, quarter) so the cache is small.

LOOK-AHEAD: 13F for quarter-end Q is public ~45 days later. Trade quarter-Q
breadth only from ~month 2 of Q+1 (lag enforced in the sleeve builder).
Screened through alpha_factory.gate(); GREEN-only deploys.
"""
from __future__ import annotations

import io
import logging
import os
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_UA = "research ${USER_EMAIL}"
_BASE = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/%dq%d_form13f.zip"
_AGG_CACHE = "data/cache/_13f_security_quarter.parquet"


def _quarters(y0, y1):
    for y in range(y0, y1 + 1):
        for q in range(1, 5):
            yield y, q


def fetch_13f_breadth(y0: int = 2014, y1: int = 2023, force: bool = False) -> pd.DataFrame:
    """Download SEC 13F quarters, aggregate INFOTABLE to (cusip8, year, quarter):
    n_managers (breadth), tot_shares, tot_value. Cached + resumable."""
    if os.path.exists(_AGG_CACHE) and not force:
        return pd.read_parquet(_AGG_CACHE)
    parts = []
    done = set()
    if os.path.exists(_AGG_CACHE + ".partial"):
        prev = pd.read_parquet(_AGG_CACHE + ".partial")
        parts = [prev]
        done = set(zip(prev["year"], prev["quarter"]))
    for y, q in _quarters(y0, y1):
        if (y, q) in done:
            continue
        url = _BASE % (y, q)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=180).read()))
        except Exception as exc:
            logger.warning("13f %dq%d download failed: %s", y, q, exc)
            continue
        it = pd.read_csv(z.open("INFOTABLE.tsv"), sep="\t", low_memory=False,
                         usecols=["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE"])
        it = it[it["SSHPRNAMTTYPE"].astype(str).str.upper() == "SH"]
        it["cusip8"] = it["CUSIP"].astype(str).str.strip().str.slice(0, 8)
        it = it[it["cusip8"].str.len() == 8]
        it["SSHPRNAMT"] = pd.to_numeric(it["SSHPRNAMT"], errors="coerce")
        it["VALUE"] = pd.to_numeric(it["VALUE"], errors="coerce")
        g = it.groupby("cusip8").agg(
            n_managers=("ACCESSION_NUMBER", "nunique"),
            tot_shares=("SSHPRNAMT", "sum"),
            tot_value=("VALUE", "sum"),
        ).reset_index()
        g["year"] = y
        g["quarter"] = q
        parts.append(g)
        pd.concat(parts, ignore_index=True).to_parquet(_AGG_CACHE + ".partial", index=False)
        logger.info("13f %dq%d: %d securities, %d holding rows", y, q, len(g), len(it))
    agg = pd.concat(parts, ignore_index=True)
    agg.to_parquet(_AGG_CACHE, index=False)
    logger.info("13f breadth: %d security-quarters, %d cusips", len(agg), agg["cusip8"].nunique())
    return agg
