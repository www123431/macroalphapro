"""engine/data/fetchers/_common.py — shared utilities for real fetchers.

Provides:
- HTTP session with retry + backoff
- Canonical UTC date conversion
- Senior-quant care: sentinel-NaN cleanup, calendar alignment
- User-Agent management (SEC fair access requires identifying UA)

Per project_senior_quant_data_pitfalls_2026-05-30, every fetcher should
use these utilities to ensure consistent data hygiene.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# SEC requires identifying User-Agent for fair access
DEFAULT_UA = "macro-alpha-research/1.0 (research@local; iterate-and-solve)"


def http_session(user_agent: str | None = None):
    """Build a requests.Session with sensible retry + UA defaults."""
    import requests
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except ImportError:
        Retry = None

    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent or DEFAULT_UA,
        "Accept-Encoding": "gzip, deflate",
    })
    if Retry is not None:
        retry = Retry(
            total=3, backoff_factor=2.0,
            status_forcelist=[500, 502, 503, 504, 429],
            allowed_methods=frozenset(["GET", "HEAD"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
    return s


def to_utc_dates(series: pd.Series) -> pd.Series:
    """Normalize a date series to UTC tz-aware Timestamps (then strip TZ for
    Parquet compatibility; preserved as UTC canonical)."""
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s.dt.tz_localize(None)


def replace_sentinel_values(df: pd.DataFrame,
                              sentinels: list[float | str] | None = None
                              ) -> pd.DataFrame:
    """NaN-ify common sentinel placeholders that some sources use for missing.

    Default sentinels: 9999.99, -9999, 9999, -1.0e30 for numeric;
    'NA', 'null', '' for string.
    """
    sentinels = sentinels if sentinels is not None else [
        9999.99, -9999.0, 9999.0, -1.0e30,
    ]
    out = df.copy()
    for col in out.select_dtypes(include="number").columns:
        for s in sentinels:
            out.loc[out[col] == s, col] = float("nan")
    return out


def get_secret(key: str) -> str | None:
    """Look up a secret from environment variable or Streamlit secrets.toml.

    Used by fetchers needing API keys (FRED, etc.) without hardcoding."""
    env_value = os.environ.get(key.upper())
    if env_value:
        return env_value
    try:
        import streamlit as st
        return st.secrets.get(key) or st.secrets.get(key.upper())
    except Exception:
        return None


def rate_limit_sleep(rate_limit_rps: float = 2.0) -> None:
    """Polite pacing between requests. Caller invokes between fetches."""
    if rate_limit_rps > 0:
        time.sleep(1.0 / rate_limit_rps)
