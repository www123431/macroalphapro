"""engine/data/fetchers/api_fred.py — FRED macro series.

Free with API key registration at fred.stlouisfed.org.

Senior-quant care:
- FRED data revisions: API returns latest revision by default. For PIT,
  add `realtime_start` and `realtime_end` parameters. v1 returns latest;
  v2 will support `as_of` parameter for PIT-vintage.
- Series can change over time (discontinued / renamed). Handle 404.
- Rate limit: 120 req/minute. We pace at 2 req/sec default.
"""
from __future__ import annotations

import logging

import pandas as pd

from engine.data.orchestrator import ProbeResult
from engine.data.fetchers._common import get_secret, to_utc_dates

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _fred_client():
    """Build a Fred client with API key from secrets."""
    key = get_secret("fred_api_key") or get_secret("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY not configured. Register at fred.stlouisfed.org "
            "and add to .streamlit/secrets.toml or env."
        )
    from fredapi import Fred
    return Fred(api_key=key)


def probe(start: str, end: str, *, target_function: str | None = None,
            series_id: str | None = None, **kw) -> ProbeResult:
    """Probe FRED by fetching 1 observation of a small series."""
    import time
    t0 = time.time()
    try:
        from fredapi import Fred    # check importable
    except ImportError as exc:
        return ProbeResult(
            available=False, error=f"fredapi not installed: {exc}",
            error_class="schema_unknown", elapsed_secs=time.time() - t0,
        )
    if not (get_secret("fred_api_key") or get_secret("FRED_API_KEY")):
        return ProbeResult(
            available=False, error="FRED_API_KEY not configured",
            error_class="auth_missing", elapsed_secs=time.time() - t0,
        )
    try:
        client = _fred_client()
        # Use UNRATE (a stable, long-running series) as probe target
        # Limit to 1 observation for minimal load
        s = client.get_series("UNRATE", observation_start="2020-01-01",
                                observation_end="2020-01-31")
        if s is None or len(s) == 0:
            return ProbeResult(
                available=False, error="FRED probe returned empty",
                error_class="network", elapsed_secs=time.time() - t0,
            )
    except Exception as exc:
        msg = str(exc).lower()
        ec = "auth_missing" if ("400" in msg or "key" in msg) else "network"
        return ProbeResult(
            available=False, error=f"FRED probe failed: {exc}",
            error_class=ec, elapsed_secs=time.time() - t0,
        )
    return ProbeResult(
        available=True, error=None, error_class=None,
        elapsed_secs=time.time() - t0,
    )


def fetch_series(start: str, end: str, *,
                   series_id: str = "UNRATE", **kw) -> pd.DataFrame:
    """Fetch one FRED series. Returns long format: date, series_id, value.

    Args:
      series_id: FRED series ID (e.g. UNRATE, GS10, DGS10, CPIAUCSL)
    """
    client = _fred_client()
    s = client.get_series(series_id,
                            observation_start=start,
                            observation_end=end)
    if s is None or len(s) == 0:
        return pd.DataFrame(columns=["date", "series_id", "value"])
    df = pd.DataFrame({
        "date":      pd.to_datetime(s.index),
        "series_id": series_id,
        "value":     s.values,
    })
    df["date"] = to_utc_dates(df["date"])
    return df


def fetch_series_batch(start: str, end: str, *,
                        series_ids: list[str], **kw) -> pd.DataFrame:
    """Fetch multiple FRED series and concat."""
    frames = []
    for sid in series_ids:
        try:
            df = fetch_series(start, end, series_id=sid)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("FRED %s failed: %s", sid, exc)
    if not frames:
        return pd.DataFrame(columns=["date", "series_id", "value"])
    return pd.concat(frames, ignore_index=True)
