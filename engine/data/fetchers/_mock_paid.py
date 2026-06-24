"""engine/data/fetchers/_mock_paid.py — mock paid-tier fetcher for testing.

Demonstrates the probe()+fetch() contract. Used by orchestrator tests to
validate auto-selection / fallback / cache behavior without real WRDS access.

The TEST_STATE module global controls probe and fetch behavior so tests
can simulate availability, access-denied, and partial-coverage scenarios
without monkeypatching the fetcher directly.
"""
from __future__ import annotations

import time

import pandas as pd

from engine.data.orchestrator import ProbeResult

SCHEMA_VERSION = 1

# Test-state knobs — manipulate from tests via this module's namespace
TEST_STATE = {
    "probe_available":   True,
    "probe_error":       None,
    "probe_error_class": None,
    "fetch_raises":      None,        # Exception class or None
    "fetch_rows":        100,
    "fetch_partial":     False,        # True = return only half the date range
}


def probe(start: str, end: str, *, target_function: str | None = None,
            **kw) -> ProbeResult:
    """Mock probe — controlled by TEST_STATE."""
    t0 = time.time()
    return ProbeResult(
        available=TEST_STATE["probe_available"],
        error=TEST_STATE["probe_error"],
        error_class=TEST_STATE["probe_error_class"],
        elapsed_secs=time.time() - t0,
        estimated_rows=TEST_STATE["fetch_rows"]
        if TEST_STATE["probe_available"] else None,
    )


def fetch_dsf(start: str, end: str, **kw) -> pd.DataFrame:
    """Mock CRSP DSF fetcher — produces synthetic data."""
    if TEST_STATE["fetch_raises"]:
        raise TEST_STATE["fetch_raises"]("simulated paid fetch failure")
    end_date = end
    if TEST_STATE["fetch_partial"]:
        midpoint_ts = (pd.to_datetime(start) +
                       (pd.to_datetime(end) - pd.to_datetime(start)) / 2)
        end_date = midpoint_ts.strftime("%Y-%m-%d")
    dates = pd.date_range(start, end_date, freq="ME")
    n = TEST_STATE["fetch_rows"]
    rows = []
    for i in range(n):
        ticker = f"T{i:04d}"
        for d in dates:
            rows.append({"date": d, "permno_or_ticker": ticker,
                          "ret": 0.01, "prc": 100.0,
                          "vol": 1000, "shrout": 10000})
    return pd.DataFrame(rows)


def fetch_msf(start: str, end: str, **kw) -> pd.DataFrame:
    """Mock CRSP MSF fetcher."""
    return fetch_dsf(start, end, **kw)
