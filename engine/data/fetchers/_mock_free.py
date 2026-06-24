"""engine/data/fetchers/_mock_free.py — mock free-tier fetcher for testing.

Lower fidelity than mock_paid; demonstrates the orchestrator's quality_caveat
flow when fallback to a free source occurs.
"""
from __future__ import annotations

import time

import pandas as pd

from engine.data.orchestrator import ProbeResult

SCHEMA_VERSION = 1

TEST_STATE = {
    "probe_available":  True,
    "probe_error":      None,
    "probe_error_class": None,
    "fetch_raises":     None,
    "fetch_rows":       50,
}


def probe(start: str, end: str, *, target_function: str | None = None,
            **kw) -> ProbeResult:
    t0 = time.time()
    return ProbeResult(
        available=TEST_STATE["probe_available"],
        error=TEST_STATE["probe_error"],
        error_class=TEST_STATE["probe_error_class"],
        elapsed_secs=time.time() - t0,
        estimated_rows=TEST_STATE["fetch_rows"]
        if TEST_STATE["probe_available"] else None,
    )


def fetch_equity_daily(start: str, end: str, **kw) -> pd.DataFrame:
    """Mock free-source equity daily. Lower row count than paid."""
    if TEST_STATE["fetch_raises"]:
        raise TEST_STATE["fetch_raises"]("simulated free fetch failure")
    dates = pd.date_range(start, end, freq="ME")
    n = TEST_STATE["fetch_rows"]
    rows = []
    for i in range(n):
        ticker = f"F{i:04d}"
        for d in dates:
            rows.append({"date": d, "permno_or_ticker": ticker,
                          "ret": 0.008, "prc": 100.0,
                          "vol": 500, "shrout": 5000})
    return pd.DataFrame(rows)


def fetch_equity_monthly(start: str, end: str, **kw) -> pd.DataFrame:
    return fetch_equity_daily(start, end, **kw)
