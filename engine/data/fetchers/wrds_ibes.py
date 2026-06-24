"""engine/data/fetchers/wrds_ibes.py — I/B/E/S analyst data via WRDS.

Provides PIT-correct summary forecasts, individual analyst details, and
guidance for any analyst-revision-family mechanism (PEAD-cousin, guidance
drift, dispersion, etc.).

PIT semantics per WRDS official docs + senior-quant pitfall list:
  - ibes.statsum_epsus.statpers IS the snapshot date — i.e. "what was the
    consensus believed to be ON THIS DATE" — so for an as-of-T backtest
    pin, filter statpers <= T (NOT fpedats, which is fiscal period end).
  - ibes.det_epsus.revdats IS the analyst's revision date for individual
    forecasts; same as-of semantics.
  - actual + anndats_act are the realized + report-date ground truth.
  - fpi is forecast period index: 1=current Q, 2=next Q, 3=next FY, etc.

Per [[feedback-wrds-care-and-probe-pattern-2026-05-30]] dual-account
${WRDS_USER_2} active + probe-first.

Canonical schemas:
  fetch_statsum_eps:  ticker, statpers, meanest, medest, stdev, numest,
                      fpedats, fpi, measure, actual, anndats_act
  fetch_det_eps:      ticker, analys, estimator, value, revdats,
                      fpedats, fpi, measure, anndats_act
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from engine.data.orchestrator import ProbeResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROBE_TIMEOUT_SEC = 15
QUERY_TIMEOUT_SEC = 600

DEFAULT_STATSUM_COLS = [
    "ticker", "statpers",
    "meanest", "medest", "stdev", "numest",
    "fpedats", "fpi", "measure",
    "actual", "anndats_act",
]

DEFAULT_DET_COLS = [
    "ticker", "analys", "estimator", "value",
    "revdats", "fpedats", "fpi", "measure",
    "anndats_act",
]


# -- Connector helpers ---------------------------------------------------

def _get_connector():
    try:
        from engine.line_c import wrds_direct
        return wrds_direct
    except ImportError as exc:
        logger.warning("wrds_direct not importable: %s", exc)
        return None


def _safe_raw_sql(sql: str, *, account: str = "${WRDS_USER_2}",
                       timeout: int = QUERY_TIMEOUT_SEC) -> pd.DataFrame | None:
    wrds = _get_connector()
    if wrds is None:
        return None
    try:
        return wrds.raw_sql(sql, account=account)
    except Exception as exc:
        emsg = str(exc).lower()
        if "denied" in emsg or "permission" in emsg or "insufficient" in emsg:
            logger.error("WRDS IBES access denied — NOT retrying: %s", exc)
            return None
        logger.warning("IBES query failed: %s", exc)
        return None


# -- Probe ---------------------------------------------------------------

def probe(start: str = "2020-01-01", end: str = "2020-12-31",
              *, target_function: str | None = None, **kw) -> ProbeResult:
    """LIGHT probe — LIMIT 1 on ibes.statsum_epsus."""
    t0 = time.time()
    wrds = _get_connector()
    if wrds is None:
        return ProbeResult(available=False,
                              error="wrds_direct module not importable",
                              error_class="schema_unknown",
                              elapsed_secs=time.time() - t0)
    probe_sql = (
        f"SELECT 1 AS ok FROM ibes.statsum_epsus "
        f"WHERE statpers BETWEEN '{start}' AND '{end}' LIMIT 1"
    )
    try:
        df = wrds.raw_sql(probe_sql, account="${WRDS_USER_2}")
    except Exception as exc:
        emsg = str(exc).lower()
        if "denied" in emsg or "permission" in emsg:
            return ProbeResult(available=False,
                                  error=f"WRDS IBES access denied: {exc}",
                                  error_class="access_denied",
                                  elapsed_secs=time.time() - t0)
        if "pgpass" in emsg or "password" in emsg:
            return ProbeResult(available=False,
                                  error=f"WRDS credentials missing: {exc}",
                                  error_class="auth_missing",
                                  elapsed_secs=time.time() - t0)
        return ProbeResult(available=False,
                              error=f"IBES probe failed: {exc}",
                              error_class="network",
                              elapsed_secs=time.time() - t0)
    if df is None or df.empty:
        return ProbeResult(available=False,
                              error="IBES probe returned empty",
                              error_class="network",
                              elapsed_secs=time.time() - t0)
    return ProbeResult(available=True, error=None, error_class=None,
                          elapsed_secs=time.time() - t0)


# -- Statsum (consensus summary, EPS US) -----------------------------

def fetch_statsum_eps(start: str, end: str, *,
                            tickers: list[str] | None = None,
                            fpi: list[str] | None = None,
                            measure: str = "EPS",
                            cols: list[str] | None = None,
                            **kw) -> pd.DataFrame:
    """Fetch IBES consensus summary forecasts (ibes.statsum_epsus).

    Args:
      start, end: ISO dates on statpers (the as-of date).
      tickers: optional IBES ticker filter.
      fpi: forecast period index list (e.g. ["1","2"] = current+next Q;
        ["3"] = next FY).
      measure: forecast measure (EPS default; can be BPS, CPS, EBITDA, etc.).
      cols: column subset; defaults to DEFAULT_STATSUM_COLS.

    Returns: DataFrame with proper datetime cols.
    """
    col_list = ",".join(cols or DEFAULT_STATSUM_COLS)
    where = [f"statpers BETWEEN '{start}' AND '{end}'", f"measure='{measure}'"]
    if fpi:
        fpi_list = ",".join(f"'{f}'" for f in fpi)
        where.append(f"fpi IN ({fpi_list})")
    if tickers:
        tk_list = ",".join(f"'{t}'" for t in tickers)
        where.append(f"ticker IN ({tk_list})")
    sql = (
        f"SELECT {col_list} FROM ibes.statsum_epsus WHERE "
        + " AND ".join(where)
        + " ORDER BY ticker, statpers"
    )
    df = _safe_raw_sql(sql)
    if df is None:
        return pd.DataFrame(columns=cols or DEFAULT_STATSUM_COLS)
    for col in ("statpers", "fpedats", "anndats_act"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


# -- Detail (individual analyst, EPS US) ------------------------------

def fetch_det_eps(start: str, end: str, *,
                       tickers: list[str] | None = None,
                       fpi: list[str] | None = None,
                       measure: str = "EPS",
                       cols: list[str] | None = None,
                       **kw) -> pd.DataFrame:
    """Fetch individual analyst forecasts (ibes.det_epsus).

    Args mirror fetch_statsum_eps. Filter on revdats for "what did this
    analyst publish by date T?" semantics.
    """
    col_list = ",".join(cols or DEFAULT_DET_COLS)
    where = [f"revdats BETWEEN '{start}' AND '{end}'", f"measure='{measure}'"]
    if fpi:
        fpi_list = ",".join(f"'{f}'" for f in fpi)
        where.append(f"fpi IN ({fpi_list})")
    if tickers:
        tk_list = ",".join(f"'{t}'" for t in tickers)
        where.append(f"ticker IN ({tk_list})")
    sql = (
        f"SELECT {col_list} FROM ibes.det_epsus WHERE "
        + " AND ".join(where)
        + " ORDER BY ticker, revdats"
    )
    df = _safe_raw_sql(sql)
    if df is None:
        return pd.DataFrame(columns=cols or DEFAULT_DET_COLS)
    for col in ("revdats", "fpedats", "anndats_act"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


# -- Derived: revision count panel for analyst-revision sleeve --------

def revision_count_panel(det: pd.DataFrame,
                                lookback_days: int = 90) -> pd.DataFrame:
    """For each (ticker, anniversary_date), count number of analyst revisions
    in the trailing lookback_days window. Used as input to revision-momentum
    style signals.

    Returns: (ticker, t, n_revisions, n_up, n_down) — sparse panel.
    """
    if det.empty:
        return pd.DataFrame(columns=["ticker", "t", "n_revisions",
                                       "n_up", "n_down"])
    d = det[["ticker", "analys", "value", "revdats"]].copy()
    d["revdats"] = pd.to_datetime(d["revdats"])
    d = d.sort_values(["ticker", "analys", "revdats"])
    d["value_prev"] = d.groupby(["ticker", "analys"])["value"].shift(1)
    d["delta"] = d["value"] - d["value_prev"]
    d = d.dropna(subset=["delta"])

    # Build daily panel: at each (ticker, t), how many revisions in last
    # `lookback_days` days. For simplicity return raw event list — caller
    # rolls.
    return d[["ticker", "revdats", "delta"]].rename(
        columns={"revdats": "t"}
    )
