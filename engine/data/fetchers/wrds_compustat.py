"""engine/data/fetchers/wrds_compustat.py — Compustat fundamentals via WRDS.

Provides PIT-correct funda (annual) + fundq (quarterly) fetchers for the
BARRA-equivalent risk model Phase 2 (HML / QMJ style factors per Asness-
Frazzini-Pedersen 2019) and any fundamental-anomaly mechanism (PEAD,
revision, quality, etc.).

PIT semantics per [[feedback-wrds-care-and-probe-pattern-2026-05-30]] +
senior-quant pitfall list:
  - Filter to indfmt='INDL' AND consol='C' AND popsrc='D' AND datafmt='STD'
    (this is the standardized Industrial-Consolidated-Domestic snapshot
    everyone uses; mixing financial-services format will break ratios).
  - datadate is FISCAL period end (NOT when the data became known).
    For backtest pinning, MUST shift forward by the public-availability
    lag (~90 days post-quarter end for fundq, ~120 days for funda).
    The fetchers DO NOT do this automatically — caller must apply lag.
  - Quarterly: use fundq + rdq for the actual public-release date.

Probe-first pattern per dual-account WRDS workflow (${WRDS_USER_2} active).

Canonical output columns:
  fetch_funda: gvkey, datadate, fyear, tic, at, ceq, lt, ni, oibdp,
               sale, cogs, xsga, xrd, ppent, dlc, dltt, dvc
  fetch_fundq: gvkey, datadate, fyearq, fqtr, tic, rdq, atq, ceqq, niq,
               oibdpq, saleq, cogsq, xsgaq
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from engine.data.orchestrator import ProbeResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROBE_TIMEOUT_SEC = 15
QUERY_TIMEOUT_SEC = 600   # Compustat funda is big — bigger budget

# Standard PIT filter — every Compustat query should include these.
PIT_FILTER = (
    "indfmt='INDL' AND consol='C' AND popsrc='D' AND datafmt='STD'"
)

# Default annual columns (the fundamental risk-model + anomaly subset).
DEFAULT_FUNDA_COLS = [
    "gvkey", "datadate", "fyear", "tic",
    "at",     # total assets
    "ceq",    # common equity (book value)
    "lt",     # total liabilities
    "ni",     # net income
    "oibdp",  # operating income before D&A
    "sale",   # net sales
    "cogs",   # cost of goods sold
    "xsga",   # SG&A
    "xrd",    # R&D
    "ppent",  # net PP&E
    "dlc",    # short-term debt
    "dltt",   # long-term debt
    "dvc",    # cash dividends common
]

DEFAULT_FUNDQ_COLS = [
    "gvkey", "datadate", "fyearq", "fqtr", "tic",
    "rdq",     # actual public release date
    "atq",
    "ceqq",
    "niq",
    "oibdpq",
    "saleq",
    "cogsq",
    "xsgaq",
]


# -- Connector helpers (mirror wrds_crsp pattern) ------------------------

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
            logger.error("WRDS Compustat access denied — NOT retrying: %s", exc)
            return None
        logger.warning("Compustat query failed: %s", exc)
        return None


# -- Probe ---------------------------------------------------------------

def probe(start: str = "2000-01-01", end: str = "2000-12-31",
              *, target_function: str | None = None, **kw) -> ProbeResult:
    """LIGHT probe — LIMIT 1 on comp.funda.

    Returns (available, error, error_class, elapsed) via ProbeResult.
    """
    t0 = time.time()
    wrds = _get_connector()
    if wrds is None:
        return ProbeResult(available=False,
                              error="wrds_direct module not importable",
                              error_class="schema_unknown",
                              elapsed_secs=time.time() - t0)
    probe_sql = (
        f"SELECT 1 AS ok FROM comp.funda "
        f"WHERE datadate BETWEEN '{start}' AND '{end}' AND {PIT_FILTER} LIMIT 1"
    )
    try:
        df = wrds.raw_sql(probe_sql, account="${WRDS_USER_2}")
    except Exception as exc:
        emsg = str(exc).lower()
        if "denied" in emsg or "permission" in emsg:
            return ProbeResult(available=False,
                                  error=f"WRDS access denied: {exc}",
                                  error_class="access_denied",
                                  elapsed_secs=time.time() - t0)
        if "pgpass" in emsg or "password" in emsg:
            return ProbeResult(available=False,
                                  error=f"WRDS credentials missing: {exc}",
                                  error_class="auth_missing",
                                  elapsed_secs=time.time() - t0)
        return ProbeResult(available=False,
                              error=f"Compustat probe failed: {exc}",
                              error_class="network",
                              elapsed_secs=time.time() - t0)
    if df is None or df.empty:
        return ProbeResult(available=False,
                              error="Compustat probe returned empty",
                              error_class="network",
                              elapsed_secs=time.time() - t0)
    return ProbeResult(available=True, error=None, error_class=None,
                          elapsed_secs=time.time() - t0)


# -- Funda (annual) -----------------------------------------------------

def fetch_funda(start: str, end: str, *,
                    gvkeys: list[str] | None = None,
                    cols: list[str] | None = None,
                    **kw) -> pd.DataFrame:
    """Fetch Compustat annual fundamentals (comp.funda).

    Args:
      start, end: ISO dates filtered on datadate (fiscal year end).
      gvkeys: optional list of Compustat firm keys (faster).
      cols: optional column subset; defaults to DEFAULT_FUNDA_COLS.

    Returns: DataFrame with PIT filter applied. CALLER must shift datadate
    forward by the public-availability lag (~120 days) before joining to
    a backtest panel — this fetcher returns RAW Compustat semantics.
    """
    col_list = ",".join(cols or DEFAULT_FUNDA_COLS)
    where = [f"datadate BETWEEN '{start}' AND '{end}'", PIT_FILTER]
    if gvkeys:
        gv_list = ",".join(f"'{g}'" for g in gvkeys)
        where.append(f"gvkey IN ({gv_list})")
    sql = (
        f"SELECT {col_list} FROM comp.funda WHERE "
        + " AND ".join(where)
        + " ORDER BY gvkey, datadate"
    )
    df = _safe_raw_sql(sql)
    if df is None:
        return pd.DataFrame(columns=cols or DEFAULT_FUNDA_COLS)
    if "datadate" in df.columns:
        df["datadate"] = pd.to_datetime(df["datadate"])
    return df


# -- Fundq (quarterly) -------------------------------------------------

def fetch_fundq(start: str, end: str, *,
                    gvkeys: list[str] | None = None,
                    cols: list[str] | None = None,
                    **kw) -> pd.DataFrame:
    """Fetch Compustat quarterly fundamentals (comp.fundq).

    rdq column is the REPORTING DATE (when public). For backtest, filter
    on rdq <= as_of_date rather than datadate (which is fiscal end).
    """
    col_list = ",".join(cols or DEFAULT_FUNDQ_COLS)
    where = [f"datadate BETWEEN '{start}' AND '{end}'", PIT_FILTER]
    if gvkeys:
        gv_list = ",".join(f"'{g}'" for g in gvkeys)
        where.append(f"gvkey IN ({gv_list})")
    sql = (
        f"SELECT {col_list} FROM comp.fundq WHERE "
        + " AND ".join(where)
        + " ORDER BY gvkey, datadate"
    )
    df = _safe_raw_sql(sql)
    if df is None:
        return pd.DataFrame(columns=cols or DEFAULT_FUNDQ_COLS)
    for col in ("datadate", "rdq"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


# -- Derived helpers for BARRA Phase 2 ---------------------------------

def book_to_market_panel(funda: pd.DataFrame,
                              market_caps: pd.DataFrame) -> pd.DataFrame:
    """Build a panel of (gvkey, datadate, book_to_market) for HML factor
    construction.

    book_value = CEQ (common equity, $M).
    market_caps must have (date, gvkey, market_cap_usd_m) at fiscal year end.
    Joins on (gvkey, datadate ~= date with 0-day tolerance).

    Returns: (gvkey, datadate, ceq, market_cap_m, b_to_m)
    """
    if funda.empty or market_caps.empty:
        return pd.DataFrame(columns=["gvkey", "datadate", "ceq",
                                       "market_cap_m", "b_to_m"])
    f = funda[["gvkey", "datadate", "ceq"]].copy()
    f["datadate"] = pd.to_datetime(f["datadate"])
    m = market_caps.copy()
    m.columns = [c.lower() for c in m.columns]
    if "date" in m.columns:
        m = m.rename(columns={"date": "datadate"})
    m["datadate"] = pd.to_datetime(m["datadate"])
    j = f.merge(m, on=["gvkey", "datadate"], how="left")
    j["b_to_m"] = j["ceq"] / j["market_cap_m"].replace(0, pd.NA)
    return j


def roe_panel(funda: pd.DataFrame) -> pd.DataFrame:
    """Build (gvkey, datadate, roe) panel for QMJ profitability proxy.

    ROE = NI_t / avg(CEQ_t, CEQ_t-1). For simplicity, uses NI / CEQ
    (acceptable for QMJ ranking purposes; full QMJ uses 4-component avg).
    """
    if funda.empty:
        return pd.DataFrame(columns=["gvkey", "datadate", "roe"])
    f = funda[["gvkey", "datadate", "ni", "ceq"]].copy()
    f["roe"] = f["ni"] / f["ceq"].replace(0, pd.NA)
    return f[["gvkey", "datadate", "roe"]]
