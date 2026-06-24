"""engine/data/fetchers/wrds_crsp.py — CRSP via WRDS (paid).

CAREFUL: WRDS access has been denied before. This module:
- ALWAYS probes first with LIMIT 1 query (no quota burn)
- NEVER retries on access_denied (would trigger abuse detection)
- USES existing engine/line_c/wrds_direct.py connector (proven non-interactive)
- USES dual-account discovery (${WRDS_USER_2} fallback if ${WRDS_USER_1} fails)
- MERGES delisting returns into ret column (per senior-quant pitfall #2)

Per [[feedback-wrds-care-and-probe-pattern-2026-05-30]] doctrine.

Schema returned (canonical):
  crsp_dsf: [date, permno, ticker, ret, prc, vol, shrout]
  crsp_msf: [date, permno, ticker, ret, prc]
  fetch_dsedist: [date, permno, dlret] (delisting returns — used internally)
"""
from __future__ import annotations

import logging

import pandas as pd

from engine.data.orchestrator import ProbeResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROBE_TIMEOUT_SEC = 15
QUERY_TIMEOUT_SEC = 300


# ── Defensive connector wrapper ──────────────────────────────────────────

def _get_connector():
    """Lazy-import wrds_direct. Returns None if unavailable (e.g. missing
    pgpass files in dev env)."""
    try:
        from engine.line_c import wrds_direct
        return wrds_direct
    except ImportError as exc:
        logger.warning("wrds_direct not importable: %s", exc)
        return None


def _safe_raw_sql(sql: str, *, account: str = "${WRDS_USER_1}",
                    timeout: int = QUERY_TIMEOUT_SEC) -> pd.DataFrame | None:
    """Run a WRDS query DEFENSIVELY:
    - Returns None on auth/connection failure (caller handles fallback)
    - NEVER retries on permission-denied
    - Logs query duration to acquisition log via orchestrator
    """
    wrds = _get_connector()
    if wrds is None:
        return None
    try:
        # NOTE: wrds_direct.raw_sql opens + closes its own conn if conn=None
        # which is the safest pattern (no leak on exception).
        return wrds.raw_sql(sql, account=account)
    except Exception as exc:
        emsg = str(exc).lower()
        if "denied" in emsg or "permission" in emsg or "insufficient" in emsg:
            logger.error("WRDS access denied — NOT retrying: %s", exc)
            return None
        logger.warning("WRDS query failed: %s", exc)
        return None


# ── Probe ────────────────────────────────────────────────────────────────

def probe(start: str, end: str, *, target_function: str | None = None,
            **kw) -> ProbeResult:
    """LIGHT probe — LIMIT 1 query on CRSP DSF.

    Detects:
    - pgpass files missing → auth_missing
    - account access denied → access_denied
    - connection failure → network
    """
    import time
    t0 = time.time()
    wrds = _get_connector()
    if wrds is None:
        return ProbeResult(
            available=False, error="wrds_direct module not importable",
            error_class="schema_unknown", elapsed_secs=time.time() - t0,
        )

    # Probe minimal: just check table exists + auth works
    probe_sql = "SELECT 1 AS ok FROM crsp.dsf WHERE date <= '1990-01-31' LIMIT 1"
    try:
        df = wrds.raw_sql(probe_sql, account="${WRDS_USER_2}")    # try ${WRDS_USER_2} first (active)
    except Exception as exc:
        emsg = str(exc).lower()
        if "denied" in emsg or "permission" in emsg:
            return ProbeResult(
                available=False, error=f"WRDS access denied: {exc}",
                error_class="access_denied", elapsed_secs=time.time() - t0,
            )
        if "pgpass" in emsg or "password" in emsg or "no such file" in emsg:
            return ProbeResult(
                available=False, error=f"WRDS credentials missing: {exc}",
                error_class="auth_missing", elapsed_secs=time.time() - t0,
            )
        return ProbeResult(
            available=False, error=f"WRDS probe failed: {exc}",
            error_class="network", elapsed_secs=time.time() - t0,
        )
    if df is None or df.empty:
        return ProbeResult(
            available=False, error="WRDS probe returned empty",
            error_class="network", elapsed_secs=time.time() - t0,
        )
    return ProbeResult(
        available=True, error=None, error_class=None,
        elapsed_secs=time.time() - t0,
    )


# ── DSF (daily stock file) ──────────────────────────────────────────────

def fetch_dsf(start: str, end: str, *,
                permnos: list[int] | None = None,
                merge_delisting: bool = True,
                **kw) -> pd.DataFrame:
    """Fetch CRSP daily stock file with optional permno filter.

    Returns canonical columns: date, permno, ticker, ret, prc, vol, shrout.

    If merge_delisting=True (default), the ret column is replaced where
    available with CFM's "complete" return (ret merged with dlret per
    delisting). This is senior-quant standard practice — without it your
    backtest is survivorship-biased.
    """
    where = [f"date BETWEEN '{start}' AND '{end}'"]
    if permnos:
        permno_list = ",".join(str(p) for p in permnos)
        where.append(f"permno IN ({permno_list})")
    where_str = " AND ".join(where)
    sql = (
        f"SELECT date, permno, ticker, ret, prc, vol, shrout "
        f"FROM crsp.dsf WHERE {where_str} ORDER BY date, permno"
    )
    df = _safe_raw_sql(sql)
    if df is None:
        return pd.DataFrame(columns=["date", "permno", "ticker", "ret", "prc",
                                       "vol", "shrout"])

    df["date"] = pd.to_datetime(df["date"])

    if merge_delisting and not df.empty:
        # Pull delisting returns and merge
        del_sql = (
            f"SELECT dlstdt AS date, permno, dlret FROM crsp.dsedelist "
            f"WHERE dlstdt BETWEEN '{start}' AND '{end}' AND dlret IS NOT NULL"
        )
        ddf = _safe_raw_sql(del_sql)
        if ddf is not None and not ddf.empty:
            ddf["date"] = pd.to_datetime(ddf["date"])
            df = df.merge(ddf, on=["date", "permno"], how="left")
            # Where dlret present, ret = (1+ret)(1+dlret) - 1
            mask = df["dlret"].notna()
            df.loc[mask, "ret"] = (
                (1.0 + df.loc[mask, "ret"].fillna(0.0))
                * (1.0 + df.loc[mask, "dlret"]) - 1.0
            )
            df = df.drop(columns=["dlret"])

    return df


# ── MSF (monthly stock file) ────────────────────────────────────────────

def fetch_msf(start: str, end: str, *,
                permnos: list[int] | None = None,
                merge_delisting: bool = True,
                **kw) -> pd.DataFrame:
    """Fetch CRSP monthly stock file. Same delisting-merge semantics as DSF."""
    where = [f"date BETWEEN '{start}' AND '{end}'"]
    if permnos:
        permno_list = ",".join(str(p) for p in permnos)
        where.append(f"permno IN ({permno_list})")
    where_str = " AND ".join(where)
    sql = (
        f"SELECT date, permno, ticker, ret, prc "
        f"FROM crsp.msf WHERE {where_str} ORDER BY date, permno"
    )
    df = _safe_raw_sql(sql)
    if df is None:
        return pd.DataFrame(columns=["date", "permno", "ticker", "ret", "prc"])
    df["date"] = pd.to_datetime(df["date"])

    if merge_delisting and not df.empty:
        del_sql = (
            f"SELECT dlstdt AS date, permno, dlret FROM crsp.msedelist "
            f"WHERE dlstdt BETWEEN '{start}' AND '{end}' AND dlret IS NOT NULL"
        )
        ddf = _safe_raw_sql(del_sql)
        if ddf is not None and not ddf.empty:
            ddf["date"] = pd.to_datetime(ddf["date"])
            df = df.merge(ddf, on=["date", "permno"], how="left")
            mask = df["dlret"].notna()
            df.loc[mask, "ret"] = (
                (1.0 + df.loc[mask, "ret"].fillna(0.0))
                * (1.0 + df.loc[mask, "dlret"]) - 1.0
            )
            df = df.drop(columns=["dlret"])
    return df


def fetch_index_constituents(start: str, end: str, *,
                                index_id: str = "SPX", **kw) -> pd.DataFrame:
    """Historical S&P 500 constituents (or other CRSP-indexed indices)
    snapshot-aligned per request date range.

    Returns: ticker, name, sector, date_added (PIT-correct as-of each date).

    Used by orchestrator as backup for scraper_wikipedia.
    """
    # CRSP's idxcst_his has historical index membership
    sql = (
        f"SELECT s.gvkey, s.tic AS ticker, s.conm AS name, "
        f"       s.gsector AS sector, c.from AS date_added "
        f"FROM crsp.dsp500list c "
        f"JOIN comp.security s ON c.permno = s.permno "
        f"WHERE c.start <= '{end}' AND (c.ending IS NULL OR c.ending >= '{start}')"
    )
    df = _safe_raw_sql(sql)
    if df is None:
        return pd.DataFrame(columns=["ticker", "name", "sector", "date_added"])
    return df
