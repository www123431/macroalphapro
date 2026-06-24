"""engine/line_c/wrds_direct.py — non-interactive dual-account WRDS connector.

The `wrds` Python wrapper ALWAYS calls input() for the username inside
connect(), which EOFs in a non-interactive subprocess. We bypass it and connect
straight to the WRDS Postgres (wrds-pgdata.wharton.upenn.edu:9737/wrds) via
psycopg2, reading passwords from the pgpass files the wrds wrapper auto-created.

Two accounts (discovered 2026-05-21):
  - '${WRDS_USER_1}' — has ciq_transcripts.* (full CIQ earnings-call history 2004-2026)
                password in %APPDATA%/postgresql/pgpass.conf.${WRDS_USER_1}.bak
  - '${WRDS_USER_2}' — has I/B/E/S; only ciqsamp transcripts (20 rows). Active pgpass.
                password in %APPDATA%/postgresql/pgpass.conf

Use account='${WRDS_USER_1}' for any ciq_transcripts pull; either works for crsp/comp.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"

_PG_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "postgresql"
# account -> pgpass file holding that account's password line
_PGPASS_FILES = {
    "${WRDS_USER_2}": _PG_DIR / "pgpass.conf",
    "${WRDS_USER_1}": _PG_DIR / "pgpass.conf.${WRDS_USER_1}.bak",
}


def _read_password(account: str) -> str:
    """Read an account's password from its pgpass file (host:port:db:user:pw)."""
    candidates = []
    pf = _PGPASS_FILES.get(account)
    if pf is not None:
        candidates.append(pf)
    # also scan both files (a password line for `account` may live in either)
    candidates += [p for p in _PGPASS_FILES.values() if p not in candidates]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split(":", 4)  # split on first 4 colons; pw may contain ':'
            if len(parts) == 5 and parts[3] == account:
                return parts[4]
    raise RuntimeError(
        f"No pgpass password for account '{account}'. Looked in: "
        + ", ".join(str(p) for p in candidates)
    )


def connect(account: str = "${WRDS_USER_1}", *, connect_timeout: int = 60):
    """Open a psycopg2 connection to WRDS as `account` (non-interactive)."""
    import psycopg2

    pw = _read_password(account)
    conn = psycopg2.connect(
        host=WRDS_HOST, port=WRDS_PORT, dbname=WRDS_DB,
        user=account, password=pw, sslmode="require",
        connect_timeout=connect_timeout,
    )
    logger.info("WRDS connected (direct psycopg2) as %s", account)
    return conn


def raw_sql(
    sql: str,
    *,
    account: str = "${WRDS_USER_1}",
    params: Optional[dict] = None,
    conn=None,
) -> pd.DataFrame:
    """Run a query and return a DataFrame. Opens/closes its own conn if none given."""
    own = conn is None
    if own:
        conn = connect(account)
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        if own:
            conn.close()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Access probe: which schemas does each account actually have?
    probes = {
        "ciq_transcripts.wrds_transcript_detail": "SELECT COUNT(*) AS n FROM ciq_transcripts.wrds_transcript_detail WHERE keydeveventtypename='Earnings Calls'",
        "comp.fundq":  "SELECT COUNT(*) AS n FROM comp.fundq WHERE datadate >= '2011-01-01' AND datadate < '2011-04-01'",
        "crsp.msf":    "SELECT COUNT(*) AS n FROM crsp.msf WHERE date = '2011-03-31'",
        "crsp.dsf":    "SELECT COUNT(*) AS n FROM crsp.dsf WHERE date = '2011-03-31'",
        "crsp.ccmxpf_lnkhist": "SELECT COUNT(*) AS n FROM crsp.ccmxpf_lnkhist",
    }
    for acct in ("${WRDS_USER_1}", "${WRDS_USER_2}"):
        print(f"\n=== account {acct} ===")
        try:
            conn = connect(acct)
        except Exception as e:
            print("  CONNECT FAIL:", e); continue
        for label, q in probes.items():
            try:
                n = pd.read_sql(q, conn)["n"].iloc[0]
                print(f"  {label:42s} OK  n={int(n)}")
            except Exception as e:
                conn.rollback()
                print(f"  {label:42s} DENIED/ERR: {str(e).splitlines()[0][:70]}")
        conn.close()
