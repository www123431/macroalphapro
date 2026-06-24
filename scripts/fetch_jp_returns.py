"""scripts/fetch_jp_returns.py — fetch JP daily returns for IBES
tickers in our JP EPS panel.

Linking chain:
  IBES ticker -> ibes.id.cusip -> comp.g_security.cusip -> gvkey
  -> comp.g_secd daily prices

Build daily return panel for JP firms with quarterly EPS coverage.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_jp_returns")

REPO_ROOT = Path(__file__).resolve().parent.parent
JP_EPS = REPO_ROOT / "data" / "cache" / "_jp_ibes_eps_actuals.parquet"
LINK_OUT = REPO_ROOT / "data" / "cache" / "_jp_ibes_gvkey_link.parquet"
RET_OUT = REPO_ROOT / "data" / "cache" / "_jp_compg_daily_ret.parquet"

_WRDS_USER = "${WRDS_USER_2}"


def _connect():
    appdata = os.environ.get("APPDATA")
    if appdata:
        os.environ["PGPASSFILE"] = os.path.join(
            appdata, "postgresql", "pgpass.conf",
        )
    import wrds
    return wrds.Connection(wrds_username=_WRDS_USER)


def main() -> int:
    print("=" * 80)
    print(" Fetch JP daily returns: IBES → cusip → comp.g_secd")
    print("=" * 80)

    jp = pd.read_parquet(JP_EPS)
    tickers = jp["ticker"].dropna().unique().tolist()
    print(f"\n  JP IBES tickers in EPS panel: {len(tickers)}")

    conn = _connect()

    # Step 1: IBES ticker → cusip
    print(f"\n[link IBES → cusip]")
    tk_sql = ",".join(repr(t) for t in tickers)
    q1 = f"""
    SELECT DISTINCT ticker, cusip FROM ibes.id
    WHERE ticker IN ({tk_sql}) AND cusip IS NOT NULL
    """
    t0 = time.time()
    ix = conn.raw_sql(q1)
    logger.info(f"  IBES ticker → cusip: {len(ix):,} pairs ({time.time()-t0:.1f}s)")

    if ix.empty:
        print("  no cusip links found — abort")
        return 1

    # Step 2: cusip → comp.g_security
    print(f"\n[link cusip → gvkey via comp.g_security]")
    cusips = ix["cusip"].unique().tolist()
    # Compustat global stores cusip differently (8 or 9 chars); try both
    cu_sql_8 = ",".join(repr(c[:8]) for c in cusips if c)
    q2 = f"""
    SELECT gvkey, cusip, isin, tic, exchg
    FROM comp.g_security
    WHERE cusip IN ({cu_sql_8})
    """
    t0 = time.time()
    try:
        gx = conn.raw_sql(q2)
        logger.info(f"  cusip → gvkey: {len(gx):,} rows ({time.time()-t0:.1f}s)")
    except Exception as exc:
        logger.warning(f"  cusip query failed: {exc}")
        gx = pd.DataFrame()

    # Step 3: build full link
    if not gx.empty:
        link = ix.merge(gx, on="cusip", how="inner")
        print(f"  full link IBES ticker → gvkey: {len(link):,}")
        link.to_parquet(LINK_OUT)
        print(f"  saved link: {LINK_OUT}")

        # Step 4: pull comp.g_secd daily prices for gvkeys
        print(f"\n[fetch comp.g_secd daily prices 2014-2024]")
        gvkeys = link["gvkey"].unique().tolist()
        print(f"  unique gvkeys: {len(gvkeys)}")

        # chunked fetch to avoid massive query
        chunk_size = 200
        all_dfs = []
        for i in range(0, len(gvkeys), chunk_size):
            chunk = gvkeys[i:i+chunk_size]
            gv_sql = ",".join(repr(g) for g in chunk)
            q3 = f"""
            SELECT gvkey, datadate, prccd, ajexdi, trfd
            FROM comp.g_secd
            WHERE gvkey IN ({gv_sql})
              AND datadate BETWEEN '2014-01-01' AND '2024-12-31'
            """
            t0 = time.time()
            try:
                df = conn.raw_sql(q3)
                all_dfs.append(df)
                logger.info(f"  chunk {i//chunk_size+1}/{(len(gvkeys)+chunk_size-1)//chunk_size}: "
                            f"{len(df):,} rows ({time.time()-t0:.1f}s)")
            except Exception as exc:
                logger.warning(f"  chunk {i} failed: {exc}")

        if all_dfs:
            secd = pd.concat(all_dfs, ignore_index=True)
            print(f"\n  total daily price rows: {len(secd):,}")
            secd.to_parquet(RET_OUT)
            print(f"  saved: {RET_OUT}")
    else:
        print(f"  no gvkey linkage found")
        return 1

    print(f"\n[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
