"""scripts/fetch_optionm_spx_iv.py — fetch OptionMetrics IV surface for SPX.

Uses ${WRDS_USER_2} account (has FULL optionm + optionm_all access — confirmed
2026-06-14 LIVE probe; the historical comment saying ${WRDS_USER_2} was IBES-only
is OUTDATED).

Unlocks signals:
  - SPX put-skew (25-delta put IV - 25-delta call IV) — TAIL_RISK family
  - SPX term slope (90d IV - 30d IV) — VRP term structure
  - SPX IV innovation (today IV - 5d EMA IV) — VOL momentum
  - SPX put-call ratio proxy via opvold — SENTIMENT family

Strategy
========
1. Find SPX secid (108105 typically, but verify via optionm.indexd).
2. Fetch vsurfd yearly tables 2000-2024 for SPX only, days IN (30,90,365),
   delta IN (-25, -50, 25, 50). About ~6300 rows per year × 25 years
   = 158k rows. Tiny.
3. Save as data/cache/_spx_iv_surface_daily.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import time
import psycopg2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB   = "wrds"
WRDS_USER = "${WRDS_USER_2}"

OUT_PATH = REPO_ROOT / "data" / "cache" / "_spx_iv_surface_daily.parquet"


def _load_secret_pwd() -> str:
    pgp = Path.home() / "AppData" / "Roaming" / "postgresql" / "pgpass.conf"
    line = pgp.read_text().strip().splitlines()[0]
    return line.rsplit(":", 1)[-1]


def main() -> int:
    pwd = _load_secret_pwd()
    conn = psycopg2.connect(
        host=WRDS_HOST, port=WRDS_PORT, dbname=WRDS_DB,
        user=WRDS_USER, password=pwd, connect_timeout=60,
    )
    print(f"[wrds] connected as {WRDS_USER}")

    # Find SPX secid
    spx_secid = pd.read_sql(
        "SELECT secid FROM optionm.indexd WHERE ticker='SPX'", conn
    )
    if spx_secid.empty:
        print("[err] SPX secid not found in optionm.indexd")
        return 1
    secid = int(spx_secid.iloc[0]["secid"])
    print(f"[secid] SPX = {secid}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    t_total = time.time()

    for year in range(2000, 2025):
        sql = f"""
        SELECT date, secid, days, delta, cp_flag,
               impl_volatility, impl_strike, impl_premium, dispersion
        FROM optionm.vsurfd{year}
        WHERE secid = {secid}
          AND days IN (30, 60, 91, 122, 152, 182, 273, 365, 547, 730)
          AND delta IN (-25, -50, -75, 25, 50, 75)
        ORDER BY date, days, delta
        """
        try:
            df = pd.read_sql(sql, conn)
        except Exception as exc:
            print(f"[{year}] FAIL: {str(exc)[:200]}")
            continue
        if df.empty:
            print(f"[{year}] (empty)")
            continue
        df["date"] = pd.to_datetime(df["date"])
        total_rows += len(df)
        print(f"[{year}] {len(df):>6,} rows  total {total_rows:,}")
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(OUT_PATH), table.schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()
    conn.close()

    if OUT_PATH.exists():
        size_mb = OUT_PATH.stat().st_size / 1024 / 1024
        print(f"\n[done] {OUT_PATH.name} {size_mb:.1f} MB, "
                f"{total_rows:,} rows, elapsed {time.time()-t_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
