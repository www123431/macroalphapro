"""scripts/fetch_ibes_revisions_us.py — fetch IBES analyst EPS revision
data for US equities via ${WRDS_USER_2}.

Unlocks REVISION / SENTIMENT family templates:
  - Mean estimate revision signal (Chan-Jegadeesh-Lakonishok 1996)
  - Revision dispersion signal (Diether-Malloy-Scherbina 2002)
  - Up/down revision ratio signal (Womack 1996)

Data source
===========
  ibes.statsumu_epsus — monthly summary of analyst EPS estimates per
  firm-fiscal-period:
    ticker / cusip / statpers (statistic period date) / measure (EPS) /
    fiscalp (Q1/Q2/Q3/Q4/A) / fpi (1=FY1, 2=FY2 etc) / numest /
    numup / numdn (?) / meanest / medest / stdev

We fetch 1990-2024 with fpi=1 (FY1 = current fiscal year) since most
revision literature focuses on near-horizon estimates.

Estimated size: ~50-80MB parquet (1990-2024 monthly × ~10k tickers ×
FY1 only).
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
WRDS_USER = "${WRDS_USER_2}"
OUT_PATH = REPO_ROOT / "data" / "cache" / "_ibes_eps_summary_us_fy1.parquet"


def _pwd() -> str:
    pgp = Path.home() / "AppData" / "Roaming" / "postgresql" / "pgpass.conf"
    line = pgp.read_text().strip().splitlines()[0]
    return line.rsplit(":", 1)[-1]


def main() -> int:
    conn = psycopg2.connect(host=WRDS_HOST, port=WRDS_PORT, dbname="wrds",
                              user=WRDS_USER, password=_pwd(),
                              connect_timeout=60)
    print(f"[wrds] connected as {WRDS_USER}", flush=True)

    # First: probe what columns actually exist (some IBES tables vary)
    probe = pd.read_sql(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='ibes' AND table_name='statsumu_epsus' "
        "ORDER BY ordinal_position",
        conn,
    )
    cols = probe["column_name"].tolist()
    print(f"[probe] statsumu_epsus columns: {cols}", flush=True)

    # Pick the columns we know exist
    select_cols = ["ticker", "cusip", "statpers", "measure", "fiscalp",
                    "fpi", "numest", "meanest", "medest", "stdev"]
    select_cols = [c for c in select_cols if c in cols]
    # numup is sometimes there
    for opt in ("numup", "numdown", "highest", "lowest"):
        if opt in cols:
            select_cols.append(opt)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    writer: pq.ParquetWriter | None = None
    total = 0
    t0 = time.time()
    for year in range(1990, 2025):
        sql = f"""
        SELECT {', '.join(select_cols)}
        FROM ibes.statsumu_epsus
        WHERE statpers BETWEEN '{year}-01-01' AND '{year}-12-31'
          AND fpi = '1'
          AND measure = 'EPS'
        ORDER BY ticker, statpers
        """
        try:
            df = pd.read_sql(sql, conn)
        except Exception as exc:
            print(f"[{year}] FAIL: {str(exc)[:200]}", flush=True)
            continue
        if df.empty:
            print(f"[{year}] (empty)", flush=True)
            continue
        df["statpers"] = pd.to_datetime(df["statpers"])
        total += len(df)
        print(f"[{year}] {len(df):>7,} rows  total {total:,}", flush=True)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(OUT_PATH), tbl.schema,
                                         compression="snappy")
        writer.write_table(tbl)

    if writer is not None:
        writer.close()
    conn.close()
    if OUT_PATH.exists():
        size_mb = OUT_PATH.stat().st_size / 1024 / 1024
        print(f"\n[done] {OUT_PATH.name} {size_mb:.1f} MB, "
                f"{total:,} rows, elapsed {time.time()-t0:.1f}s",
                flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
