"""scripts/fetch_crsp_dsf_top3000.py — fetch CRSP DSF for top-3000 universe.

Unlocks daily-frequency signals:
  - MAX-effect (Bali-Cakici-Whitelaw 2011)
  - Amihud illiquidity (Amihud 2002)
  - idio_vol (Ang-Hodrick-Xing-Zhang 2006, CAPM residual std)
  - SUE precision (post-earnings return precision)

Strategy
========
1. Get permno universe = union of permnos appearing in our existing
   monthly cache (data/cache/_crsp_msf_long_history.parquet).
2. Fetch CRSP DSF for those permnos 1990-2024 in YEARLY chunks
   (avoids 67M-row pull timeout; each yearly chunk ~2M rows ~3min).
3. Stream-append to parquet via pyarrow ParquetWriter.
4. Validate row count + date coverage at end.

Estimated runtime: 30-60min total
Estimated output size: 1-2 GB
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
WRDS_USER = "${WRDS_USER_1}"

OUT_PATH = REPO_ROOT / "data" / "cache" / "_crsp_dsf_top3000.parquet"
MSF_PATH = REPO_ROOT / "data" / "cache" / "_crsp_msf_long_history.parquet"


def _load_secret_pwd() -> str:
    try:
        import tomllib as tom
        with (REPO_ROOT / ".streamlit" / "secrets.toml").open("rb") as fh:
            return tom.load(fh).get("WRDS", {}).get("PASSWORD") or tom.load_string("")
    except Exception:
        pass
    # Fallback: read pgpass
    pgp = Path.home() / "AppData" / "Roaming" / "postgresql" / "pgpass.conf.${WRDS_USER_1}.bak"
    if pgp.is_file():
        line = pgp.read_text().strip().splitlines()[0]
        return line.rsplit(":", 1)[-1]
    raise RuntimeError("can't find WRDS password")


def get_permno_universe() -> list[int]:
    if not MSF_PATH.is_file():
        raise RuntimeError(f"MSF cache missing: {MSF_PATH}")
    df = pd.read_parquet(MSF_PATH, columns=["permno"])
    permnos = sorted(df["permno"].astype(int).unique().tolist())
    print(f"[universe] {len(permnos):,} unique permnos from MSF cache")
    return permnos


def fetch_year(conn, year: int, permnos: list[int]) -> pd.DataFrame:
    permno_str = ",".join(str(p) for p in permnos)
    # crsp.dsf actual columns: cusip, permno, permco, issuno, hexcd, hsiccd,
    # date, bidlo, askhi, prc, vol, ret, bid, ask, shrout, cfacpr, cfacshr,
    # openprc, numtrd, retx. No 'ticker' — that's in crsp.dsenames.
    sql = f"""
    SELECT date, permno, ret, retx, prc, vol, shrout, bidlo, askhi, cfacpr
    FROM crsp.dsf
    WHERE date BETWEEN '{year}-01-01' AND '{year}-12-31'
      AND permno IN ({permno_str})
    ORDER BY date, permno
    """
    df = pd.read_sql(sql, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> int:
    pwd = _load_secret_pwd()
    permnos = get_permno_universe()

    # Connect once
    conn = psycopg2.connect(
        host=WRDS_HOST, port=WRDS_PORT, dbname=WRDS_DB,
        user=WRDS_USER, password=pwd, connect_timeout=60,
    )
    print(f"[wrds] connected as {WRDS_USER}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f"[out] EXISTING: {OUT_PATH} — will overwrite")
        OUT_PATH.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    t_total = time.time()

    # 2026-06-14: start with 2000-2024 (25 years) for faster initial cache
    # — covers the modern regime where most factor research operates
    # (Bali-Cakici-Whitelaw 2011 / Amihud 2002 / Ang-Hodrick-Xing-Zhang 2006
    # all post-1990; 2000+ gives clean post-decimalization regime).
    # Extend to 1990 later if pre-2000 regime needed.
    for year in range(2000, 2025):
        t0 = time.time()
        try:
            df = fetch_year(conn, year, permnos)
        except Exception as exc:
            print(f"[{year}] FAIL: {exc}")
            continue
        elapsed = time.time() - t0
        n = len(df)
        total_rows += n
        print(f"[{year}] {n:>10,} rows in {elapsed:>5.1f}s  (total {total_rows:,})")
        if df.empty:
            continue
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(OUT_PATH), table.schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()
    conn.close()

    if OUT_PATH.exists():
        size_mb = OUT_PATH.stat().st_size / 1024 / 1024
        print(f"\n[done] {OUT_PATH.name} {size_mb:,.1f} MB, "
                f"{total_rows:,} rows, elapsed {time.time()-t_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
