"""scripts/fetch_crsp_msf_long_history.py — pull CRSP monthly stock
file 1990-2024 for LTR (needs longer history than our 2013-2024
daily cache provides).

CRSP msf schema (monthly):
  permno, date, prc, ret, shrout, vol

Filter: NYSE/AMEX/NASDAQ common stocks (shrcd 10 or 11), exclude micro
caps (prc >= $5) per De Bondt-Thaler convention.
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
logger = logging.getLogger("crsp_msf")

OUT = Path("data/cache/_crsp_msf_long_history.parquet")
_WRDS_USER = "${WRDS_USER_2}"
START_DATE = "1990-01-01"
END_DATE = "2024-12-31"


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
    print(f" Fetch CRSP monthly stock file {START_DATE} to {END_DATE}")
    print("=" * 80)
    conn = _connect()

    # Probe row count
    print(f"\n[probe row count]")
    q_probe = f"""
    SELECT COUNT(*) AS n FROM crsp.msf
    WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    t0 = time.time()
    n = int(conn.raw_sql(q_probe).iloc[0, 0])
    print(f"  total rows: {n:,} ({time.time()-t0:.1f}s)")

    # Join to msenames for shrcd filter (common stock only)
    print(f"\n[fetch monthly returns + identifiers]")
    q = f"""
    SELECT m.permno, m.date, m.prc, m.ret, m.shrout
    FROM crsp.msf m
    JOIN crsp.msenames n ON m.permno = n.permno
      AND n.namedt <= m.date AND (n.nameendt >= m.date OR n.nameendt IS NULL)
    WHERE m.date BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND n.shrcd IN (10, 11)
      AND n.exchcd IN (1, 2, 3)
      AND m.ret IS NOT NULL
    """
    t0 = time.time()
    df = conn.raw_sql(q)
    logger.info(f"  fetched: {len(df):,} rows ({time.time()-t0:.1f}s)")

    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype(int)
    # Market cap proxy = |prc| × shrout (CRSP prc can be negative for bid-ask spread)
    df["mktcap"] = df["prc"].abs() * df["shrout"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(f"\n[saved] {OUT}")
    print(f"  rows: {len(df):,}")
    print(f"  date range: {df.date.min().date()} → {df.date.max().date()}")
    print(f"  unique permnos: {df.permno.nunique():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
