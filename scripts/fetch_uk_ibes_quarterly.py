"""scripts/fetch_uk_ibes_quarterly.py — pull UK quarterly EPS actuals
+ consensus from IBES International for cross-country PEAD strategy.

WRDS tables:
  ibes.act_epsint    — actual quarterly EPS reported (international)
  ibes.actpsum_epsint — analyst consensus summary at quarter
  ibes.id            — IBES identifier mapping (ticker → cusip / sedol)

Probe + range fetch:
  - Filter to UK firms via IBES id table (country code 'UK' or via SEDOL)
  - Quarterly (pdicity='QTR')
  - 2014-01 to 2024-12 to align with US PIT SN window

Output:
  data/cache/_uk_ibes_eps_actuals.parquet
  data/cache/_uk_ibes_eps_consensus.parquet
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_uk_ibes")

REPO_ROOT = Path(__file__).resolve().parent.parent
ACT_OUT = REPO_ROOT / "data" / "cache" / "_uk_ibes_eps_actuals.parquet"
CONS_OUT = REPO_ROOT / "data" / "cache" / "_uk_ibes_eps_consensus.parquet"
ID_OUT = REPO_ROOT / "data" / "cache" / "_uk_ibes_id_map.parquet"

_WRDS_USER = "${WRDS_USER_2}"


def _connect():
    import os
    appdata = os.environ.get("APPDATA")
    if appdata and "PGPASSFILE" not in os.environ:
        os.environ["PGPASSFILE"] = os.path.join(
            appdata, "postgresql", "pgpass.conf",
        )
    import wrds
    return wrds.Connection(wrds_username=_WRDS_USER)


def probe_uk_eps_via_gbp(conn) -> int:
    """Count UK quarterly EPS via curr_act='GBP' filter.
    Most UK firms report EPS in GBP — this is a robust IMPLICIT UK
    filter without needing a separate country mapping table."""
    q = """
    SELECT COUNT(DISTINCT ticker) AS n_tickers,
           COUNT(*) AS n_obs
    FROM ibes.act_epsint
    WHERE pdicity = 'QTR'
      AND curr_act = 'GBP'
      AND anndats BETWEEN '2014-01-01' AND '2024-12-31'
    """
    t0 = time.time()
    df = conn.raw_sql(q)
    logger.info(f"  probe curr_act='GBP': {df.iloc[0,1]:,} obs, "
                f"{df.iloc[0,0]:,} tickers ({time.time()-t0:.1f}s)")
    return int(df.iloc[0, 1])


def main() -> int:
    print("=" * 80)
    print(" Fetch UK IBES quarterly EPS (act_epsint + actpsum_epsint)")
    print("=" * 80)

    conn = _connect()
    print(f"\n[UK EPS rows via curr_act='GBP' (implicit UK filter)]")
    n_obs = probe_uk_eps_via_gbp(conn)

    if n_obs > 0:
        print(f"\n[fetch actuals]")
        q = """
        SELECT ticker, oftic, cname, anndats, pends, value AS eps_actual,
               curr_act
        FROM ibes.act_epsint
        WHERE pdicity = 'QTR'
          AND curr_act = 'GBP'
          AND anndats BETWEEN '2014-01-01' AND '2024-12-31'
        """
        t0 = time.time()
        df_act = conn.raw_sql(q)
        logger.info(f"  fetched actuals: {len(df_act):,} rows ({time.time()-t0:.1f}s)")
        ACT_OUT.parent.mkdir(parents=True, exist_ok=True)
        df_act.to_parquet(ACT_OUT)
        print(f"  saved: {ACT_OUT}")

    print(f"\n[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
