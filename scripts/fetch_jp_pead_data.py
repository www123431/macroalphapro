"""scripts/fetch_jp_pead_data.py — pull Japan IBES quarterly EPS +
TOPIX/JPX returns for cross-country PEAD strategy.

Per pivot 2026-05-31 (UK has only 5 quarterly EPS rows; UK reports
semi-annually). Switching to Japan which has 213,759 rows.

Strategy:
  - Filter ibes.act_epsint curr_act='JPY' for quarterly EPS actuals
  - Fetch ibes.actpsum_epsint for analyst consensus (compute surprise)
  - Map ticker → security via ibes.id cusip
  - Fetch daily returns from comp.g_secd via cusip-sedol link
  - Save panel + return series for downstream PIT SN signal build
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
logger = logging.getLogger("fetch_jp_pead")

REPO_ROOT = Path(__file__).resolve().parent.parent
ACT_OUT = REPO_ROOT / "data" / "cache" / "_jp_ibes_eps_actuals.parquet"
CONS_OUT = REPO_ROOT / "data" / "cache" / "_jp_ibes_eps_consensus.parquet"

_WRDS_USER = "${WRDS_USER_2}"


def _connect():
    appdata = os.environ.get("APPDATA")
    if appdata and "PGPASSFILE" not in os.environ:
        os.environ["PGPASSFILE"] = os.path.join(
            appdata, "postgresql", "pgpass.conf",
        )
    import wrds
    return wrds.Connection(wrds_username=_WRDS_USER)


def main() -> int:
    print("=" * 80)
    print(" Fetch Japan IBES quarterly EPS (act_epsint + actpsum_epsint, JPY)")
    print("=" * 80)

    conn = _connect()

    # ── Probe ───────────────────────────────────────────────────────
    print(f"\n[probe]")
    q_probe = """
    SELECT COUNT(DISTINCT ticker) AS n_tickers, COUNT(*) AS n_obs
    FROM ibes.act_epsint
    WHERE pdicity = 'QTR' AND curr_act = 'JPY'
      AND anndats BETWEEN '2014-01-01' AND '2024-12-31'
    """
    t0 = time.time()
    pr = conn.raw_sql(q_probe).iloc[0]
    print(f"  JP quarterly EPS rows: {pr['n_obs']:,}  "
          f"unique tickers: {pr['n_tickers']:,}  ({time.time()-t0:.1f}s)")

    # ── Actuals ─────────────────────────────────────────────────────
    print(f"\n[fetch actuals]")
    q_act = """
    SELECT ticker, oftic, cname, anndats, pends, value AS eps_actual,
           curr_act
    FROM ibes.act_epsint
    WHERE pdicity = 'QTR' AND curr_act = 'JPY'
      AND anndats BETWEEN '2014-01-01' AND '2024-12-31'
      AND value IS NOT NULL
    """
    t0 = time.time()
    df_act = conn.raw_sql(q_act)
    logger.info(f"  fetched: {len(df_act):,} rows ({time.time()-t0:.1f}s)")
    ACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_act.to_parquet(ACT_OUT)
    print(f"  saved: {ACT_OUT}")
    print(f"  date range: {df_act['anndats'].min()} → {df_act['anndats'].max()}")

    # ── Consensus ───────────────────────────────────────────────────
    # IBES actpsum_epsint has consensus pre-announcement
    print(f"\n[fetch consensus pre-announcement]")
    # Need to filter for forecasts of the SAME quarter that was announced
    q_cons = """
    SELECT ticker, statpers, fpedats, fpi, meanest, medest, stdev, numest
    FROM ibes.actpsum_epsint
    WHERE fpi = '6'    -- quarter ahead
      AND statpers BETWEEN '2014-01-01' AND '2024-12-31'
      AND meanest IS NOT NULL
    """
    t0 = time.time()
    df_cons = conn.raw_sql(q_cons)
    logger.info(f"  fetched consensus: {len(df_cons):,} rows ({time.time()-t0:.1f}s)")
    df_cons.to_parquet(CONS_OUT)
    print(f"  saved: {CONS_OUT}")

    # ── Returns: cusip → comp.g_secd ────────────────────────────────
    # We need a different approach for JP returns. Let's:
    # 1. Get distinct cusips from our EPS panel
    # 2. Query comp.g_secd for those cusips' daily returns
    print(f"\n[fetch JP daily returns via cusip → g_secd]")
    cusips = df_act["ticker"].dropna().unique().tolist()[:50]  # first 50 for now
    print(f"  using sample of {len(cusips)} tickers")

    # ibes.id has cusip column for tickers
    q_link = f"""
    SELECT DISTINCT ticker, cusip FROM ibes.id
    WHERE ticker IN ({','.join(repr(t) for t in cusips)})
      AND cusip IS NOT NULL
    """
    try:
        df_link = conn.raw_sql(q_link)
        print(f"  linked {len(df_link)} tickers to CUSIPs")
        if not df_link.empty:
            cusip_list = df_link["cusip"].unique().tolist()
            cusip_sql = ",".join(repr(c) for c in cusip_list)
            q_ret = f"""
            SELECT cusip, datadate, prccd, ajexdi, trfd
            FROM comp.g_secd
            WHERE cusip IN ({cusip_sql})
              AND datadate BETWEEN '2014-01-01' AND '2024-12-31'
            LIMIT 1000
            """
            df_ret = conn.raw_sql(q_ret)
            print(f"  sample g_secd rows: {len(df_ret)}")
    except Exception as exc:
        logger.warning(f"link/returns fetch failed: {exc}")

    print(f"\n[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
