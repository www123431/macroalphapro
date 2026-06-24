"""
engine/universe_singlename/top1500_panel_loader.py — top-1500 US stocks
survivorship-bias-free weekly panel builder for Path AD (Path V re-spec).

Spec: docs/spec_path_ad_cumulative_momentum_top1500_v3_v1.md (pending)

Builds a point-in-time top-1500 by market cap universe from CRSP MSF/DSF,
then pulls weekly close prices for the union of all permnos that ever made
the top-1500 between 2014-2014 → 2023-12-31.

Cached at `data/factor_ensemble_singlename/_crsp_dsf_top1500_panel.parquet`.

Survivorship-bias-free by construction (CRSP includes delisted stocks).
"""
from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_START: str = "2014-09-12"
WINDOW_END:   str = "2023-12-29"
TOP_N:        int = 1500

# Quarter-end ranking dates for universe inclusion (rolled to nearest CRSP trading date)
QUARTER_END_TARGETS = [
    "2014-12-31", "2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31",
    "2016-03-31", "2016-06-30", "2016-09-30", "2016-12-31",
    "2017-03-31", "2017-06-30", "2017-09-30", "2017-12-31",
    "2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31",
    "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31",
    "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
    "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
    "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
    "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-29",
]


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _REPO_ROOT / "data" / "factor_ensemble_singlename"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PANEL_CACHE_PATH = _CACHE_DIR / "_crsp_dsf_top1500_panel.parquet"
_META_CACHE_PATH  = _CACHE_DIR / "_crsp_dsf_top1500_panel.meta.json"


def build_top1500_universe(conn) -> tuple[set[int], pd.DataFrame]:
    """For each quarter-end target, rank stocks by market cap; union top-1500 permnos.

    Returns:
      (set of permnos, DataFrame of quarter-end → top-1500 membership)
    """
    targets_str = "', '".join(QUARTER_END_TARGETS)
    sql = f"""
    WITH targets AS (
        SELECT unnest(ARRAY['{targets_str}']::date[]) AS target_date
    ),
    actual_dates AS (
        SELECT t.target_date,
               (SELECT MAX(m.date) FROM crsp.msf m
                WHERE m.date <= t.target_date
                  AND m.date >= t.target_date - interval '14 days') AS rank_date
        FROM targets t
    ),
    ranked AS (
        SELECT a.target_date, msf.permno, msf.date AS rank_date,
               abs(msf.prc) * msf.shrout AS mcap,
               ROW_NUMBER() OVER (
                   PARTITION BY msf.date
                   ORDER BY abs(msf.prc) * msf.shrout DESC NULLS LAST
               ) AS rk
        FROM crsp.msf msf
        INNER JOIN actual_dates a ON msf.date = a.rank_date
        WHERE abs(msf.prc) > 0 AND msf.shrout > 0
    )
    SELECT target_date, rank_date, permno, mcap, rk
    FROM ranked
    WHERE rk <= {TOP_N}
    ORDER BY target_date, rk
    """
    logger.info("Querying top-%d universe across %d quarter-ends...", TOP_N, len(QUARTER_END_TARGETS))
    t0 = time.time()
    df = conn.raw_sql(sql)
    logger.info("Universe query done in %.1fs — %d rows", time.time() - t0, len(df))

    df["permno"] = df["permno"].astype(int)
    permno_union = set(df["permno"].unique().tolist())
    return permno_union, df


def fetch_daily_prices(conn, permnos: list[int], start: str, end: str) -> pd.DataFrame:
    """Pull daily adjusted close prices from crsp.dsf for the union of permnos."""
    chunk_size = 500
    chunks: list[pd.DataFrame] = []
    permnos_sorted = sorted(permnos)
    for i in range(0, len(permnos_sorted), chunk_size):
        sub = permnos_sorted[i : i + chunk_size]
        sub_tup = tuple(sub)
        logger.info("Fetching dsf chunk %d-%d / %d permnos...",
                    i + 1, i + len(sub), len(permnos_sorted))
        t0 = time.time()
        df_chunk = conn.raw_sql(
            f"""
            SELECT permno, date, prc, ret, cfacpr
            FROM crsp.dsf
            WHERE permno IN %(permnos)s
              AND date BETWEEN %(start)s AND %(end)s
              AND abs(prc) > 0
            """,
            params={"permnos": sub_tup, "start": start, "end": end},
            date_cols=["date"],
        )
        logger.info("  chunk %d-%d done in %.1fs — %d rows",
                    i + 1, i + len(sub), time.time() - t0, len(df_chunk))
        chunks.append(df_chunk)
    df = pd.concat(chunks, ignore_index=True)
    df["prc"] = df["prc"].abs()  # negative prices = mid-quote, take absolute
    df["adj_close"] = df["prc"] / df["cfacpr"]   # split/dividend adjusted
    df["permno"] = df["permno"].astype(int)
    return df[["permno", "date", "adj_close", "ret"]]


def fetch_permno_tickers(conn, permnos: list[int]) -> pd.DataFrame:
    """Map permno → most recent ticker from msenames."""
    sub_tup = tuple(sorted(permnos))
    df = conn.raw_sql(
        """
        SELECT DISTINCT ON (permno) permno, ticker, namedt, nameendt
        FROM crsp.msenames
        WHERE permno IN %(permnos)s
        ORDER BY permno, nameendt DESC NULLS FIRST
        """,
        params={"permnos": sub_tup},
        date_cols=["namedt", "nameendt"],
    )
    df["permno"] = df["permno"].astype(int)
    return df[["permno", "ticker"]]


def build_top1500_panel(force_rebuild: bool = False) -> pd.DataFrame:
    """End-to-end build of top-1500 weekly W-FRI close panel.

    Returns:
      pandas.DataFrame indexed by W-FRI dates, columns = permnos with at least
      80% non-NaN coverage in the window. Cached at _PANEL_CACHE_PATH.
    """
    if _PANEL_CACHE_PATH.exists() and not force_rebuild:
        logger.info("Top-1500 panel cache HIT: %s", _PANEL_CACHE_PATH)
        return pd.read_parquet(_PANEL_CACHE_PATH)

    from engine.universe_singlename.crsp_loader import _open_wrds_connection
    conn = _open_wrds_connection()
    try:
        permno_set, q_membership = build_top1500_universe(conn)
        logger.info("Union top-1500 permnos: %d distinct names", len(permno_set))

        daily = fetch_daily_prices(conn, list(permno_set), WINDOW_START, WINDOW_END)
        tickers = fetch_permno_tickers(conn, list(permno_set))

        # Pivot to wide format: rows = date, columns = permno, values = adj_close
        wide = daily.pivot(index="date", columns="permno", values="adj_close").sort_index()
        wide.index = pd.to_datetime(wide.index)

        # Resample to weekly W-FRI close
        weekly = wide.resample("W-FRI").last()

        # Filter to permnos with sufficient coverage (>= 80% non-NaN)
        coverage = weekly.notna().mean(axis=0)
        keep_permnos = coverage[coverage >= 0.20].index.tolist()
        weekly_kept = weekly[keep_permnos]
        logger.info("Coverage filter: kept %d / %d permnos (>= 20%% non-NaN weekly)",
                    len(keep_permnos), len(wide.columns))

        # Save panel + ticker map + quarter membership
        weekly_kept.to_parquet(_PANEL_CACHE_PATH)
        tickers.to_parquet(_CACHE_DIR / "_crsp_top1500_tickers.parquet")
        q_membership.to_parquet(_CACHE_DIR / "_crsp_top1500_q_membership.parquet")

        import json
        meta = {
            "built_at":     datetime.datetime.utcnow().isoformat() + "Z",
            "window_start": WINDOW_START,
            "window_end":   WINDOW_END,
            "top_n":        TOP_N,
            "n_quarter_ends":   len(QUARTER_END_TARGETS),
            "n_unique_permnos": len(permno_set),
            "n_kept_permnos":   len(keep_permnos),
            "n_weeks":          int(len(weekly_kept)),
        }
        _META_CACHE_PATH.write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
        logger.info("Top-1500 panel saved: %s", _PANEL_CACHE_PATH)
        logger.info("Meta: %s", meta)

        return weekly_kept
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_top1500_panel(force_rebuild="--rebuild" in sys.argv)
    print(f"Top-1500 weekly W-FRI panel: shape={df.shape}, "
          f"idx={df.index.min()} -> {df.index.max()}")
