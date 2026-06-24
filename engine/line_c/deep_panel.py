"""engine/line_c/deep_panel.py — extend PIT top-1500 universe + SUE panel to 2011-2024.

Reuses the LOCKED SUE math + SQL from engine.path_c.pead_ts_signal_panel
(Bernard-Thomas 1989 time-series SUE), but drives every query through ONE
reused ${WRDS_USER_1} psycopg2 connection (engine.line_c.wrds_direct) so we never hit
the interactive wrds wrapper and never hammer auth (WRDS locks accounts on
repeated failed logins — open one connection, reuse it, close once).

Outputs (data/line_c/):
  _universe_top1500_2011_2024.parquet   permno, ticker  (PIT union membership)
  _q_membership_2011_2024.parquet       target_date, permno, mcap, rk
  _sue_panel_2011_2024.parquet          permno, ticker, gvkey, rdq, sue, ...
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.line_c import wrds_direct
from engine.path_c.pead_ts_signal_panel import (
    SEASONAL_LAG_QUARTERS, SIGMA_WINDOW_QUARTERS, SIGMA_MIN_PERIODS,
    SIGMA_MIN_VALUE, SUE_WINSORIZE_LOW, SUE_WINSORIZE_HIGH,
)

logger = logging.getLogger(__name__)

CACHE = Path("data/line_c")
CACHE.mkdir(parents=True, exist_ok=True)
TOP_N = 1500

WINDOW_START = datetime.date(2011, 1, 1)
WINDOW_END   = datetime.date(2024, 6, 30)   # CRSP daily cache currently ends 2024-06


def _quarter_end_targets(start="2011-03-31", end="2024-06-30") -> list[str]:
    qs = pd.date_range(start, end, freq="QE")  # quarter-end dates
    return [d.date().isoformat() for d in qs]


def build_universe(conn) -> tuple[list[int], pd.DataFrame, pd.DataFrame]:
    """PIT top-1500-by-mcap union over quarter-ends 2011-2024 (survivorship-free)."""
    targets = _quarter_end_targets()
    targets_str = "', '".join(targets)
    sql = f"""
    WITH targets AS (SELECT unnest(ARRAY['{targets_str}']::date[]) AS target_date),
    actual_dates AS (
        SELECT t.target_date,
               (SELECT MAX(m.date) FROM crsp.msf m
                WHERE m.date <= t.target_date AND m.date >= t.target_date - interval '14 days') AS rank_date
        FROM targets t),
    ranked AS (
        SELECT a.target_date, msf.permno, msf.date AS rank_date,
               abs(msf.prc) * msf.shrout AS mcap,
               ROW_NUMBER() OVER (PARTITION BY msf.date
                   ORDER BY abs(msf.prc) * msf.shrout DESC NULLS LAST) AS rk
        FROM crsp.msf msf INNER JOIN actual_dates a ON msf.date = a.rank_date
        WHERE abs(msf.prc) > 0 AND msf.shrout > 0)
    SELECT target_date, rank_date, permno, mcap, rk
    FROM ranked WHERE rk <= {TOP_N} ORDER BY target_date, rk
    """
    t0 = time.time()
    q = pd.read_sql(sql, conn, parse_dates=["target_date", "rank_date"])
    logger.info("universe: %d quarter-end rows in %.1fs", len(q), time.time() - t0)
    q["permno"] = q["permno"].astype(int)
    permnos = sorted(q["permno"].unique().tolist())

    tdf = pd.read_sql(
        "SELECT DISTINCT ON (permno) permno, ticker FROM crsp.msenames "
        "WHERE permno IN %(p)s ORDER BY permno, nameendt DESC NULLS FIRST",
        conn, params={"p": tuple(permnos)},
    )
    tdf["permno"] = tdf["permno"].astype(int)
    tdf = tdf.dropna(subset=["ticker"])
    logger.info("universe: %d distinct permnos, %d with tickers", len(permnos), len(tdf))
    return permnos, tdf, q


def build_sue_panel(conn, tickers: list[int], start=WINDOW_START, end=WINDOW_END) -> pd.DataFrame:
    """B-T 1989 SUE panel for `tickers` over [start, end]; faithful to path_c locks.

    ticker -> permno (msenames) -> gvkey (ccmxpf_lnkhist) -> comp.fundq, then
    seasonal ΔEPS / σ_8q (8 prior quarters) winsorized ±10.
    """
    tickers = sorted(set(str(t).upper() for t in tickers))
    # 1) ticker -> permno
    mse = pd.read_sql(
        "SELECT DISTINCT permno, ticker FROM crsp.msenames "
        "WHERE ticker IN %(t)s AND nameendt >= %(s)s AND namedt <= %(e)s",
        conn, params={"t": tuple(tickers), "s": start.isoformat(), "e": end.isoformat()},
    )
    mse = mse.sort_values(["ticker", "permno"]).drop_duplicates("ticker", keep="first")
    permnos = sorted(int(p) for p in mse["permno"].dropna())
    permno_to_ticker = {int(r.permno): str(r.ticker).strip() for r in mse.itertuples()}

    # 2) permno -> gvkey
    link = pd.read_sql(
        "SELECT gvkey, lpermno AS permno, linkdt, linkenddt FROM crsp.ccmxpf_lnkhist "
        "WHERE lpermno IN %(p)s AND linktype IN ('LU','LC') AND linkprim IN ('P','C') "
        "AND (linkenddt IS NULL OR linkenddt >= %(s)s) AND linkdt <= %(e)s",
        conn, params={"p": tuple(permnos), "s": start.isoformat(), "e": end.isoformat()},
        parse_dates=["linkdt", "linkenddt"],
    )
    link = link.dropna(subset=["gvkey", "permno"]).sort_values(["permno", "gvkey"]).drop_duplicates("permno", keep="first")
    permno_to_gvkey = {int(r.permno): str(r.gvkey) for r in link.itertuples()}
    gvkey_to_permno = {v: k for k, v in permno_to_gvkey.items()}
    gvkeys = sorted(set(permno_to_gvkey.values()))

    # 3) comp.fundq with 12Q lookback buffer (8Q σ + 4Q seasonal)
    buffer_start = start - datetime.timedelta(days=1095 + 90)
    fundq = pd.read_sql(
        "SELECT gvkey, fyearq, fqtr, datadate, rdq, epspxq, ajexq, atq, cshoq, prccq "
        "FROM comp.fundq WHERE datadate BETWEEN %(b)s AND %(e)s AND gvkey IN %(g)s "
        "AND indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C' "
        "ORDER BY gvkey, fyearq, fqtr",
        conn, params={"b": buffer_start.isoformat(), "e": end.isoformat(), "g": tuple(gvkeys)},
        parse_dates=["datadate", "rdq"],
    )
    logger.info("sue: fundq rows=%d for %d gvkeys", len(fundq), len(gvkeys))
    if fundq.empty:
        return pd.DataFrame()

    f = fundq.copy()
    f["gvkey"] = f["gvkey"].astype(str)
    f["permno"] = f["gvkey"].map(gvkey_to_permno)
    f["ticker"] = f["permno"].map(permno_to_ticker)
    f = f.dropna(subset=["permno", "ticker", "fyearq", "fqtr"])
    f["eps_adj"] = f["epspxq"].astype(float) * f["ajexq"].astype(float).fillna(1.0)
    f["market_cap_at_q"] = f["cshoq"].astype(float) * f["prccq"].astype(float)
    f = f.sort_values(["gvkey", "fyearq", "fqtr"]).reset_index(drop=True)

    f["eps_adj_lag4"] = f.groupby("gvkey")["eps_adj"].shift(SEASONAL_LAG_QUARTERS)
    f["delta_eps"] = f["eps_adj"] - f["eps_adj_lag4"]

    def _rolling_prior_std(s: pd.Series) -> pd.Series:
        return s.rolling(window=SIGMA_WINDOW_QUARTERS, min_periods=SIGMA_MIN_PERIODS).std().shift(1)

    f["sigma_8q"] = f.groupby("gvkey")["delta_eps"].transform(_rolling_prior_std)
    f.loc[f["sigma_8q"].astype(float) < SIGMA_MIN_VALUE, "sigma_8q"] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        f["sue_raw"] = f["delta_eps"] / f["sigma_8q"]
    f["sue"] = f["sue_raw"].clip(SUE_WINSORIZE_LOW, SUE_WINSORIZE_HIGH)

    f = f.dropna(subset=["rdq"])
    f["rdq"] = pd.to_datetime(f["rdq"]).dt.date
    out = f[(f["rdq"] >= start) & (f["rdq"] <= end)].copy()
    out["fiscal_yearq"] = out["fyearq"].astype(int).astype(str) + "Q" + out["fqtr"].astype(int).astype(str)
    out = out[["permno", "ticker", "gvkey", "fiscal_yearq", "rdq", "eps_adj", "eps_adj_lag4",
               "delta_eps", "sigma_8q", "sue_raw", "sue", "market_cap_at_q"]]
    out["permno"] = out["permno"].astype(int)
    return out.sort_values(["rdq", "ticker"]).reset_index(drop=True)


RETURNS_PATH = CACHE / "_crsp_daily_ret_2011_2024.parquet"


def build_returns(conn, permnos: list[int], start=WINDOW_START, end=WINDOW_END,
                  chunk=500) -> pd.DataFrame:
    """Daily total returns (crsp.dsf) for universe permnos over [start, end].

    Self-contained Line C artifact (avoids universe-mismatch with the older
    crsp_hist_daily_ret cache). Chunked 500 permnos/query on ONE connection.
    """
    permnos = sorted(set(int(p) for p in permnos))
    frames = []
    for i in range(0, len(permnos), chunk):
        sub = permnos[i:i + chunk]
        t0 = time.time()
        df = pd.read_sql(
            "SELECT permno, date, ret FROM crsp.dsf "
            "WHERE permno IN %(p)s AND date BETWEEN %(s)s AND %(e)s AND ret IS NOT NULL",
            conn, params={"p": tuple(sub), "s": start.isoformat(), "e": end.isoformat()},
            parse_dates=["date"],
        )
        df["permno"] = df["permno"].astype(int)
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
        frames.append(df.dropna(subset=["ret"]))
        logger.info("  returns chunk %d-%d/%d: %d rows in %.1fs",
                    i + 1, i + len(sub), len(permnos), len(df), time.time() - t0)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(RETURNS_PATH)
    logger.info("returns saved: %d rows, %s -> %s",
                len(out), out["date"].min().date(), out["date"].max().date())
    return out


def returns_main():
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tdf = pd.read_parquet(CACHE / "_universe_top1500_2011_2024.parquet")
    conn = wrds_direct.connect("${WRDS_USER_1}")
    try:
        r = build_returns(conn, tdf["permno"].tolist())
        print(f"returns rows={len(r)} permnos={r['permno'].nunique()} "
              f"{r['date'].min().date()}->{r['date'].max().date()}")
    finally:
        conn.close()


def main():
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = wrds_direct.connect("${WRDS_USER_1}")        # ONE connection, reused
    try:
        permnos, tdf, qmem = build_universe(conn)
        tdf.to_parquet(CACHE / "_universe_top1500_2011_2024.parquet")
        qmem.to_parquet(CACHE / "_q_membership_2011_2024.parquet")

        sue = build_sue_panel(conn, tdf["ticker"].tolist())
        sue.to_parquet(CACHE / "_sue_panel_2011_2024.parquet")
        rdq = pd.to_datetime(sue["rdq"])
        meta = {
            "built_at": datetime.datetime.utcnow().isoformat() + "Z",
            "n_universe_permnos": len(permnos),
            "n_universe_tickers": int(tdf["ticker"].nunique()),
            "n_sue_firm_quarters": int(len(sue)),
            "sue_rdq_min": str(rdq.min().date()), "sue_rdq_max": str(rdq.max().date()),
            "n_distinct_permno_in_sue": int(sue["permno"].nunique()),
        }
        (CACHE / "_deep_panel_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
        print("\nby year:")
        print(sue.assign(yr=rdq.dt.year).groupby("yr").size().to_string())
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    if "--returns" in sys.argv:
        returns_main()
    else:
        main()
