"""engine/validation/inst_flow_13f.py — 2nd-alpha shot: institutional flow (Thomson 13F).

Newly confirmed SELECT-OK on ${WRDS_USER_2}: tr_13f.s34 = 124M rows of institutional
holdings (mgrno, rdate, cusip, shares, change, prc). Cleaner + longer than the
free-SEC version (inst_13f.py, which was RED-ish on breadth alone).

"Follow the smart money" — institutional POSITIONING, orthogonal to the earnings
mechanism. Signals (Chen-Hong-Stein 2002 breadth; Gompers-Metrick 2001 ownership):
  - n_mgr (breadth = # institutions holding) and its quarterly change;
  - total institutional shares and its change (net institutional buying);
  - HHI-style concentration (built later if breadth/flow show promise).

Server-side aggregation to (cusip, rdate-quarter) — 124M rows is too big raw.
13F is public ~45 days after rdate → trade with that lag (no look-ahead).
Map cusip->permno via cached CRSP stocknames ncusip. Screened via alpha_factory.

VERDICT (2026-05-21, top-1500, Thomson tr_13f.s34 708k cusip-quarters): RED.
  - Δbreadth (Chen-Hong-Stein): gross 3.25%/yr, Sharpe 0.30, FF5+UMD residual
    t=0.74 (n.s.), net deflSR 0.21. AND corr 0.54 w/ D_PEAD — not even orthogonal
    (institutions buy AFTER good earnings → breadth-change overlaps PEAD).
  - net institutional buying (Δownership): DEAD, gross 0.40%/yr, t=0.02.
  => the breadth-of-ownership anomaly is arbitraged (matches the free-SEC version
  inst_13f.py + the post-publication decay literature) and not a clean diversifier.
  Institutional 13F flow is not a 2nd alpha source.
"""
from __future__ import annotations

import logging
import os
import socket
import time

import pandas as pd

logger = logging.getLogger(__name__)

_STOCKNAMES = "data/cache/_stocknames_ncusip.parquet"
_CACHE = "data/cache/_tr13f_cusip_quarter.parquet"


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine("postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
                         connect_args={"sslmode": "require"})


def fetch_13f_agg(force: bool = False) -> pd.DataFrame:
    """ONE WRDS connection: server-side aggregate tr_13f.s34 to (cusip, rdate):
    n_mgr (breadth), tot_shares, n_pos. Cached."""
    if os.path.exists(_CACHE) and not force:
        return pd.read_parquet(_CACHE)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        q = ("select cusip, rdate, count(distinct mgrno) as n_mgr, "
             "sum(shares) as tot_sh, count(*) as n_pos "
             "from tr_13f.s34 where rdate >= '2012-06-01' "
             "group by cusip, rdate")
        agg = pd.read_sql(text(q), eng)
    finally:
        eng.dispose()
    agg["rdate"] = pd.to_datetime(agg["rdate"])
    agg.to_parquet(_CACHE, index=False)
    logger.info("13F agg: %d cusip-quarters, %d cusips, %s..%s",
                len(agg), agg["cusip"].nunique(), agg["rdate"].min(), agg["rdate"].max())
    return agg


_RET = "data/cache/crsp_hist_daily_ret.parquet"


def build_flow_signals():
    """Quarterly institutional-flow panel keyed to permno: Δbreadth, net buying.
    Mapped via cached CRSP stocknames ncusip(8). Returns wide monthly L/S inputs."""
    agg = fetch_13f_agg()
    agg["cusip8"] = agg["cusip"].astype(str).str.slice(0, 8)
    sn = pd.read_parquet(_STOCKNAMES).rename(columns={"ncusip": "cusip"})
    sn["cusip8"] = sn["cusip"].astype(str).str.slice(0, 8)
    a = agg.merge(sn[["cusip8", "permno"]].drop_duplicates(), on="cusip8", how="inner")
    a["permno"] = a["permno"].astype(int)
    a = a.sort_values(["permno", "rdate"])
    a["d_breadth"] = a.groupby("permno")["n_mgr"].diff()
    a["d_breadth_pct"] = a["d_breadth"] / a.groupby("permno")["n_mgr"].shift(1)
    a["d_own"] = a.groupby("permno")["tot_sh"].pct_change()
    return a


def build_flow_ls(signal: str = "d_breadth_pct", q: float = 0.2, lag_q: int = 1):
    """Quarterly L/S on an institutional-flow signal; trade with `lag_q` quarter
    lag (13F public ~45d after rdate); hold to next quarter; monthly returns."""
    import numpy as np
    a = build_flow_signals().dropna(subset=[signal])
    ret = pd.read_parquet(_RET); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    mret = mret.where(daily.resample("ME").count() > 5)
    # tradeable month = rdate + lag_q quarters (+~45d filing lag baked into lag_q>=1)
    a["trade_m"] = (a["rdate"] + pd.offsets.QuarterEnd(lag_q)).dt.to_period("M").dt.to_timestamp("M")
    rows = []
    months = mret.index
    for tm, g in a.groupby("trade_m"):
        # hold the quarter following trade_m
        hold = [m for m in months if tm < m <= tm + pd.DateOffset(months=3)]
        if len(g) < 100 or not hold:
            continue
        hi = set(g[g[signal] >= g[signal].quantile(1 - q)]["permno"])
        lo = set(g[g[signal] <= g[signal].quantile(q)]["permno"])
        for m in hold:
            nr = mret.loc[m]
            rl = nr.reindex(list(hi)).dropna(); rs = nr.reindex(list(lo)).dropna()
            if len(rl) < 10 or len(rs) < 10:
                continue
            rows.append((m, float(rl.mean() - rs.mean())))
    return pd.Series(dict(rows)).groupby(level=0).mean().sort_index().rename(f"13f_{signal}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    a = fetch_13f_agg()
    logger.info("DONE %s", a.shape)
