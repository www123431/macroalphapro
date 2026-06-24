"""engine/validation/short_interest.py — short-interest ratio reversal sleeve.

Direction 2 candidate #2: a behavioral, EVENT-light sibling to D_PEAD that is
NOT price-momentum (the GH 52-week-high sibling was rejected for being just
decayed momentum, corr 0.63 with D_PEAD). The short-interest hypothesis
(Asquith-Pathak-Ritter 2005 JFE; Boehmer-Jones-Zhang 2008 JF): heavily
shorted stocks underperform — short sellers are informed, so a high
short-interest ratio (SIR) is a bearish signal that the market under-reacts
to. Long low-SIR, short high-SIR.

Why this is a real sibling test (vs GH 52w-high):
  (1) different trigger — short-seller positioning, NOT price path. So the
      sharp test is whether it is UNCORRELATED with D_PEAD (earnings surprise)
      AND with simple momentum.
  (2) residual alpha AFTER FF5+UMD — is there anything beyond size/value/mom?
  (3) is it small-cap concentrated like PEAD? (short constraints bind hardest
      in small/illiquid names — APR 2005 found the effect there.)

Construction: monthly, cross-sectional. SIR = shares-short / shares-outstanding,
taken as the latest bi-monthly short-interest reading as of month-end. Long
bottom decile (least shorted), short top decile (most shorted), equal-weight,
hold one month.

Honest cost note: the SHORT leg here shorts the MOST-shorted stocks — exactly
the names with the highest borrow cost and squeeze risk. The alpha factory
run uses ss_small cost class but borrow cost on hard-to-borrow shorts is
ADDITIONAL and not captured; flag it in the verdict.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RET_CACHE = "data/cache/crsp_hist_daily_ret.parquet"
_PANEL = "data/cache/_pead_ts_panel_2014_2023.parquet"
_MEMBERSHIP = "data/factor_ensemble_singlename/_crsp_top1500_q_membership.parquet"
_SHORTINT_CACHE = "data/cache/_sec_shortint_2014_2024.parquet"
_SHROUT_CACHE = "data/cache/_crsp_shrout_2014_2024.parquet"

_WRDS_USER = "${WRDS_USER_1}"


def _permno_gvkey_map() -> pd.DataFrame:
    """Unique permno<->gvkey pairs from the PEAD panel (one gvkey per permno)."""
    p = pd.read_parquet(_PANEL, columns=["permno", "gvkey"]).dropna().drop_duplicates()
    p["permno"] = p["permno"].astype(int)
    # comp gvkey is a 6-char zero-padded string
    p["gvkey6"] = p["gvkey"].astype(int).astype(str).str.zfill(6)
    return p.reset_index(drop=True)


def fetch_short_interest(force: bool = False) -> pd.DataFrame:
    """Pull comp.sec_shortint (shares short, bi-monthly) for our gvkeys.
    Cached. Primary issue (iid='01') only."""
    if os.path.exists(_SHORTINT_CACHE) and not force:
        return pd.read_parquet(_SHORTINT_CACHE)
    import wrds

    m = _permno_gvkey_map()
    gvkeys = tuple(sorted(m["gvkey6"].unique()))
    db = wrds.Connection(wrds_username=_WRDS_USER)
    try:
        in_list = ",".join("'%s'" % g for g in gvkeys)
        sql = (
            "select gvkey, iid, datadate, shortint, shortintadj "
            "from comp.sec_shortint "
            "where iid = '01' and datadate >= '2013-10-01' and datadate <= '2024-06-30' "
            "and gvkey in (%s)" % in_list
        )
        df = db.raw_sql(sql)
    finally:
        db.close()
    df["datadate"] = pd.to_datetime(df["datadate"])
    df = df.dropna(subset=["shortint"])
    df.to_parquet(_SHORTINT_CACHE, index=False)
    logger.info("short interest: %d rows, %d gvkeys", len(df), df["gvkey"].nunique())
    return df


def fetch_shares_outstanding(force: bool = False) -> pd.DataFrame:
    """Pull CRSP monthly shares outstanding (shrout, in thousands) for our
    permnos. Cached."""
    if os.path.exists(_SHROUT_CACHE) and not force:
        return pd.read_parquet(_SHROUT_CACHE)
    import wrds

    m = _permno_gvkey_map()
    permnos = tuple(int(x) for x in sorted(m["permno"].unique()))
    db = wrds.Connection(wrds_username=_WRDS_USER)
    try:
        in_list = ",".join(str(p) for p in permnos)
        sql = (
            "select permno, date, shrout "
            "from crsp.msf "
            "where date >= '2013-10-01' and date <= '2024-06-30' "
            "and permno in (%s)" % in_list
        )
        df = db.raw_sql(sql)
    finally:
        db.close()
    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype(int)
    df = df.dropna(subset=["shrout"])
    df.to_parquet(_SHROUT_CACHE, index=False)
    logger.info("shrout: %d rows, %d permnos", len(df), df["permno"].nunique())
    return df


def build_sir_panel() -> pd.DataFrame:
    """Monthly short-interest-ratio panel: index=month-end, columns=permno,
    value=SIR (shares short / shares outstanding). Uses the latest short-int
    reading and the latest shrout as of each month-end (point-in-time, no
    look-ahead — short interest is reported with a settlement-date lag, which
    is conservative)."""
    si = fetch_short_interest()
    so = fetch_shares_outstanding()
    m = _permno_gvkey_map()[["permno", "gvkey6"]]

    # map shortint gvkey -> permno
    si = si.merge(m, left_on="gvkey", right_on="gvkey6", how="inner")
    # month-end stamp the bi-monthly readings, take the LAST reading in a month
    si["month"] = si["datadate"].dt.to_period("M").dt.to_timestamp("M")
    si_m = (si.sort_values("datadate")
              .groupby(["permno", "month"], as_index=False)["shortint"].last())

    so["month"] = so["date"].dt.to_period("M").dt.to_timestamp("M")
    so_m = (so.sort_values("date")
              .groupby(["permno", "month"], as_index=False)["shrout"].last())

    # forward-fill both onto a common monthly grid per permno, then divide.
    # shrout is in thousands -> shares = shrout*1000.
    si_w = si_m.pivot(index="month", columns="permno", values="shortint").sort_index()
    so_w = so_m.pivot(index="month", columns="permno", values="shrout").sort_index()
    grid = si_w.index.union(so_w.index)
    si_w = si_w.reindex(grid).ffill(limit=2)        # short-int valid ~1 month
    so_w = so_w.reindex(grid).ffill(limit=3) * 1000.0
    sir = si_w / so_w
    # sane bounds: SIR in (0, 1]; drop absurd values
    sir = sir.where((sir > 0) & (sir <= 1.0))
    return sir


def _wide_daily_returns(ret_path: str = _RET_CACHE) -> pd.DataFrame:
    r = pd.read_parquet(ret_path)
    r["date"] = pd.to_datetime(r["date"])
    return r.pivot_table(index="date", columns="permno", values="ret").sort_index()


def build_si_sleeve(decile: float = 0.1) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly L/S short-interest reversal sleeve. Long LOW-SIR (least
    shorted), short HIGH-SIR (most shorted). Returns (monthly_ls_returns,
    monthly_selection)."""
    sir = build_sir_panel()
    daily = _wide_daily_returns()
    monthly_ret = (1.0 + daily.fillna(0.0)).resample("ME").prod() - 1.0
    monthly_ret = monthly_ret.where(daily.resample("ME").count() > 5)

    # align SIR month-end stamps to the return month-end index
    sir_me = sir.copy()
    sir_me.index = sir_me.index + pd.offsets.MonthEnd(0)

    rows_ret, rows_sel = [], []
    months = sir_me.index
    for i in range(len(months) - 1):
        t, t1 = months[i], months[i + 1]
        s = sir_me.loc[t].dropna()
        if len(s) < 50:
            continue
        lo = s[s <= s.quantile(decile)].index          # least shorted -> long
        hi = s[s >= s.quantile(1 - decile)].index       # most shorted  -> short
        if t1 not in monthly_ret.index:
            continue
        nxt = monthly_ret.loc[t1]
        rl = nxt.reindex(lo).dropna()
        rs = nxt.reindex(hi).dropna()
        if len(rl) < 5 or len(rs) < 5:
            continue
        rows_ret.append((t1, float(rl.mean() - rs.mean())))
        rows_sel.append((t1, list(lo), list(hi)))
    sleeve = pd.Series(dict(rows_ret), name="short_interest_rev").sort_index()
    sel = pd.DataFrame(rows_sel, columns=["month", "long", "short"]).set_index("month")
    return sleeve, sel


def si_smallcap_concentration(sel: pd.DataFrame) -> dict:
    """Is the LONG leg (low-SIR) concentrated in small caps? Tags each month's
    long-leg permnos by cap tertile using nearest-prior quarterly membership."""
    mem = pd.read_parquet(_MEMBERSHIP)
    mem["target_date"] = pd.to_datetime(mem["target_date"])
    out = {"small": 0, "mid": 0, "large": 0, "n": 0}
    mem_by_q = {d: g for d, g in mem.groupby("target_date")}
    q_dates = sorted(mem_by_q.keys())
    for month, row in sel.iterrows():
        prior = [d for d in q_dates if d <= month]
        if not prior:
            continue
        g = mem_by_q[prior[-1]].dropna(subset=["mcap"])
        if len(g) < 30:
            continue
        q1, q2 = g["mcap"].quantile(1 / 3), g["mcap"].quantile(2 / 3)
        cap = g.set_index("permno")["mcap"]
        for pn in row["long"]:
            mc = cap.get(pn)
            if mc is None or not np.isfinite(mc):
                continue
            out["n"] += 1
            out["small" if mc <= q1 else "mid" if mc <= q2 else "large"] += 1
    if out["n"] == 0:
        return {"error": "no cap-tagged long names"}
    return {k: (out[k] / out["n"] if k != "n" else out["n"]) for k in out}
