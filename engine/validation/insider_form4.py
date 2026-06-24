"""engine/validation/insider_form4.py — second-alpha search: insider transactions (SEC Form 4).

Per the D_PEAD-family spec id=76 mandate: search for a second alpha where a solo
operator can actually win — unique/alternative data, NOT published-factor
replication. Insider trading qualifies: the SIGNAL is corporate-insider
conviction (a different trigger than D_PEAD's earnings surprise → should be
low-correlation), and the EDGE is in CONSTRUCTION (open-market trades by
officers/directors; the Cohen-Malloy-Pomorski 2012 routine-vs-opportunistic
filter is the refinement) rather than a raw published factor.

Data: SEC DERA Form 345 structured datasets (free, complete, 2006+). WRDS only
exposes a tiny sample (no tr_insiders permission), so we go to the source.

Construction (first pass — Lakonishok-Lee 2001 / Jeng-Metrick-Zeckhauser 2003):
  - keep open-market transactions only: TRANS_CODE in {P (purchase), S (sale)}
  - keep Form 4 by OFFICERS or DIRECTORS (exclude pure 10% owners = often funds)
  - firm-month net insider buying = signed dollar value (P=+, S=−)
  - monthly cross-section: long net-buyers, short net-sellers, hold one month
Refinement (if first pass shows life): CMP-2012 opportunistic-trade filter.

Everything is screened through alpha_factory.gate(); GREEN-only deploys.
"""
from __future__ import annotations

import io
import logging
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MAP        = "data/cache/_cik_gvkey_permno_map.parquet"      # top-1500 universe
_CIK_PERMNO_FULL = "data/cache/_cik_permno_map_FULL.parquet"  # full CRSP-Compustat
_RET_CACHE  = "data/cache/crsp_hist_daily_ret.parquet"
_MEMBERSHIP = "data/factor_ensemble_singlename/_crsp_top1500_q_membership.parquet"
_PANEL_CACHE = "data/cache/_insider_firmmonth_panel.parquet"           # top-1500
_PANEL_CACHE_FULL = "data/cache/_insider_firmmonth_panel_FULL.parquet" # full universe
_RET_BROAD = "data/cache/_crsp_msf_insider_universe.parquet"           # monthly ret, broad
_UA = "research ${USER_EMAIL}"
_BASE = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/%dq%d_form345.zip"


def _quarters(y0: int, y1: int):
    for y in range(y0, y1 + 1):
        for q in range(1, 5):
            yield y, q


def fetch_insider_panel(y0: int = 2014, y1: int = 2023, force: bool = False,
                        map_path: str = _MAP, cache_path: str = _PANEL_CACHE) -> pd.DataFrame:
    """Download SEC Form 345 quarters, filter to our CIK universe + open-market
    officer/director trades, aggregate to firm-month net insider buying.
    Cached. Returns columns: cik, month, net_dollars, net_shares, n_buy, n_sell.

    map_path selects the CIK universe: _MAP = top-1500; _CIK_PERMNO_FULL =
    full CRSP-Compustat (small/micro caps included, the documented insider
    alpha habitat)."""
    if os.path.exists(cache_path) and not force:
        return pd.read_parquet(cache_path)

    cik_set = set(pd.read_parquet(map_path)["cik"].astype(int).unique())
    rows = []
    for y, q in _quarters(y0, y1):
        url = _BASE % (y, q)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=120).read()))
        except Exception as exc:
            logger.warning("insider: %dq%d download failed: %s", y, q, exc)
            continue
        sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", low_memory=False,
                          usecols=["ACCESSION_NUMBER", "DOCUMENT_TYPE", "ISSUERCIK"])
        sub = sub[sub["DOCUMENT_TYPE"].astype(str) == "4"]
        sub["ISSUERCIK"] = pd.to_numeric(sub["ISSUERCIK"], errors="coerce")
        sub = sub[sub["ISSUERCIK"].isin(cik_set)]
        if sub.empty:
            continue
        own = pd.read_csv(z.open("REPORTINGOWNER.tsv"), sep="\t", low_memory=False,
                          usecols=["ACCESSION_NUMBER", "RPTOWNER_RELATIONSHIP"])
        rel = own["RPTOWNER_RELATIONSHIP"].astype(str).str.lower()
        own = own[rel.str.contains("director") | rel.str.contains("officer")]
        nd = pd.read_csv(z.open("NONDERIV_TRANS.tsv"), sep="\t", low_memory=False,
                         usecols=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
                                  "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                                  "TRANS_ACQUIRED_DISP_CD"])
        nd = nd[nd["TRANS_CODE"].isin(["P", "S"])]
        nd = nd.merge(sub[["ACCESSION_NUMBER", "ISSUERCIK"]], on="ACCESSION_NUMBER", how="inner")
        nd = nd[nd["ACCESSION_NUMBER"].isin(set(own["ACCESSION_NUMBER"]))]
        if nd.empty:
            continue
        nd["TRANS_DATE"] = pd.to_datetime(nd["TRANS_DATE"], errors="coerce")
        nd["TRANS_SHARES"] = pd.to_numeric(nd["TRANS_SHARES"], errors="coerce")
        nd["TRANS_PRICEPERSHARE"] = pd.to_numeric(nd["TRANS_PRICEPERSHARE"], errors="coerce")
        nd = nd.dropna(subset=["TRANS_DATE", "TRANS_SHARES"])
        sign = np.where(nd["TRANS_ACQUIRED_DISP_CD"].astype(str) == "A", 1.0, -1.0)
        nd["signed_sh"] = sign * nd["TRANS_SHARES"]
        nd["signed_dol"] = nd["signed_sh"] * nd["TRANS_PRICEPERSHARE"].fillna(0.0)
        nd["month"] = nd["TRANS_DATE"].dt.to_period("M").dt.to_timestamp("M")
        g = nd.groupby(["ISSUERCIK", "month"]).agg(
            net_dollars=("signed_dol", "sum"),
            net_shares=("signed_sh", "sum"),
            n_buy=("TRANS_CODE", lambda s: (s == "P").sum()),
            n_sell=("TRANS_CODE", lambda s: (s == "S").sum()),
        ).reset_index().rename(columns={"ISSUERCIK": "cik"})
        rows.append(g)
        logger.info("insider %dq%d: %d firm-month rows", y, q, len(g))

    panel = (pd.concat(rows, ignore_index=True) if rows else
             pd.DataFrame(columns=["cik", "month", "net_dollars", "net_shares", "n_buy", "n_sell"]))
    # a firm may appear in two quarters for the same month boundary — sum
    panel = panel.groupby(["cik", "month"], as_index=False).sum()
    panel.to_parquet(cache_path, index=False)
    logger.info("insider panel: %d firm-months, %d CIKs", len(panel), panel["cik"].nunique())
    return panel


def _wide_daily_returns(ret_path: str = _RET_CACHE) -> pd.DataFrame:
    r = pd.read_parquet(ret_path)
    r["date"] = pd.to_datetime(r["date"])
    return r.pivot_table(index="date", columns="permno", values="ret").sort_index()


def build_insider_sleeve(min_names: int = 20) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly L/S: long net-buyer firms, short net-seller firms (among firms
    with officer/director open-market activity that month), equal-weight, hold
    one month. Returns (monthly_ls, selection)."""
    panel = fetch_insider_panel()
    cmap = pd.read_parquet(_MAP)[["cik", "permno"]].drop_duplicates()
    panel = panel.merge(cmap, on="cik", how="inner")
    daily = _wide_daily_returns()
    monthly_ret = (1.0 + daily.fillna(0.0)).resample("ME").prod() - 1.0
    monthly_ret = monthly_ret.where(daily.resample("ME").count() > 5)

    rows_ret, rows_sel = [], []
    months = sorted(panel["month"].unique())
    for i in range(len(months) - 1):
        t, t1 = months[i], months[i + 1]
        g = panel[panel["month"] == t]
        # one signal per permno (a firm may map to one permno)
        sig = g.groupby("permno")["net_dollars"].sum()
        buyers = sig[sig > 0].index
        sellers = sig[sig < 0].index
        if len(buyers) + len(sellers) < min_names:
            continue
        if t1 not in monthly_ret.index:
            continue
        nxt = monthly_ret.loc[t1]
        rb = nxt.reindex(buyers).dropna()
        rs = nxt.reindex(sellers).dropna()
        if len(rb) < 3 or len(rs) < 3:
            continue
        rows_ret.append((t1, float(rb.mean() - rs.mean())))
        rows_sel.append((t1, list(buyers), list(sellers)))
    sleeve = pd.Series(dict(rows_ret), name="insider_form4").sort_index()
    sel = pd.DataFrame(rows_sel, columns=["month", "long", "short"]).set_index("month")
    return sleeve, sel


def build_cluster_sleeve_broad(nb_min: int = 3, hold_months: int = 1,
                               map_path: str = _CIK_PERMNO_FULL,
                               panel_path: str = _PANEL_CACHE_FULL,
                               ret_path: str = _RET_BROAD) -> tuple[pd.Series, float]:
    """Broad (small-cap-inclusive) cluster-buy sleeve: long firms with >=nb_min
    officer/director open-market BUYS in the last `hold_months`, excess over the
    equal-weight insider universe, monthly. Returns (excess_series, ann_turnover).

    Verdict (2026-05-20): residual alpha is REAL (>=3 cluster, 1-month hold:
    +6.07%/yr gross, FF5+UMD residual t=3.02, corr 0.22 w/ D_PEAD) but it is
    FRONT-LOADED in month 1 (decays to t<1 by month 3), so capturing it needs
    ~21x annual turnover, and the names are small-cap (ss_small cost) -> the
    net deflated SR collapses. A genuine uncorrelated alpha that the solo
    small-cap-cost constraint makes NOT net-tradeable. Shelved as a validated
    lead, not deployed."""
    panel = pd.read_parquet(panel_path)
    fmap = pd.read_parquet(map_path)[["cik", "permno"]].drop_duplicates()
    panel = panel.merge(fmap, on="cik", how="inner")
    panel["month"] = pd.to_datetime(panel["month"])
    ret = pd.read_parquet(ret_path)
    mret = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret.index = mret.index + pd.offsets.MonthEnd(0)
    months = sorted([m for m in panel["month"].unique() if m in mret.index])

    cluster = {}
    for t in months:
        g = panel[panel["month"] == t].groupby("permno").agg(
            net=("net_dollars", "sum"), nb=("n_buy", "sum"))
        cluster[t] = set(g[(g["net"] > 0) & (g["nb"] >= nb_min)].index)

    rows, turns, prev = [], [], set()
    for i in range(len(months) - 1):
        t1 = months[i + 1]
        if t1 not in mret.index:
            continue
        held = set().union(*[cluster.get(months[j], set())
                             for j in range(max(0, i - hold_months + 1), i + 1)])
        nxt = mret.loc[t1]
        rl = nxt.reindex(list(held)).dropna()
        uni = nxt.dropna().mean()
        if len(rl) < 5:
            continue
        rows.append((t1, float(rl.mean() - uni)))
        turns.append(len(held ^ prev) / max(len(held), 1))
        prev = held
    return pd.Series(dict(rows)).sort_index().rename("insider_cluster"), float(np.mean(turns) * 12)


_OWNER_PANEL = "data/cache/_insider_owner_panel_FULL.parquet"


def fetch_insider_owner_panel(y0: int = 2012, y1: int = 2023, force: bool = False) -> pd.DataFrame:
    """Owner-level Form 4 panel for the CMP routine/opportunistic filter: keeps
    (issuercik, rptownercik, month, year, n_buy, n_sell, net_dollars) for
    open-market P/S by officers/directors, full CRSP-Compustat CIK universe.
    Starts 2012 to give 2-year classification history before the 2014 trade
    start. Cached."""
    if os.path.exists(_OWNER_PANEL) and not force:
        return pd.read_parquet(_OWNER_PANEL)
    cik_set = set(pd.read_parquet(_CIK_PERMNO_FULL)["cik"].astype(int).unique())
    rows = []
    for y, q in _quarters(y0, y1):
        try:
            req = urllib.request.Request(_BASE % (y, q), headers={"User-Agent": _UA})
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=120).read()))
        except Exception as exc:
            logger.warning("insider owner %dq%d download failed: %s", y, q, exc)
            continue
        sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", low_memory=False,
                          usecols=["ACCESSION_NUMBER", "DOCUMENT_TYPE", "ISSUERCIK"])
        sub = sub[sub["DOCUMENT_TYPE"].astype(str) == "4"]
        sub["ISSUERCIK"] = pd.to_numeric(sub["ISSUERCIK"], errors="coerce")
        sub = sub[sub["ISSUERCIK"].isin(cik_set)]
        if sub.empty:
            continue
        own = pd.read_csv(z.open("REPORTINGOWNER.tsv"), sep="\t", low_memory=False,
                          usecols=["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
        rel = own["RPTOWNER_RELATIONSHIP"].astype(str).str.lower()
        own = own[rel.str.contains("director") | rel.str.contains("officer")]
        own["RPTOWNERCIK"] = pd.to_numeric(own["RPTOWNERCIK"], errors="coerce")
        own = own.dropna(subset=["RPTOWNERCIK"])
        nd = pd.read_csv(z.open("NONDERIV_TRANS.tsv"), sep="\t", low_memory=False,
                         usecols=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
                                  "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
        nd = nd[nd["TRANS_CODE"].isin(["P", "S"])]
        nd = nd.merge(sub[["ACCESSION_NUMBER", "ISSUERCIK"]], on="ACCESSION_NUMBER", how="inner")
        nd = nd.merge(own[["ACCESSION_NUMBER", "RPTOWNERCIK"]], on="ACCESSION_NUMBER", how="inner")
        if nd.empty:
            continue
        nd["TRANS_DATE"] = pd.to_datetime(nd["TRANS_DATE"], errors="coerce")
        nd["TRANS_SHARES"] = pd.to_numeric(nd["TRANS_SHARES"], errors="coerce")
        nd["TRANS_PRICEPERSHARE"] = pd.to_numeric(nd["TRANS_PRICEPERSHARE"], errors="coerce")
        nd = nd.dropna(subset=["TRANS_DATE", "TRANS_SHARES"])
        sign = np.where(nd["TRANS_ACQUIRED_DISP_CD"].astype(str) == "A", 1.0, -1.0)
        nd["signed_dol"] = sign * nd["TRANS_SHARES"] * nd["TRANS_PRICEPERSHARE"].fillna(0.0)
        nd["month"] = nd["TRANS_DATE"].dt.to_period("M").dt.to_timestamp("M")
        nd["cal_month"] = nd["TRANS_DATE"].dt.month
        nd["year"] = nd["TRANS_DATE"].dt.year
        g = nd.groupby(["ISSUERCIK", "RPTOWNERCIK", "month", "year", "cal_month"]).agg(
            net_dollars=("signed_dol", "sum"),
            n_buy=("TRANS_CODE", lambda s: (s == "P").sum()),
            n_sell=("TRANS_CODE", lambda s: (s == "S").sum()),
        ).reset_index().rename(columns={"ISSUERCIK": "cik", "RPTOWNERCIK": "owner"})
        rows.append(g)
        logger.info("insider owner %dq%d: %d rows", y, q, len(g))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[(panel["year"] >= y0) & (panel["year"] <= y1)]
    panel.to_parquet(_OWNER_PANEL, index=False)
    logger.info("owner panel: %d rows, %d owners", len(panel), panel["owner"].nunique())
    return panel
