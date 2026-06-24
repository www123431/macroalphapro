"""engine/validation/network_momentum.py — un-arbitraged 2nd-alpha candidate on a
DIFFERENT mechanism: shared-analyst connected-firm momentum (Ali-Hirshleifer 2020,
"Shared Analyst Coverage: Unifying Momentum Spillover Effects").

Mechanism = SLOW INFORMATION DIFFUSION across economically-linked firms under
limited attention (Hong-Stein; Cohen-Frazzini): investors update a firm fast but are
slow to propagate the implication to its connected firms, so the connected firms'
recent returns predict the focal firm's next-month return. A MOMENTUM-spillover
channel — genuinely different from the earnings-information underreaction family
(D_PEAD/revision/guidance) → real orthogonality potential.

Why less arbitraged: it needs a firm-firm GRAPH built from messy data. We use the
CLEANEST link — analyst co-coverage (two firms connected if a common analyst covers
both) — which avoids the customer-NAME matching of Cohen-Frazzini and which
Ali-Hirshleifer show SUBSUMES customer/supply-chain/industry/geographic momentum.

DISCIPLINE (this is a momentum-family signal, so the #1 trap is it's just UMD in
disguise): the L/S return is RESIDUALIZED vs FF5+UMD+PEAD — only orthogonal alpha
counts — then run through the corrected deflated-Sharpe + audit battery.

VERDICT (2026-05-21, _network_run.py; co-coverage graph, 2228 linked permnos,
2014-2024): **RED — triple failure.** (1) gross ~zero: L/S ann +1.2%, Sharpe 0.08,
t=0.25 — co-coverage momentum is arbitraged/absent in the liquid large-cap universe;
(2) TURNOVER WALL: 9.2x turnover -> after cost net Sharpe -0.24 (same wall that
killed reversal/stat-arb — momentum-diffusion signals are turnover-locked for a
solo); (3) NOT orthogonal: residual alpha vs FF5+UMD+PEAD = -6.48%/yr t=-1.39
(negative), PEAD loading 0.67, corr +0.39 D_PEAD / +0.58 with the cached
supply-chain mom sleeve. deflated SR 0.013. The existing customer-supplier
supply-chain sleeve is the same family (gross Sharpe 0.49 t=1.63, dies after cost).
LESSON: the network/momentum-diffusion family is ALSO arbitraged + turnover-walled
in our accessible liquid universe — confirming that on plain WRDS data BOTH the
earnings-information family AND the momentum family are competed away; genuine
un-arbitraged alpha needs NOVEL DATA (alt-data) or the small/illiquid/speed edges we
don't have. Joins the graveyard. (Ali-Hirshleifer's stronger result lives in a
broader/older/smaller-cap universe + pre-2017.)
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_COV = "data/cache/_ibes_coverage_panel.parquet"
_STOCKNAMES = "data/cache/_stocknames_ncusip.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine(
        "postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
        connect_args={"sslmode": "require"})


def fetch_coverage(force: bool = False) -> pd.DataFrame:
    """ONE WRDS connection: distinct (analyst, ticker, cusip, year) analyst-coverage
    pairs from ibes.detu_epsus (US firms, 2010-2024). Each individual EPS estimate =
    that analyst covers that firm that year. Cached. Returns the coverage panel."""
    import socket
    import time
    if os.path.exists(_COV) and not force:
        return pd.read_parquet(_COV)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        sql = ("select distinct analys, ticker, cusip, extract(year from anndats)::int as yr "
               "from ibes.detu_epsus where usfirm=1 and measure='EPS' "
               "and anndats >= '2010-01-01' and analys is not null and cusip is not null")
        cov = pd.read_sql(text(sql), eng)
    finally:
        eng.dispose()
    cov.to_parquet(_COV, index=False)
    logger.info("coverage panel: %d rows, %d analysts, %d tickers, yr %d..%d",
                len(cov), cov["analys"].nunique(), cov["ticker"].nunique(),
                cov["yr"].min(), cov["yr"].max())
    return cov


def _ticker_to_permno(cov: pd.DataFrame) -> pd.DataFrame:
    """Map IBES ticker -> permno via cusip(8) <-> crsp.stocknames ncusip."""
    sn = pd.read_parquet(_STOCKNAMES).rename(columns={"ncusip": "cusip8"})
    c = cov.copy(); c["cusip8"] = c["cusip"].astype(str).str[:8]
    link = (c.merge(sn[["cusip8", "permno"]].drop_duplicates(), on="cusip8", how="inner")
            [["ticker", "permno"]].drop_duplicates("ticker"))
    return link


def _monthly_returns():
    ret = pd.read_parquet(_RET); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    return mret.where(daily.resample("ME").count() > 5)


def build_cmom_signal(min_shared: int = 1, form: int = 1):
    """Connected-firm momentum (CMOM). For each year's co-coverage graph (firms
    sharing >= min_shared analysts that year), CMOM_{i,month} = equal-weight mean of
    connected firms' return over the past `form` month(s). Predicts firm i's NEXT
    month. Returns a (month x permno) wide CMOM signal aligned to permno returns.

    Built per coverage-YEAR (graph fixed within year, using that year's coverage to
    avoid look-ahead: the prior year's graph is applied to the current year)."""
    cov = fetch_coverage()
    link = _ticker_to_permno(cov)
    cov = cov.merge(link, on="ticker", how="inner")          # analys, permno, yr
    mret = _monthly_returns()
    # past-`form`-month return per permno (the diffusion source), aligned by month
    pastret = (1 + mret.fillna(0)).rolling(form).apply(np.prod, raw=True) - 1 if form > 1 else mret
    permnos = list(mret.columns)
    pidx = {p: k for k, p in enumerate(permnos)}
    sig = pd.DataFrame(index=mret.index, columns=permnos, dtype=float)
    for yr in range(cov["yr"].min() + 1, cov["yr"].max() + 1):
        # USE PRIOR-YEAR coverage graph for the current year (causal, no look-ahead)
        cy = cov[cov["yr"] == yr - 1]
        cy = cy[cy["permno"].isin(pidx)]
        if cy["permno"].nunique() < 50:
            continue
        # firm x analyst incidence -> co-coverage adjacency = B @ B.T
        firms = sorted(cy["permno"].unique())
        fidx = {p: k for k, p in enumerate(firms)}
        an = sorted(cy["analys"].unique()); aidx = {a: k for k, a in enumerate(an)}
        B = np.zeros((len(firms), len(an)), dtype=np.float32)
        B[cy["permno"].map(fidx).values, cy["analys"].map(aidx).values] = 1.0
        A = B @ B.T                                          # shared-analyst counts
        np.fill_diagonal(A, 0.0)
        Adj = (A >= min_shared).astype(np.float32)
        deg = Adj.sum(axis=1)
        ok = deg > 0
        months = mret.index[(mret.index.year == yr)]
        for m in months:
            if m not in pastret.index:
                continue
            pr = pastret.loc[m].reindex(firms).values.astype(float)
            valid = ~np.isnan(pr)
            prc = np.where(valid, pr, 0.0)
            cnt = Adj @ valid.astype(np.float32)             # # valid connected
            num = Adj @ prc
            cm = np.where((cnt > 0) & ok, num / np.maximum(cnt, 1), np.nan)
            row = np.full(len(permnos), np.nan)
            for p, k in fidx.items():
                row[pidx[p]] = cm[k]
            sig.loc[m] = row
    return sig.astype(float), mret


def build_cmom_sleeve(min_shared: int = 1, form: int = 1, q: float = 0.2,
                      sigret=None):
    """Monthly L/S: long high-CMOM / short low-CMOM, 1-month hold (next month).
    Returns (ls, long_only, ann_turnover)."""
    if sigret is None:
        sig, mret = build_cmom_signal(min_shared, form)
    else:
        sig, mret = sigret
    months = list(mret.index)
    ls, lo, ent, prevL = [], [], [], set()
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        s = sig.loc[m].dropna() if m in sig.index else pd.Series(dtype=float)
        if len(s) < 50 or nxt not in mret.index:
            continue
        hi = s[s >= s.quantile(1 - q)].index
        loq = s[s <= s.quantile(q)].index
        nr = mret.loc[nxt]
        rl = nr.reindex(hi).dropna(); rs = nr.reindex(loq).dropna(); rm = nr.reindex(s.index).dropna()
        if len(rl) < 10 or len(rs) < 10:
            continue
        ls.append((nxt, float(rl.mean() - rs.mean()))); lo.append((nxt, float(rl.mean() - rm.mean())))
        ent.append(len(set(hi) - prevL) / max(len(hi), 1)); prevL = set(hi)
    return (pd.Series(dict(ls)).sort_index().rename("cmom_ls"),
            pd.Series(dict(lo)).sort_index().rename("cmom_long"),
            float(np.mean(ent) * 12) if ent else float("nan"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cov = fetch_coverage()
    link = _ticker_to_permno(cov)
    logger.info("coverage %d rows; ticker->permno linked %d tickers", len(cov), len(link))
