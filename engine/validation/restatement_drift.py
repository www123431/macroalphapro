"""engine/validation/restatement_drift.py — un-arbitraged 3rd-alpha candidate on a
DIFFERENT mechanism + trigger: post-restatement-announcement drift.

Mechanism = accounting-CREDIBILITY shock + slow incorporation of bad news. When a firm
files a restatement (esp. income-DECREASING), it reveals the prior financials overstated
reality; the market under-reacts and the stock drifts DOWN over the following months
(Hribar-Jenkins 2004; Palmrose-Richardson-Scholz 2004; Anderson-Yohn). Genuinely DIFFERENT
from the book's mechanisms:
  - NOT earnings-surprise (D_PEAD/revision = forward-earnings-INFO family); restatement is a
    BACKWARD credibility/integrity event with its own timing → potential orthogonality.
  - NOT carry (cross-asset roll-yield), NOT momentum (so NOT turnover-walled like the RED
    network/co-coverage diffusion family — this is a discrete event with a multi-month hold).
  - LIMITS-TO-ARB: restating firms skew small / distressed / hard-to-borrow → exactly the
    edge institutions can't scale (the doctrine's "C" frontier) and a solo might harvest.

DATA: Audit Analytics audit.f39_restatement_filings (file_date = announcement; change_cum_
net_income / change_cum_eps = direction+magnitude) ⨝ audit.wrds_lookup_edgar_company_block
(company_fkey → cusip_number) → permno (crsp.stocknames ncusip, cached) → CRSP monthly ret.

HONEST PRIORS (recorded before running): the strong income-decreasing subset is THIN
(~906 since 2013, ~6/mo) and the sample is POST-SOX/Reg-FD (2013-2026) — the regime where
this drift most plausibly decayed (its strong era was pre-2005). Base rate LOW, like the
~20 RED accessible-data candidates. The genuine hope is the limits-to-arb (small/distressed)
short leg. Verdict by the SAME gate: gross + NET (cost/turnover) Sharpe, residual alpha vs
FF5+UMD(+PEAD), corr vs D_PEAD, corrected deflated SR. 0-LLM, deterministic.

VERDICT (2026-05-22, audit.f39 2013-2026): **RED — well-powered null, post-SOX decay.**
- income-decreasing subset (~7 names/mo, 35 mo): gross Sharpe 0.08 (t 0.14), NET -0.00,
  residual alpha vs FF5+UMD +7.0%/yr but t=0.83 (n.s., noise on 35 mo), corr +0.58 w/ D_PEAD
  (NOT orthogonal — same accounting-info family), deflated SR 0.033.
- all-restatements (~98 names/mo, 128 mo): gross Sharpe 0.31 (t 1.01 n.s.), NET 0.19,
  residual alpha +1.9%/yr t=1.16 (n.s.), corr +0.11 (more orthogonal but NO alpha), deflSR 0.162.
- Triple failure: gross ~zero/insignificant, net killed by cost, residual alpha t << 3.0.
  Strong (income-decreasing) subset is thin AND corr 0.58 w/ D_PEAD; broad set has no alpha.
  Did NOT p-hack horizons/subsets for a significant slice.
LESSON: post-SOX/Reg-FD (2013-2026) restatement drift has decayed/arbitraged like the ~20
other accessible-data candidates; even the limits-to-arb small/distressed short-leg hope
does not produce harvestable orthogonal alpha. Joins the graveyard. A genuine NEW alpha needs
NOVEL data (gated alt-data Revelio/consumer-tx — DENIED on this account) or edges we lack.
"""
from __future__ import annotations

import logging
import os
import socket
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REST = "data/cache/_restatement_events.parquet"
_STOCKNAMES = "data/cache/_stocknames_ncusip.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"
_FF = "data/cache/ff_factors_weekly.parquet"
_DPEAD = "data/cache/_dpead_recon_base.parquet"
RT_BPS = 30.0  # single-name round-trip cost (bps), matches the equity sleeves


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine("postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
                         connect_args={"sslmode": "require", "connect_timeout": 30})


def fetch_restatements(force: bool = False) -> pd.DataFrame:
    """ONE WRDS connection: restatement filings ⨝ EDGAR cusip link. Cached + resumable.
    Returns (cusip8, file_date, chg_ni, chg_eps)."""
    if os.path.exists(_REST) and not force:
        return pd.read_parquet(_REST)
    for _ in range(6):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(3)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        sql = ("select substring(l.cusip_number from 1 for 8) as cusip8, "
               "       f.file_date, f.change_cum_net_income as chg_ni, f.change_cum_eps as chg_eps "
               "from audit.f39_restatement_filings f "
               "join audit.wrds_lookup_edgar_company_block l on f.company_fkey = l.company_fkey "
               "where f.file_date >= '2013-01-01' and l.cusip_number is not null")
        df = pd.read_sql(text(sql), eng)
    finally:
        eng.dispose()
    df["file_date"] = pd.to_datetime(df["file_date"])
    df = df.dropna(subset=["cusip8"]).drop_duplicates(["cusip8", "file_date"])
    df.to_parquet(_REST, index=False)
    logger.info("restatements: %d events, %d firms, %s..%s; income-decreasing=%d",
                len(df), df["cusip8"].nunique(), df["file_date"].min().date(),
                df["file_date"].max().date(), int((df["chg_ni"] < 0).sum()))
    return df


def _monthly_returns():
    ret = pd.read_parquet(_RET); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    return mret.where(daily.resample("ME").count() > 5)


def _cusip_to_permno(cusips) -> pd.DataFrame:
    sn = pd.read_parquet(_STOCKNAMES).rename(columns={"ncusip": "cusip8"})
    return sn[sn["cusip8"].isin(set(cusips))][["cusip8", "permno"]].drop_duplicates("cusip8")


def build_restatement_panel(income_decreasing_only: bool = True):
    """Per (permno, month): flag = a qualifying restatement was filed in this firm in the
    PRIOR month (causal: signal at month m from filings <= m-end, traded next month)."""
    ev = fetch_restatements()
    if income_decreasing_only:
        ev = ev[ev["chg_ni"] < 0]                        # the credibility-damaging subset
    link = _cusip_to_permno(ev["cusip8"].unique())
    ev = ev.merge(link, on="cusip8", how="inner")
    ev["month"] = ev["file_date"] + pd.offsets.MonthEnd(0)   # month-end, aligns with mret index
    mret = _monthly_returns()
    flagged = ev.groupby("month")["permno"].apply(lambda s: set(s)).to_dict()
    return flagged, mret, ev


def build_sleeve(hold_months: int = 3, income_decreasing_only: bool = True):
    """Short firms with a qualifying restatement in the trailing `hold_months`, vs the
    equal-weight universe (short-leg drift is the documented effect). Monthly. Returns
    (ls_short_minus_mkt, ann_turnover, n_per_month)."""
    flagged, mret, ev = build_restatement_panel(income_decreasing_only)
    months = list(mret.index)
    rows, ent, prevS, ns = [], [], set(), []
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        # active short book = union of firms flagged in [m-hold+1 .. m]
        win = [mm for mm in months[max(0, i - hold_months + 1):i + 1]]
        short = set().union(*[flagged.get(mm, set()) for mm in win]) if win else set()
        short = {p for p in short if p in mret.columns}
        nr = mret.loc[nxt]
        rs = nr.reindex(list(short)).dropna()
        rm = nr.dropna()
        if len(rs) < 5:
            continue
        # signal = SHORT the restaters: return = mkt - restaters (positive if they underperform)
        rows.append((nxt, float(rm.mean() - rs.mean())))
        ns.append(len(rs))
        ent.append(len(short - prevS) / max(len(short), 1)); prevS = short
    ls = pd.Series(dict(rows)).sort_index().rename("restate_ls")
    return ls, (float(np.mean(ent) * 12) if ent else float("nan")), (float(np.mean(ns)) if ns else 0.0)


def _ann(r):
    r = r.dropna(); v = r.std() * np.sqrt(12)
    return dict(ann=r.mean() * 12, vol=v, sharpe=r.mean() * 12 / v if v > 0 else np.nan,
                t=r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else np.nan, n=len(r))


def _residual_alpha(ls: pd.Series):
    """Residual alpha vs FF5+UMD (monthly). Returns (alpha_ann, t_alpha)."""
    ff = pd.read_parquet(_FF); ff.index = pd.to_datetime(ff.index)
    cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "Mom"] if c in ff.columns]
    ffm = (1 + ff[cols]).resample("ME").prod() - 1
    J = pd.concat([ls.rename("y"), ffm], axis=1).dropna()
    if len(J) < 24:
        return np.nan, np.nan
    import numpy as _np
    X = _np.column_stack([_np.ones(len(J))] + [J[c].values for c in cols])
    y = J["y"].values
    beta, *_ = _np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = _np.sqrt((resid @ resid) / (len(J) - X.shape[1]) * _np.linalg.inv(X.T @ X)[0, 0])
    return float(beta[0] * 12), float(beta[0] / se if se > 0 else _np.nan)


def _corr_dpead(ls: pd.Series) -> float:
    try:
        d = pd.read_parquet(_DPEAD).iloc[:, 0]; d.index = pd.to_datetime(d.index)
        dm = ((1 + d.clip(-0.2, 0.2)).resample("ME").prod() - 1)
        J = pd.concat([ls.rename("a"), dm.rename("b")], axis=1).dropna()
        return float(J["a"].corr(J["b"])) if len(J) > 6 else float("nan")
    except Exception:
        return float("nan")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    print("\n" + "=" * 78 + "\nRESTATEMENT-DRIFT — different mechanism (accounting-credibility event)\n" + "=" * 78)
    for label, ido in (("income-decreasing", True), ("all-restatements", False)):
        ls, turn, npm = build_sleeve(hold_months=3, income_decreasing_only=ido)
        net = (ls - turn * RT_BPS / 10000.0 / 12).rename(ls.name + "_net")
        g, n = _ann(ls), _ann(net)
        a_ann, a_t = _residual_alpha(ls)
        corr = _corr_dpead(ls)
        dsr = deflated_sharpe_ratio(ls.dropna().values, n_trials=24, periods_per_year=12)
        print(f"\n[{label}]  n_months {g['n']}  ~{npm:.0f} names/mo  turnover {turn:.1f}x")
        print(f"  GROSS  ann {g['ann']:+.1%}  Sharpe {g['sharpe']:+.2f}  t {g['t']:+.2f}")
        print(f"  NET    ann {n['ann']:+.1%}  Sharpe {n['sharpe']:+.2f}")
        print(f"  residual alpha vs FF5+UMD: {a_ann:+.1%}/yr  t {a_t:+.2f}   (gate: t>=~3.0)")
        print(f"  corr vs D_PEAD: {corr:+.2f}   deflated SR (gross): {dsr.deflated_sr:.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
