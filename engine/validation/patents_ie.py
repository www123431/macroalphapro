"""engine/validation/patents_ie.py — 2nd-alpha: innovative efficiency (patents / R&D).

Enabled by the full-access WRDS account (${WRDS_USER_2}): wrdsapps_patents (USPTO
patents + citations linked to Compustat gvkey) is now SELECT-able.

WHY this candidate, after the alt-data search kept hitting the cost wall:
every prior signal with real residual alpha (analyst revision t=2.71, supply-
chain t=1.54) was a HIGH-TURNOVER signal that died on transaction cost. Patents
are the opposite — innovative efficiency is a SLOW, near-annual signal. Quintile
membership turns over ~once a year, so the cost drag is structurally tiny. If
the alpha is real it has a real chance of surviving net, unlike the fast signals.

Signal (Hirshleifer-Hsu-Li 2013, "Innovative Efficiency and Stock Returns";
cf. Cohen-Diether-Malloy 2013): firms that convert R&D dollars into patents
efficiently earn higher future returns — the market underweights the value of
innovative output relative to the R&D input it can see on the income statement.

  IE_count = patents granted (trailing) / R&D capital
  IE_cite  = forward citations of those patents / R&D capital   (secondary;
             SEVERELY truncation-biased for recent grant years because citing
             patents are granted later and aren't observable yet — reported
             with that caveat, not used as the headline)

  R&D capital = Chan-Lakonishok-Sougiannis 5-yr accumulation, 20%/yr decay:
      RDC_t = xrd_t + .8 xrd_{t-1} + .6 xrd_{t-2} + .4 xrd_{t-3} + .2 xrd_{t-4}

Portfolio: each June of year Y, among firms with RDC>0 (R&D-active), rank IE on
the most recent fiscal year ending in Y-1 and patents granted through Y-1 (both
public by June Y → no look-ahead). Long top quintile, short bottom, equal-weight,
hold 12 months (annual rebalance). Map gvkey→permno via the CCM link, restricted
to the tradeable top-1500 universe (crsp_hist_daily_ret). Screened through
alpha_factory.gate(); GREEN-only deploys.

VERDICT (2026-05-21, top-1500, account ${WRDS_USER_2}): RED. The low-turnover thesis
HELD — annual rebalance → one-way turnover ~0.28/yr, so the cost drag barely
moved the verdict (gross deflated SR 0.51 → net 0.47). For once cost was NOT the
killer. But the alpha itself isn't there in the tradeable universe:
  - IE_count Q5-Q1: gross +3.84%/yr, gross t=1.51, FF5+UMD residual alpha
    +2.09%/yr t=1.13 (NOT significant), net deflated SR 0.47. corr 0.24 w/ D_PEAD.
  - Quintile gradient is weakly positive but NON-monotonic (Q1 1.45% → Q3 1.93%
    → Q5 1.77%/mo) — direction right, magnitude decayed, no clean spread.
  - IE_cite is dead (t=-0.10): forward-citation truncation (recent grants have
    no observable citing patents yet) destroys the signal, as flagged up front.
  - Decile (d10) weaker than quintile (t=0.58) — thinner tails, more noise.
Same arbitraged-in-large-cap modern-era pattern as the price factors: HHL 2013
found IE with NYSE breakpoints across ALL caps over 1981-2007 (alpha t~3-4); in
top-1500 over 2014-2020 it has decayed to a non-significant +2%/yr. DATA LIMIT:
the WRDS uspatents vintage ends at grant-year 2019, capping the test at 7 annual
cohorts (holds Jul-2014..Jun-2021) — underpowered, but t=1.13 + non-monotone is
fairly conclusive. A small-cap version (where HHL is strongest) would hit the
SAME micro-cap cost wall that killed insider/others, so not pursued.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PANEL = "data/cache/_pead_ts_panel_2014_2023.parquet"
_GVKEY_PERMNO = "data/cache/_sc_gvkey_permno.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"
_PAT_CACHE = "data/cache/_patents_gvkey_year.parquet"
_XRD_CACHE = "data/cache/_compustat_xrd.parquet"


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine(
        "postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
        connect_args={"sslmode": "require"})


def _universe_gvkeys() -> list[int]:
    """gvkeys of the tradeable top-1500 universe = gvkeys whose CCM-linked permno
    appears in the daily-return panel. Drives the WHERE clause so the pulls stay
    small (one task, one connection — connection discipline)."""
    ret = pd.read_parquet(_RET, columns=["permno"])
    permnos = set(ret["permno"].astype(int).unique())
    link = pd.read_parquet(_GVKEY_PERMNO)
    link = link[link["permno"].astype(int).isin(permnos)]
    gv = set(link["gvkey"].astype(int).unique())
    gv |= set(pd.read_parquet(_PANEL)["gvkey"].dropna().astype(int).unique())
    return sorted(gv)


def fetch_patents_and_xrd(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ONE WRDS connection: pull (a) patent counts + forward citations per
    gvkey-grant-year from wrdsapps_patents, and (b) R&D expense xrd (+ at, sale)
    per gvkey-year from comp.funda, both restricted to the universe gvkeys.
    Cached; reused thereafter."""
    if os.path.exists(_PAT_CACHE) and os.path.exists(_XRD_CACHE) and not force:
        return pd.read_parquet(_PAT_CACHE), pd.read_parquet(_XRD_CACHE)

    from sqlalchemy import text
    gvkeys = _universe_gvkeys()
    gv6 = ",".join("'%s'" % str(g).zfill(6) for g in gvkeys)
    eng = _pg_engine()
    try:
        pat_q = (
            "select l.gvkey, extract(year from m.grantdate)::int as gyear, "
            "count(*) as n_pat, "
            "sum(coalesce(m.forward_cites,0))::bigint as fcites "
            "from wrdsapps_patents.uspatents_gvkey_linking l "
            "join wrdsapps_patents.uspatents_meta m on l.patnum = m.patnum "
            "where l.gvkey in (%s) and m.grantdate is not null "
            "group by l.gvkey, gyear" % gv6)
        pat = pd.read_sql(text(pat_q), eng)

        xrd_q = (
            "select gvkey, datadate, fyear, xrd, sale, at, revt "
            "from comp.funda where gvkey in (%s) "
            "and datadate between '2007-01-01' and '2023-12-31' "
            "and indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'" % gv6)
        xrd = pd.read_sql(text(xrd_q), eng)
    finally:
        eng.dispose()

    pat["gvkey"] = pat["gvkey"].astype(int)
    xrd["gvkey"] = xrd["gvkey"].astype(int)
    pat.to_parquet(_PAT_CACHE, index=False)
    xrd.to_parquet(_XRD_CACHE, index=False)
    logger.info("patents: %d gvkey-years (%d gvkeys); xrd: %d firm-years (%d gvkeys)",
                len(pat), pat["gvkey"].nunique(), len(xrd), xrd["gvkey"].nunique())
    return pat, xrd


def _rd_capital(xrd: pd.DataFrame) -> pd.DataFrame:
    """Chan-Lakonishok-Sougiannis 5-yr R&D capital (20%/yr decay) per gvkey-fyear."""
    x = xrd.dropna(subset=["fyear"]).copy()
    x["fyear"] = x["fyear"].astype(int)
    x["xrd"] = pd.to_numeric(x["xrd"], errors="coerce").fillna(0.0).clip(lower=0.0)
    x = (x.sort_values(["gvkey", "fyear"])
           .drop_duplicates(["gvkey", "fyear"], keep="last"))
    out = []
    for gv, g in x.groupby("gvkey"):
        g = g.set_index("fyear")["xrd"]
        # reindex to a continuous year range so the lagged sum is correct
        g = g.reindex(range(g.index.min(), g.index.max() + 1), fill_value=0.0)
        rdc = (g + 0.8 * g.shift(1) + 0.6 * g.shift(2)
               + 0.4 * g.shift(3) + 0.2 * g.shift(4))
        d = pd.DataFrame({"gvkey": gv, "fyear": rdc.index, "rdc": rdc.values,
                          "xrd": g.values})
        out.append(d)
    return pd.concat(out, ignore_index=True)


def build_ie_panel() -> pd.DataFrame:
    """Per gvkey, formation-year Y → IE_count / IE_cite using patents granted
    through Y-1 and R&D capital through fiscal year ending Y-1.

    Returns columns: gvkey, form_year (June-Y formation), ie_count, ie_cite,
    n_pat_ttm, rdc."""
    pat, xrd = fetch_patents_and_xrd()
    rdc = _rd_capital(xrd)

    # patents granted in the trailing TWO calendar years before formation, to
    # smooth the lumpy annual grant counts (HHL use a multi-year output window).
    pat = pat.dropna(subset=["gyear"]).copy()
    pat["gyear"] = pat["gyear"].astype(int)
    patw = pat.pivot_table(index="gvkey", columns="gyear",
                           values=["n_pat", "fcites"], aggfunc="sum").fillna(0.0)

    # The WRDS uspatents vintage ends at grant-year 2019, so the trailing
    # 2-year patent window is COMPLETE only for formation years <= 2020
    # (2020 uses grants 2018+2019). Formations 2021-23 would see ~zero patents
    # → degenerate cross-section, so the testable window is 2014-2020 (7 annual
    # cohorts, holds Jul-2014 .. Jun-2021). Honest data limitation, documented.
    rows = []
    max_grant = int(pat["gyear"].max()) if len(pat) else 2019
    last_form = min(2020, max_grant + 1)
    for form_year in range(2014, last_form + 1):
        # public by June(form_year): patents granted in form_year-1 and -2,
        # R&D capital through fiscal year ending form_year-1.
        rd_y = rdc[rdc["fyear"] == form_year - 1][["gvkey", "rdc"]]
        rd_y = rd_y[rd_y["rdc"] > 0.0]
        for _, rr in rd_y.iterrows():
            gv = int(rr["gvkey"])
            npat = fcite = 0.0
            for yy in (form_year - 1, form_year - 2):
                if ("n_pat", yy) in patw.columns and gv in patw.index:
                    npat += float(patw.loc[gv, ("n_pat", yy)])
                    fcite += float(patw.loc[gv, ("fcites", yy)])
            rows.append((gv, form_year, npat / rr["rdc"], fcite / rr["rdc"],
                         npat, float(rr["rdc"])))
    return pd.DataFrame(rows, columns=["gvkey", "form_year", "ie_count",
                                       "ie_cite", "n_pat_ttm", "rdc"])


def _gvkey_to_permno(gvkeys, asof: pd.Timestamp) -> dict[int, int]:
    """Map gvkey→permno valid at `asof` via the CCM link window."""
    link = pd.read_parquet(_GVKEY_PERMNO).copy()
    link["gvkey"] = link["gvkey"].astype(int)
    link["permno"] = link["permno"].astype(int)
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt"] = pd.to_datetime(link["linkenddt"]).fillna(pd.Timestamp("2030-01-01"))
    v = link[(link["linkdt"] <= asof) & (link["linkenddt"] >= asof)]
    v = v[v["gvkey"].isin(gvkeys)].drop_duplicates("gvkey", keep="last")
    return dict(zip(v["gvkey"], v["permno"]))


def _wide_monthly_returns() -> pd.DataFrame:
    r = pd.read_parquet(_RET); r["date"] = pd.to_datetime(r["date"])
    daily = r.pivot_table(index="date", columns="permno", values="ret").sort_index()
    m = (1.0 + daily.fillna(0.0)).resample("ME").prod() - 1.0
    return m.where(daily.resample("ME").count() > 5)


def build_ie_sleeve(signal_col: str = "ie_count", q: float = 0.2,
                    require_patents: bool = True) -> tuple[pd.Series, float]:
    """Annual June-rebalanced quintile L/S on innovative efficiency. Long top-q
    IE, short bottom-q, equal-weight, held 12 months. Returns (monthly_ls,
    ann_turnover). require_patents: bottom leg must also be R&D-active (RDC>0,
    already enforced) — zero-patent firms legitimately sit in the low-IE short."""
    panel = build_ie_panel()
    mret = _wide_monthly_returns()
    rows, ent, prevL = [], [], set()

    for form_year in sorted(panel["form_year"].unique()):
        fy = panel[panel["form_year"] == form_year].copy()
        if require_patents:
            # keep all R&D firms; low-IE (incl. zero-patent) is a valid short
            fy = fy[fy["rdc"] > 0]
        if len(fy) < 50:
            continue
        hi_cut = fy[signal_col].quantile(1 - q)
        lo_cut = fy[signal_col].quantile(q)
        Lg = fy[fy[signal_col] >= hi_cut]["gvkey"].astype(int).tolist()
        Sg = fy[fy[signal_col] <= lo_cut]["gvkey"].astype(int).tolist()
        asof = pd.Timestamp(form_year, 6, 30)
        g2p = _gvkey_to_permno(set(Lg + Sg), asof)
        L = {g2p[g] for g in Lg if g in g2p}
        S = {g2p[g] for g in Sg if g in g2p}
        if len(L) < 10 or len(S) < 10:
            continue
        # hold July(form_year) .. June(form_year+1)
        hold = [d for d in mret.index
                if pd.Timestamp(form_year, 7, 1) <= d <= pd.Timestamp(form_year + 1, 6, 30)]
        for d in hold:
            nxt = mret.loc[d]
            rl = nxt.reindex(list(L)).dropna(); rs = nxt.reindex(list(S)).dropna()
            if len(rl) < 8 or len(rs) < 8:
                continue
            rows.append((d, float(rl.mean() - rs.mean())))
        ent.append(len(L - prevL) / max(len(L), 1)); prevL = L

    ser = pd.Series(dict(rows)).sort_index().rename(signal_col)
    return ser, float(np.mean(ent)) if ent else float("nan")
