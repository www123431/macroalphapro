"""engine/validation/intl_value.py — track A+: locally-dominant anomaly per market.

Insight (not just "PEAD everywhere"): each market's investor base makes a
DIFFERENT mechanism dominant, so doing the LOCALLY-strongest anomaly gives
geographic AND mechanism diversification at once.
  - Japan: VALUE is the historic edge; momentum is famously absent (Asness
    "Value and Momentum Everywhere"); + the 2023 TSE "PBR<1" governance-reform
    catalyst. VALUE is a DIFFERENT mechanism than the US/China PEAD underreaction.
  - China A / Korea: retail-driven PEAD + short-term reversal (built elsewhere).

Data: Compustat Global on WRDS (comp.g_funda fundamentals + comp.g_secd daily
prices) — SELECT-OK with huge JP/KR coverage (Worldscope is denied, Compustat
Global covers it). I/B/E/S international (ibes.actu_epsint) available for a PEAD
extension. wind_ashare (China) is SELECT-denied, hence China uses AkShare.

Refinements vs the US port (per the senior-review brief):
  - currency-neutral L/S (long & short both local-ccy → FX cancels);
  - total return from prccd/ajexdi*trfd (split+dividend correct);
  - Japan FYE is March → fundamentals lagged >=6 months (point-in-time);
  - liquid universe only (top-N by mcap), mirroring US top-1500 discipline;
  - honest multiple-testing (same mechanism across 3 markets => higher n_trials).

VERDICT — Japan VALUE (2026-05-21, top-800 JPY, 2012-2024): RED full-sample,
regime-dependent recent strength (NOT a clean deployable alpha):
  - plain B/M L/S +4%/yr but ENTIRELY market beta (beta 0.26, JP-market-adjusted
    alpha t=-0.05); E/P +1.2%/yr t=0.48. Full-sample gate RED (net deflSR 0.26).
    The 2010s were a global value winter; Japan no exception.
  - GOVERNANCE-REFORM hypothesis (2023 TSE "PBR<1"): pre-2023 dead (B/M t=0.31,
    E/P t=-0.38) -> post-2023 STRONG (B/M +24%/yr t=2.97, E/P +17%/yr t=3.76),
    same pattern in BOTH metrics (not a single-metric fluke).
  - BUT honest caveats kill the "deploy it" read: (a) post-2023 is only 18 months
    (t=2.97 there is suggestive, not robust — same small-sample risk as the
    Lazy-Prices false YELLOW); (b) value actually revived from 2021 (2021 +12.6%,
    2022 +19%), BEFORE the 2023 reform -> this is more likely the GLOBAL value
    cycle (post-2020-growth-peak rebound) than a Japan-reform alpha; the data
    cannot separate the two. => Japan value is a REGIME BET (dead 2010s, alive
    2021+), not a persistent in-sample alpha. corr -0.23 w/ US D_PEAD (regime
    diversifier value), but a regime-dependent tilt is not a clean alpha. NOT
    deployed; did not over-fit the 18-month window into a "win".
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FUNDA_CACHE = "data/cache/intl/g_funda_{cc}.parquet"
_SECD_CACHE = "data/cache/intl/g_secd_{cc}.parquet"


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine(
        "postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
        connect_args={"sslmode": "require"})


def fetch_country(cc: str = "JPN", top_n: int = 800, force: bool = False):
    """ONE WRDS connection: pull annual fundamentals (value fields) for country
    `cc`, pick the top_n liquid names by total assets, then pull their daily
    prices. Cached. Returns (funda, secd)."""
    import socket
    import time
    fpath = _FUNDA_CACHE.format(cc=cc); spath = _SECD_CACHE.format(cc=cc)
    if os.path.exists(fpath) and os.path.exists(spath) and not force:
        return pd.read_parquet(fpath), pd.read_parquet(spath)
    os.makedirs("data/cache/intl", exist_ok=True)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        # Compustat GLOBAL uses different mnemonics than NA — introspect columns
        # and pick the right value fields (book equity / earnings / sales).
        cols = set(pd.read_sql(text(
            "select column_name from information_schema.columns "
            "where table_schema='comp' and table_name='g_funda'"), eng)["column_name"])

        def pick(*cands):
            for c in cands:
                if c in cols:
                    return c
            return None
        f_book = pick("ceq", "seq", "teq")          # book/common equity
        f_earn = pick("ni", "ib", "nicon", "ibcom")  # earnings
        f_sale = pick("sale", "revt")
        desired = ["gvkey", "datadate", "fyr", "indfmt", "datafmt", "consol", "popsrc",
                   "at", "curcd", f_book, f_earn, f_sale]
        # intersect with actual columns so any missing Global field is auto-dropped
        sel = [s for s in dict.fromkeys(desired) if s and s in cols]
        logger.info("g_funda value fields: book=%s earn=%s sale=%s", f_book, f_earn, f_sale)
        fq = ("select %s from comp.g_funda where fic='%s' "
              "and datadate between '2009-01-01' and '2024-06-30' "
              "and consol='C' and indfmt='INDL' and datafmt='HIST_STD' and popsrc='I'"
              % (", ".join(sel), cc))
        funda = pd.read_sql(text(fq), eng)
        funda = funda.rename(columns={f_book: "book", f_earn: "earn", f_sale: "sale"})
        funda["datadate"] = pd.to_datetime(funda["datadate"])
        # liquid universe = top_n gvkeys by latest total assets
        latest_at = (funda.dropna(subset=["at"]).sort_values("datadate")
                     .groupby("gvkey")["at"].last())
        univ = latest_at.nlargest(top_n).index.tolist()
        gv_in = ",".join("'%s'" % g for g in univ)
        # daily prices for the universe (total-return inputs + shares)
        sq = ("select gvkey, iid, datadate, prccd, ajexdi, trfd, cshoc, curcdd "
              "from comp.g_secd where fic='%s' and gvkey in (%s) "
              "and datadate between '2012-06-01' and '2024-06-30' "
              "and prccd is not null" % (cc, gv_in))
        secd = pd.read_sql(text(sq), eng)
    finally:
        eng.dispose()
    secd["datadate"] = pd.to_datetime(secd["datadate"])
    funda.to_parquet(fpath, index=False)
    secd.to_parquet(spath, index=False)
    logger.info("%s: funda %d rows / %d gvkeys; secd %d rows / %d gvkeys",
                cc, len(funda), funda["gvkey"].nunique(), len(secd), secd["gvkey"].nunique())
    return funda, secd


def fetch_quarterly(cc: str = "JPN", force: bool = False) -> pd.DataFrame:
    """ONE WRDS connection: Compustat Global quarterly earnings + announcement
    date (rdq) for country `cc` — for a PEAD test (the proven low-turnover
    mechanism). Column-adaptive (Global mnemonics differ). Checks rdq coverage
    (often sparse internationally; if so, fall back to ibes.actu_epsint)."""
    import socket
    import time
    qpath = f"data/cache/intl/g_fundq_{cc}.parquet"
    if os.path.exists(qpath) and not force:
        return pd.read_parquet(qpath)
    os.makedirs("data/cache/intl", exist_ok=True)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        cols = set(pd.read_sql(text(
            "select column_name from information_schema.columns "
            "where table_schema='comp' and table_name='g_fundq'"), eng)["column_name"])

        def pick(*cands):
            for x in cands:
                if x in cols:
                    return x
            return None
        f_eps = pick("epspxq", "epspxq", "epspiq", "epsf12")   # EPS quarterly
        f_ni = pick("niq", "ibq", "ni")
        f_sale = pick("saleq", "revtq")
        want = ["gvkey", "datadate", "fyearq", "fqtr", "rdq", "curcdq",
                f_eps, f_ni, f_sale]
        sel = [c for c in dict.fromkeys(want) if c and c in cols]
        logger.info("g_fundq fields: eps=%s ni=%s sale=%s rdq_in_cols=%s",
                    f_eps, f_ni, f_sale, "rdq" in cols)
        q = ("select %s from comp.g_fundq where fic='%s' "
             "and datadate between '2011-01-01' and '2024-06-30' "
             "and consol='C' and indfmt='INDL' and datafmt='HIST_STD' and popsrc='I'"
             % (", ".join(sel), cc))
        fq = pd.read_sql(text(q), eng)
    finally:
        eng.dispose()
    fq["datadate"] = pd.to_datetime(fq["datadate"])
    if "rdq" in fq.columns:
        fq["rdq"] = pd.to_datetime(fq["rdq"], errors="coerce")
    fq = fq.rename(columns={f_eps: "epsq", f_ni: "niq", f_sale: "saleq"})
    fq.to_parquet(qpath, index=False)
    rdq_cov = fq["rdq"].notna().mean() if "rdq" in fq.columns else 0.0
    logger.info("%s g_fundq: %d rows / %d gvkeys; rdq coverage %.0f%%",
                cc, len(fq), fq["gvkey"].nunique(), rdq_cov * 100)
    return fq


def fetch_ibes_intl(cc: str = "KOR", curr: str = "KRW", force: bool = False):
    """ONE WRDS connection: I/B/E/S international quarterly EPS actuals + EXACT
    announce dates (anndats) for currency `curr`, + the g_security ibtic<->gvkey
    link (international link key — cusip is NULL abroad, ibtic is the bridge).
    Returns (actuals, link). Cached."""
    import socket
    import time
    apath = f"data/cache/intl/ibes_act_{cc}.parquet"
    lpath = f"data/cache/intl/ibes_link_{cc}.parquet"
    if os.path.exists(apath) and os.path.exists(lpath) and not force:
        return pd.read_parquet(apath), pd.read_parquet(lpath)
    os.makedirs("data/cache/intl", exist_ok=True)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        aq = ("select ticker, pends, anndats, value, curr_act "
              "from ibes.actu_epsint where curr_act='%s' and measure='EPS' "
              "and pdicity='QTR' and anndats is not null and value is not null "
              "and anndats >= '2010-01-01'" % curr)
        act = pd.read_sql(text(aq), eng)
        lq = ("select gvkey, iid, ibtic, isin from comp.g_security "
              "where excntry='%s' and ibtic is not null" % cc)
        link = pd.read_sql(text(lq), eng)
    finally:
        eng.dispose()
    act["anndats"] = pd.to_datetime(act["anndats"])
    act["pends"] = pd.to_datetime(act["pends"])
    act.to_parquet(apath, index=False); link.to_parquet(lpath, index=False)
    logger.info("%s I/B/E/S: %d actuals (%d tickers), %d ibtic links",
                cc, len(act), act["ticker"].nunique(), link["ibtic"].nunique())
    return act, link


def build_pead_ibes(cc: str = "KOR", local_ccy: str = "KRW", hold: int = 2, q: float = 0.2):
    """Korea PEAD with EXACT I/B/E/S announce dates + EPS-based SUE (the precise
    confirmation of the proxy-timed niq version). Seasonal-RW SUE on I/B/E/S
    quarterly actual EPS; event = anndats (exact); ibtic->gvkey->price returns.
    Returns (ls, long_only).

    VERDICT (2026-05-21): RED — and this is the TRUSTWORTHY result that overturns
    the proxy-niq GREEN. On 539 tradeable I/B/E/S-covered Korean firms (healthy
    sample, NOT under-powered): L/S gross 7.9%/yr, Sharpe 0.60, net deflSR 0.286,
    t=1.81, with 3 NEGATIVE years (2013 -16%, 2014 -9%, 2022 -10%); long-only net
    deflSR 0.238, t=1.66. corr 0.14-0.17 w/ US D_PEAD. So Korea PEAD on the
    liquid/analyst-covered tradeable set is a WEAK, marginal, RED effect — the
    proxy-niq GREEN was an artifact (niq-level vs EPS-surprise signal + broader
    less-covered universe where PEAD is stronger but not net-harvestable). Same
    arc as US PEAD (A.2): the drift is small-cap-dense, gone in the liquid set.
    LESSON RE-CONFIRMED: proxy-timed / wrong-signal backtests can manufacture a
    false GREEN; only exact-event + correct-signal + tradeable-universe counts.
    """
    act, link = fetch_ibes_intl(cc, local_ccy)
    _, secd = fetch_country(cc)
    mret, _ = _monthly_returns_and_mcap(secd, local_ccy)
    a = act.merge(link[["ibtic", "gvkey"]].drop_duplicates("ibtic"),
                  left_on="ticker", right_on="ibtic", how="inner")
    a["gvkey"] = a["gvkey"].astype(str)
    a = a.sort_values(["gvkey", "pends"])
    out = []
    for _, g in a.groupby("gvkey"):
        g = g.drop_duplicates("pends").sort_values("pends")
        g["d"] = g["value"] - g["value"].shift(4)
        g["sig"] = g["d"].shift(1).rolling(8, min_periods=3).std()
        g["sue"] = g["d"] / g["sig"]
        out.append(g)
    s = pd.concat(out).dropna(subset=["sue"])
    s = s[np.isfinite(s["sue"])]
    s["ev_m"] = s["anndats"].dt.to_period("M").dt.to_timestamp("M")
    months = mret.index
    ls, lo = [], []
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        act_w = s[(s["ev_m"] <= m) & (s["ev_m"] > m - pd.DateOffset(months=hold))]
        if len(act_w) < 40:
            continue
        hi = act_w[act_w["sue"] >= act_w["sue"].quantile(1 - q)]["gvkey"]
        loq = act_w[act_w["sue"] <= act_w["sue"].quantile(q)]["gvkey"]
        nr = mret.loc[nxt]; nr.index = nr.index.astype(str)
        rl = nr.reindex(hi.unique()).dropna(); rs = nr.reindex(loq.unique()).dropna()
        rm = nr.reindex(act_w["gvkey"].astype(str).unique()).dropna()
        if len(rl) < 10 or len(rs) < 10:
            continue
        ls.append((nxt, float(rl.mean() - rs.mean())))
        lo.append((nxt, float(rl.mean() - rm.mean())))
    return (pd.Series(dict(ls)).sort_index().rename(f"{cc}_pead_ibes_ls"),
            pd.Series(dict(lo)).sort_index().rename(f"{cc}_pead_ibes_long"))


def _monthly_returns_and_mcap(secd: pd.DataFrame, local_ccy: str):
    """From g_secd: keep the primary local-ccy issue per gvkey, build total return
    (prccd/ajexdi*trfd), return (monthly_ret_wide, monthly_mcap_wide). FX-neutral
    for L/S since all in local ccy."""
    s = secd[secd["curcdd"] == local_ccy].copy()
    s["datadate"] = pd.to_datetime(s["datadate"])
    for c in ("prccd", "ajexdi", "trfd", "cshoc"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["prccd", "ajexdi"])
    s = s[s["ajexdi"] > 0]
    # pick the primary issue per gvkey = the iid with the most observations
    primary = (s.groupby(["gvkey", "iid"]).size().reset_index(name="n")
               .sort_values("n").groupby("gvkey").last()["iid"])
    s = s.merge(primary.rename("piid"), left_on="gvkey", right_index=True)
    s = s[s["iid"] == s["piid"]]
    s["trf"] = s["trfd"].fillna(1.0)
    s["tri"] = s["prccd"] / s["ajexdi"] * s["trf"]
    s["mcap"] = s["prccd"] * s["cshoc"]
    s = s.sort_values("datadate")
    tri = s.pivot_table(index="datadate", columns="gvkey", values="tri")
    mcap = s.pivot_table(index="datadate", columns="gvkey", values="mcap")
    # daily ret -> guard splits/errors -> monthly compound
    dret = tri.pct_change()
    dret = dret.where(dret.abs() < 1.0)        # drop >100%/day artifacts
    mret = (1 + dret.fillna(0)).resample("ME").prod() - 1
    mret = mret.where(dret.resample("ME").count() > 10)
    mmcap = mcap.resample("ME").last()
    return mret, mmcap


def build_value_ls(cc: str = "JPN", local_ccy: str = "JPY", q: float = 0.2,
                   lag_months: int = 6) -> tuple[pd.Series, pd.Series]:
    """Monthly book-to-market (value) L/S: long high-B/M, short low-B/M, within
    the liquid universe, equal-weight, 1-month hold, local-ccy (FX-neutral).
    B/M = book equity (most recent fiscal year >= lag_months before rebalance,
    PIT — handles Japan's March FYE) / market cap at rebalance.
    Returns (monthly_ls, jp_market_ew) — the latter for a market-adjusted check."""
    funda, secd = fetch_country(cc, top_n=800)
    mret, mmcap = _monthly_returns_and_mcap(secd, local_ccy)

    fb = funda.dropna(subset=["book", "datadate"]).copy()
    fb["datadate"] = pd.to_datetime(fb["datadate"])
    fb["gvkey"] = fb["gvkey"].astype(str)
    fb = fb[fb["book"] > 0].sort_values("datadate")

    months = mret.index
    rows, mkt_rows = [], []
    for i in range(len(months) - 1):
        R = months[i]; nxt = months[i + 1]
        cutoff = R - pd.DateOffset(months=lag_months)
        # PIT book: latest fiscal year ending on/before cutoff, per gvkey
        avail = fb[fb["datadate"] <= cutoff]
        if avail.empty:
            continue
        book = avail.groupby("gvkey")["book"].last()
        mc = mmcap.loc[R].dropna()
        mc.index = mc.index.astype(str)
        common = book.index.intersection(mc.index)
        if len(common) < 50:
            continue
        bm = (book.loc[common] / mc.loc[common]).replace([np.inf, -np.inf], np.nan).dropna()
        hi = bm[bm >= bm.quantile(1 - q)].index
        lo = bm[bm <= bm.quantile(q)].index
        nr = mret.loc[nxt]; nr.index = nr.index.astype(str)
        rl = nr.reindex(hi).dropna(); rs = nr.reindex(lo).dropna()
        if len(rl) < 10 or len(rs) < 10:
            continue
        rows.append((nxt, float(rl.mean() - rs.mean())))
        mkt_rows.append((nxt, float(nr.reindex(bm.index).dropna().mean())))
    ls = pd.Series(dict(rows)).sort_index().rename(f"{cc}_value")
    mkt = pd.Series(dict(mkt_rows)).sort_index().rename(f"{cc}_mkt_ew")
    return ls, mkt


def build_pead_ls(cc: str = "KOR", local_ccy: str = "KRW", seasonal: int = 4,
                  hold: int = 2, q: float = 0.2):
    """PEAD (the PROVEN mechanism) ported to market `cc`. Bernard-Thomas seasonal-
    random-walk SUE on net income (niq, since Compustat Global has no EPS/rdq),
    proxy event = datadate + 45d (Global lacks rdq; +45d ~ the quarterly filing
    deadline → CONSERVATIVE, understates if anything). Monthly calendar-time:
    long top-SUE quintile / short bottom (hold `hold` months). Returns
    (ls_monthly, long_only_vs_market) — the latter is deployable under a short
    ban (Korea). seasonal=4 quarterly; =2 for semi-annual reporters (Japan).

    VERDICT (2026-05-21, KOR top-~400, proxy-timed niq-SUE): looked GREEN
    (net deflSR 0.993 t=5.14, 12/12 yrs positive) BUT this was a FALSE POSITIVE —
    it did NOT survive the precise confirmation (build_pead_ibes below). The
    proxy-niq GREEN is a construction artifact: net-income (niq) seasonal RW
    instead of proper EPS surprise, + datadate+45d proxy instead of exact announce
    dates, + a broader less-analyst-covered universe (where PEAD is stronger but
    less tradeable). The discipline rule HELD: confirm with exact I/B/E/S anndats
    before locking — and confirmation killed it. DO NOT trust the niq/proxy GREEN.
    See build_pead_ibes for the RED rigorous result. (Japan via this route is
    EMPTY: Japan is a SEMI-ANNUAL reporter — fqtr almost all 2&4 — quarterly SUE
    degenerates; needs semi-annual / I/B/E/S.)
    """
    fq = fetch_quarterly(cc)
    _, secd = fetch_country(cc)
    mret, _ = _monthly_returns_and_mcap(secd, local_ccy)
    f = fq.dropna(subset=["niq", "datadate"]).copy()
    f["gvkey"] = f["gvkey"].astype(str)
    f = f.sort_values(["gvkey", "datadate"])
    out = []
    for _, g in f.groupby("gvkey"):
        g = g.drop_duplicates("datadate").sort_values("datadate")
        g["dniq"] = g["niq"] - g["niq"].shift(seasonal)
        g["sig"] = g["dniq"].shift(1).rolling(8, min_periods=3).std()
        g["sue"] = g["dniq"] / g["sig"]
        out.append(g)
    s = pd.concat(out).dropna(subset=["sue"])
    s = s[np.isfinite(s["sue"])]
    s["ev_m"] = (s["datadate"] + pd.Timedelta(days=45)).dt.to_period("M").dt.to_timestamp("M")
    months = mret.index
    ls, lo = [], []
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        a = s[(s["ev_m"] <= m) & (s["ev_m"] > m - pd.DateOffset(months=hold))]
        if len(a) < 40:
            continue
        hi = a[a["sue"] >= a["sue"].quantile(1 - q)]["gvkey"]
        loq = a[a["sue"] <= a["sue"].quantile(q)]["gvkey"]
        nr = mret.loc[nxt]; nr.index = nr.index.astype(str)
        rl = nr.reindex(hi.unique()).dropna(); rs = nr.reindex(loq.unique()).dropna()
        rm = nr.reindex(a["gvkey"].astype(str).unique()).dropna()
        if len(rl) < 10 or len(rs) < 10:
            continue
        ls.append((nxt, float(rl.mean() - rs.mean())))
        lo.append((nxt, float(rl.mean() - rm.mean())))
    return (pd.Series(dict(ls)).sort_index().rename(f"{cc}_pead_ls"),
            pd.Series(dict(lo)).sort_index().rename(f"{cc}_pead_long"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    f, s = fetch_country("JPN", top_n=800)
    print("JAPAN funda:", f.shape, "gvkeys:", f["gvkey"].nunique())
    print("  datadate:", f["datadate"].min(), "..", f["datadate"].max())
    print("JAPAN secd:", s.shape, "gvkeys:", s["gvkey"].nunique(), "iids:", s["iid"].nunique())
    print("  datadate:", s["datadate"].min(), "..", s["datadate"].max())
    print("  currencies:", s["curcdd"].value_counts().to_dict())
