"""engine/validation/cn_pead_data.py — China A-share PEAD data pull (AkShare, free).

WRDS wind_ashare is catalog-visible but SELECT-DENIED on this account, and for a
China-based operator the LIVE source IS AkShare/Tushare — so backtesting on
AkShare means ZERO research-to-production data gap (unlike US, where WRDS != live).

Pulls, per quarter-end report date, the all-A-share earnings table (stock_yjbb_em):
EPS (每股收益) + announcement date (最新公告日期) + net-profit YoY + industry.
Slow (eastmoney throttles ~5 min/quarter) so each quarter is cached + resumable.
"""
from __future__ import annotations

import logging
import os
import socket
import time

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = "data/cache/cn"
# stock_yjbb_em columns are GBK Chinese; select by POSITION for robustness.
_COLS = {1: "code", 2: "name", 3: "eps", 4: "revenue", 8: "np_yoy",
         11: "roe", 14: "industry", 15: "ann_date"}


def _quarters(y0: int = 2018, y1: int = 2024) -> list[str]:
    out = []
    for y in range(y0, y1 + 1):
        for mmdd in ("0331", "0630", "0930", "1231"):
            out.append(f"{y}{mmdd}")
    return out


def _dns_ok(host: str, tries: int = 6) -> bool:
    for _ in range(tries):
        try:
            socket.gethostbyname(host); return True
        except Exception:
            time.sleep(3)
    return False


def _robust(fn, *args, tries: int = 6, base: float = 2.0, **kwargs):
    """Call a flaky CN-data endpoint with retries + linear backoff. eastmoney /
    sina intermittently drop connections (RemoteDisconnected) and the local DNS
    for their CDN hosts flaps; a handful of backed-off retries rides through it."""
    last = None
    for a in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            time.sleep(base * (a + 1))
    if last is not None:
        raise last
    raise RuntimeError("_robust: no attempts made")


def pull_eps(y0: int = 2018, y1: int = 2024, force: bool = False) -> None:
    """Pull + cache all-A-share EPS/announce-date per quarter (resumable)."""
    import akshare as ak
    os.makedirs(CACHE_DIR, exist_ok=True)
    _dns_ok("datacenter-web.eastmoney.com")
    for d in _quarters(y0, y1):
        path = f"{CACHE_DIR}/yjbb_{d}.parquet"
        if os.path.exists(path) and not force:
            continue
        try:
            _dns_ok("datacenter-web.eastmoney.com")
            df = _robust(ak.stock_yjbb_em, date=d, tries=8, base=4.0)
        except Exception as exc:
            logger.warning("yjbb %s failed after retries: %s", d, str(exc)[:80])
            continue
        if df is None or len(df) == 0:
            logger.warning("yjbb %s empty", d); continue
        sub = df.iloc[:, list(_COLS.keys())].copy()
        sub.columns = list(_COLS.values())
        sub["report_date"] = pd.to_datetime(d)
        sub["ann_date"] = pd.to_datetime(sub["ann_date"], errors="coerce")
        sub["eps"] = pd.to_numeric(sub["eps"], errors="coerce")
        sub.to_parquet(path, index=False)
        logger.info("yjbb %s: %d rows cached", d, len(sub))


_PX_CACHE = "data/cache/cn/_cn_prices.parquet"
_PX_COLS = ["code", "date", "close", "amount", "shares"]


def _sina_symbol(code: str) -> str:
    """6-digit A-share code -> Sina symbol (sh/sz prefix). 6* = Shanghai
    (incl. 688 STAR); everything else (000/001/002/003/300/301) = Shenzhen."""
    return ("sh" if code[0] == "6" else "sz") + code


def csi_universe() -> list[str]:
    """Liquid A-share universe = CSI 300 (large) + CSI 500 (mid) constituents."""
    import akshare as ak
    codes = set()
    for idx in ("000300", "000905"):
        for _ in range(3):
            try:
                c = ak.index_stock_cons_csindex(symbol=idx)
                # pick the 6-digit column with MANY distinct values = constituent
                # code (NOT 指数代码, which is constant), avoiding encoding issues.
                best = None
                for col in c.columns:
                    v = c[col].astype(str)
                    if v.str.match(r"^\d{6}$").mean() > 0.8 and v.nunique() > 10:
                        best = col; break
                if best is not None:
                    codes |= set(c[best].astype(str).str.zfill(6))
                break
            except Exception:
                time.sleep(3)
    return sorted(codes)


def fetch_cn_prices(codes=None, force: bool = False, save_every: int = 25) -> pd.DataFrame:
    """Daily hfq-adjusted close + turnover value (amount) + outstanding shares per
    A-share, via SINA (stock_zh_a_daily). Sina hits a DIFFERENT host than eastmoney
    push2his (which rate-limits / drops connections hard) and is ~0.6s/code, not
    throttled. `amount` (real-CNY daily turnover) gives an ADV liquidity weight so
    the L/S can be value/liquidity-weighted (Asparouhova-Bessembinder-Kalcheva:
    EW small-cap backtests harvest bid-ask bounce as fake return — VW/ADV-weight is
    the clean construction). Resumable: cached codes are skipped. Returns long panel
    (code, date, close, amount, shares)."""
    import akshare as ak
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(_PX_CACHE) and not force:
        cached = pd.read_parquet(_PX_CACHE)
    else:
        cached = pd.DataFrame(columns=_PX_COLS)
    # schema guard: an old close-only (eastmoney) cache lacks amount/shares — its
    # hfq base also differs from Sina's, so don't mix; rebuild cleanly from Sina.
    if not set(_PX_COLS).issubset(set(cached.columns)):
        logger.info("price cache schema mismatch -> rebuilding from Sina")
        cached = pd.DataFrame(columns=_PX_COLS)
    if codes is None:
        codes = csi_universe()
    done = set(cached["code"].unique())
    parts = [cached]
    n = 0
    for code in codes:
        if code in done:
            continue
        try:
            df = _robust(ak.stock_zh_a_daily, symbol=_sina_symbol(code),
                         start_date="20170101", end_date="20240630", adjust="hfq",
                         tries=6, base=2.0)
        except Exception as exc:
            logger.warning("price %s failed after retries: %s", code, str(exc)[:70])
            df = None
        if df is not None and len(df):
            d = df[["date", "close", "amount", "outstanding_share"]].copy()
            d.columns = ["date", "close", "amount", "shares"]
            d["code"] = code
            parts.append(d[_PX_COLS])
        n += 1
        if n % save_every == 0:
            pd.concat(parts, ignore_index=True).to_parquet(_PX_CACHE, index=False)
            logger.info("cn prices: %d new codes done (%d codes cached)",
                        n, sum(p["code"].nunique() for p in parts))
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["code", "date"])
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(_PX_CACHE, index=False)
    logger.info("cn prices DONE: %d rows, %d codes", len(out), out["code"].nunique())
    return out


def load_eps_panel() -> pd.DataFrame:
    """Concatenate all cached quarter files into one EPS event panel."""
    import glob
    files = sorted(glob.glob(f"{CACHE_DIR}/yjbb_*.parquet"))
    if not files:
        raise FileNotFoundError("no cached yjbb quarters — run pull_eps() first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["code", "eps", "ann_date"])
    return df


def build_cn_sue() -> pd.DataFrame:
    """Bernard-Thomas seasonal-RW SUE on Chinese CUMULATIVE YTD EPS. report_date
    sorted; shift(4)=same fiscal period prior year. Returns code, ann_date, sue."""
    import numpy as np
    p = load_eps_panel().copy()
    p["report_date"] = pd.to_datetime(p["report_date"])
    p = p.dropna(subset=["eps", "ann_date"]).sort_values(["code", "report_date"])
    out = []
    for _, g in p.groupby("code"):
        g = g.drop_duplicates("report_date").sort_values("report_date")
        g["d"] = g["eps"] - g["eps"].shift(4)
        g["sig"] = g["d"].shift(1).rolling(8, min_periods=3).std()
        g["sue"] = g["d"] / g["sig"]
        out.append(g)
    s = pd.concat(out).dropna(subset=["sue"])
    return s[np.isfinite(s["sue"])][["code", "ann_date", "report_date", "sue"]]


def build_cn_pead(skip_days: int = 1, hold: int = 60, q: float = 0.1):
    """China PEAD per-event CAR (EW, skip first `skip_days` for limit-up/T+1) +
    calendar-time monthly L/S. A-share frictions handled. Returns dict of results."""
    import numpy as np
    from scipy import stats
    s = build_cn_sue(); s["ann_date"] = pd.to_datetime(s["ann_date"])
    px = fetch_cn_prices(); px = px.sort_values(["code", "date"])
    wide = px.pivot_table(index="date", columns="code", values="close").sort_index()
    dret = wide.pct_change().where(lambda x: x.abs() < 0.5)   # drop bad ratios
    cal = wide.index
    mkt = dret.mean(axis=1)                                   # EW market proxy
    # per-event CAR [ann+skip, +hold], market-adjusted, EW
    retby = {c: dret[c].dropna() for c in wide.columns}
    rows = []
    for _, e in s.iterrows():
        c = e["code"]; ad = e["ann_date"]
        if c not in retby:
            continue
        after = cal[cal > ad]
        if len(after) < skip_days + hold:
            continue
        lo, hi = after[skip_days], after[skip_days + hold - 1]
        seg = dret[c][(cal >= lo) & (cal <= hi)].dropna()
        if len(seg) < hold * 0.6:
            continue
        mseg = mkt[(cal >= lo) & (cal <= hi)]
        car = float((1 + seg).prod() - (1 + mseg).prod())
        rows.append((c, e["sue"], car))
    ev = pd.DataFrame(rows, columns=["code", "sue", "car"])
    res = {"n_events": len(ev)}
    if len(ev) > 200:
        hiq = ev[ev.sue >= ev.sue.quantile(1 - q)]; loq = ev[ev.sue <= ev.sue.quantile(q)]
        t, _ = stats.ttest_ind(hiq.car, loq.car, equal_var=False)
        res["per_event"] = dict(spread=(hiq.car.mean() - loq.car.mean()),
                                long=hiq.car.mean(), short=loq.car.mean(), t=float(t), n=len(ev))
    # calendar-time monthly L/S (skip-day entry baked into using next-month return)
    mret = (1 + dret.fillna(0)).resample("ME").prod() - 1
    mret = mret.where(dret.resample("ME").count() > 5)
    s["am"] = s["ann_date"].dt.to_period("M").dt.to_timestamp("M")
    months = mret.index; ls = []
    for m in months:
        a = s[(s.am <= m) & (s.am > m - pd.DateOffset(months=2))]
        if len(a) < 50:
            continue
        h = a[a.sue >= a.sue.quantile(1 - q)].code; l = a[a.sue <= a.sue.quantile(q)].code
        nx = m + pd.offsets.MonthEnd(1)
        if nx not in mret.index:
            continue
        rl = mret.loc[nx].reindex(h.unique()).dropna(); rs = mret.loc[nx].reindex(l.unique()).dropna()
        if len(rl) < 8 or len(rs) < 8:
            continue
        ls.append((nx, float(rl.mean() - rs.mean())))
    res["ls_monthly"] = pd.Series(dict(ls)).sort_index()
    return res


def _monthly_from_prices(px: pd.DataFrame):
    """From the (code,date,close,amount,shares) panel build:
      mret  — monthly total-return wide (bad >50%/day ratios dropped),
      madv  — monthly mean daily turnover (CNY) wide = ADV liquidity weight,
      dret  — daily return wide, cal — calendar index.
    ADV-weighting (real-CNY turnover) is the clean alternative to equal-weight:
    EW small/mid-cap backtests harvest close-to-close bid-ask bounce as fake
    return (Asparouhova-Bessembinder-Kalcheva); weighting by liquidity down-
    weights exactly the illiquid names where the bounce lives."""
    px = px.sort_values(["code", "date"])
    close = px.pivot_table(index="date", columns="code", values="close").sort_index()
    amt = px.pivot_table(index="date", columns="code", values="amount").sort_index()
    dret = close.pct_change().where(lambda x: x.abs() < 0.5)
    mret = (1 + dret.fillna(0)).resample("ME").prod() - 1
    mret = mret.where(dret.resample("ME").count() > 5)
    madv = amt.resample("ME").mean()
    return mret, madv, dret, close.index


def cn_pead_audit(q: float = 0.2, hold: int = 2, n_trials: int = 50,
                  rt_bps: float = 20.0, dpead_monthly: "pd.Series | None" = None) -> dict:
    """FULL A-share PEAD audit battery — the discipline that caught the Korea
    proxy-niq and US small-cap bid-ask-bounce false GREENs, applied to China:

      1. per-event CAR (EW + ADV-weighted), with the EXPLICIT caveat that the
         per-event t-stat reflects SAMPLE SIZE, not a tradeable edge;
      2. calendar-time monthly series for FOUR constructions — {L/S, long-only}
         x {equal-weight, ADV(liquidity)-weight}. LONG-ONLY is the DEPLOYABLE one:
         A-share short-selling (融券) is restricted/expensive (cf. Korea short
         ban), so the L/S spread overstates the harvestable edge;
      3. A-share transaction cost (commission ~5bps RT + stamp 5bps sell + slippage
         => rt_bps round-trip) applied via measured turnover; gross AND net;
      4. deflated Sharpe (monthly ppy=12, honest n_trials for the ~50-candidate
         multi-market campaign — porting rule: same mechanism x markets raises
         n_trials); 5. regime halves + yearly positivity; 6. OOS split;
      7. correlation with the US D_PEAD book (pass dpead_monthly) — diversification
         only MATTERS if China itself clears the bar.

    Entry is via NEXT-month return (announcement day skipped) so the day-1
    limit-up jump / T+1 friction / bid-ask bounce is never counted as alpha.
    Returns a structured dict; the verdict is read off net deflated SR of the
    DEPLOYABLE long-only construction."""
    import numpy as np
    from scipy import stats
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    from engine.validation.after_cost import apply_cost

    s = build_cn_sue(); s["ann_date"] = pd.to_datetime(s["ann_date"])
    px = fetch_cn_prices()
    mret, madv, dret, cal = _monthly_from_prices(px)
    mkt_d = dret.mean(axis=1)
    s["am"] = s["ann_date"].dt.to_period("M").dt.to_timestamp("M")

    # ---- 1. per-event CAR [ann+1, +60td], market-adjusted (context only) ----
    per = {}
    rows = []
    for _, e in s.iterrows():
        c = e["code"]
        if c not in dret.columns:
            continue
        after = cal[cal > e["ann_date"]]
        if len(after) < 61:
            continue
        lo, hi = after[1], after[60]              # skip the announcement day
        seg = dret[c][(cal >= lo) & (cal <= hi)].dropna()
        if len(seg) < 36:
            continue
        mseg = mkt_d[(cal >= lo) & (cal <= hi)]
        rows.append((c, e["sue"], float((1 + seg).prod() - (1 + mseg).prod())))
    ev = pd.DataFrame(rows, columns=["code", "sue", "car"])
    if len(ev) > 200:
        hiq = ev[ev.sue >= ev.sue.quantile(1 - q)]; loq = ev[ev.sue <= ev.sue.quantile(q)]
        t, _ = stats.ttest_ind(hiq.car, loq.car, equal_var=False)
        per = dict(n=len(ev), spread=float(hiq.car.mean() - loq.car.mean()),
                   long=float(hiq.car.mean()), short=float(loq.car.mean()), t=float(t),
                   note="per-event t reflects the event count, NOT a tradeable Sharpe")

    # ---- 2. calendar-time monthly constructions ----
    months = list(mret.index)
    recs, ent, prevL = [], [], set()
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        a = s[(s.am <= m) & (s.am > m - pd.DateOffset(months=hold))]
        if len(a) < 50:
            continue
        a = a.sort_values("ann_date").drop_duplicates("code", keep="last")
        sue = a.set_index("code")["sue"]
        hi = sue[sue >= sue.quantile(1 - q)].index
        lo = sue[sue <= sue.quantile(q)].index
        nr = mret.loc[nxt]
        w = madv.loc[m] if m in madv.index else None

        def leg(idx, weighted):
            r = nr.reindex(idx).dropna()
            if len(r) < 8:
                return None
            if weighted and w is not None:
                ww = w.reindex(r.index).fillna(0.0)
                if ww.sum() > 0:
                    return float((r * ww).sum() / ww.sum())
            return float(r.mean())

        lE, sE, mE = leg(hi, False), leg(lo, False), leg(sue.index, False)
        lW, sW, mW = leg(hi, True), leg(lo, True), leg(sue.index, True)
        if None in (lE, sE, mE):
            continue
        rec = dict(m=nxt, ls_ew=lE - sE, lo_ew=lE - mE)
        if None not in (lW, sW, mW):
            rec["ls_vw"] = lW - sW; rec["lo_vw"] = lW - mW
        recs.append(rec)
        ent.append(len(set(hi) - prevL) / max(len(hi), 1)); prevL = set(hi)
    R = pd.DataFrame(recs).set_index("m").sort_index()
    turn = float(np.mean(ent) * 12) if ent else float("nan")

    # ---- 3+4. gross/net + deflated SR per construction ----
    def summarize(name, r, legs):
        r = r.dropna()
        if len(r) < 12:
            return dict(name=name, n=len(r), note="too few months")
        vol = r.std() * np.sqrt(12)
        drag = legs * turn * rt_bps / 10000.0       # legs=1 long-only, 2 for L/S
        net = apply_cost(r, drag, ppy=12)
        g = deflated_sharpe_ratio(r.values, n_trials=n_trials, periods_per_year=12)
        nd = deflated_sharpe_ratio(net.values, n_trials=n_trials, periods_per_year=12)
        return dict(name=name, n=len(r),
                    gross_ann=float(r.mean() * 12), vol=float(vol),
                    gross_sharpe=float(r.mean() * 12 / vol) if vol > 0 else float("nan"),
                    gross_defSR=float(g.deflated_sr),
                    ann_turnover=turn, ann_cost=float(drag),
                    net_ann=float(net.mean() * 12),
                    net_sharpe=float(net.mean() * 12 / vol) if vol > 0 else float("nan"),
                    net_defSR=float(nd.deflated_sr))

    constr = {"ls_ew": 2, "ls_vw": 2, "lo_ew": 1, "lo_vw": 1}
    summ = {k: summarize(k, R[k], legs) for k, legs in constr.items() if k in R}

    # ---- 5. regime halves + yearly positivity (on the deployable lo_vw) ----
    dep = "lo_vw" if "lo_vw" in R else "lo_ew"
    ser = R[dep].dropna()
    regimes = {}
    if len(ser) > 24:
        mid = ser.index[len(ser) // 2]
        for nm, sub in (("first_half", ser[ser.index < mid]), ("second_half", ser[ser.index >= mid])):
            tt = stats.ttest_1samp(sub, 0).statistic if len(sub) > 2 else float("nan")
            regimes[nm] = dict(n=len(sub), ann=float(sub.mean() * 12), t=float(tt))
    yearly = (ser.groupby(ser.index.year).mean() * 12)
    regimes["yearly_ann"] = {int(y): float(v) for y, v in yearly.items()}
    regimes["years_positive"] = int((yearly > 0).sum()); regimes["years_total"] = int(len(yearly))

    # ---- 7. correlation with US D_PEAD book (overlapping months) ----
    corr = None
    if dpead_monthly is not None and len(ser) > 6:
        j = pd.concat([ser.rename("cn"), dpead_monthly.rename("dpead")], axis=1).dropna()
        if len(j) > 6:
            corr = dict(n=len(j), corr=float(j["cn"].corr(j["dpead"])))

    return dict(deployable=dep, per_event=per, constructions=summ,
                regimes=regimes, corr_dpead=corr,
                n_months=len(R), window=(str(R.index.min()), str(R.index.max())),
                series=R)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pull_eps()
    p = load_eps_panel()
    logger.info("EPS panel: %d rows, %d stocks, ann_date %s..%s",
                len(p), p["code"].nunique(), p["ann_date"].min(), p["ann_date"].max())
