"""engine/validation/dividend_drift.py — 2nd-alpha candidate on a DIFFERENT mechanism:
dividend-change drift (Michaely-Thaler-Womack 1995).

NOT earnings-information underreaction (the D_PEAD / revision / guidance family, which
this session proved is one correlated forward-earnings factor). Dividend changes are
a PAYOUT-SIGNALING / clientele event — a genuinely different channel, so real
orthogonality potential for a true 2nd alpha.

Data: CRSP distributions (crsp.dsedist) — distcd 12xx = ordinary taxable cash
dividends, dclrdt = the declaration/ANNOUNCE date (the information event), divamt =
$/share. Signal = dividend change at the declaration vs the firm's prior regular
dividend of the same code (raise / cut / initiation). Long recent raisers/initiators,
short recent cutters; post-declaration drift. Link permno -> reuse cached CRSP daily
returns (no new price pull). Audited with the corrected deflated-Sharpe methodology
(actual cross-trial variance, not raw grid size).

VERDICT (2026-05-21, _dividend_run.py; 58,125 change events / 1408 permnos / 2009-2024):
**RED — fully arbitraged.** The deployable long-only nets Sharpe -0.09 (gross only
0.21 -> negative after cost); L/S net Sharpe -0.29, PSR-vs-0 0.187 (no real edge);
deflated SR 0.157; grid net Sharpe -0.18..0.24. AND not orthogonal: corr +0.43-0.47
with D_PEAD (firms raise dividends when earnings are good). Michaely-Thaler-Womack
(1995) dividend-change drift is ~zero in 2009-2024 liquid large-caps — 30 years of
post-publication arbitrage. A textbook example of why re-testing PUBLISHED anomalies
is low-EV: they are competed away. The un-arbitraged edge lives in NOVEL DATA
(alt-data acquisition) or NOVEL PROCESSING (text NLP = Line C; network/linked-firm
graph signals) — NOT in another named cross-sectional anomaly on plain WRDS data.
Joins the graveyard.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DIST = "data/cache/_crsp_dsedist_cash.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine(
        "postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
        connect_args={"sslmode": "require"})


def fetch_distributions(force: bool = False) -> pd.DataFrame:
    """ONE WRDS connection: ordinary cash dividends (distcd 1200-1299) with the
    declaration date, 2009-2024, for the permnos in our cached return panel.
    Cached. Returns (permno, distcd, divamt, dclrdt, exdt)."""
    import socket
    import time
    if os.path.exists(_DIST) and not force:
        return pd.read_parquet(_DIST)
    universe = sorted(pd.read_parquet(_RET)["permno"].unique().tolist())
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        pl = ",".join(str(int(p)) for p in universe)
        sql = ("select permno, distcd, divamt, dclrdt, exdt from crsp.dsedist "
               "where distcd between 1200 and 1299 and divamt > 0 "
               "and dclrdt >= '2009-01-01' and permno in (%s)" % pl)
        df = pd.read_sql(text(sql), eng)
    finally:
        eng.dispose()
    df["dclrdt"] = pd.to_datetime(df["dclrdt"])
    df["exdt"] = pd.to_datetime(df["exdt"], errors="coerce")
    df.to_parquet(_DIST, index=False)
    logger.info("cash dividends: %d rows / %d permnos, dclrdt %s..%s",
                len(df), df["permno"].nunique(), df["dclrdt"].min(), df["dclrdt"].max())
    return df


def build_div_change_signal() -> pd.DataFrame:
    """Per declaration event: dividend change vs the firm's prior regular dividend
    of the SAME distcd (same payment frequency), event = dclrdt. Split-artifact
    guard: |change| > 0.6 clipped out (a 2:1 split mechanically halves $/share).
    Returns (permno, dclrdt, div_chg)."""
    d = fetch_distributions().dropna(subset=["dclrdt", "divamt"])
    d = d.sort_values(["permno", "distcd", "dclrdt"])
    # prior dividend of the SAME code (same frequency) -> sequential change
    d["prev"] = d.groupby(["permno", "distcd"])["divamt"].shift(1)
    d = d.dropna(subset=["prev"])
    d = d[d["prev"] > 0]
    d["div_chg"] = d["divamt"] / d["prev"] - 1.0
    # drop split-mechanical artifacts (real raises are a few %..~25%, cuts to ~-100%)
    d = d[d["div_chg"].abs() <= 0.6]
    return d[["permno", "dclrdt", "div_chg"]].dropna()


def _monthly_returns():
    ret = pd.read_parquet(_RET); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    return mret.where(daily.resample("ME").count() > 5)


def build_div_drift_sleeve(hold: int = 3, q: float = 0.2, weight: str = "equal",
                           signal: "pd.DataFrame | None" = None,
                           mret: "pd.DataFrame | None" = None):
    """Calendar-time post-dividend-change drift. Each month, among firms that
    changed their dividend in the last `hold` months, long top-`q` change / short
    bottom-`q`; entry via NEXT month's return (declaration day skipped). Returns
    (ls_monthly, long_only_monthly, ann_turnover). long_only = long − event market."""
    s = build_div_change_signal() if signal is None else signal.copy()
    s["em"] = s["dclrdt"].dt.to_period("M").dt.to_timestamp("M")
    if mret is None:
        mret = _monthly_returns()
    months = list(mret.index)
    ls, lo, ent, prevL = [], [], [], set()
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        a = s[(s.em <= m) & (s.em > m - pd.DateOffset(months=hold))]
        if len(a) < 40:
            continue
        a = a.sort_values("dclrdt").drop_duplicates("permno", keep="last")
        sv = a.set_index("permno")["div_chg"]
        hi = sv[sv >= sv.quantile(1 - q)].index
        loq = sv[sv <= sv.quantile(q)].index
        nr = mret.loc[nxt]
        rl = nr.reindex(hi).dropna(); rs = nr.reindex(loq).dropna(); rm = nr.reindex(sv.index).dropna()
        if len(rl) < 10 or len(rs) < 10:
            continue
        if weight == "mag":
            wl = sv.reindex(rl.index).abs(); wl = wl / wl.sum()
            ws = sv.reindex(rs.index).abs(); ws = ws / ws.sum()
            lr = float((rl * wl).sum()); sr = float((rs * ws).sum())
        else:
            lr = float(rl.mean()); sr = float(rs.mean())
        ls.append((nxt, lr - sr)); lo.append((nxt, lr - float(rm.mean())))
        ent.append(len(set(hi) - prevL) / max(len(hi), 1)); prevL = set(hi)
    return (pd.Series(dict(ls)).sort_index().rename("div_ls"),
            pd.Series(dict(lo)).sort_index().rename("div_long"),
            float(np.mean(ent) * 12) if ent else float("nan"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = build_div_change_signal()
    logger.info("dividend-change events: %d, %d permnos, %s..%s; raises %.0f%% cuts %.0f%%",
                len(s), s["permno"].nunique(), s["dclrdt"].min(), s["dclrdt"].max(),
                (s["div_chg"] > 0.005).mean() * 100, (s["div_chg"] < -0.005).mean() * 100)
