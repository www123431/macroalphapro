"""engine/validation/guidance_drift.py — 2nd-alpha cousin: management-guidance drift.

Different TRIGGER than analyst revisions or realized earnings surprises: a firm's
own MANAGEMENT forward guidance (IBES Company-Issued-Guidance, ibes.det_guidance,
~2M rows, SELECT-OK on ${WRDS_USER_2} — never tested before). Same information-
underreaction family, but a distinct event → potential LOW correlation with the
D_PEAD + revision book.

Signal: guidance SURPRISE = (guidance midpoint − prevailing consensus) scaled.
ibes.det_guidance carries `mean_at_date` = the consensus mean AT the guidance
announcement date, so the surprise is computable in-table:
    surprise = (mid(val_1,val_2) − mean_at_date) / |mean_at_date|.
Long high-surprise (management guides above the street), short low-surprise; hold;
post-guidance drift (the underreaction). Event = anndats (exact). Link ticker →
cusip (ibes.id_guidance) → permno (crsp.stocknames ncusip) → reuse the cached CRSP
daily-return panel (no new price pull).

Audited with the corrected deflated-Sharpe methodology (actual cross-trial variance,
not raw grid size) per feedback-deflated-sharpe-n-trials-methodology-2026-05-21.

VERDICT (2026-05-21, _guidance_run.py; 56,380 EPS-guidance events / 1157 permnos /
2011-2026, returns 2013-2024): **RED — real but sub-threshold AND not orthogonal.**
  - deployable LONG-ONLY: gross Sharpe 0.92 -> after ss_large cost (26bps RT) NET
    ann +2.7%, Sharpe 0.58, t=1.90, PSR-vs-0 0.974, deflated SR (correct, N=12 grid,
    actual cross-trial V=0.005) = 0.716 — well under the 0.90 / t~3 bar.
  - L/S cost-eaten (turnover 6x): NET Sharpe 0.33, t=1.09.
  - both regime halves weak (t 1.76 / 1.23); 9/12 years positive but small.
  - ORTHOGONALITY (decisive): corr(guidance L/S, analyst-revision) = +0.48 — shares
    ~half its variance with the revision signal (both are forward-earnings-
    information underreaction). corr(long-only, D_PEAD) = -0.14 (low, but the signal
    is too weak to matter). => even the marginally-real edge adds no independent
    breadth on top of the revision sleeve.
  CONCLUSION: not a deployable 3rd alpha. Reinforces the broader finding that the
  US large-cap information-underreaction family (SUE / revision / guidance) is
  driven by ONE correlated forward-earnings factor (pairwise 0.3-0.6), so a genuinely
  independent 3rd alpha won't come from another earnings-information trigger.
  Joins the graveyard. Book stays D_PEAD GREEN + analyst-revision conditional GREEN.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_GUID = "data/cache/_ibes_det_guidance_eps.parquet"
_GLINK = "data/cache/_ibes_id_guidance_link.parquet"
_STOCKNAMES = "data/cache/_stocknames_ncusip.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine(
        "postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
        connect_args={"sslmode": "require"})


def fetch_guidance(measure: str = "EPS", force: bool = False):
    """ONE WRDS connection: pull EPS company-issued guidance (det_guidance) +
    the ticker->cusip id link (id_guidance, column-adaptive). US firms only,
    anndats >= 2011 (drift window 2013-2024). Cached. Returns (guid, link)."""
    import socket
    import time
    if os.path.exists(_GUID) and os.path.exists(_GLINK) and not force:
        return pd.read_parquet(_GUID), pd.read_parquet(_GLINK)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        gq = ("select ticker, pdicity, measure, curr, units, anndats, prd_yr, prd_mon, "
              "val_1, val_2, mean_at_date, usfirm "
              "from ibes.det_guidance where measure='%s' and usfirm=1 "
              "and anndats >= '2011-01-01' and mean_at_date is not null "
              "and val_1 is not null" % measure)
        guid = pd.read_sql(text(gq), eng)
        # id_guidance: introspect for the cusip column (ticker<->cusip bridge)
        cols = set(pd.read_sql(text(
            "select column_name from information_schema.columns where "
            "table_schema='ibes' and table_name='id_guidance'"), eng)["column_name"])
        cusip_col = "cusip" if "cusip" in cols else ("cusip8" if "cusip8" in cols else None)
        selc = [c for c in ("ticker", cusip_col, "oftic", "cname") if c and c in cols]
        link = pd.read_sql(text("select %s from ibes.id_guidance" % ", ".join(selc)), eng)
        if cusip_col and cusip_col != "cusip":
            link = link.rename(columns={cusip_col: "cusip"})
    finally:
        eng.dispose()
    guid["anndats"] = pd.to_datetime(guid["anndats"])
    guid.to_parquet(_GUID, index=False); link.to_parquet(_GLINK, index=False)
    logger.info("guidance %s: %d rows / %d tickers; id_guidance link %d rows (cols %s)",
                measure, len(guid), guid["ticker"].nunique(), len(link), list(link.columns))
    return guid, link


def build_guidance_surprise() -> pd.DataFrame:
    """Per guidance event: surprise = (mid(val_1,val_2) − mean_at_date)/|mean_at_date|,
    mapped to permno. Returns (permno, anndats, surprise, pdicity)."""
    guid, link = fetch_guidance()
    g = guid.copy()
    g["mid"] = np.where(g["val_2"].notna(), (g["val_1"] + g["val_2"]) / 2.0, g["val_1"])
    g = g[g["mean_at_date"].abs() > 1e-6]
    g["surprise"] = (g["mid"] - g["mean_at_date"]) / g["mean_at_date"].abs()
    g = g[np.isfinite(g["surprise"])]
    # winsorize extreme surprises (guidance vs tiny consensus can explode)
    lo, hi = g["surprise"].quantile([0.01, 0.99])
    g["surprise"] = g["surprise"].clip(lo, hi)
    # ticker -> cusip -> permno
    link = link.dropna(subset=["cusip"]).copy()
    link["cusip8"] = link["cusip"].astype(str).str[:8]
    sn = pd.read_parquet(_STOCKNAMES).rename(columns={"ncusip": "cusip8"})
    t2p = (link.merge(sn[["cusip8", "permno"]].drop_duplicates(), on="cusip8", how="inner")
           [["ticker", "permno"]].drop_duplicates("ticker"))
    g = g.merge(t2p, on="ticker", how="inner")
    return g[["permno", "anndats", "surprise", "pdicity"]].dropna()


def _monthly_returns():
    ret = pd.read_parquet(_RET); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    return mret.where(daily.resample("ME").count() > 5)


def build_guidance_sleeve(hold: int = 2, q: float = 0.2, weight: str = "equal",
                          surprise: "pd.DataFrame | None" = None,
                          mret: "pd.DataFrame | None" = None):
    """Calendar-time post-guidance drift. Long top-`q` guidance-surprise / short
    bottom-`q`, held `hold` months; entry via NEXT month's return (guidance day
    skipped). Returns (ls_monthly, long_only_monthly, ann_turnover). long_only =
    long − event-universe market (the deployable leg)."""
    s = build_guidance_surprise() if surprise is None else surprise.copy()
    s["em"] = s["anndats"].dt.to_period("M").dt.to_timestamp("M")
    if mret is None:
        mret = _monthly_returns()
    months = list(mret.index)
    ls, lo, ent, prevL = [], [], [], set()
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        a = s[(s.em <= m) & (s.em > m - pd.DateOffset(months=hold))]
        if len(a) < 40:
            continue
        a = a.sort_values("anndats").drop_duplicates("permno", keep="last")
        sv = a.set_index("permno")["surprise"]
        hi = sv[sv >= sv.quantile(1 - q)].index
        loq = sv[sv <= sv.quantile(q)].index
        nr = mret.loc[nxt]
        rl = nr.reindex(hi).dropna(); rs = nr.reindex(loq).dropna()
        rm = nr.reindex(sv.index).dropna()
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
    return (pd.Series(dict(ls)).sort_index().rename("guid_ls"),
            pd.Series(dict(lo)).sort_index().rename("guid_long"),
            float(np.mean(ent) * 12) if ent else float("nan"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = build_guidance_surprise()
    logger.info("guidance surprise panel: %d events, %d permnos, %s..%s",
                len(s), s["permno"].nunique(), s["anndats"].min(), s["anndats"].max())
