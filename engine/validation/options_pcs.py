"""engine/validation/options_pcs.py — 2nd-alpha candidate, DIFFERENT mechanism:
options-implied informed-direction = the ATM put-call implied-vol SPREAD
(Cremers-Weinbaum 2010, "Deviations from Put-Call Parity and Stock Return
Predictability", RFS). Distinct from the tail SKEW (already RED) and the variance
risk premium VRP (already RED): the ATM call-IV minus put-IV reflects DIRECTIONAL
informed/hedging demand (calls rich vs puts -> informed bullish) -> a non-earnings
information channel with orthogonality potential vs the D_PEAD/revision book.

Data: OptionMetrics volatility surface (delta=50, 30-day) ATM IV for BOTH C and P,
pulled fresh together so the maturities match; secid->permno via the cached link;
reuse cached CRSP returns. Run through the standard gate: residual-alpha-t vs
FF5+UMD (FIRST gate), orthogonality vs D_PEAD, cost/turnover, corrected deflated SR.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ATM = "data/cache/_optionm_atm_cp_iv.parquet"
_LINK = "data/cache/_optionm_secid_permno.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine(
        "postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
        connect_args={"sslmode": "require"})


def fetch_atm_cp_iv(force: bool = False) -> pd.DataFrame:
    """ONE WRDS connection: ATM (delta=50, 30-day) implied vol for BOTH calls and
    puts from the OptionMetrics vol surface, 2014-2023. Cached. Adaptive to the
    surface table name (optionm.vsurfd / optionm_all.vsurfd)."""
    import socket
    import time
    if os.path.exists(_ATM) and not force:
        return pd.read_parquet(_ATM)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        # OptionMetrics vol surface is PER-YEAR: optionm.vsurfd{YYYY}. Confirm cols
        # on one year, then pull ATM (delta +-50, 30-day) C & P for each year.
        cols = set(pd.read_sql(text(
            "select column_name from information_schema.columns where "
            "table_schema='optionm' and table_name='vsurfd2014'"), eng)["column_name"])
        logger.info("vsurfd2014 cols: %s", sorted(cols))
        dcol = "days" if "days" in cols else "dte"
        # restrict to OUR universe secids (OptionMetrics tables index on secid) so
        # the per-year scan is fast — without this it full-scans ~100M rows/year.
        secids = pd.read_parquet(_LINK)["secid"].dropna().astype(int).unique().tolist()
        sid_in = ",".join(str(s) for s in secids)
        parts = []
        for yr in range(2014, 2024):
            sql = (f"select secid, date, cp_flag, impl_volatility from optionm.vsurfd{yr} "
                   f"where secid in ({sid_in}) and delta in (50,-50) and {dcol}=30")
            part = pd.read_sql(text(sql), eng)
            parts.append(part)
            logger.info("vsurfd%d: %d rows", yr, len(part))
        df = pd.concat(parts, ignore_index=True)
    finally:
        eng.dispose()
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(_ATM, index=False)
    logger.info("ATM C/P IV: %d rows, %d secids, cp %s",
                len(df), df["secid"].nunique(), df["cp_flag"].value_counts().to_dict())
    return df


def build_pcs_signal() -> pd.DataFrame:
    """Per secid-month: ATM put-call IV spread = callIV - putIV (last obs in month),
    mapped to permno. Cremers-Weinbaum: calls rich (spread high) -> bullish informed
    -> predicts positive next-month return. Returns (permno, month, pcs)."""
    df = fetch_atm_cp_iv()
    # ATM put rows can be cp_flag 'P' with delta -50 OR 50 depending on convention
    c = df[df["cp_flag"] == "C"][["secid", "date", "impl_volatility"]].rename(columns={"impl_volatility": "civ"})
    p = df[df["cp_flag"] == "P"][["secid", "date", "impl_volatility"]].rename(columns={"impl_volatility": "piv"})
    m = c.merge(p, on=["secid", "date"], how="inner")
    m["pcs"] = m["civ"] - m["piv"]
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp("M")
    mm = (m.sort_values("date").groupby(["secid", "month"], as_index=False)
          .agg(pcs=("pcs", "last")))
    link = pd.read_parquet(_LINK)
    lc = {c.lower(): c for c in link.columns}
    link = link.rename(columns={lc.get("secid", "secid"): "secid", lc.get("permno", "permno"): "permno"})
    mm = mm.merge(link[["secid", "permno"]].drop_duplicates(), on="secid", how="inner")
    return mm[["permno", "month", "pcs"]].dropna()


def _monthly_returns():
    ret = pd.read_parquet(_RET); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    return mret.where(daily.resample("ME").count() > 5)


def build_pcs_sleeve(q: float = 0.2, signal=None, mret=None):
    """Monthly L/S: long high put-call-spread (calls rich) / short low; next-month
    hold. Returns (ls, long_only, ann_turnover)."""
    s = build_pcs_signal() if signal is None else signal.copy()
    if mret is None:
        mret = _monthly_returns()
    months = list(mret.index)
    ls, lo, ent, prevL = [], [], [], set()
    for i in range(len(months) - 1):
        m, nxt = months[i], months[i + 1]
        a = s[s["month"] == m]
        if len(a) < 50 or nxt not in mret.index:
            continue
        sv = a.set_index("permno")["pcs"]
        hi = sv[sv >= sv.quantile(1 - q)].index
        loq = sv[sv <= sv.quantile(q)].index
        nr = mret.loc[nxt]
        rl = nr.reindex(hi).dropna(); rs = nr.reindex(loq).dropna(); rm = nr.reindex(sv.index).dropna()
        if len(rl) < 10 or len(rs) < 10:
            continue
        ls.append((nxt, float(rl.mean() - rs.mean()))); lo.append((nxt, float(rl.mean() - rm.mean())))
        ent.append(len(set(hi) - prevL) / max(len(hi), 1)); prevL = set(hi)
    return (pd.Series(dict(ls)).sort_index().rename("pcs_ls"),
            pd.Series(dict(lo)).sort_index().rename("pcs_long"),
            float(np.mean(ent) * 12) if ent else float("nan"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = build_pcs_signal()
    logger.info("put-call IV spread panel: %d rows, %d permnos, %s..%s",
                len(s), s["permno"].nunique(), s["month"].min(), s["month"].max())
