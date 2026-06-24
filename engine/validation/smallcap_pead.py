"""engine/validation/smallcap_pead.py — settle the small-cap PEAD question rigorously.

The deferred A.2 follow-up: PEAD alpha is small-cap-dense, but is there a market-
cap band where it is BOTH alpha-rich AND net-harvestable at SOLO scale? Done
properly: true small-caps (<$3B, below the top-1500 panel floor), per-event CAR
(clean PEAD, not the crude monthly constructor), + a CONSISTENT cost model
(spread by liquidity tier + size-dependent sqrt impact at a stated AUM, per
[[feedback-cost-model-consistency-aum-capacity-2026-05-21]]).

Reuses cached broad monthly mcap (_crsp_msf_insider_mcap, has true small-caps:
median $669M) + gvkey<->permno map. Pulls comp.fundq US (EPS + rdq announcement
date) + crsp.dsf daily returns for the small-cap universe (one WRDS connection).

VERDICT (2026-05-21, US small-caps $150M-$3B): RED — the deferred A.2 small-cap
question is now DEFINITIVELY CLOSED. The drift is REAL but NOT net-harvestable,
even at solo scale, even long-only. The full chain (a 2nd false-GREEN caught this
session):
  - per-event 60d CAR: STRONG and monotone by cap (micro L/S +5.49% t=6.20, small
    +3.56%, small-mid +2.36%) — the alpha EXISTS.
  - BUT a high per-event t is a LARGE-SAMPLE artifact (23k events), NOT a tradeable
    Sharpe. Calendar-time monthly L/S: net deflSR 0.02-0.06 RED (small-cap returns
    too volatile → low Sharpe).
  - Daily EQUAL-WEIGHT L/S looked GREEN (Sharpe 1.8, deflSR 1.0) — but that is
    BID-ASK BOUNCE: daily EW rebalancing of wide-spread small-caps harvests the
    close-to-close bounce as fake return (Asparouhova-Bessembinder-Kalcheva). The
    micro>small>mid monotonicity of the "alpha" is the artifact's signature.
  - DISCRIMINATING TESTS: skip-first-day collapses it 20.6%→4.7% (Sharpe 1.83→0.44)
    — ~75% of the "alpha" is the first day = bounce + un-capturable announcement
    jump. The CLEANEST tradeable construction (VALUE-weighted + skip first day) is
    RED across ALL bands (best micro long-only Sharpe 0.74 / net deflSR 0.32).
  => small-cap PEAD alpha is locked in the un-tradeable first-day microstructure;
  the clean multi-day drift is too weak + cost-eaten. Same conclusion as insider
  (small-cap alpha = illiquidity premium, not net-harvestable for a solo). No
  sweet spot. Deployable book stays the 2 US large-cap alphas (D_PEAD + revision).
  LESSON: daily-EW small-cap backtests manufacture fake alpha via bid-ask bounce —
  ALWAYS cross-check value-weighted + skip-first-day before believing a GREEN.
"""
from __future__ import annotations

import logging
import os
import socket
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MCAP = "data/cache/_crsp_msf_insider_mcap.parquet"
_MAP = "data/cache/_cik_permno_map_FULL.parquet"
_FUNDQ = "data/cache/_smallcap_fundq.parquet"
_DSF = "data/cache/_smallcap_dsf.parquet"
_MKT = "data/cache/crsp_vwretd_daily.parquet"

BAND_LO, BAND_HI = 150e3, 3.0e6   # mcap in $thousands → $150M .. $3B


def smallcap_permnos() -> list[int]:
    """Permnos whose MEDIAN market cap over the sample sits in the small-cap
    tradeable band ($150M-$3B). Below the top-1500 panel floor (~$1.3B) so this
    genuinely tests true small-caps."""
    mc = pd.read_parquet(_MCAP)
    med = mc.groupby("permno")["mcap"].median()
    return sorted(med[(med >= BAND_LO) & (med < BAND_HI)].index.astype(int))


def _dns():
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); return
        except Exception:
            time.sleep(4)


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine("postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
                         connect_args={"sslmode": "require"})


def fetch_smallcap_data(force: bool = False):
    """ONE WRDS connection: comp.fundq (EPS+rdq) for small-cap gvkeys + crsp.dsf
    daily returns for small-cap permnos. Cached."""
    if os.path.exists(_FUNDQ) and os.path.exists(_DSF) and not force:
        return pd.read_parquet(_FUNDQ), pd.read_parquet(_DSF)
    _dns()
    from sqlalchemy import text
    permnos = smallcap_permnos()
    gmap = pd.read_parquet(_MAP)
    gvkeys = sorted(gmap[gmap["permno"].astype(int).isin(permnos)]["gvkey"].dropna().astype(int).unique())
    gv6 = ",".join("'%s'" % str(g).zfill(6) for g in gvkeys)
    pn_in = ",".join(str(p) for p in permnos)
    eng = _pg_engine()
    try:
        fq = ("select gvkey, datadate, rdq, fyearq, fqtr, epspxq, cshoq "
              "from comp.fundq where datadate between '2011-06-01' and '2024-06-30' "
              "and gvkey in (%s) and rdq is not null and epspxq is not null "
              "and indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'" % gv6)
        fundq = pd.read_sql(text(fq), eng)
        dq = ("select permno, date, ret from crsp.dsf "
              "where permno in (%s) and date between '2013-06-01' and '2024-06-30' "
              "and ret is not null" % pn_in)
        dsf = pd.read_sql(text(dq), eng)
    finally:
        eng.dispose()
    fundq["datadate"] = pd.to_datetime(fundq["datadate"])
    fundq["rdq"] = pd.to_datetime(fundq["rdq"])
    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["ret"] = pd.to_numeric(dsf["ret"], errors="coerce")
    fundq.to_parquet(_FUNDQ, index=False); dsf.to_parquet(_DSF, index=False)
    logger.info("smallcap: fundq %d rows/%d gvkeys; dsf %d rows/%d permnos",
                len(fundq), fundq["gvkey"].nunique(), len(dsf), dsf["permno"].nunique())
    return fundq, dsf


def _build_sue(fundq: pd.DataFrame) -> pd.DataFrame:
    """Bernard-Thomas seasonal-RW SUE on EPS (epspxq) per gvkey-quarter + rdq."""
    f = fundq.dropna(subset=["epspxq", "rdq"]).copy()
    f["gvkey"] = f["gvkey"].astype(int)
    f = f.sort_values(["gvkey", "datadate"])
    out = []
    for _, g in f.groupby("gvkey"):
        g = g.drop_duplicates("datadate").sort_values("datadate")
        g["d"] = g["epspxq"] - g["epspxq"].shift(4)
        g["sig"] = g["d"].shift(1).rolling(8, min_periods=4).std()
        g["sue"] = g["d"] / g["sig"]
        out.append(g)
    s = pd.concat(out).dropna(subset=["sue"])
    return s[np.isfinite(s["sue"])]


def per_event_car(decile: float = 0.1, entry_lag: int = 1, hold: int = 60,
                  band: tuple = (BAND_LO, BAND_HI)):
    """Clean per-event PEAD CAR: market-adjusted cumulative return [rdq+entry_lag,
    +hold trading days] for each event; long top-SUE-decile / short bottom-decile;
    restricted to events whose mcap-at-rdq is in `band`. Returns the per-event
    frame (sue, mcap, car) — ready for L/S spread + the cost model."""
    fundq, dsf = fetch_smallcap_data()
    s = _build_sue(fundq)
    gmap = pd.read_parquet(_MAP)[["gvkey", "permno"]].dropna().drop_duplicates()
    gmap["gvkey"] = gmap["gvkey"].astype(int); gmap["permno"] = gmap["permno"].astype(int)
    s = s.merge(gmap, on="gvkey", how="inner")
    mc = pd.read_parquet(_MCAP); mc["date"] = pd.to_datetime(mc["date"])
    # mcap as of the month before rdq
    s["mkey"] = s["rdq"].dt.to_period("M").dt.to_timestamp("M")
    mcm = mc.assign(mkey=mc["date"].dt.to_period("M").dt.to_timestamp("M"))[["permno", "mkey", "mcap"]]
    s = s.merge(mcm, on=["permno", "mkey"], how="left").dropna(subset=["mcap"])
    s = s[(s["mcap"] >= band[0]) & (s["mcap"] < band[1])]

    mkt = pd.read_parquet(_MKT)["vwretd"]; mkt.index = pd.to_datetime(mkt.index)
    cal = mkt.sort_index().index
    ret_by = {pn: g.sort_values("date") for pn, g in dsf.groupby("permno")}
    rows = []
    for _, ev in s.iterrows():
        pn = int(ev["permno"]); rdq = ev["rdq"]
        g = ret_by.get(pn)
        if g is None:
            continue
        after = cal[cal > rdq]
        if len(after) < entry_lag + hold:
            continue
        lo_d, hi_d = after[entry_lag - 1], after[entry_lag + hold - 1]
        seg = g[(g["date"] >= lo_d) & (g["date"] <= hi_d)]["ret"]
        if len(seg) < hold * 0.6:
            continue
        mseg = mkt[(mkt.index >= lo_d) & (mkt.index <= hi_d)]
        car = float((1 + seg).prod() - (1 + mseg).prod())
        rows.append({"permno": pn, "rdq": rdq, "sue": ev["sue"],
                     "mcap": float(ev["mcap"]), "car": car})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pn = smallcap_permnos()
    logger.info("small-cap universe ($150M-$3B): %d permnos", len(pn))
    fq, dsf = fetch_smallcap_data()
    logger.info("DONE fundq %s dsf %s", fq.shape, dsf.shape)
