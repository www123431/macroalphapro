"""engine/validation/commodity_carry.py — 2nd-alpha candidate, DIFFERENT mechanism:
genuine COMMODITY CARRY (Koijen-Moskowitz-Pedersen-Vrugt 2018, "Carry", JFE).

UNLOCKED by tr_ds_fut (Datastream Futures, SELECT-OK): we have the per-contract
settlement CURVE (multiple maturities per commodity), so carry is the TRUE roll-yield
(near vs deferred settlement), NOT the momentum-proxy it degenerates to on spot-only
free data (which is why carry_crossasset.py was RED + already in CTA).

Mechanism = storage cost / convenience yield / hedging pressure (Keynes-Hicks normal
backwardation) — orthogonal to BOTH equity earnings underreaction AND momentum. And
commodity futures are liquid + low-turnover + no short-ban => SOLO-EXECUTABLE (it
sidesteps the bid-ask / turnover-wall / short-ban walls that killed our equity shots).

Signal: carry_i = (F_near - F_next)/F_next annualized by the gap between expiries
(>0 = backwardation = positive expected carry). Cross-sectional monthly L/S: long
top-carry commodities, short bottom. Front-contract returns with roll. GATE: residual
alpha vs FF5+UMD AND the INCREMENT over the CTA(PQTIX) sleeve + commodity momentum.

VERDICT (2026-05-21, _commodity_carry_run.py; 20 USD commodities, 2000-2026):
**YELLOW — a REAL but modest, strongly REGIME-DEPENDENT, genuinely-different-mechanism
signal.** Full-sample L/S Sharpe 0.43, t=2.23, 20/27 yrs positive, PSR0 0.985; cost
~nil (liquid futures, turnover 3.9x) -> NET Sharpe 0.42. The WINS none of our equity
shots had: a different mechanism (roll yield / hedging pressure) + SOLO-EXECUTABLE
(no bid-ask/turnover/short-ban wall). BUT: regime-dependent — Sharpe 2.75 (2000-13
commodity supercycle) -> 0.11 (2013-18 bear) -> 0.46 (2018-26 inflation revival);
recent residual alpha vs FF5+UMD ~0 (t=-0.15, n=133 2014+); corr 0.29 w/ D_PEAD,
0.52 w/ commodity momentum -> increment over the existing CTA(commodity-trend) sleeve
is small. So: deploy only as a SMALL, regime-aware commodity sleeve (like the CTA/
insurance overlays), NOT a conviction alpha. KEY: for the "D_PEAD backup" goal a
modest DIFFERENT-mechanism YELLOW (commodity carry) diversifies the MECHANISM better
than a strong SAME-mechanism signal (analyst-revision) can. Confirms the doctrine:
the solo-harvestable un-arbitraged frontier is OTHER ASSET CLASSES (futures), not
more equity factors. data tr_ds_fut (Datastream Futures, SELECT-OK).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CONTR = "data/cache/_cmdty_contracts.parquet"
_PX = "data/cache/_cmdty_settle.parquet"
_PXDIR = "data/cache/_cmdty_settle_chunks"

# 24 liquid USD commodity classes (clscode -> symbol), from tr_ds_fut.wrds_contract_info.
# 2026-05-29 spec 77 §11 amendment: extended 20→24 to match canonical Moskowitz-Ooi-
# Pedersen 2012 ("Time Series Momentum") universe. The 4 additions (Wheat / Coffee /
# Sugar / Cotton) are the obvious published-precedent gaps in the original 20 — NOT
# add-until-significant. Same logic as FX 5→9 expansion that pushed combined carry over
# institutional bars (project_cross_asset_breadth_focus_2026-05-28). The 4 contracts
# selected match the AQR "Value & Momentum Everywhere" instruments exactly: CBOT
# Wheat composite (ZW), ICE Coffee C arabica (KC), ICE Sugar #11 (SB), ICE Cotton #2 (CT).
COMMODITIES = {
    1482: "CL_WTI", 1175: "BRN_Brent", 1514: "HO_HeatOil", 1539: "NG_NatGas",
    1562: "RB_Gasoline", 1176: "G_GasOil", 1508: "GC_Gold", 1574: "SI_Silver",
    1512: "HG_Copper", 1549: "PL_Platinum", 1542: "PA_Palladium", 2420: "ZC_Corn",
    2522: "ZS_Soybean", 2429: "ZM_SoyMeal", 3968: "ZL_SoyOil", 1478: "CC_Cocoa",
    1520: "OJ_OrangeJuice", 1996: "LE_LiveCattle", 2423: "GF_FeederCattle", 3893: "HE_LeanHogs",
    # ── §11 additions 2026-05-29 (MOP canonical universe completion) ──────────
    2442: "ZW_Wheat",      # CBOT Wheat composite
    3289: "KC_Coffee",     # ICE Coffee C (Arabica)
    3299: "SB_Sugar",      # ICE Sugar #11
    1487: "CT_Cotton",     # ICE Cotton #2
}


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine(
        "postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
        connect_args={"sslmode": "require"})


def fetch_commodity_futures(force: bool = False):
    """ONE WRDS connection: contract master (futcode, clscode, expiry=lasttrddate)
    + per-contract daily settlement, for the 20 commodity classes, 2000-2024.
    Cached. Returns (contracts, prices)."""
    import socket
    import time
    if os.path.exists(_CONTR) and os.path.exists(_PX) and not force:
        return pd.read_parquet(_CONTR), pd.read_parquet(_PX)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    import glob
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        cls_in = ",".join(str(c) for c in COMMODITIES)
        contracts = pd.read_sql(text(
            "select futcode, clscode, lasttrddate, startdate, contrname, isocurrcode "
            f"from tr_ds_fut.wrds_contract_info where clscode in ({cls_in}) "
            "and isocurrcode='USD' and lasttrddate is not null"), eng)
        contracts["lasttrddate"] = pd.to_datetime(contracts["lasttrddate"])
        contracts.to_parquet(_CONTR, index=False)            # cache immediately
        futs = contracts["futcode"].dropna().astype(int).unique().tolist()
        logger.info("commodity contracts: %d futcodes across %d classes",
                    len(futs), contracts["clscode"].nunique())
        # SMALL chunks (server OOMs on 4000) + per-chunk partial cache (resumable)
        os.makedirs(_PXDIR, exist_ok=True)
        CH = 1000
        for i in range(0, len(futs), CH):
            cpath = f"{_PXDIR}/chunk_{i // CH:03d}.parquet"
            if os.path.exists(cpath):
                continue
            chunk = ",".join(str(f) for f in futs[i:i + CH])
            # DISTINCT: tr_ds_fut.wrds_fut_contract returns duplicate (futcode, date_, settlement) rows
            # for ~22% of commodity rows; without DISTINCT they break front-vs-next ranking in
            # build_carry_and_returns. Spec 77 §9 amendment 2026-05-28.
            part = pd.read_sql(text(
                "select distinct futcode, date_, settlement from tr_ds_fut.wrds_fut_contract "
                f"where futcode in ({chunk}) and date_ >= '2000-01-01' "
                "and settlement is not null"), eng)
            part.to_parquet(cpath, index=False)
            logger.info("settle chunk %d: %d rows", i // CH, len(part))
    finally:
        eng.dispose()
    prices = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{_PXDIR}/chunk_*.parquet"))],
                       ignore_index=True)
    prices["date_"] = pd.to_datetime(prices["date_"])
    prices.to_parquet(_PX, index=False)
    logger.info("DONE: %d contracts, %d settle rows", len(contracts), len(prices))
    return contracts, prices


def build_carry_and_returns(daily: bool = False):
    """For each commodity, each date: identify front (F1) and next (F2) contracts by
    expiry; carry = (F1-F2)/F2 annualized by expiry gap. Front-contract daily return
    with roll on the day the front contract expires. Returns (carry_df monthly,
    ret_df) wide by commodity symbol. ret_df is monthly-compounded by default; with
    daily=True it is the DAILY front-return panel (for daily-marking the monthly-
    rebalanced sleeve). The carry signal (carry_df) is monthly either way; daily=True
    leaves the validated monthly path (daily=False) byte-identical."""
    contracts, prices = fetch_commodity_futures()
    # Dedupe (futcode, date_) on read — see crossasset_carry._carry_and_returns and
    # spec 77 §9 amendment 2026-05-28. tr_ds_fut.wrds_fut_contract returns ~22%
    # duplicate rows on commodity clscodes; the rank-by-expiry below treats dupes as
    # separate rank slots, collapsing carry to ~0. Fix at source in fetch_commodity_
    # futures via SELECT DISTINCT, AND defensively here for cached parquets pre-fix.
    prices = prices.drop_duplicates(["futcode", "date_"])
    contracts = contracts.dropna(subset=["lasttrddate"]).copy()
    contracts["sym"] = contracts["clscode"].map(COMMODITIES)
    exp = contracts.set_index("futcode")["lasttrddate"]
    sym = contracts.set_index("futcode")["sym"]
    px = prices.merge(contracts[["futcode", "sym", "lasttrddate"]], on="futcode", how="inner")
    px = px[px["settlement"] > 0]

    # per (sym, date): front F1 + next F2 by expiry (vectorized via sort + groupby head)
    px = px[px["lasttrddate"] > px["date_"]].sort_values(["sym", "date_", "lasttrddate"])
    px["rank"] = px.groupby(["sym", "date_"]).cumcount()
    f1 = px[px["rank"] == 0][["sym", "date_", "futcode", "settlement", "lasttrddate"]]
    f2 = px[px["rank"] == 1][["sym", "date_", "settlement", "lasttrddate"]]
    m12 = f1.merge(f2, on=["sym", "date_"], suffixes=("_1", "_2"))
    gap = (m12["lasttrddate_2"] - m12["lasttrddate_1"]).dt.days
    m12 = m12[(gap > 0) & (m12["settlement_2"] > 0)]
    m12["carry"] = (m12["settlement_1"] - m12["settlement_2"]) / m12["settlement_2"] * (365.0 / (
        (m12["lasttrddate_2"] - m12["lasttrddate_1"]).dt.days))
    # front-contract daily return (VECTORIZED): pct_change of F1 settlement within sym,
    # masked NaN on roll days (front futcode changed) to avoid the contract-switch jump.
    fr = m12.sort_values(["sym", "date_"]).rename(columns={"futcode": "front_fut",
                                                           "settlement_1": "front_px"})
    fr["ret"] = fr.groupby("sym")["front_px"].pct_change()
    rolled = fr.groupby("sym")["front_fut"].shift(1) != fr["front_fut"]
    fr.loc[rolled, "ret"] = np.nan
    fr = fr[fr["ret"].abs() < 0.5]
    # monthly carry (last) + monthly compounded front return
    fr["m"] = fr["date_"].dt.to_period("M").dt.to_timestamp("M")
    cwide = (fr.groupby(["m", "sym"])["carry"].last().unstack("sym").sort_index())
    if daily:
        rwide = fr.pivot(index="date_", columns="sym", values="ret").sort_index()
    else:
        rwide = (fr.set_index("date_").groupby("sym")["ret"]
                 .apply(lambda x: (1 + x).resample("ME").prod() - 1).unstack("sym").sort_index())
    return cwide, rwide


def build_carry_sleeve(q: float = 0.3):
    """Monthly cross-sectional carry L/S: long top-`q` carry commodities, short
    bottom-`q`, next-month front return. Returns (ls, long_only, ann_turnover)."""
    cwide, rwide = build_carry_and_returns()
    months = [m for m in cwide.index if m in rwide.index]
    ls, lo, ent, prevL = [], [], [], set()
    allm = sorted(set(cwide.index) | set(rwide.index))
    for i in range(len(allm) - 1):
        m, nxt = allm[i], allm[i + 1]
        if m not in cwide.index or nxt not in rwide.index:
            continue
        c = cwide.loc[m].dropna()
        if len(c) < 8:
            continue
        hi = c[c >= c.quantile(1 - q)].index
        loq = c[c <= c.quantile(q)].index
        nr = rwide.loc[nxt]
        rl = nr.reindex(hi).dropna(); rs = nr.reindex(loq).dropna(); rm = nr.reindex(c.index).dropna()
        if len(rl) < 2 or len(rs) < 2:
            continue
        ls.append((nxt, float(rl.mean() - rs.mean()))); lo.append((nxt, float(rl.mean() - rm.mean())))
        ent.append(len(set(hi) - prevL) / max(len(hi), 1)); prevL = set(hi)
    return (pd.Series(dict(ls)).sort_index().rename("cmdty_carry_ls"),
            pd.Series(dict(lo)).sort_index().rename("cmdty_carry_long"),
            float(np.mean(ent) * 12) if ent else float("nan"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    c, p = fetch_commodity_futures()
    logger.info("contracts %d, settle rows %d, span %s..%s",
                len(c), len(p), p["date_"].min(), p["date_"].max())
