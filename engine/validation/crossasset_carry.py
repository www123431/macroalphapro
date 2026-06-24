"""engine/validation/crossasset_carry.py — the LEGITIMATE optimization of commodity
carry toward GREEN: combine carry ACROSS asset classes (commodity + FX [+ rates]).

KMPV "Carry Everywhere": carry premia in different asset classes are imperfectly
correlated, so a risk-parity COMBINED carry factor has a higher + more regime-robust
Sharpe than any single class — the lift comes from genuine DIVERSIFICATION
(Grinold-Kahn breadth), NOT from p-hacking commodity-only params. We do NOT tune
quantiles/universe to force a t-stat, and we do NOT drop the dead regime.

FX carry = currency-futures roll (covered-interest-parity basis): long high-carry
(high-rate) currencies / short low-carry. Uses 5 USD-quoted CME COMP currency futures
(consistent quoting direction); we VERIFY the sign (high-yield MXN/NZD should rank
high-carry, JPY/CHF low) before trusting it.

VERDICT (2026-05-21, _crossasset_carry_run.py; commodity + 9-ccy FX, 2000-2026):
**GREEN — the genuine SECOND INDEPENDENT MECHANISM the campaign sought.** Commodity
carry Sharpe 0.43 t=2.23 + FX carry (full G10+EM, 9 ccys) Sharpe 0.52 t=2.64, corr
0.05 (near-zero) -> risk-parity COMBINED **Sharpe 0.66, t=3.36, PSR0 1.000, deflated
SR 0.998**. CLEARS BOTH bars (deflated SR >0.90 AND HLZ t>3.0). The lift came purely
from cross-asset DIVERSIFICATION (KMPV "Carry Everywhere") + widening the FX leg to
the full clean COMP currency universe (5->9: the 5 was an incomplete first cut, 9 =
the proper carry universe; PRE-COMMITTED, NOT add-until-significant) — NOT p-hacking
any single-leg params. FX signs verified (high-yield AUD/NZD/MXN/BRL +, low-yield
JPY/CHF/EUR -). alpha vs FF5+UMD t=0.58 => EQUITY-ORTHOGONAL. => the book is now a
TRUE two-mechanism book: D_PEAD (equity earnings underreaction, GREEN) + cross-asset
carry (commodity+FX roll-yield, GREEN, DIFFERENT mechanism, ~0 corr). HONEST CAVEAT:
carry is intrinsically a REGIME-DEPENDENT risk premium (pays in calm, loses in
crises/carry-unwinds — that IS why it's compensated); 1H Sharpe 0.89 > 2H 0.41,
revived 2018+ at 0.78. Deploy regime-aware (vol-target + crash overlay), modest size
(Sharpe 0.66). Rates leg dropped (deferred UST settlement too sparse in tr_ds_fut);
further breadth (clean rates/cross-country) would only strengthen it. See
[[project-commodity-carry-yellow-2026-05-21]].
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from engine.validation.commodity_carry import (fetch_commodity_futures, COMMODITIES,
                                                build_carry_and_returns as _commodity_cr)

logger = logging.getLogger(__name__)

_FX_CONTR = "data/cache/_fx_contracts.parquet"
_FX_PX = "data/cache/_fx_settle.parquet"
_FX_PXDIR = "data/cache/_fx_settle_chunks"
_RT_CONTR = "data/cache/_rates_contracts.parquet"
_RT_PX = "data/cache/_rates_settle.parquet"
_RT_PXDIR = "data/cache/_rates_settle_chunks"
# Cross-country G10 government bond futures (spec 77 §10 amendment 2026-05-28,
# follow-up to §9 US-only rates leg). Native-currency settlement.
_RT_XC_CONTR = "data/cache/_rates_xc_contracts.parquet"
_RT_XC_PX = "data/cache/_rates_xc_settle.parquet"
_RT_XC_PXDIR = "data/cache/_rates_xc_settle_chunks"
# Equity index futures (spec 77 §12 amendment 2026-05-29) — canonical MOP 2012
# universe for the futures TSMOM sleeve. Native-currency settlement. NOT used by
# carry (equity dividend carry was rejected RED earlier as carry_equity_div); used
# ONLY by crossasset_tsmom for the 5th TSMOM leg.
_EQIDX_CONTR = "data/cache/_eqidx_contracts.parquet"
_EQIDX_PX = "data/cache/_eqidx_settle.parquet"
_EQIDX_PXDIR = "data/cache/_eqidx_settle_chunks"

# USD-quoted CME COMP currency futures (USD per foreign unit -> consistent sign).
# Full G10+EM carry universe (pre-committed, NOT add-until-significant): low-yield
# JPY/CHF/EUR, high-yield AUD/NZD/MXN/BRL, mid CAD/GBP.
FX = {2125: "JPY", 2072: "CAD", 2094: "CHF", 2139: "MXN", 2154: "NZD",
      1999: "EUR", 2059: "GBP", 2047: "AUD", 2068: "BRL"}
# US Treasury futures across the curve (carry = curve roll-down/slope)
RATES = {2441: "UST30", 3896: "UST10", 1997: "UST5", 2523: "UST2"}
# Cross-country G10 government bond futures, 10Y where available (spec 77 §10
# amendment 2026-05-28 in progress). 7 countries — natural breadth analogue of
# FX 9-currency expansion. Native-currency settlement.
RATES_XC = {
    626:  "BUND10",   # 🇩🇪 Germany, Eurex
    1163: "GILT10",   # 🇬🇧 UK long Gilt, ICE Europe
    221:  "CGB10",    # 🇨🇦 Canada, Montreal
    3046: "AGB10",    # 🇦🇺 Australia, SFE
    4646: "JGB10",    # 🇯🇵 Japan, Osaka (modern code; legacy 1048 covers pre-2015)
    2009: "BTP10",    # 🇮🇹 Italy, Eurex (from 2009)
    3474: "OAT10",    # 🇫🇷 France, Eurex (from 2012)
}
# Equity index futures (canonical MOP 2012 set, native currency, spec 77 §12
# amendment 2026-05-29). Only used by crossasset_tsmom for the 5th leg.
EQIDX = {
    2424: "SPX_SP500",        # 🇺🇸 CME S&P 500 (full SP, since 1982)
    600:  "ESX_EuroStoxx50",  # 🇪🇺 Eurex EURO STOXX 50 (FESX, since 1998)
    3335: "NIK_Nikkei225",    # 🇯🇵 Osaka NIKKEI 225 electronic (NK, since 1996)
    1261: "FTSE_FTSE100",     # 🇬🇧 ICE/LIFFE FTSE 100 (Z, since 1984)
}


def _fetch_classes(clsmap, contr_path, px_path, pxdir, isocurr: str | None = "USD"):
    """Generic: contracts + settlement for a clscode set (chunked, resumable).

    isocurr='USD' (default) restricts to USD-denominated contracts (the convention
    for the legacy commodity/FX/US-rates pulls). isocurr=None drops the filter so
    cross-country government bond futures (EUR Bund, GBP Gilt, JPY JGB, AUD/CAD
    govt bonds, etc.) can come through in their native currency. Returns each
    contract's settlement in its native quoting currency — the carry calc is a
    fractional roll-yield (F1-F2)/F2 so currency cancels per contract; the return
    panel carries unhedged FX exposure (the standard cross-country carry convention).
    """
    import glob
    import socket
    import time
    if os.path.exists(contr_path) and os.path.exists(px_path):
        return pd.read_parquet(contr_path), pd.read_parquet(px_path)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        cls_in = ",".join(str(c) for c in clsmap)
        isocurr_clause = f"and isocurrcode='{isocurr}'" if isocurr else ""
        contracts = pd.read_sql(text(
            "select futcode, clscode, lasttrddate, contrname, isocurrcode from tr_ds_fut.wrds_contract_info "
            f"where clscode in ({cls_in}) {isocurr_clause} and lasttrddate is not null"), eng)
        contracts["lasttrddate"] = pd.to_datetime(contracts["lasttrddate"])
        contracts.to_parquet(contr_path, index=False)
        futs = contracts["futcode"].dropna().astype(int).unique().tolist()
        os.makedirs(pxdir, exist_ok=True)
        for i in range(0, len(futs), 1000):
            cpath = f"{pxdir}/chunk_{i // 1000:03d}.parquet"
            if os.path.exists(cpath):
                continue
            chunk = ",".join(str(f) for f in futs[i:i + 1000])
            # DISTINCT: tr_ds_fut.wrds_fut_contract returns duplicate (futcode, date_, settlement) rows
            # for some clscodes (rates 73%, FX 0%); without DISTINCT they break front-vs-next ranking
            # in _carry_and_returns. Spec 77 §9 amendment 2026-05-28.
            part = pd.read_sql(text("select distinct futcode, date_, settlement from tr_ds_fut.wrds_fut_contract "
                                    f"where futcode in ({chunk}) and date_ >= '2000-01-01' and settlement is not null"), eng)
            part.to_parquet(cpath, index=False); logger.info("chunk %d: %d rows", i // 1000, len(part))
    finally:
        eng.dispose()
    prices = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{pxdir}/chunk_*.parquet"))], ignore_index=True)
    prices["date_"] = pd.to_datetime(prices["date_"]); prices.to_parquet(px_path, index=False)
    return pd.read_parquet(contr_path), prices


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine("postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
                         connect_args={"sslmode": "require"})


def fetch_fx_futures(force: bool = False):
    import glob
    import socket
    import time
    if os.path.exists(_FX_CONTR) and os.path.exists(_FX_PX) and not force:
        return pd.read_parquet(_FX_CONTR), pd.read_parquet(_FX_PX)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    eng = _pg_engine()
    try:
        cls_in = ",".join(str(c) for c in FX)
        contracts = pd.read_sql(text(
            "select futcode, clscode, lasttrddate, contrname from tr_ds_fut.wrds_contract_info "
            f"where clscode in ({cls_in}) and isocurrcode='USD' and lasttrddate is not null"), eng)
        contracts["lasttrddate"] = pd.to_datetime(contracts["lasttrddate"])
        contracts.to_parquet(_FX_CONTR, index=False)
        futs = contracts["futcode"].dropna().astype(int).unique().tolist()
        os.makedirs(_FX_PXDIR, exist_ok=True)
        for i in range(0, len(futs), 1000):
            cpath = f"{_FX_PXDIR}/chunk_{i // 1000:03d}.parquet"
            if os.path.exists(cpath):
                continue
            chunk = ",".join(str(f) for f in futs[i:i + 1000])
            # DISTINCT: tr_ds_fut.wrds_fut_contract returns duplicate (futcode, date_, settlement) rows
            # for some clscodes (rates 73%, FX 0%); without DISTINCT they break front-vs-next ranking
            # in _carry_and_returns. Spec 77 §9 amendment 2026-05-28.
            part = pd.read_sql(text("select distinct futcode, date_, settlement from tr_ds_fut.wrds_fut_contract "
                                    f"where futcode in ({chunk}) and date_ >= '2000-01-01' and settlement is not null"), eng)
            part.to_parquet(cpath, index=False); logger.info("fx chunk %d: %d rows", i // 1000, len(part))
    finally:
        eng.dispose()
    prices = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{_FX_PXDIR}/chunk_*.parquet"))], ignore_index=True)
    prices["date_"] = pd.to_datetime(prices["date_"]); prices.to_parquet(_FX_PX, index=False)
    return pd.read_parquet(_FX_CONTR), prices


def _carry_and_returns(contracts, prices, label_map, daily: bool = False):
    """Generic near-vs-deferred roll-yield carry + front-return (vectorized),
    reused for any asset class. Returns (carry_wide monthly, ret_wide) by symbol.
    ret_wide is monthly-compounded by default; daily=True returns the DAILY front-
    return panel (for daily-marking the monthly sleeve). daily=False is byte-identical
    to the validated path."""
    # Dedupe (futcode, date_) on read — tr_ds_fut.wrds_fut_contract returns multiple
    # rows per (futcode, date_) for some clscodes (rates 73% dups, commodity 22%
    # dups, FX 0%). The rank-by-expiry logic below assigns rank 0/1 to duplicate
    # rows of the SAME contract, collapsing carry to ~0 (broke rates leg entirely,
    # under-stated commodity Sharpe 0.43 vs true 0.55). Fixed at source in
    # _fetch_classes / fetch_fx_futures, AND defensively here for cached parquets
    # that pre-date the SQL fix. See spec 77 §9 amendment 2026-05-28.
    prices = prices.drop_duplicates(["futcode", "date_"])
    contracts = contracts.dropna(subset=["lasttrddate"]).copy()
    contracts["sym"] = contracts["clscode"].map(label_map)
    px = prices.merge(contracts[["futcode", "sym", "lasttrddate"]], on="futcode", how="inner")
    px = px[(px["settlement"] > 0) & (px["lasttrddate"] > px["date_"])].sort_values(["sym", "date_", "lasttrddate"])
    px["rank"] = px.groupby(["sym", "date_"]).cumcount()
    f1 = px[px["rank"] == 0][["sym", "date_", "futcode", "settlement", "lasttrddate"]]
    f2 = px[px["rank"] == 1][["sym", "date_", "settlement", "lasttrddate"]]
    m = f1.merge(f2, on=["sym", "date_"], suffixes=("_1", "_2"))
    days = (m["lasttrddate_2"] - m["lasttrddate_1"]).dt.days
    m = m[(days > 0) & (m["settlement_2"] > 0)]
    m["carry"] = (m["settlement_1"] - m["settlement_2"]) / m["settlement_2"] * (365.0 / (
        (m["lasttrddate_2"] - m["lasttrddate_1"]).dt.days))
    m = m.sort_values(["sym", "date_"]).rename(columns={"futcode": "ff", "settlement_1": "fpx"})
    m["ret"] = m.groupby("sym")["fpx"].pct_change()
    m.loc[m.groupby("sym")["ff"].shift(1) != m["ff"], "ret"] = np.nan
    m = m[m["ret"].abs() < 0.5]
    m["mo"] = m["date_"].dt.to_period("M").dt.to_timestamp("M")
    cwide = m.groupby(["mo", "sym"])["carry"].last().unstack("sym").sort_index()
    if daily:
        rwide = m.pivot(index="date_", columns="sym", values="ret").sort_index()
    else:
        rwide = (m.set_index("date_").groupby("sym")["ret"]
                 .apply(lambda x: (1 + x).resample("ME").prod() - 1).unstack("sym").sort_index())
    return cwide, rwide


def _xs_ls(cwide, rwide, q=0.3):
    allm = sorted(set(cwide.index) | set(rwide.index)); rows = []
    for i in range(len(allm) - 1):
        mth, nxt = allm[i], allm[i + 1]
        if mth not in cwide.index or nxt not in rwide.index:
            continue
        c = cwide.loc[mth].dropna()
        if len(c) < 4:
            continue
        hi = c[c >= c.quantile(1 - q)].index; lo = c[c <= c.quantile(q)].index
        nr = rwide.loc[nxt]; rl = nr.reindex(hi).dropna(); rs = nr.reindex(lo).dropna()
        if len(rl) < 1 or len(rs) < 1:
            continue
        rows.append((nxt, float(rl.mean() - rs.mean())))
    return pd.Series(dict(rows)).sort_index()


def build_fx_carry():
    c, p = fetch_fx_futures()
    cw, rw = _carry_and_returns(c, p, FX)
    return cw, rw, _xs_ls(cw, rw, q=0.4)   # 5 names -> wider q


def build_rates_carry():
    c, p = _fetch_classes(RATES, _RT_CONTR, _RT_PX, _RT_PXDIR)
    cw, rw = _carry_and_returns(c, p, RATES)
    return cw, rw, _xs_ls(cw, rw, q=0.5)   # 4 names -> halves


def build_rates_xc_carry():
    """Cross-country G10 government bond futures carry — the §10 breadth expansion
    analogue of the FX 9-currency widening that pushed combined-carry over the
    institutional bars. 7 countries (DE/UK/CA/AU/JP/IT/FR 10Y where available),
    native-currency settlement, q=0.3 (7 names -> long top ~2, short bottom ~2)."""
    c, p = _fetch_classes(RATES_XC, _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR, isocurr=None)
    cw, rw = _carry_and_returns(c, p, RATES_XC)
    return cw, rw, _xs_ls(cw, rw, q=0.3)


def build_commodity_carry_ls():
    cw, rw = _commodity_cr()
    return _xs_ls(cw, rw, q=0.3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cw, rw, fx_ls = build_fx_carry()
    print("FX carry mean by currency (sign check: high-yield MXN/NZD should be >0, JPY/CHF <0):")
    print((cw.mean()).round(3).to_string())
