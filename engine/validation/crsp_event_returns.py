"""engine/validation/crsp_event_returns.py — clean per-event CARs from WRDS.

The A.2 event-conditioning blocker was data quality: the local weekly
price panel had split artifacts + 14.6% coverage and produced a
sign-inverted reconstruction. The fix is CRSP's `ret` field (already
split + dividend adjusted) pulled live from WRDS via the configured
pgpass.

Pipeline:
  1. event permnos + rdq + sue + market_cap from the DHS signal panel
  2. daily ret for those permnos (crsp.dsf) over the window, cached
  3. CRSP value-weight market return (crsp.dsi vwretd), cached
  4. per-event CAR = compound(ret, t+1 .. t+K trading days)
                     − compound(vwretd, same window)

K = 60 trading days matches the D_PEAD spec drift horizon. Caches to
data/cache/ so the (slow) WRDS pull happens once.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SIGNAL_PANEL  = "data/path_c_dhs/_pead_ts_signal_panel.parquet"
_RET_CACHE     = Path("data/cache/crsp_event_daily_ret.parquet")
_MKT_CACHE     = Path("data/cache/crsp_vwretd_daily.parquet")
_WRDS_USER     = "${WRDS_USER_1}"
HOLD_TRADING_DAYS = 60


def _event_permnos(signal_path: str = _SIGNAL_PANEL) -> tuple[pd.DataFrame, list[int]]:
    sig = pd.read_parquet(signal_path).dropna(subset=["permno", "rdq", "sue"]).copy()
    sig["permno"] = sig["permno"].astype(int)
    sig["rdq"] = pd.to_datetime(sig["rdq"])
    permnos = sorted(sig["permno"].unique().tolist())
    return sig, permnos


def fetch_clean_returns(
    start: str = "2013-10-01",
    end:   str = "2024-06-30",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Pull daily CRSP `ret` for the event permnos + value-weight market
    return (vwretd), cached. Returns (ret_long_df, mkt_series).

    ret_long_df columns: permno, date, ret (long form).
    mkt_series indexed by date (vwretd, decimal).
    """
    if (_RET_CACHE.exists() and _MKT_CACHE.exists() and not force_refresh):
        try:
            return (pd.read_parquet(_RET_CACHE),
                    pd.read_parquet(_MKT_CACHE)["vwretd"])
        except Exception as exc:
            logger.warning("crsp_event_returns: cache read failed: %s", exc)

    import wrds
    _sig, permnos = _event_permnos()
    conn = wrds.Connection(wrds_username=_WRDS_USER)
    try:
        # Chunk the permno IN-list to keep the query reasonable.
        chunks = []
        CH = 400
        for i in range(0, len(permnos), CH):
            sub = permnos[i:i + CH]
            inlist = ",".join(str(p) for p in sub)
            q = (f"SELECT permno, date, ret FROM crsp.dsf "
                 f"WHERE date BETWEEN '{start}' AND '{end}' "
                 f"AND permno IN ({inlist}) AND ret IS NOT NULL")
            chunks.append(conn.raw_sql(q))
            logger.info("crsp pull chunk %d/%d", i // CH + 1,
                        (len(permnos) + CH - 1) // CH)
        ret = pd.concat(chunks, ignore_index=True)
        ret["date"] = pd.to_datetime(ret["date"])
        ret["permno"] = ret["permno"].astype(int)
        ret["ret"] = ret["ret"].astype(float)

        mkt = conn.raw_sql(
            f"SELECT date, vwretd FROM crsp.dsi "
            f"WHERE date BETWEEN '{start}' AND '{end}'")
        mkt["date"] = pd.to_datetime(mkt["date"])
        mkt_s = pd.Series(mkt["vwretd"].astype(float).values,
                          index=pd.DatetimeIndex(mkt["date"]),
                          name="vwretd").sort_index()
    finally:
        conn.close()

    _RET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ret.to_parquet(_RET_CACHE)
    mkt_s.to_frame().to_parquet(_MKT_CACHE)
    return ret, mkt_s


def compute_cars(
    sig:       pd.DataFrame,
    ret:       pd.DataFrame,
    mkt:       pd.Series,
    hold_days: int = HOLD_TRADING_DAYS,
) -> pd.DataFrame:
    """Pure per-event CAR compute from a signal frame + daily ret long-frame
    + market series. Enter the first trading day AFTER rdq (no look-ahead),
    hold `hold_days` trading days; CAR = stock compound minus value-weight
    market compound over the same window. Requires >=60% window fill.

    sig must have columns: permno, rdq, sue, market_cap_at_q.
    ret must have columns: permno, date, ret.
    mkt: market return series indexed by date.
    """
    cal = mkt.index.sort_values()
    mkt_sorted = mkt.sort_index()
    rows = []
    ret_by_permno = {pn: g.sort_values("date") for pn, g in ret.groupby("permno")}
    for _, ev in sig.iterrows():
        pn, rdq, sue = int(ev["permno"]), pd.Timestamp(ev["rdq"]), float(ev["sue"])
        g = ret_by_permno.get(pn)
        if g is None:
            continue
        after = cal[cal > rdq]
        if len(after) < hold_days + 1:
            continue
        w_start, w_end = after[0], after[hold_days]
        seg = g[(g["date"] >= w_start) & (g["date"] <= w_end)]["ret"]
        if len(seg) < hold_days * 0.6:
            continue
        stock_cum = float((1.0 + seg).prod() - 1.0)
        mseg = mkt_sorted[(mkt_sorted.index >= w_start) & (mkt_sorted.index <= w_end)]
        mkt_cum = float((1.0 + mseg).prod() - 1.0) if len(mseg) else 0.0
        rows.append({
            "permno": pn, "rdq": rdq, "sue": sue,
            "market_cap_m": float(ev.get("market_cap_at_q", np.nan)),
            "fwd_raw": stock_cum, "mkt_fwd": mkt_cum,
            "car": stock_cum - mkt_cum,
        })
    return pd.DataFrame(rows)


def compute_event_cars_clean(
    hold_days:   int = HOLD_TRADING_DAYS,
    signal_path: str = _SIGNAL_PANEL,
) -> pd.DataFrame:
    """Per-event CAR using the default (production) signal panel + the
    cached WRDS `ret` pull. Thin wrapper around compute_cars."""
    sig, _ = _event_permnos(signal_path)
    ret, mkt = fetch_clean_returns()
    return compute_cars(sig, ret, mkt, hold_days)
