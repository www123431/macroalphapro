"""engine/validation/dpead_reaction.py — D_PEAD axis A.3: announcement-reaction conditioning.

A.1 reduced the weak short leg (tilt). A.2 found the drift is almost entirely
in small caps. A.3 asks a different conditioning question, grounded in the
UNDERREACTION mechanism itself: does the immediate announcement-window
reaction sharpen the drift signal?

Mechanism (Chan-Jegadeesh-Lakonishok 1996, JF): the post-earnings drift
"follows" the earnings-announcement return. SUE (the fundamental surprise)
and the announcement-window abnormal return are PARTIALLY INDEPENDENT
signals; combining them ("two confirming signals") should isolate the
events where the market started to move but under-reacted -> higher drift
density.

No look-ahead: the production D_PEAD enters the first trading day after rdq.
Here the reaction window is [rdq, rdq+1] (2 trading days; the announcement
move is observable by its close regardless of before/after-close timing) and
the drift window is [rdq+2, rdq+60]. The reaction is therefore KNOWN at the
(slightly delayed) entry — it is a legitimate point-in-time FILTER.

Sharp test: take the high-SUE long / low-SUE short baseline and add the
reaction-confirmation filter. Does the long-short CAR spread WIDEN (more
alpha per name) — and at what cost in event count (capacity)?
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PANEL_2014_2023 = "data/cache/_pead_ts_panel_2014_2023.parquet"
_RET_CACHE       = "data/cache/crsp_hist_daily_ret.parquet"
_MKT_CACHE       = "data/cache/crsp_vwretd_daily.parquet"

REACTION_DAYS = 2     # [rdq, rdq+1] trading days
HOLD_DAYS     = 60    # drift window length, starting day rdq+2


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    sig = pd.read_parquet(_PANEL_2014_2023).dropna(subset=["permno", "rdq", "sue"]).copy()
    sig["permno"] = sig["permno"].astype(int)
    sig["rdq"] = pd.to_datetime(sig["rdq"])
    ret = pd.read_parquet(_RET_CACHE)
    ret["date"] = pd.to_datetime(ret["date"])
    ret["permno"] = ret["permno"].astype(int)
    ret["ret"] = ret["ret"].astype(float)
    mkt = pd.read_parquet(_MKT_CACHE)["vwretd"]
    mkt.index = pd.to_datetime(mkt.index)
    return sig, ret, mkt.sort_index()


def compute_reaction_and_drift(
    sig:           pd.DataFrame,
    ret:           pd.DataFrame,
    mkt:           pd.Series,
    reaction_days: int = REACTION_DAYS,
    hold_days:     int = HOLD_DAYS,
) -> pd.DataFrame:
    """Per-event (reaction_car, drift_car, sue, mcap). Reaction = market-
    adjusted compound over the [rdq, rdq+reaction_days-1] trading days; drift
    = market-adjusted compound over the next `hold_days` trading days. Pure."""
    cal = mkt.index.sort_values()
    mkt_sorted = mkt.sort_index()
    ret_by_permno = {pn: g.sort_values("date") for pn, g in ret.groupby("permno")}

    def _cum(series_idx_lo, series_idx_hi, stock_ret) -> tuple[float, float]:
        seg = stock_ret[(stock_ret["date"] >= series_idx_lo) &
                        (stock_ret["date"] <= series_idx_hi)]["ret"]
        mseg = mkt_sorted[(mkt_sorted.index >= series_idx_lo) &
                          (mkt_sorted.index <= series_idx_hi)]
        s = float((1.0 + seg).prod() - 1.0) if len(seg) else np.nan
        m = float((1.0 + mseg).prod() - 1.0) if len(mseg) else 0.0
        return s, m

    rows = []
    for _, ev in sig.iterrows():
        pn, rdq, sue = int(ev["permno"]), pd.Timestamp(ev["rdq"]), float(ev["sue"])
        g = ret_by_permno.get(pn)
        if g is None:
            continue
        on_after = cal[cal >= rdq]                      # rdq day onward
        if len(on_after) < reaction_days + hold_days + 1:
            continue
        # reaction window [rdq, rdq+reaction_days-1]
        r_lo, r_hi = on_after[0], on_after[reaction_days - 1]
        s_r, m_r = _cum(r_lo, r_hi, g)
        if not np.isfinite(s_r):
            continue
        # drift window [rdq+reaction_days, rdq+reaction_days+hold_days-1]
        d_lo = on_after[reaction_days]
        d_hi = on_after[reaction_days + hold_days - 1]
        seg = g[(g["date"] >= d_lo) & (g["date"] <= d_hi)]["ret"]
        if len(seg) < hold_days * 0.6:
            continue
        s_d = float((1.0 + seg).prod() - 1.0)
        mseg = mkt_sorted[(mkt_sorted.index >= d_lo) & (mkt_sorted.index <= d_hi)]
        m_d = float((1.0 + mseg).prod() - 1.0) if len(mseg) else 0.0
        rows.append({
            "permno": pn, "rdq": rdq, "sue": sue,
            "mcap": float(ev.get("market_cap_at_q", np.nan)),
            "reaction_car": s_r - m_r,
            "drift_car":    s_d - m_d,
        })
    return pd.DataFrame(rows)


def double_sort(ev: pd.DataFrame, n_sue: int = 5, n_react: int = 3) -> pd.DataFrame:
    """Mean drift CAR in each SUE-quantile x reaction-quantile cell (and the
    cell event count). Reveals where the drift concentrates."""
    e = ev.dropna(subset=["sue", "reaction_car", "drift_car"]).copy()
    e["sue_q"] = pd.qcut(e["sue"], n_sue, labels=False, duplicates="drop") + 1
    e["react_q"] = pd.qcut(e["reaction_car"], n_react, labels=False, duplicates="drop") + 1
    tab = e.groupby(["sue_q", "react_q"]).agg(
        drift_mean=("drift_car", "mean"),
        n=("drift_car", "size"),
    ).reset_index()
    return tab


def confirmed_long_short(ev: pd.DataFrame, decile: float = 0.1) -> dict:
    """Compare the SUE-only baseline L/S to the reaction-CONFIRMED L/S.

    Baseline: long top-SUE-decile, short bottom-SUE-decile (the D_PEAD core
    cross-section). Confirmed: additionally require the announcement reaction
    to AGREE in sign (long leg reaction>0, short leg reaction<0).

    Reports per-event mean drift CAR for each leg + the spread, and the event
    counts (capacity cost of the filter)."""
    e = ev.dropna(subset=["sue", "reaction_car", "drift_car"]).copy()
    hi_cut = e["sue"].quantile(1 - decile)
    lo_cut = e["sue"].quantile(decile)
    long_all  = e[e["sue"] >= hi_cut]
    short_all = e[e["sue"] <= lo_cut]

    def leg(df):
        return float(df["drift_car"].mean()), int(len(df))

    bl_l, bl_ln = leg(long_all)
    bl_s, bl_sn = leg(short_all)
    long_conf  = long_all[long_all["reaction_car"] > 0]
    short_conf = short_all[short_all["reaction_car"] < 0]
    cf_l, cf_ln = leg(long_conf)
    cf_s, cf_sn = leg(short_conf)
    return {
        "baseline":  {"long_car": bl_l, "short_car": bl_s,
                      "spread": bl_l - bl_s, "n_long": bl_ln, "n_short": bl_sn},
        "confirmed": {"long_car": cf_l, "short_car": cf_s,
                      "spread": cf_l - cf_s, "n_long": cf_ln, "n_short": cf_sn},
        "spread_gain": (cf_l - cf_s) - (bl_l - bl_s),
        "events_retained_long":  cf_ln / bl_ln if bl_ln else float("nan"),
        "events_retained_short": cf_sn / bl_sn if bl_sn else float("nan"),
    }


def holding_period_decay(
    sig: pd.DataFrame,
    ret: pd.DataFrame,
    mkt: pd.Series,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20, 40, 60),
    decile: float = 0.1,
) -> pd.DataFrame:
    """Single-pass: market-adjusted cumulative CAR from entry at rdq+1 to
    each horizon, then the top-vs-bottom SUE-decile spread at each horizon.
    Answers: given the front-loaded edge, what hold captures most of it?

    Entry is the first trading day AFTER rdq (production convention). The
    horizon h means: hold h trading days from entry. Returns one row per
    horizon: spread, t-stat, long CAR, short CAR, n per leg, and the
    spread-per-day (capital efficiency)."""
    from scipy import stats

    cal = mkt.index.sort_values()
    mkt_sorted = mkt.sort_index()
    ret_by_permno = {pn: g.sort_values("date") for pn, g in ret.groupby("permno")}
    hmax = max(horizons)

    rows = []  # one dict per event: sue + car at each horizon
    for _, ev in sig.iterrows():
        pn, rdq, sue = int(ev["permno"]), pd.Timestamp(ev["rdq"]), float(ev["sue"])
        g = ret_by_permno.get(pn)
        if g is None:
            continue
        after = cal[cal > rdq]
        if len(after) < hmax + 1:
            continue
        entry = after[0]
        rec = {"sue": sue}
        ok = True
        for h in horizons:
            w_end = after[h - 1]
            seg = g[(g["date"] >= entry) & (g["date"] <= w_end)]["ret"]
            if len(seg) < h * 0.6:
                ok = False
                break
            s = float((1.0 + seg).prod() - 1.0)
            mseg = mkt_sorted[(mkt_sorted.index >= entry) & (mkt_sorted.index <= w_end)]
            m = float((1.0 + mseg).prod() - 1.0) if len(mseg) else 0.0
            rec[f"car_{h}"] = s - m
        if ok:
            rows.append(rec)
    e = pd.DataFrame(rows)

    out = []
    hi = e["sue"].quantile(1 - decile)
    lo = e["sue"].quantile(decile)
    L = e[e["sue"] >= hi]
    S = e[e["sue"] <= lo]
    for h in horizons:
        c = f"car_{h}"
        t, _ = stats.ttest_ind(L[c], S[c], equal_var=False)
        spread = float(L[c].mean() - S[c].mean())
        out.append({
            "horizon_days": h,
            "spread_pct": spread * 100,
            "t": float(t),
            "long_car_pct": float(L[c].mean()) * 100,
            "short_car_pct": float(S[c].mean()) * 100,
            "spread_per_day_bps": spread / h * 1e4,
            "n_leg": int(len(L)),
        })
    return pd.DataFrame(out)


def reaction_independence(ev: pd.DataFrame) -> dict:
    """How independent is the reaction from SUE? If corr is high, the reaction
    adds little (it's just SUE again). CJL 1996 found them partially
    independent — the source of the incremental signal."""
    e = ev.dropna(subset=["sue", "reaction_car", "drift_car"])
    return {
        "corr_sue_reaction": float(e["sue"].corr(e["reaction_car"])),
        "corr_sue_drift":    float(e["sue"].corr(e["drift_car"])),
        "corr_reaction_drift": float(e["reaction_car"].corr(e["drift_car"])),
        "n": int(len(e)),
    }
