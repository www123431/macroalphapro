"""engine/validation/spinoff.py — 2nd-alpha search: spin-off drift (parent leg).

Spin-offs were chosen as a SLOW, low-turnover special-situation candidate:
the insider lesson was that fast signals die on small-cap turnover cost, so a
1-2 year drift (Cusatis-Miles-Woolridge 1993; forced-selling limits-to-arbitrage)
should avoid that wall. Different trigger than D_PEAD (restructuring vs earnings).

Identification: CRSP `msedist` distribution code 5523 (spin-off) gives the
PARENT permno + ex-date (2748 events, 2014-2023). The spun CHILD is the stronger
classic leg but `acperm`=0 here (needs CUSIP/new-listing matching) — left
unbuilt; this module tests the feasible PARENT leg.

VERDICT (2026-05-20): parent post-spin drift is significantly NEGATIVE in
2014-2023 (6-mo hold −24%/yr, FF5+UMD residual t=−3.25) — the CMW-1993
parent-outperformance has REVERSED/decayed. BUT the 2014-2023 spin-off universe
is heavy in energy (hsiccd 1311) spun during the 2015-16 oil crash, so the
negative is likely sector/timing-contaminated — neither a clean long (anomaly
gone) nor a clean short (sector artifact). RED; not pursued to the child leg.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EVENTS = "data/cache/_spinoff_events.parquet"          # crsp.msedist distcd=5523
_PARENT_RET = "data/cache/_spinoff_parent_ret.parquet"  # crsp.msf parent returns
_MKT = "data/cache/_crsp_vwretd_monthly.parquet"        # crsp.msi vwretd


def build_parent_spinoff_drift(hold_months: int = 6) -> tuple[pd.Series, float]:
    """Long parents that executed a spin-off in the last `hold_months`, monthly,
    excess over the value-weight market. Returns (excess_series, ann_turnover)."""
    sp = pd.read_parquet(_EVENTS)
    sp["spin_month"] = pd.to_datetime(sp["exdt"]).dt.to_period("M").dt.to_timestamp("M")
    sp["permno"] = sp["permno"].astype(int)
    ret = pd.read_parquet(_PARENT_RET)
    mret = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret.index = mret.index + pd.offsets.MonthEnd(0)
    mkt = pd.read_parquet(_MKT).set_index("date")["vwretd"]
    mkt.index = pd.to_datetime(mkt.index) + pd.offsets.MonthEnd(0)
    months = sorted(mret.index)

    rows, turns, prev = [], [], set()
    for i in range(len(months) - 1):
        t, t1 = months[i], months[i + 1]
        held = set(sp[(sp["spin_month"] <= t) &
                      (sp["spin_month"] > t - pd.DateOffset(months=hold_months))]["permno"])
        rl = mret.loc[t1].reindex(list(held)).dropna()
        if len(rl) < 5:
            continue
        m = mkt.loc[t1] if t1 in mkt.index else 0.0
        rows.append((t1, float(rl.mean() - m)))
        turns.append(len(held ^ prev) / max(len(held), 1))
        prev = held
    return pd.Series(dict(rows)).sort_index().rename("spinoff_parent"), float(np.mean(turns) * 12)
