"""engine/validation/carry_crossasset.py — Direction-3 cross-asset CARRY legs.

User picked carry as the first Direction-3 cross-asset domain. The prior
spec id=50 (factor_ensemble_v1) already established the key constraint: on
FREE data, commodity / FX / fixed-income carry DEGENERATES to a 1-month-return
momentum proxy (= our existing TSMOM), so that ensemble kept Carry equity-only.
This module evaluates the two carry legs where the carry signal is genuinely
OBSERVABLE on free data and NOT a momentum proxy:

  1. EQUITY carry  — trailing-12mo dividend yield across 24 sector/country
     ETFs (the spec id=50 signal), as a standalone cross-sectional L/S.
  2. BOND carry    — yield-curve slope (DGS10 - DGS3MO) timing duration
     exposure (IEF/TLT). The slope is observable & forward-looking — the
     rates analog of FX carry, distinct from bond momentum (trailing return).

Both are screened through engine.validation.alpha_factory.gate(). Verdict
(2026-05-20): BOTH RED — equity carry is a value/sector tilt that lost over
the growth decade (residual alpha t=-2.28); bond carry's slope-timing loses
to passive duration and collapses under cost+trials (net deflated SR 0.65,
recent decay). Carry domain yields no deployable sleeve under our data.

Literature: Koijen-Moskowitz-Pedersen-Vrugt 2018 (JFE) "Carry".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CARRY_PX  = "data/factor_carry_equity/_prices_full_history.parquet"
_CARRY_DIV = "data/factor_carry_equity/_dividends_full_history.parquet"
_FRED_YLD  = "data/cache/_fred_cmt_yields.parquet"
_BOND_PX   = "data/cache/_bond_etf_px.parquet"


def build_equity_div_carry_ls(min_names: int = 9) -> pd.Series:
    """Monthly cross-sectional L/S on trailing-12mo dividend yield: long top
    tertile (high carry), short bottom tertile (low carry), hold one month.
    The spec id=50 equity-carry signal as a standalone sleeve."""
    p = pd.read_parquet(_CARRY_PX).sort_index().drop(columns=["TEST"], errors="ignore")
    d = pd.read_parquet(_CARRY_DIV).sort_index().drop(columns=["TEST"], errors="ignore")
    p.index, d.index = pd.to_datetime(p.index), pd.to_datetime(d.index)
    pme = p.resample("ME").last()
    ret = pme.pct_change()

    months = pme.index
    ttm_yield = pd.DataFrame(index=months, columns=p.columns, dtype=float)
    for t in months:
        lo = t - pd.Timedelta(days=365)
        divsum = d[(d.index > lo) & (d.index <= t)].sum()
        px = pme.loc[t]
        ttm_yield.loc[t] = (divsum / px).where(px > 0)

    rows = []
    for i in range(len(months) - 1):
        t, t1 = months[i], months[i + 1]
        y = ttm_yield.loc[t].dropna()
        y = y[y > 0]
        if len(y) < min_names:
            continue
        hi = y[y >= y.quantile(2 / 3)].index
        lo = y[y <= y.quantile(1 / 3)].index
        nxt = ret.loc[t1]
        rl, rs = nxt.reindex(hi).dropna(), nxt.reindex(lo).dropna()
        if len(rl) < 3 or len(rs) < 3:
            continue
        rows.append((t1, float(rl.mean() - rs.mean())))
    return pd.Series(dict(rows), name="carry_equity_div").sort_index()


def build_bond_carry_slope(inst: str = "IEF", slope_scale: float = 1.5) -> pd.Series:
    """Monthly bond carry: position in duration ETF `inst` proportional to the
    curve slope (DGS10 - DGS3MO, observable, lagged 1 month), clipped [-1,1].
    Steep curve = positive carry = long duration; inverted = short. Distinct
    from bond momentum (no trailing-return signal)."""
    y = pd.read_parquet(_FRED_YLD).ffill()
    px = pd.read_parquet(_BOND_PX).sort_index()
    slope_me = (y["DGS10"] - y["DGS3MO"]).resample("ME").last()
    ret = px.resample("ME").last().pct_change()
    pos = (slope_me / slope_scale).clip(-1, 1).shift(1)
    return (pos * ret[inst]).dropna().rename("bond_carry")
