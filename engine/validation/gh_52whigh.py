"""engine/validation/gh_52whigh.py — 52-week-high proximity momentum (George-Hwang 2004).

Direction 2 candidate: a behavioral, price-only sibling to D_PEAD. The
anchoring hypothesis (George-Hwang 2004, JF): investors anchor on the
52-week high, so stocks near their high UNDERREACT to good news and
keep drifting up; stocks far from their high keep lagging. GH found this
"nearness to 52-week high" momentum subsumes standard price momentum.

Why this sibling: price-only (no I/B/E/S — the analyst-revision sibling
is blocked by the I/B/E/S gap that parked spec id=57), so it's the
cleanest to validate now using the cached WRDS daily returns. The sharp
tests: (1) residual alpha AFTER the momentum factor (UMD) — is it more
than just momentum beta? (2) is it small-cap concentrated like PEAD?
(3) is it uncorrelated with D_PEAD (different trigger: price-anchoring
vs earnings-surprise)?

Construction: monthly, cross-sectional. GH ratio = price / trailing-252d
high. Long top decile (near high), short bottom decile, equal-weight,
hold one month.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RET_CACHE = "data/cache/crsp_hist_daily_ret.parquet"
_MEMBERSHIP = "data/factor_ensemble_singlename/_crsp_top1500_q_membership.parquet"


def _wide_daily_returns(ret_path: str = _RET_CACHE) -> pd.DataFrame:
    r = pd.read_parquet(ret_path)
    r["date"] = pd.to_datetime(r["date"])
    return r.pivot_table(index="date", columns="permno", values="ret").sort_index()


def gh_ratio(daily_ret: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """52-week-high proximity ratio per stock = price / trailing-252d high,
    where price is the return-cumulated index (scale-invariant, so the
    arbitrary start level is irrelevant)."""
    price = (1.0 + daily_ret.fillna(0.0)).cumprod()
    # mask leading all-NaN so a stock's index only counts after it starts
    valid = daily_ret.notna().cumsum() > 0
    price = price.where(valid)
    high = price.rolling(window, min_periods=window).max()
    return price / high


def build_gh_sleeve(
    ret_path:  str = _RET_CACHE,
    decile:    float = 0.1,
) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly L/S 52w-high momentum sleeve. Returns (monthly_ls_returns,
    monthly_selection) where selection records which permnos were long/
    short each month (for the small-cap conditioning)."""
    daily = _wide_daily_returns(ret_path)
    ratio = gh_ratio(daily)
    # month-end signal + next-month constituent returns
    monthly_ret = (1.0 + daily.fillna(0.0)).resample("ME").prod() - 1.0
    monthly_ret = monthly_ret.where(daily.resample("ME").count() > 5)  # need data
    ratio_me = ratio.resample("ME").last()

    rows_ret = []
    rows_sel = []
    months = ratio_me.index
    for i in range(len(months) - 1):
        t, t1 = months[i], months[i + 1]
        r = ratio_me.loc[t].dropna()
        if len(r) < 50:
            continue
        hi = r[r >= r.quantile(1 - decile)].index   # near 52w high -> long
        lo = r[r <= r.quantile(decile)].index        # far from high -> short
        nxt = monthly_ret.loc[t1]
        rl = nxt.reindex(hi).dropna()
        rs = nxt.reindex(lo).dropna()
        if len(rl) < 5 or len(rs) < 5:
            continue
        rows_ret.append((t1, float(rl.mean() - rs.mean())))
        rows_sel.append((t1, list(hi), list(lo)))
    sleeve = pd.Series(dict(rows_ret), name="gh_52whigh").sort_index()
    sel = pd.DataFrame(rows_sel, columns=["month", "long", "short"]).set_index("month")
    return sleeve, sel


@dataclass(frozen=True)
class GHCapSplit:
    long_small_frac:  float    # fraction of long leg in small-cap tertile
    note:             str


def gh_smallcap_concentration(sel: pd.DataFrame) -> dict:
    """Is the GH long leg concentrated in small caps (like PEAD)? Uses the
    quarterly membership mcap to tag each month's long-leg permnos by cap
    tertile and report the small-cap share."""
    mem = pd.read_parquet(_MEMBERSHIP)
    mem["target_date"] = pd.to_datetime(mem["target_date"])
    # latest mcap per permno per quarter; tag tertiles within each quarter
    out = {"small": 0, "mid": 0, "large": 0, "n": 0}
    mem_by_q = {d: g for d, g in mem.groupby("target_date")}
    q_dates = sorted(mem_by_q.keys())
    for month, row in sel.iterrows():
        # nearest prior quarter membership
        prior = [d for d in q_dates if d <= month]
        if not prior:
            continue
        g = mem_by_q[prior[-1]].dropna(subset=["mcap"])
        if len(g) < 30:
            continue
        q1, q2 = g["mcap"].quantile(1/3), g["mcap"].quantile(2/3)
        cap = g.set_index("permno")["mcap"]
        for pn in row["long"]:
            mc = cap.get(pn)
            if mc is None or not np.isfinite(mc):
                continue
            out["n"] += 1
            out["small" if mc <= q1 else "mid" if mc <= q2 else "large"] += 1
    if out["n"] == 0:
        return {"error": "no cap-tagged long names"}
    return {k: (out[k] / out["n"] if k != "n" else out["n"]) for k in out}
