"""engine/validation/gross_profitability.py — Quality / Novy-Marx 2013 POC.

Phase 2 §I.A.1 of docs/decisions/research_agenda_2026-05-29.md — adds the
QUALITY factor family to the candidate roster. The book currently has only
the earnings-information family on the equity side (D-PEAD + revision, ρ 0.64).
A Quality sleeve is mechanism-orthogonal to both and to FF5 (HML/CMA are
value/investment, not profitability).

Academic anchor
---------------
- Novy-Marx 2013 "The Other Side of Value: The Gross Profitability Premium",
  JFE. Defines profitability as (Revenue - COGS) / Assets — the highest signal-
  to-noise ratio profitability measure, BEFORE depreciation/interest/etc. that
  introduce balance-sheet noise.
- Asness-Frazzini-Pedersen 2019 "Quality minus Junk" — multifactor Quality
  composite. This POC starts with the single-factor Novy-Marx version because
  it is the simplest, cleanest, and most replicated.

Construction (pre-committed Novy-Marx 2013 standard)
----------------------------------------------------
- Signal: GP_t = (revt - cogs) / at  (annual, gvkey-level)
- Lag: 6 months after fiscal year-end (Fama-French standard; ensures data
  was actually public when we trade it)
- Universe: all gvkeys in the cached Compustat funda panel (~2,322 names)
- Returns: CRSP daily returns by permno, compounded to monthly
- Mapping: _cik_gvkey_permno_map links the panels
- Portfolio: monthly rebalance, top 30% L / bottom 30% S, equal-weighted
  within each leg (the Fama-French / Novy-Marx default)
- Costs: 30 bp/side execution (RT_EQ; matches our equity sleeve convention)
- Output: monthly net L/S return series

Strict-gate framing
-------------------
- Pre-committed parameters above; NO grid search per
  [[feedback-strict-gate-no-lowering-2026-05-28]]
- Sample expected ~84-100 monthly observations after fundamentals warmup
- Expected Sharpe historically 0.4-0.7 unconditional → Sharpe-t may be just
  below HLZ 3.0 on this short sample; ABSOLUTE verdict will be honest
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Pre-committed parameters ────────────────────────────────────────────────
LAG_MONTHS = 6           # months after fiscal year-end before tradeable
TOP_Q = 0.30             # top 30% long
BOT_Q = 0.30             # bottom 30% short
RT_EQ_BPS = 30.0         # single-side execution cost
MIN_NAMES_PER_LEG = 10   # require at least 10 stocks per leg for the month


@lru_cache(maxsize=1)
def _load_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and join the three data panels. Cached on first call."""
    funda = pd.read_parquet("data/cache/_compustat_funda.parquet")
    rets  = pd.read_parquet("data/cache/crsp_hist_daily_ret.parquet")
    mp    = pd.read_parquet("data/cache/_cik_gvkey_permno_map.parquet")
    funda["datadate"] = pd.to_datetime(funda["datadate"])
    funda["gvkey"] = funda["gvkey"].astype(int)
    return funda, rets, mp


def _build_signal_panel() -> pd.DataFrame:
    """Build monthly panel: (month_end, permno, gp_signal). The signal is the
    most recent fundamentals available with LAG_MONTHS publication lag."""
    funda, _, mp = _load_panels()
    # Filter funda to rows with all required fields, drop nonsense
    f = funda.dropna(subset=["revt", "cogs", "at"]).copy()
    f = f[(f["at"] > 0)]   # avoid div-by-zero / negative assets
    f["gp"] = (f["revt"] - f["cogs"]) / f["at"]
    # Tradeable date = datadate + LAG_MONTHS (the first day this is public)
    f["public_date"] = f["datadate"] + pd.DateOffset(months=LAG_MONTHS)
    f = f[["gvkey", "public_date", "gp"]].sort_values(["gvkey", "public_date"])

    # Join gvkey -> permno (some gvkeys may map to multiple permnos historically;
    # we use the most recent mapping per gvkey)
    mp_unique = mp.drop_duplicates("gvkey", keep="last")[["gvkey", "permno"]]
    f = f.merge(mp_unique, on="gvkey", how="inner")

    # For each (gvkey, month), pick the most recent public_date ≤ month_end.
    # We expand annual funda to monthly grid using as-of merge per permno.
    months = pd.date_range("2012-01-31", "2026-05-31", freq="ME")
    rows: list[dict] = []
    for permno, g in f.groupby("permno"):
        g = g.sort_values("public_date")
        for m in months:
            sub = g[g["public_date"] <= m]
            if sub.empty:
                continue
            rows.append({"month": m, "permno": int(permno),
                          "gp": float(sub.iloc[-1]["gp"])})
    return pd.DataFrame(rows)


def _build_monthly_returns() -> pd.DataFrame:
    """Compound CRSP daily returns to monthly. Returns long-form panel."""
    _, rets, _ = _load_panels()
    rets["date"] = pd.to_datetime(rets["date"])
    rets["month"] = rets["date"].dt.to_period("M").dt.to_timestamp("M")
    # Compound: (1+r).prod() - 1 per (permno, month)
    monthly = rets.groupby(["permno", "month"])["ret"].apply(
        lambda x: float((1.0 + x).prod() - 1.0))
    return monthly.reset_index().rename(columns={"ret": "monthly_ret"})


def build_gross_profitability_ls() -> pd.Series:
    """Build the monthly long-short return series for Novy-Marx 2013 gross
    profitability, NET of 30bp/side cost.

    Returns
    -------
    pd.Series indexed by month-end timestamps, name='gross_profitability'.
    """
    signal = _build_signal_panel()
    monthly = _build_monthly_returns()

    # Position month t signal → traded for month t+1 (no look-ahead)
    signal = signal.sort_values(["permno", "month"])
    signal["traded_month"] = signal["month"] + pd.offsets.MonthEnd(1)
    panel = signal.merge(
        monthly, left_on=["permno", "traded_month"], right_on=["permno", "month"],
        how="inner", suffixes=("_sig", "_ret"))
    panel = panel.rename(columns={"month_ret": "ret_month"})

    rows: list[dict] = []
    for m, g in panel.groupby("ret_month"):
        g = g.dropna(subset=["gp", "monthly_ret"])
        if len(g) < MIN_NAMES_PER_LEG * 2 / max(TOP_Q, BOT_Q):
            continue
        hi_cut = g["gp"].quantile(1 - TOP_Q)
        lo_cut = g["gp"].quantile(BOT_Q)
        longs = g[g["gp"] >= hi_cut]
        shorts = g[g["gp"] <= lo_cut]
        if len(longs) < MIN_NAMES_PER_LEG or len(shorts) < MIN_NAMES_PER_LEG:
            continue
        long_ret = float(longs["monthly_ret"].mean())
        short_ret = float(shorts["monthly_ret"].mean())
        # 100% turnover assumption monthly (worst case) — 2 sides
        cost = 2.0 * RT_EQ_BPS / 10000.0
        ls_net = long_ret - short_ret - cost
        rows.append({"month": m, "net_ret": ls_net,
                      "n_long": int(len(longs)), "n_short": int(len(shorts))})

    out = pd.DataFrame(rows).set_index("month")["net_ret"].sort_index()
    return out.rename("gross_profitability")


def diagnostic_summary() -> dict:
    """Compute basic stats for the gate eval."""
    r = build_gross_profitability_ls().dropna()
    n = len(r)
    if n < 12:
        return {"n_months": n, "error": "insufficient data"}
    sh = float(r.mean() * 12 / (r.std() * np.sqrt(12)))
    vol = float(r.std() * np.sqrt(12))
    cum = (1.0 + r).cumprod()
    dd = float((cum / cum.cummax() - 1.0).min())
    return {
        "n_months": n,
        "range":    f"{r.index.min().date()} -> {r.index.max().date()}",
        "ann_ret":  round(float(r.mean() * 12), 4),
        "ann_vol":  round(vol, 4),
        "sharpe":   round(sh, 3),
        "t_stat":   round(sh * np.sqrt(n / 12), 2),
        "maxdd":    round(dd, 4),
        "best":     round(float(r.max()), 4),
        "worst":    round(float(r.min()), 4),
        "hit_rate": round(float((r > 0).mean()), 3),
    }
