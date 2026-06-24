"""engine/validation/bond_xsmom.py — Bond Cross-Sectional Momentum POC
(Asness-Moskowitz-Pedersen 2013).

Phase 2 §I.A.3 of research_agenda_2026-05-29 (added late). After 4 consecutive
Phase 2 RED candidates on equity/equity-cousin/news-sentiment samples, this
tests a TRULY DIFFERENT mechanism class that matches our 3 GREEN sleeve pattern:
  - Cross-asset (bonds, not equity)
  - Long history available (15+ years futures data via WRDS tr_ds_fut)
  - Genuine economic insurance role (bonds underperform in inflation, outperform
    in deflation — momentum captures the regime persistence)

Academic anchor
---------------
- Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere" (J. of Finance):
  documents cross-sectional momentum across 8 asset classes. Bond momentum sub-sample
  shows Sharpe 0.55-0.75 in 1972-2011.
- Brooks-Moskowitz 2018 "Yield Curve Premia" — confirms bond momentum holds in
  recent samples on G10 country panel.
- Distinct from MOP 2012 (Moskowitz-Ooi-Pedersen) time-series momentum: that's
  per-instrument trend; THIS is cross-sectional ranking across the panel.
- DISTINCT from our deployed TSMOM (crossasset_tsmom.py): TSMOM does long/short
  per-instrument signs (own past return). XSMOM ranks across the panel and
  goes long-top / short-bottom.

Construction (pre-committed AM 2013 standard)
---------------------------------------------
- Universe: 11 government bond futures from existing WRDS cache
    UST2 / UST5 / UST10 / UST30 (US Treasury 4-tenor, RATES dict)
    BUND10 / GILT10 / CGB10 / AGB10 / JGB10 / BTP10 / OAT10 (G10 cross-country)
- Signal: 12-1 month cumulative log return per instrument
- Cross-sectional ranking: long top tercile (top ~4 of 11), short bottom tercile
- Equal-weight within each leg
- Monthly rebalance, RT_CY = 12 bps cost (matches our carry / TSMOM convention)
- Vol-target 10% sleeve-level (matches existing book convention)
- Output: monthly L/S return series

Strict-gate framing
-------------------
- Pre-committed parameters above. NO grid search per
  [[feedback-strict-gate-no-lowering-2026-05-28]]
- Sample window: limited by latest 11-instrument intersection (2014+ likely
  given some XC contracts start later)
- Expected outcome: mechanism-class-and-sample distinct from prior 4 REDs,
  so the base rate is genuinely higher than the equity-on-2013-2024 path
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Pre-committed parameters ────────────────────────────────────────────────
LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1                # 12-1 standard (skip latest month, mean-reversion shield)
LONG_QUANTILE_TOP = 0.36        # top tercile (~ 4 of 11)
SHORT_QUANTILE_BOTTOM = 0.36
RT_CY_BPS = 12.0                # single-side execution cost
MIN_INSTRUMENTS = 6             # require 6+ live instruments per month
SLEEVE_VOL_TARGET = 0.10        # 10% annualized vol target


@lru_cache(maxsize=1)
def _load_bond_returns_wide() -> pd.DataFrame:
    """Load monthly returns for the 11-instrument bond universe (UST + G10 XC)
    by reusing the validated carry-pipeline data loaders. Returns wide
    DataFrame (rows=month, cols=instrument)."""
    from engine.validation.crossasset_carry import (
        _carry_and_returns, _fetch_classes,
        RATES, RATES_XC,
        _RT_CONTR, _RT_PX, _RT_PXDIR,
        _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR,
    )

    # US Treasury 4-tenor (USD-denominated)
    c_us, p_us = _fetch_classes(RATES, _RT_CONTR, _RT_PX, _RT_PXDIR)
    _, rw_us = _carry_and_returns(c_us, p_us, RATES)

    # G10 cross-country (native currency)
    c_xc, p_xc = _fetch_classes(RATES_XC, _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR,
                                  isocurr=None)
    _, rw_xc = _carry_and_returns(c_xc, p_xc, RATES_XC)

    # Concat columns; keep monthly index alignment
    out = pd.concat([rw_us, rw_xc], axis=1)
    out = out.sort_index().dropna(how="all")
    return out


def _xsmom_signal(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """12-1 cumulative log return per instrument, cross-sectionally rankable.

    Returns wide DataFrame of signals (same shape as returns_wide). The
    signal at month t is the SUM of log returns over months [t-12, t-1].
    """
    log_ret = np.log1p(returns_wide.astype(float))
    # Rolling 11-month sum (12 minus 1 skip), shifted so signal at t uses t-12..t-2
    rolled = log_ret.shift(SKIP_MONTHS).rolling(LOOKBACK_MONTHS - SKIP_MONTHS).sum()
    return rolled


def _xsmom_ls_per_month(signal: pd.DataFrame, returns_wide: pd.DataFrame) -> pd.Series:
    """For each month t, rank instruments by signal_{t-1}, long top quantile,
    short bottom quantile, equal-weight within each leg. Return monthly L/S
    (gross of cost)."""
    rows = []
    for i, t in enumerate(signal.index):
        if i + 1 >= len(signal.index):
            break
        next_t = signal.index[i + 1]
        if next_t not in returns_wide.index:
            continue
        sig_row = signal.loc[t].dropna()
        if len(sig_row) < MIN_INSTRUMENTS:
            continue
        ret_row = returns_wide.loc[next_t].reindex(sig_row.index).dropna()
        common = sig_row.index.intersection(ret_row.index)
        if len(common) < MIN_INSTRUMENTS:
            continue
        sig_row = sig_row.loc[common]
        ret_row = ret_row.loc[common]

        hi_cut = sig_row.quantile(1 - LONG_QUANTILE_TOP)
        lo_cut = sig_row.quantile(SHORT_QUANTILE_BOTTOM)
        longs = sig_row[sig_row >= hi_cut].index
        shorts = sig_row[sig_row <= lo_cut].index
        if len(longs) < 2 or len(shorts) < 2:
            continue
        long_ret = float(ret_row.loc[longs].mean())
        short_ret = float(ret_row.loc[shorts].mean())
        rows.append({"month": next_t, "gross_ret": long_ret - short_ret,
                     "n_long": int(len(longs)), "n_short": int(len(shorts))})

    if not rows:
        return pd.Series(dtype=float, name="bond_xsmom_gross")
    df = pd.DataFrame(rows).set_index("month").sort_index()
    return df["gross_ret"].rename("bond_xsmom_gross")


def build_bond_xsmom_gross() -> pd.Series:
    """Monthly gross L/S series, no cost deduction."""
    rw = _load_bond_returns_wide()
    sig = _xsmom_signal(rw)
    return _xsmom_ls_per_month(sig, rw)


def build_bond_xsmom_ls() -> pd.Series:
    """Monthly NET L/S series, cost = 2 × RT_CY (one round trip / month).

    Vol-targeted to 10% sleeve level (matches book convention)."""
    gross = build_bond_xsmom_gross()
    cost = 2.0 * RT_CY_BPS / 10000.0    # 100% turnover assumption monthly
    net = (gross - cost).rename("bond_xsmom")
    # Vol target (full-sample scalar — matches existing carry / TSMOM sleeve convention)
    realized = float(net.std() * np.sqrt(12))
    if realized > 0:
        net = (net * SLEEVE_VOL_TARGET / realized).rename("bond_xsmom")
    return net


def diagnostic_summary() -> dict:
    r = build_bond_xsmom_ls().dropna()
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
