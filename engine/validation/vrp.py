"""engine/validation/vrp.py — Direction-3 cross-asset domain: Volatility Risk Premium.

The VRP is the most-cited "harvestable" premium: implied variance trades rich
to realized variance because investors overpay for crash insurance, so the
variance SELLER earns a premium (Carr-Wu 2009; Bollerslev-Tauchen-Zhou 2009).

Synthetic short variance swap (free data, long history): each month-end sell
1-month variance struck at the VIX-implied variance, pay the realized variance
of SPX over the next month. P&L = implied_var − realized_var (the VRP).

WHY IT IS RED (gate, 2026-05-20) — and why the factory matters here:
Naive view: monthly Sharpe 0.71, 84% hit rate, gross deflated SR 0.92 (survives
even multiple-testing over the 1990-2026 sample). A naive screen DEPLOYS this.
The gate exposes it:
  - residual alpha vs the equity MARKET is significantly NEGATIVE (t=-2.03):
    the VRP is NOT alpha, it is compensation for bearing equity crash risk
    (short vol ~= leveraged short equity tail). market_only strips it negative.
  - skew -9.16, kurtosis 113: "picking up pennies in front of a steamroller".
    The worst months are every vol spike (2020-03, 2008-09/10).
  - It SELLS the crash insurance our AC/TSMOM sleeves BUY (corr w/ D_PEAD
    -0.35) — adding it would CANCEL the book's crisis protection.

Units caveat: the P&L is in variance units, so a bps-based cost drag is not
meaningful in return space; the RED verdict stands on the scale-invariant
residual-alpha + skew evidence, NOT the after-cost number. (Real vol-trading
costs — option bid/ask, VIX-future roll — are separately high.)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VIX_SPX = "data/cache/_vix_spx_daily.parquet"


def build_vrp_short_variance() -> pd.Series:
    """Monthly synthetic short-variance-swap P&L = implied_var(VIX_t) −
    realized_var(SPX over [t, t+1mo]), indexed at t+1 (no look-ahead).
    Positive on average (the VRP), catastrophic in vol spikes."""
    df = pd.read_parquet(_VIX_SPX).dropna().sort_index()
    r = np.log(df["SPX"]).diff()
    rv = (r ** 2).resample("ME").sum()                 # realized monthly variance
    vix_me = df["VIX"].resample("ME").last()
    iv = (vix_me / 100.0) ** 2 / 12.0                  # implied monthly variance
    pnl = (iv.shift(1) - rv).dropna()                  # sell at t, pay realized [t,t+1]
    return pnl.rename("vrp_short")
