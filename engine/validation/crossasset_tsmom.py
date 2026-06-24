"""engine/validation/crossasset_tsmom.py — Axis B: cross-asset Futures TSMOM
(Moskowitz-Ooi-Pedersen 2012, "Time Series Momentum", JFE — KMPV classic).

Reuses crossasset_carry's data infrastructure (same WRDS futures pulls, same
_carry_and_returns helper that returns per-instrument monthly returns rwide).
Different signal: each instrument's OWN past-12-month return sign drives its
own long/short position — NO cross-sectional ranking (that's TSMOM vs XSMOM).

Pre-committed parameters (Moskowitz 2012 standard, not searched):
  - Formation window: 12 months
  - Skip:             1 month (avoid 1-month reversal noise)
  - Per-instrument vol scaling target: 40% annualized (canonical MOP)
  - Per-leg combine:   equal-weight across instruments (after vol scaling)
  - 4-leg combine:     risk-parity (inverse-vol) across cmdty / fx / rates_us / rates_xc

Cost model: monthly rebalance, RT_CY = 12 bps (matches carry sleeve deployment
convention, Phase A.4 amendment 2026-05-28).

This file ONLY computes the returns and reports diagnostics. Deployment
into combined_book.py is a SEPARATE step gated on strict gate evidence:
  Sharpe-t ≥ 3.0 (HLZ)
  Deflated SR ≥ 0.90 at honest n_trials
  OOS (last 1/3) Sharpe > 0
  Subperiod-robust (1H Sharpe > 0 AND 2H Sharpe > 0)
  FF5+UMD α-t orthogonal (|t| < 2 for α)
  Book correlation < 0.5 with existing equity ⊕ carry book
  Per-instrument sign-sensible (>50% of instruments have positive own TSMOM)

If ANY fail → record honestly, no deploy.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Pre-committed parameters (Moskowitz 2012 §2.1) ────────────────────────────

LOOKBACK_MONTHS = 12         # formation window
SKIP_MONTHS     = 1          # skip latest month (avoid 1m reversal noise)
TARGET_INSTRUMENT_VOL = 0.40 # per-instrument annualized vol target (MOP standard)
VOL_LOOKBACK_MONTHS = 12     # rolling window for instrument vol estimate
MIN_INSTRUMENTS_PER_LEG = 3  # don't trade a leg-month with fewer instruments


def _tsmom_per_instrument(rwide_monthly: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument 12-1 TSMOM positions, vol-scaled.

    For each instrument i at month t:
      signal_{i,t} = sign( cum_log_return over months [t-12, t-1] )
      sigma_{i,t}  = realized monthly vol over months [t-12, t-1] × sqrt(12)
      pos_{i,t}    = signal_{i,t} × min(TARGET_INSTRUMENT_VOL / sigma_{i,t}, 2.0)
      ret_{i,t}    = pos_{i,t} × r_{i,t}

    Returns a wide DataFrame of vol-scaled signed monthly returns
    (same shape as rwide), NaN where signal not yet observable or vol undefined.
    """
    # 12-1 cumulative log return ending at t-1 (so position uses only t-1-known info)
    log_ret = np.log1p(rwide_monthly.astype(float))
    cum_lookback = log_ret.shift(SKIP_MONTHS).rolling(LOOKBACK_MONTHS - SKIP_MONTHS).sum()
    signal = np.sign(cum_lookback)

    # Realized vol from same lookback window (also shifted, so no look-ahead)
    realized_vol = (
        rwide_monthly.shift(SKIP_MONTHS)
        .rolling(VOL_LOOKBACK_MONTHS)
        .std()
        * np.sqrt(12)
    )
    # Avoid zero / NaN division → no position when undefined
    scale = (TARGET_INSTRUMENT_VOL / realized_vol).clip(upper=2.0)

    position = signal * scale  # vol-scaled signed position size
    inst_returns = position * rwide_monthly
    return inst_returns


def _aggregate_leg(inst_returns: pd.DataFrame) -> pd.Series:
    """Equal-weight average across instruments per month (MOP §2.1 final step).
    Months with fewer than MIN_INSTRUMENTS_PER_LEG live instruments → NaN."""
    n_live = inst_returns.notna().sum(axis=1)
    leg = inst_returns.mean(axis=1, skipna=True)
    leg[n_live < MIN_INSTRUMENTS_PER_LEG] = np.nan
    return leg.dropna()


# ── Per-leg builders (reuse crossasset_carry data plumbing) ───────────────────

def build_commodity_tsmom() -> pd.Series:
    from engine.validation.commodity_carry import build_carry_and_returns
    _, rw = build_carry_and_returns(daily=False)
    return _aggregate_leg(_tsmom_per_instrument(rw)).rename("cmdty_tsmom")


def build_fx_tsmom() -> pd.Series:
    from engine.validation.crossasset_carry import fetch_fx_futures, _carry_and_returns, FX
    c, p = fetch_fx_futures()
    _, rw = _carry_and_returns(c, p, FX)
    return _aggregate_leg(_tsmom_per_instrument(rw)).rename("fx_tsmom")


def build_rates_us_tsmom() -> pd.Series:
    from engine.validation.crossasset_carry import (
        _fetch_classes, _carry_and_returns, RATES, _RT_CONTR, _RT_PX, _RT_PXDIR,
    )
    c, p = _fetch_classes(RATES, _RT_CONTR, _RT_PX, _RT_PXDIR)
    _, rw = _carry_and_returns(c, p, RATES)
    return _aggregate_leg(_tsmom_per_instrument(rw)).rename("rates_us_tsmom")


def build_rates_xc_tsmom() -> pd.Series:
    from engine.validation.crossasset_carry import (
        _fetch_classes, _carry_and_returns, RATES_XC,
        _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR,
    )
    c, p = _fetch_classes(RATES_XC, _RT_XC_CONTR, _RT_XC_PX, _RT_XC_PXDIR, isocurr=None)
    _, rw = _carry_and_returns(c, p, RATES_XC)
    return _aggregate_leg(_tsmom_per_instrument(rw)).rename("rates_xc_tsmom")


def build_eqidx_tsmom() -> pd.Series:
    """Spec 77 §12 amendment 2026-05-29: 5th leg = equity-index TSMOM (MOP 2012
    canonical 4 indices SPX/ESX/NIK/FTSE), native currency. NOT a carry leg —
    equity dividend carry was rejected RED earlier (carry_equity_div).
    """
    from engine.validation.crossasset_carry import (
        _fetch_classes, _carry_and_returns, EQIDX,
        _EQIDX_CONTR, _EQIDX_PX, _EQIDX_PXDIR,
    )
    c, p = _fetch_classes(EQIDX, _EQIDX_CONTR, _EQIDX_PX, _EQIDX_PXDIR, isocurr=None)
    _, rw = _carry_and_returns(c, p, EQIDX)
    return _aggregate_leg(_tsmom_per_instrument(rw)).rename("eqidx_tsmom")


# ── Combined sleeve (risk-parity across 4 legs, same as carry) ────────────────

def build_tsmom_sleeve_returns(
    include_cmdty: bool = True,
    include_fx: bool = True,
    include_rates_us: bool = True,
    include_rates_xc: bool = True,
    include_eqidx: bool = True,
) -> pd.Series:
    """Risk-parity combined 5-leg TSMOM sleeve (NOT cost-deducted).
    Caller deducts n_legs × RT_CY/10000/12 per month (matches carry-sleeve convention).

    Default = 5-leg (spec 77 §12 amendment 2026-05-29). To reproduce the
    pre-amendment 4-leg path, pass include_eqidx=False.
    """
    from engine.portfolio.carry_sleeve import risk_parity_combine

    legs: dict[str, pd.Series] = {}
    if include_cmdty:
        legs["cmdty"]    = build_commodity_tsmom()
    if include_fx:
        legs["fx"]       = build_fx_tsmom()
    if include_rates_us:
        legs["rates_us"] = build_rates_us_tsmom()
    if include_rates_xc:
        legs["rates_xc"] = build_rates_xc_tsmom()
    if include_eqidx:
        legs["eqidx"]    = build_eqidx_tsmom()

    return risk_parity_combine(legs).rename("tsmom_combined")


def per_instrument_diagnostics(rwide: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument TSMOM mean return, Sharpe, hit rate. Used by the strict
    gate to verify per-instrument sign-sensibility (>50% of instruments
    profitable on their own TSMOM)."""
    inst = _tsmom_per_instrument(rwide).dropna(how="all")
    rows = []
    for col in inst.columns:
        s = inst[col].dropna()
        if len(s) < 24:
            continue
        m = float(s.mean() * 12)
        v = float(s.std() * np.sqrt(12))
        sh = m / v if v > 0 else float("nan")
        pos = float((s > 0).mean())
        rows.append({"instrument": col, "ann_ret": m, "ann_vol": v, "sharpe": sh,
                     "hit_rate": pos, "n_months": int(len(s))})
    return pd.DataFrame(rows).set_index("instrument").sort_values("sharpe", ascending=False)
