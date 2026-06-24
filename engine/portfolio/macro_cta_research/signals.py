"""
Spec signal implementations for horse race candidates (Path P/Q/S/T/U).

Each function matches its locked spec §2.2 + §2.3 EXACTLY. Hash-lock in
SpecRegistry validates against `docs/spec_path_*.md` content; any
deviation between this code and the spec doc would constitute HARKing.

Function signature contract (per backtest.py framework):
  signal_fn(prices_so_far, as_of, extras) -> pd.Series[ticker → intra_weight]

Returns final intra-sleeve weights ready to apply (sizing/vol-target embedded
per spec). Framework applies these weights to compute P&L; no further sizing.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from engine.portfolio.macro_cta_research.backtest import ewma_volatility

logger = logging.getLogger(__name__)


# Common sizing parameters (locked across all 5 specs)
TARGET_SLEEVE_VOL_ANNUALIZED: float = 0.10   # 10% sleeve vol target
EWMA_LAMBDA:                  float = 0.94    # RiskMetrics standard
EWMA_LOOKBACK_WEEKS:          int   = 60      # ~14 months
PER_NAME_INTRA_CAP:           float = 0.35    # max |w| per ticker (Path P/Q)


# ─────────────────────────────────────────────────────────────────────────────
# Path P — TSMOM(12-1) + Cross-Sectional Momentum
#   spec_id=63 · hash 04cad208 · docs/spec_path_p_macro_trend_xsec_momentum_v1.md
# ─────────────────────────────────────────────────────────────────────────────

def signal_path_p(prices: pd.DataFrame, as_of: pd.Timestamp,
                  extras: Optional[dict] = None) -> pd.Series:
    """Path P spec §2.2 — combined TSMOM + Cross-Sectional Momentum z-score.

    Signal: 0.5 × z(TSMOM 12-1) + 0.5 × z(rank-based X-sec)
    Sizing: per-ticker inverse-vol × signal · cap 35% · renormalize gross 1.0
    """
    n = prices.shape[0]
    if n < 52:
        return pd.Series(0.0, index=prices.columns)

    # TSMOM 12-1: past 12-month return excluding most recent month
    raw_tsmom = prices.iloc[-4] / prices.iloc[-52] - 1.0

    # Cross-sectional momentum: signed rank centered at zero
    ranks = raw_tsmom.rank(method="average")
    xsec = ranks - ranks.mean()

    # Z-scores (cross-sectional across tickers)
    z_tsmom = _safe_z(raw_tsmom)
    z_xsec  = _safe_z(xsec)

    signal = 0.5 * z_tsmom + 0.5 * z_xsec

    # Sizing per §2.3: per-ticker inverse-vol scaling
    returns = prices.pct_change()
    vol = ewma_volatility(returns, lambda_=EWMA_LAMBDA, lookback=EWMA_LOOKBACK_WEEKS)

    intra = signal * (TARGET_SLEEVE_VOL_ANNUALIZED / vol.replace(0, np.nan))
    intra = intra.fillna(0).clip(-PER_NAME_INTRA_CAP, PER_NAME_INTRA_CAP)
    gross = intra.abs().sum()
    if gross > 1e-9:
        intra = intra / gross
    return intra


# ─────────────────────────────────────────────────────────────────────────────
# Path Q — Multi-Frequency TSMOM (1m / 3m / 12m blend)
#   spec_id=64 · hash d256c4d5 · docs/spec_path_q_macro_multifreq_tsmom_v1.md
# ─────────────────────────────────────────────────────────────────────────────

def signal_path_q(prices: pd.DataFrame, as_of: pd.Timestamp,
                  extras: Optional[dict] = None) -> pd.Series:
    """Path Q spec §2.2 — equal-weighted multi-frequency TSMOM (1m+3m+12m).

    Signal: (1/3) z(1m return) + (1/3) z(3m return) + (1/3) z(12m skip 1m)
    Sizing: same as Path P (per-ticker inverse-vol · cap 35% · renorm gross 1)
    """
    n = prices.shape[0]
    if n < 52:
        return pd.Series(0.0, index=prices.columns)

    # 3 lookback windows
    tsmom_1m  = prices.iloc[-1]  / prices.iloc[-4]  - 1.0   # past 4 weeks (~1 month)
    tsmom_3m  = prices.iloc[-1]  / prices.iloc[-12] - 1.0   # past 12 weeks (~3 months)
    tsmom_12m = prices.iloc[-4]  / prices.iloc[-52] - 1.0   # past 52w skip 4w

    z_1m  = _safe_z(tsmom_1m)
    z_3m  = _safe_z(tsmom_3m)
    z_12m = _safe_z(tsmom_12m)

    signal = (z_1m + z_3m + z_12m) / 3.0

    # Sizing identical to Path P
    returns = prices.pct_change()
    vol = ewma_volatility(returns, lambda_=EWMA_LAMBDA, lookback=EWMA_LOOKBACK_WEEKS)

    intra = signal * (TARGET_SLEEVE_VOL_ANNUALIZED / vol.replace(0, np.nan))
    intra = intra.fillna(0).clip(-PER_NAME_INTRA_CAP, PER_NAME_INTRA_CAP)
    gross = intra.abs().sum()
    if gross > 1e-9:
        intra = intra / gross
    return intra


# ─────────────────────────────────────────────────────────────────────────────
# Path S — Risk Parity (passive · inverse-vol)
#   spec_id=66 · hash e727394d · docs/spec_path_s_macro_risk_parity_v1.md
# ─────────────────────────────────────────────────────────────────────────────

def signal_path_s(prices: pd.DataFrame, as_of: pd.Timestamp,
                  extras: Optional[dict] = None) -> pd.Series:
    """Path S spec §2.3 — Risk Parity (inverse-vol weighting, ERC approximation).

    Signal: NO active signal · `signal_i = 1` for all tickers (long-only)
    Sizing:
      Step 1: raw_w_i = 1 / σ_i(t)
              intra_w_i = raw_w_i / sum(raw_w_j)   # long-only · gross = 1.0
      Step 2: portfolio_vol = sqrt(w·Σ·w) annualized
              gross_scale = target_vol_10% / portfolio_vol
              book_w_i = intra_w_i × gross_scale
    """
    n = prices.shape[0]
    if n < EWMA_LOOKBACK_WEEKS:
        return pd.Series(0.0, index=prices.columns)

    returns = prices.pct_change()
    vol = ewma_volatility(returns, lambda_=EWMA_LAMBDA, lookback=EWMA_LOOKBACK_WEEKS)

    # Inverse-vol weighting (ERC approximation)
    raw_w = 1.0 / vol.replace(0, np.nan)
    raw_w = raw_w.fillna(0)
    if raw_w.sum() <= 1e-9:
        return pd.Series(0.0, index=prices.columns)
    intra = raw_w / raw_w.sum()   # gross = 1.0

    # Portfolio vol-target
    return _portfolio_vol_target(intra, returns, TARGET_SLEEVE_VOL_ANNUALIZED)


# ─────────────────────────────────────────────────────────────────────────────
# Path T — Antonacci Dual Momentum (regime-conditional)
#   spec_id=67 · hash e28486e8 · docs/spec_path_t_macro_antonacci_dual_momentum_v1.md
# ─────────────────────────────────────────────────────────────────────────────

def signal_path_t(prices: pd.DataFrame, as_of: pd.Timestamp,
                  extras: Optional[dict] = None) -> pd.Series:
    """Path T spec §2.2 — SPY 12-1 regime triggers risk-on/risk-off tilt.

    Signal:
      regime = "risk_on"  if SPY_12m_return > 0
      regime = "risk_off" if SPY_12m_return ≤ 0
    Intra-weights:
      risk_on:  HYG 0.5 · DBC 0.5 · TLT/GLD 0.0
      risk_off: TLT 0.5 · GLD 0.5 · HYG/DBC 0.0
    Sizing: same vol-target portfolio normalization as Path S
    """
    n = prices.shape[0]
    if n < 52:
        return pd.Series(0.0, index=prices.columns)
    if extras is None or "spy_series" not in extras:
        logger.warning("Path T requires extras['spy_series']")
        return pd.Series(0.0, index=prices.columns)

    spy = extras["spy_series"]
    # SPY 12-1 = price(t) / price(t-52w) - 1, using SPY at as_of
    spy_up_to = spy.loc[:as_of]
    if len(spy_up_to) < 52:
        return pd.Series(0.0, index=prices.columns)
    spy_12m_return = float(spy_up_to.iloc[-1] / spy_up_to.iloc[-52] - 1.0)

    # Regime-conditional intra weights
    intra = pd.Series(0.0, index=prices.columns)
    if spy_12m_return > 0:
        # Risk-on: credit + commodities
        if "HYG" in intra.index: intra["HYG"] = 0.5
        if "DBC" in intra.index: intra["DBC"] = 0.5
    else:
        # Risk-off: rates + gold
        if "TLT" in intra.index: intra["TLT"] = 0.5
        if "GLD" in intra.index: intra["GLD"] = 0.5

    returns = prices.pct_change()
    return _portfolio_vol_target(intra, returns, TARGET_SLEEVE_VOL_ANNUALIZED)


# ─────────────────────────────────────────────────────────────────────────────
# Path U — Vol-Scaled Risk Parity (Moreira-Muir 2017)
#   spec_id=68 · hash d52c382a · docs/spec_path_u_macro_vol_scaled_risk_parity_v1.md
# ─────────────────────────────────────────────────────────────────────────────

def signal_path_u(prices: pd.DataFrame, as_of: pd.Timestamp,
                  extras: Optional[dict] = None) -> pd.Series:
    """Path U spec §2.3 — Risk Parity + VIX-conditional gross scaling.

    Step 1: base allocation = Risk Parity (inverse-vol, identical to Path S Step 1)
    Step 2: gross_scale based on VIX(t):
              VIX < 20     → 1.00
              20 ≤ VIX <30 → 0.70
              30 ≤ VIX <40 → 0.40
              VIX ≥ 40     → 0.10
    Step 3: portfolio vol-target normalization scaled by gross_scale
    """
    n = prices.shape[0]
    if n < EWMA_LOOKBACK_WEEKS:
        return pd.Series(0.0, index=prices.columns)
    if extras is None or "vix_series" not in extras:
        logger.warning("Path U requires extras['vix_series']")
        return pd.Series(0.0, index=prices.columns)

    vix_series = extras["vix_series"]
    # VIX at or just before as_of
    vix_up_to = vix_series.loc[:as_of]
    if vix_up_to.empty:
        return pd.Series(0.0, index=prices.columns)
    vix_val = float(vix_up_to.iloc[-1])

    # Step 1: Risk Parity base
    returns = prices.pct_change()
    vol = ewma_volatility(returns, lambda_=EWMA_LAMBDA, lookback=EWMA_LOOKBACK_WEEKS)
    raw_w = 1.0 / vol.replace(0, np.nan)
    raw_w = raw_w.fillna(0)
    if raw_w.sum() <= 1e-9:
        return pd.Series(0.0, index=prices.columns)
    intra = raw_w / raw_w.sum()   # gross = 1.0

    # Step 2: VIX-conditional gross scaling
    if vix_val < 20:
        gross_scale = 1.00
    elif vix_val < 30:
        gross_scale = 0.70
    elif vix_val < 40:
        gross_scale = 0.40
    else:
        gross_scale = 0.10

    intra_after_vix = intra * gross_scale

    # Step 3: portfolio vol-target normalization (target_vol scaled by gross_scale
    # so de-risked sleeve has proportionally lower vol target)
    effective_target_vol = TARGET_SLEEVE_VOL_ANNUALIZED * gross_scale
    return _portfolio_vol_target(intra_after_vix, returns, effective_target_vol)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_z(series: pd.Series) -> pd.Series:
    """Z-score with safe handling of zero std."""
    s = series.dropna()
    if len(s) < 2 or s.std() < 1e-12:
        return pd.Series(0.0, index=series.index)
    z = (series - s.mean()) / s.std()
    return z.fillna(0)


def _portfolio_vol_target(intra_weights: pd.Series,
                          returns: pd.DataFrame,
                          target_vol_annualized: float) -> pd.Series:
    """Portfolio-level vol-target: scale intra weights so realized port vol = target."""
    recent = returns.tail(EWMA_LOOKBACK_WEEKS).dropna()
    if len(recent) < 10:
        return intra_weights * 0
    cov = recent.cov() * 52.0   # annualized
    w = intra_weights.reindex(cov.index).fillna(0).values
    port_var = float(w @ cov.values @ w)
    if port_var <= 1e-12:
        return intra_weights * 0
    port_vol = port_var ** 0.5
    scale = target_vol_annualized / port_vol
    return intra_weights * scale


# ─────────────────────────────────────────────────────────────────────────────
# Spec → signal_fn dispatch
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_DISPATCH = {
    "P": signal_path_p,
    "Q": signal_path_q,
    "S": signal_path_s,
    "T": signal_path_t,
    "U": signal_path_u,
}
