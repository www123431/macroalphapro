"""
engine/factors_singlename/ivol.py — Tier 1 mining candidate: Idiosyncratic Volatility.

Tier 1 mining lab content (registered into FACTOR_REGISTRY_SINGLENAME).
Pre-registration: factor_kind="infrastructure_spec" (P-LAB exempt, +0 trials)
                  per `feedback_factor_research_3_tier_framework.md`.

Literature anchor
-----------------
Ang, Hodrick, Xing, Zhang (2006). "The Cross-Section of Volatility and Expected
Returns." Journal of Finance 61(1):259-299.

Original spec uses **Fama-French 3-factor** regression residual volatility:
    r_i,t = α + β_mkt·r_mkt,t + β_smb·r_smb,t + β_hml·r_hml,t + ε_i,t
    IVOL_i = std(ε_i) × √252

This module implements a **CAPM (single-factor) simplification**:
    r_i,t = β_i·r_mkt,t + ε_i,t          (no α, OLS through origin on excess returns)
    IVOL_i = std(ε_i) × √252

Honest disclose
---------------
CAPM-IVOL ≠ AHX-Z 2006 strict (which uses FF SMB + HML). The simplification is
required because Wave A retail panel (yfinance prices) does not include
SMB/HML factor returns; full AHX-Z replication would require Ken French data
library integration.

For Tier 1 mining demonstration:
  - CAPM-IVOL is a reasonable proxy of AHX-Z IVOL (correlation 0.7-0.85
    in published replications when β_mkt dominates β_smb / β_hml)
  - Tier 1 mining is exploratory; a full FF 3-factor implementation is
    Tier 2 promotion territory (requires `engine/ff_factors_loader.py`
    + amendment to spec, not yet built)
  - Verdict markdown will explicitly tag this deviation under
    `tier_1_mining_caveat`

Expected sign (per AHX-Z 2006):
  HIGH idiosyncratic vol → LOWER expected returns
  → expected_sign = -1 (short high-z, long low-z = "low-vol anomaly")

API mirror of `engine/factors_singlename/bab.py` (same window length, same
benchmark, same NaN propagation pattern) so mining_runner walk-forward can
interleave IVOL with BAB / TSMOM with no per-factor wiring.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from engine.factor_library_singlename import (
    FactorSpecSinglename,
    register_factor,
)

logger = logging.getLogger(__name__)


# Locked Tier 1 mining constants
IVOL_WINDOW_DAYS_LOCKED:        int   = 60       # matches BAB; AHX-Z uses 1-month
IVOL_BENCHMARK_LOCKED:          str   = "SPY"    # CAPM market proxy
IVOL_MIN_OBS_RATIO_LOCKED:      float = 0.5      # need ≥ 50% of window with valid returns
TRADING_DAYS_PER_YEAR:          int   = 252
MIN_UNIVERSE_FOR_ZSCORE_LOCKED: int   = 5        # mirror Wave A z-score gate


def compute_ivol_singlestock_signal(
    as_of:         datetime.date,
    universe:      list[str],
    asset_classes: Optional[dict[str, str]] = None,
    panel:         Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Cross-section z-score of CAPM-IVOL.

    Mechanism:
      1. Slice trailing 60 trading days of price panel + SPY benchmark
      2. Compute daily simple returns
      3. Per ticker: OLS regression r_i,t = β_i · r_mkt,t (origin-pinned;
         centering happens implicitly through cross-section z-score later)
      4. Residual: ε_i,t = r_i,t - β_i · r_mkt,t
      5. IVOL_i = std(ε_i, ddof=1) × √252  (annualized)
      6. Cross-section z-score across universe (high IVOL → high z)

    Args:
        as_of:          decision date (no look-ahead — uses returns ≤ as_of - 1)
        universe:       list of tickers
        asset_classes:  ignored (Wave A factor signature parity)
        panel:          pre-fetched price panel including IVOL_BENCHMARK_LOCKED

    Returns:
        pd.Series indexed by ticker, continuous z-score of IVOL.
        NaN for tickers with insufficient data or universe < 5.
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)
    if panel is None or panel.empty:
        logger.warning("compute_ivol_singlestock_signal: panel required → all-NaN")
        return pd.Series(np.nan, index=universe, dtype=float)
    if IVOL_BENCHMARK_LOCKED not in panel.columns:
        logger.warning(
            "compute_ivol_singlestock_signal: benchmark %s not in panel → all-NaN",
            IVOL_BENCHMARK_LOCKED,
        )
        return pd.Series(np.nan, index=universe, dtype=float)

    end   = as_of - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=120)   # 120 calendar ≈ 60 trading

    mask = (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    sub  = panel.loc[mask].dropna(how="all")
    if sub.empty or len(sub) < IVOL_WINDOW_DAYS_LOCKED // 2:
        return pd.Series(np.nan, index=universe, dtype=float)

    # Market returns (SPY) — the regressor
    bench_close   = sub[IVOL_BENCHMARK_LOCKED].dropna()
    bench_returns = bench_close.pct_change().dropna().tail(IVOL_WINDOW_DAYS_LOCKED)
    min_obs = max(int(IVOL_WINDOW_DAYS_LOCKED * IVOL_MIN_OBS_RATIO_LOCKED), 5)
    if len(bench_returns) < min_obs:
        return pd.Series(np.nan, index=universe, dtype=float)

    bench_var = float(bench_returns.var(ddof=1))
    if bench_var <= 1e-12:   # degenerate market (constant prices)
        return pd.Series(np.nan, index=universe, dtype=float)

    raw_ivol: dict[str, float] = {}
    for ticker in universe:
        if ticker not in sub.columns:
            raw_ivol[ticker] = np.nan
            continue
        t_returns = sub[ticker].dropna().pct_change().dropna().tail(IVOL_WINDOW_DAYS_LOCKED)
        # Align with benchmark window
        joined = pd.concat(
            [t_returns.rename("r"), bench_returns.rename("m")],
            axis=1, join="inner",
        ).dropna()
        if len(joined) < min_obs:
            raw_ivol[ticker] = np.nan
            continue

        cov = float(joined["r"].cov(joined["m"], ddof=1))
        beta = cov / bench_var
        residuals = joined["r"] - beta * joined["m"]
        sigma_eps = float(residuals.std(ddof=1))
        ivol_annual = sigma_eps * np.sqrt(TRADING_DAYS_PER_YEAR)
        raw_ivol[ticker] = ivol_annual

    return _cross_section_zscore(raw_ivol, universe)


def _cross_section_zscore(
    raw_values: dict[str, float],
    universe:   list[str],
) -> pd.Series:
    """Cross-section z-score within universe; min-5 gate (mirror W-B-3 / dividend_yield)."""
    raw_series = pd.Series(raw_values, dtype=float)
    valid = raw_series.dropna()
    if len(valid) < MIN_UNIVERSE_FOR_ZSCORE_LOCKED:
        return pd.Series(np.nan, index=universe, dtype=float)
    mean = float(valid.mean())
    std  = float(valid.std(ddof=1))
    if std <= 1e-9:
        return pd.Series(np.nan, index=universe, dtype=float)

    out: dict[str, float] = {}
    for ticker in universe:
        v = raw_series.get(ticker, np.nan)
        out[ticker] = (v - mean) / std if np.isfinite(v) else np.nan
    return pd.Series(out, dtype=float)


# ── Register into Tier 1 mining content layer ──────────────────────────────
register_factor(FactorSpecSinglename(
    factor_id        = "ivol_singlestock",
    citation         = "Ang, Hodrick, Xing, Zhang (2006) Journal of Finance 61(1):259-299",
    asset_class      = "equity_singlename",
    formula_summary  = (
        "60-day CAPM (single-factor) regression residual std × √252 → cross-section z-score. "
        "Simplification of AHX-Z 2006 FF 3-factor IVOL (FF SMB/HML unavailable in Wave A "
        "retail panel; full FF replication is Tier 2 promotion territory)."
    ),
    signal_fn        = compute_ivol_singlestock_signal,
    expected_sign    = -1,   # high IVOL → low expected returns per AHX-Z 2006
))
