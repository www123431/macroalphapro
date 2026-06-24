"""engine/research/primitives.py — Layer 0 audited strategy primitives.

PRIMITIVE_REGISTRY at bottom of file is the AUTHORITATIVE allowlist for the
Tier 2 primitive_composition template. Adding to the registry requires:
  1. Implement the primitive above (with look-ahead/NaN/dtype tests)
  2. Add to PRIMITIVE_REGISTRY
  3. Document in the registry's 'description' field

Anything not in the registry CANNOT be called by primitive_composition.
This is the alpha-safety barrier.

The ONLY building blocks templates may use. Each primitive is:
- Deterministic (same input → same output)
- Pure (no I/O, no global state)
- Anti-look-ahead by default (lags applied where canonical)
- NaN-safe (returns NaN, never raises)
- Dtype-preserving

Doctrine:
- Templates compose primitives. Templates do NOT use raw pandas operations.
- A new primitive requires: type signature + look-ahead unit test +
  NaN unit test + dtype unit test + code review.
- Existing primitive bug fix MUST add a regression test that catches it.

This module is THE single source of truth for "how do we compute
canonical academic strategy mechanics safely". A bug here propagates to
all templates; a bug in a template only affects one mechanism.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Signal construction ─────────────────────────────────────────────────

def compute_log_return(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Element-wise log-return: log(p_t / p_{t-1}).

    Args:
      price_panel: wide DataFrame, rows=dates, cols=tickers, values=prices

    Returns:
      log-return panel, same shape, first row all NaN
    """
    if price_panel.empty:
        return price_panel.copy()
    return np.log(price_panel / price_panel.shift(1))


def rolling_sum(panel: pd.DataFrame, window: int, skip: int = 0) -> pd.DataFrame:
    """Rolling sum over a window with optional skip of most-recent periods.

    Anti-look-ahead: ALWAYS shifts by at least 1 (signal at month t can NEVER
    use month t's value).

    Canonical 12-1 momentum: window=12, skip=1 — sums log-returns from t-12
    through t-2 (skip the most-recent month for reversal shield).

    Mathematics:
      signal at month t = sum of returns over months [t-window, t-skip-1]
                         (that's window-skip months total)

      Implementation: panel.shift(skip + 1).rolling(window - skip).sum()
        - shift(skip+1): value at t = original at t-(skip+1)
        - rolling(window-skip): sum of shifted values at [t-(window-skip)+1, t]
                                  = sum of original at [t-window, t-skip-1] ✓

    Args:
      panel:  wide DataFrame, time-indexed
      window: total lookback span in periods (≥1)
      skip:   most-recent periods to exclude from sum (≥0)

    Returns:
      rolling-sum panel; first (window) rows NaN
    """
    if window < 1 or skip < 0:
        raise ValueError(f"window must be ≥1 and skip ≥0; got {window=}, {skip=}")
    if window - skip < 1:
        raise ValueError(
            f"window-skip must be ≥1; got window={window}, skip={skip}"
        )
    if panel.empty:
        return panel.copy()
    return panel.shift(skip + 1).rolling(window - skip).sum()


def cross_sectional_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional rank (percent rank 0-1).

    Args:
      panel: wide DataFrame, rows=dates, cols=tickers

    Returns:
      percent-rank panel, same shape; per row: NaN if all-NaN, else
      ranks in [0, 1] (higher value → higher rank)
    """
    if panel.empty:
        return panel.copy()
    return panel.rank(axis=1, pct=True)


def cross_sectional_standardize(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date z-score (mean 0, std 1 across cols)."""
    if panel.empty:
        return panel.copy()
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(mu, axis=0).div(sd, axis=0)


def apply_lag(signal: pd.DataFrame | pd.Series,
               n_periods: int = 1) -> pd.DataFrame | pd.Series:
    """Lag a signal by N periods. Default 1 = standard anti-look-ahead.

    Args:
      signal:   DataFrame or Series, time-indexed
      n_periods: positive integer (≥1); 0 raises ValueError (would
                  preserve look-ahead)

    Returns:
      lagged copy; first n_periods rows NaN
    """
    if n_periods < 1:
        raise ValueError(
            f"apply_lag requires n_periods ≥ 1 to prevent look-ahead; "
            f"got {n_periods}"
        )
    return signal.shift(n_periods)


def residualize_against(returns: pd.Series,
                          factor_returns: pd.DataFrame) -> pd.Series:
    """OLS residualize returns vs factor returns (no constant — assumes
    factor returns are already excess).

    Args:
      returns:        time-series of strategy returns
      factor_returns: time-indexed factor panel (e.g. FF3)

    Returns:
      residuals aligned to returns.index, NaN where overlap missing
    """
    j = pd.concat([returns.rename("r"), factor_returns], axis=1).dropna()
    if len(j) < 2:
        return pd.Series(index=returns.index, dtype=float)
    y = j["r"].values
    X = j[factor_returns.columns].values
    # OLS via normal equations
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return pd.Series(index=returns.index, dtype=float)
    resid = y - X @ beta
    out = pd.Series(index=returns.index, dtype=float)
    out.loc[j.index] = resid
    return out


# ── Portfolio construction ──────────────────────────────────────────────

def top_bottom_membership(rank_panel: pd.DataFrame,
                            top_frac: float, bottom_frac: float
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-row L/S membership masks from a percent-rank panel.

    Args:
      rank_panel: percent-rank DataFrame in [0, 1]
      top_frac:    fraction in (0, 0.5] for long side
      bottom_frac: fraction in (0, 0.5] for short side

    Returns:
      (long_mask, short_mask) — bool DataFrames same shape as rank_panel
    """
    if not (0 < top_frac <= 0.5 and 0 < bottom_frac <= 0.5):
        raise ValueError(
            f"top_frac/bottom_frac must be in (0, 0.5]; "
            f"got {top_frac=}, {bottom_frac=}"
        )
    long_mask = rank_panel >= (1.0 - top_frac)
    short_mask = rank_panel <= bottom_frac
    return long_mask, short_mask


def equal_weight_long_short_returns(
    long_mask: pd.DataFrame, short_mask: pd.DataFrame,
    return_panel: pd.DataFrame
) -> pd.Series:
    """Equal-weight L/S monthly returns.

    Args:
      long_mask:    bool DataFrame, rows=dates, cols=tickers (membership)
      short_mask:   bool DataFrame, same shape
      return_panel: numeric DataFrame, same shape (returns)

    Returns:
      time-series: per-row mean(longs) - mean(shorts), NaN if either leg empty
    """
    long_ret = return_panel.where(long_mask).mean(axis=1)
    short_ret = return_panel.where(short_mask).mean(axis=1)
    return (long_ret - short_ret).rename("ls_return")


# ── Filters ─────────────────────────────────────────────────────────────

def exclude_microcap(price_panel: pd.DataFrame,
                       threshold: float = 5.0) -> pd.DataFrame:
    """Mask out tickers below price threshold for each date independently.

    Args:
      price_panel: wide DataFrame, prices
      threshold:   minimum acceptable price (USD)

    Returns:
      same shape; NaN where price below threshold
    """
    if threshold < 0:
        raise ValueError(f"threshold must be ≥0; got {threshold}")
    return price_panel.where(price_panel >= threshold)


def winsorize(panel: pd.DataFrame, lower: float = 0.01,
                upper: float = 0.99) -> pd.DataFrame:
    """Per-row winsorize at quantiles. Default 1%/99% (Q1/Q99).

    Args:
      panel: wide DataFrame
      lower, upper: quantile bounds in (0, 1); lower < upper

    Returns:
      same shape, clipped per row
    """
    if not (0 < lower < upper < 1):
        raise ValueError(f"need 0 < lower < upper < 1; got {lower=}, {upper=}")
    lo = panel.quantile(lower, axis=1)
    hi = panel.quantile(upper, axis=1)
    return panel.clip(lower=lo, upper=hi, axis=0)


# ── Execution ───────────────────────────────────────────────────────────

def vol_target_normalize(returns: pd.Series, target_vol: float = 0.10,
                           lookback: int = 36,
                           periods_per_year: int = 12) -> pd.Series:
    """Lookback-vol normalization to constant target annual vol.

    Scaling factor at month t = target_vol / realized_vol(t-lookback:t-1).
    Lookback-1 prevents look-ahead.

    Args:
      returns:          monthly L/S returns
      target_vol:       annual target vol (e.g. 0.10 = 10%)
      lookback:         rolling-window length for vol estimate
      periods_per_year: 12 for monthly, 252 for daily

    Returns:
      vol-targeted returns, same index; first `lookback` periods NaN
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be >0; got {target_vol}")
    if lookback < 2:
        raise ValueError(f"lookback must be ≥2; got {lookback}")
    realized_vol = (
        returns.rolling(lookback).std() * np.sqrt(periods_per_year)
    ).shift(1)    # anti-look-ahead: scale uses information up to t-1
    scaling = target_vol / realized_vol.replace(0, np.nan)
    return (returns * scaling).rename(returns.name)


def apply_round_trip_cost(returns: pd.Series, bps_per_side: float = 12.0,
                            turnover: float = 1.0) -> pd.Series:
    """Deduct per-period round-trip cost: 2 sides × turnover × bps.

    Default turnover=1.0 = full monthly turnover (every period rebuild).

    Args:
      returns:       pre-cost returns
      bps_per_side:  one-side cost in basis points (e.g. 12 = 12 bp)
      turnover:      fractional turnover per period (0-1+)

    Returns:
      net returns
    """
    if bps_per_side < 0 or turnover < 0:
        raise ValueError(
            f"bps_per_side and turnover must be ≥0; got "
            f"{bps_per_side=}, {turnover=}"
        )
    cost_per_period = 2.0 * turnover * bps_per_side / 10000.0
    return (returns - cost_per_period).rename(returns.name)


# ── Resampling ─────────────────────────────────────────────────────────

def monthly_resample_last(daily_series: pd.Series) -> pd.Series:
    """Resample daily to month-end last observation.

    Use for prices (last-of-month). For returns use monthly_resample_compound.
    """
    if daily_series.empty:
        return daily_series.copy()
    return daily_series.resample("ME").last()


def monthly_resample_compound(daily_returns: pd.Series) -> pd.Series:
    """Resample daily returns to monthly via compounding (1+r product - 1)."""
    if daily_returns.empty:
        return daily_returns.copy()
    return (1.0 + daily_returns).resample("ME").prod() - 1.0


# ── PRIMITIVE_REGISTRY — allowlist for Tier 2 primitive_composition ──────

# Each entry: {fn, n_outputs (1 or N for multi-return), description}.
# A primitive NOT in this dict cannot be called by primitive_composition.
PRIMITIVE_REGISTRY = {
    "compute_log_return":  {
        "fn": compute_log_return, "n_outputs": 1,
        "description": "Element-wise log-return: log(p_t / p_{t-1}).",
    },
    "rolling_sum": {
        "fn": rolling_sum, "n_outputs": 1,
        "description": "Rolling sum with anti-look-ahead skip. window=12, skip=1 = canonical 12-1.",
    },
    "cross_sectional_rank": {
        "fn": cross_sectional_rank, "n_outputs": 1,
        "description": "Per-date percent rank (0-1).",
    },
    "cross_sectional_standardize": {
        "fn": cross_sectional_standardize, "n_outputs": 1,
        "description": "Per-date z-score across columns.",
    },
    "apply_lag": {
        "fn": apply_lag, "n_outputs": 1,
        "description": "Lag by N periods. n_periods<1 raises (anti-look-ahead).",
    },
    "residualize_against": {
        "fn": residualize_against, "n_outputs": 1,
        "description": "OLS residualize returns against factor returns.",
    },
    "top_bottom_membership": {
        "fn": top_bottom_membership, "n_outputs": 2,
        "description": "Per-row long/short masks from a percent-rank panel. RETURNS TUPLE (long_mask, short_mask).",
    },
    "equal_weight_long_short_returns": {
        "fn": equal_weight_long_short_returns, "n_outputs": 1,
        "description": "Equal-weight long-short return series from masks + returns.",
    },
    "exclude_microcap": {
        "fn": exclude_microcap, "n_outputs": 1,
        "description": "Mask out tickers with price below threshold.",
    },
    "winsorize": {
        "fn": winsorize, "n_outputs": 1,
        "description": "Per-row winsorize at quantile bounds.",
    },
    "vol_target_normalize": {
        "fn": vol_target_normalize, "n_outputs": 1,
        "description": "Lookback-vol normalization to constant target annual vol.",
    },
    "apply_round_trip_cost": {
        "fn": apply_round_trip_cost, "n_outputs": 1,
        "description": "Deduct round-trip cost: 2 sides × turnover × bps_per_side.",
    },
    "monthly_resample_last": {
        "fn": monthly_resample_last, "n_outputs": 1,
        "description": "Resample daily to month-end last observation.",
    },
    "monthly_resample_compound": {
        "fn": monthly_resample_compound, "n_outputs": 1,
        "description": "Resample daily returns to monthly via (1+r) product.",
    },
}


def list_primitive_names() -> list[str]:
    """Public allowlist of primitive names available to primitive_composition."""
    return sorted(PRIMITIVE_REGISTRY.keys())


def get_primitive(name: str):
    """Look up a primitive by name. Returns None if not in registry."""
    entry = PRIMITIVE_REGISTRY.get(name)
    return entry["fn"] if entry else None
