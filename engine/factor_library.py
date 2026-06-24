"""
engine/factor_library.py — Factor signal content layer (5 BHY-validated candidates).

Spec: docs/spec_factor_library_v1.md (registered 2026-05-09, id=42)

Module boundary (per spec §4.1)
-------------------------------
This module is the **content layer** — provides FACTOR_REGISTRY (5 named factor
signal_fn closures + metadata) and ensemble construction utilities.

It does NOT implement lifecycle, state machine, power_check, or verdict logic —
those live in `engine/factor_lab/` (infrastructure layer).

Dependency direction is one-way:
    factor_library  →  factor_lab.power.power_check
    factor_lab      →/  factor_library  (FORBIDDEN; Tier R rule_factor_lab_no_factor_library_import enforces statically)

Boundary invariant (per spec §4.3): zero LLM imports — pure deterministic signals.

5 candidates (locked v1 per spec §2.1)
--------------------------------------
- bab            Frazzini & Pedersen (2014) JFE 111(1):1-25
- low_vol        Baker, Bradley, Wurgler (2011) FAJ 67(1):40-54
- tsmom_12_1     Moskowitz, Ooi, Pedersen (2012) JFE 104(2):228 eq.(3)
- csmom          Asness, Moskowitz, Pedersen (2013) JF 68(3):929
- donchian_trend Hurst, Ooi, Pedersen (2017) JPM 44(1):15-29

Inverse-Vol DROPPED v1 (spec §2.1) — concept overlap with risk-parity weighting.

Architecture pattern
--------------------
For testability and rigor, each factor is split into:

  1. Pure compute helper `_compute_<factor>_weights(closes, ...)` — deterministic
     function of price DataFrame; testable with synthetic data; no I/O.
  2. signal_fn wrapper `_<factor>_signal_fn(as_of)` — fetches yfinance prices
     for the as-of universe, calls the pure compute helper.

Tests should exercise the pure compute helpers; signal_fn wrappers are smoke-
tested only when network is available (skipped in CI by default).
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Per-factor lookback constants (locked v1 per spec §2.1).
_TRADING_DAYS_PER_YEAR = 252      # Lo (2002) convention
_TRADING_DAYS_PER_MONTH = 21      # ≈ 252 / 12
_BETA_WINDOW_DAYS       = 252     # BAB §2.1
_VOL_WINDOW_DAYS        = 252     # Low-Vol §2.1
_TSMOM_LOOKBACK_DAYS    = 252     # TSMOM 12-1 §2.1
_TSMOM_SKIP_DAYS        = 21      # TSMOM "1-month skip"
_DONCHIAN_HORIZONS      = (21, 63, 252)   # 1m / 3m / 12m §2.1
_BAB_TERTILE_PCT        = 1.0 / 3.0
_LOW_VOL_QUINTILE_PCT   = 0.20
_CSMOM_TERTILE_PCT      = 1.0 / 3.0
_TARGET_VOL_ANNUAL      = 0.10    # vol-targeting reference (Moskowitz et al. 2012)

# Spec §2.4 max-weight constraint (used downstream by build_ensemble_weights only).
_MAX_SINGLE_WEIGHT      = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# FactorSpec dataclass + FACTOR_REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# Type alias for signal_fn closures.
#   Input:  as_of (point-in-time date for walk-forward backtest)
#   Output: dict[ticker → weight]; long-short factors → weights sum to ≈ 0;
#           sign-following factors (TSMOM/Donchian) → gross exposure ≤ 1
#           (|weights|.sum() ≤ 1). Ensemble step (§2.3 + §2.4) applies final
#           normalization + max-weight cap.
SignalFn = Callable[[datetime.date], dict[str, float]]


@dataclass(frozen=True)
class FactorSpec:
    """Metadata + signal_fn closure for a single candidate factor.

    Per docs/spec_factor_library_v1.md §2.1 — these 5 are the v1 locked candidate
    pool; addition/removal requires amend_spec(kind='hypothesis_amend') which
    contributes 3 to EFFECTIVE_N_TRIALS (per AMENDMENT_KINDS).
    """
    factor_id:        str
    citation:         str               # full bibliographic reference
    asset_class:      str               # "equity_etf" / "cross_asset"
    formula_summary:  str               # one-line algorithmic summary
    signal_fn:        SignalFn          # point-in-time weights closure


# ─────────────────────────────────────────────────────────────────────────────
# Pure compute helpers (deterministic; testable with synthetic DataFrames)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_closes(closes: pd.DataFrame, name: str) -> None:
    """Common input validation: non-empty, datetime-indexed, no all-NaN columns."""
    if not isinstance(closes, pd.DataFrame):
        raise TypeError(f"{name}: closes must be a DataFrame, got {type(closes)}")
    if closes.empty:
        raise ValueError(f"{name}: closes is empty")
    if closes.shape[1] == 0:
        raise ValueError(f"{name}: closes has no columns (no tickers)")
    if not isinstance(closes.index, pd.DatetimeIndex):
        raise ValueError(
            f"{name}: closes.index must be DatetimeIndex; got {type(closes.index)}"
        )


def _daily_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns (drop first NaN row)."""
    return closes.pct_change().iloc[1:]


def _trim_to_window(returns: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Trim returns to the last `window_days` rows; drop columns with insufficient data."""
    if len(returns) >= window_days:
        windowed = returns.iloc[-window_days:]
    else:
        windowed = returns
    # Drop columns that have ≥ 20% missing in the window (insufficient for stable estimates).
    threshold = int(0.8 * len(windowed))
    keep = [c for c in windowed.columns if windowed[c].count() >= threshold]
    return windowed[keep]


def _compute_bab_weights(
    closes:           pd.DataFrame,
    benchmark_close:  pd.Series,
    beta_window_days: int   = _BETA_WINDOW_DAYS,
    tertile_pct:      float = _BAB_TERTILE_PCT,
) -> dict[str, float]:
    """BAB factor weights (Frazzini & Pedersen 2014 JFE 111(1):1-25).

    Algorithm (per spec §2.1):
      1. Compute 252d rolling β_i = Cov(r_i, r_bench) / Var(r_bench) on daily returns
      2. Rank by β; bottom tertile = low-β long leg; top tertile = high-β short leg
      3. β-neutralize: scale long leg by 1/β_long_avg, short leg by 1/β_short_avg,
         so resulting portfolio β = +1 - 1 = 0

    Args:
        closes: DataFrame[date, ticker] of daily ETF closes (universe).
        benchmark_close: Series[date] of benchmark daily closes (typically SPY).
        beta_window_days: Window for β estimation; spec locks 252.
        tertile_pct: Long/short leg cutoff fraction; spec locks 1/3.

    Returns:
        dict[ticker → weight], β-neutralized long-short. Weights sum to ≈ 0.
        Empty dict if insufficient data (< beta_window_days observations).
    """
    _validate_closes(closes, "_compute_bab_weights")
    if not isinstance(benchmark_close, pd.Series):
        raise TypeError(f"benchmark_close must be Series; got {type(benchmark_close)}")

    # Align benchmark to closes index, then compute returns
    aligned_close = closes.join(benchmark_close.rename("__bench__"), how="inner")
    if aligned_close.empty or "__bench__" not in aligned_close.columns:
        return {}

    rets = _daily_returns(aligned_close)
    rets = _trim_to_window(rets, beta_window_days)

    if len(rets) < max(60, beta_window_days // 4):  # need at least ~3 months of data
        return {}
    if "__bench__" not in rets.columns:
        return {}

    bench_ret = rets["__bench__"]
    bench_var = bench_ret.var(ddof=1)
    if not np.isfinite(bench_var) or bench_var <= 0:
        return {}

    betas: dict[str, float] = {}
    for ticker in rets.columns:
        if ticker == "__bench__":
            continue
        r = rets[ticker].dropna()
        # Re-align with bench on common dates
        common = r.index.intersection(bench_ret.index)
        if len(common) < max(60, beta_window_days // 4):
            continue
        cov = r.loc[common].cov(bench_ret.loc[common])
        if not np.isfinite(cov):
            continue
        betas[ticker] = cov / bench_var

    if len(betas) < 6:  # need ≥ 6 ETFs for tertile split (2 per leg minimum)
        return {}

    sorted_tickers = sorted(betas, key=lambda t: betas[t])
    n = len(sorted_tickers)
    n_leg = max(1, int(n * tertile_pct))
    long_tickers  = sorted_tickers[:n_leg]            # low β
    short_tickers = sorted_tickers[-n_leg:]           # high β

    beta_long_avg  = float(np.mean([betas[t] for t in long_tickers]))
    beta_short_avg = float(np.mean([betas[t] for t in short_tickers]))

    if not (np.isfinite(beta_long_avg) and np.isfinite(beta_short_avg)):
        return {}
    # Avoid division by zero / sign flip in β-neutralization
    if abs(beta_long_avg) < 1e-6 or abs(beta_short_avg) < 1e-6:
        return {}

    weights: dict[str, float] = {}
    # Long leg: each weight scaled so leg β contribution = +1
    for t in long_tickers:
        weights[t] = (1.0 / n_leg) / beta_long_avg
    # Short leg: each weight scaled so leg β contribution = -1
    for t in short_tickers:
        weights[t] = -(1.0 / n_leg) / beta_short_avg

    # Final normalization: scale gross exposure to 1 (|weights|.sum() = 1)
    gross = sum(abs(w) for w in weights.values())
    if gross <= 0:
        return {}
    return {t: w / gross for t, w in weights.items()}


def _compute_low_vol_weights(
    closes:          pd.DataFrame,
    vol_window_days: int   = _VOL_WINDOW_DAYS,
    quintile_pct:    float = _LOW_VOL_QUINTILE_PCT,
) -> dict[str, float]:
    """Low-volatility factor weights (Baker, Bradley, Wurgler 2011 FAJ 67(1):40-54).

    Algorithm (per spec §2.1):
      1. Compute 252d annualized realized vol per ETF
      2. Rank by vol; bottom quintile = long leg; top quintile = short leg
      3. Equal weight within each leg

    Args:
        closes: DataFrame[date, ticker] of daily ETF closes.
        vol_window_days: Window for vol estimation; spec locks 252.
        quintile_pct: Long/short leg cutoff; spec locks 0.20.

    Returns:
        dict[ticker → weight] long-short, equal weight within leg, summing to ≈ 0.
    """
    _validate_closes(closes, "_compute_low_vol_weights")

    rets = _daily_returns(closes)
    rets = _trim_to_window(rets, vol_window_days)

    if len(rets) < max(60, vol_window_days // 4):
        return {}

    vols: dict[str, float] = {}
    for ticker in rets.columns:
        s = rets[ticker].dropna()
        if len(s) < max(60, vol_window_days // 4):
            continue
        std = s.std(ddof=1)
        if np.isfinite(std) and std > 0:
            vols[ticker] = float(std * np.sqrt(_TRADING_DAYS_PER_YEAR))

    if len(vols) < 6:  # need ≥ 6 for quintile split (rounding floors)
        return {}

    sorted_tickers = sorted(vols, key=lambda t: vols[t])
    n = len(sorted_tickers)
    n_leg = max(1, int(n * quintile_pct))
    long_tickers  = sorted_tickers[:n_leg]            # low vol
    short_tickers = sorted_tickers[-n_leg:]           # high vol

    weights: dict[str, float] = {}
    long_w  =  1.0 / (2 * n_leg)
    short_w = -1.0 / (2 * n_leg)
    for t in long_tickers:
        weights[t] = long_w
    for t in short_tickers:
        weights[t] = short_w
    return weights


def _compute_tsmom_weights(
    closes:           pd.DataFrame,
    lookback_days:    int   = _TSMOM_LOOKBACK_DAYS,
    skip_days:        int   = _TSMOM_SKIP_DAYS,
    target_vol_annual:float = _TARGET_VOL_ANNUAL,
) -> dict[str, float]:
    """Time-Series Momentum 12-1 (Moskowitz, Ooi, Pedersen 2012 JFE 104(2):228 eq.(3)).

    Algorithm (per spec §2.1):
      1. ret_12_1 = (P_{-skip} / P_{-lookback}) - 1   (12m return excl. last 1m)
      2. sign_i = sign(ret_12_1_i)
      3. position_i = sign_i × (target_vol / σ_i)    (vol-targeted; eq.(3))

    Args:
        closes: DataFrame[date, ticker] of daily ETF closes.
        lookback_days: 252 (spec locks 12-month formation).
        skip_days: 21 (spec locks 1-month skip).
        target_vol_annual: Vol target ratio (Moskowitz 2012 convention 10%).

    Returns:
        dict[ticker → vol-scaled signed weight]. Each weight in [-target_vol/σ, +target_vol/σ];
        normalized so |weights|.sum() = 1 (gross exposure 1).
    """
    _validate_closes(closes, "_compute_tsmom_weights")

    if len(closes) < lookback_days + 5:
        return {}

    # Reference dates for 12-1 return: 'lookback_days' ago and 'skip_days' ago
    p_t_minus_lookback = closes.iloc[-lookback_days]
    p_t_minus_skip     = closes.iloc[-skip_days - 1] if skip_days > 0 else closes.iloc[-1]

    rets = _daily_returns(closes)
    # Vol estimated over the same lookback window
    vol_window = rets.iloc[-lookback_days:] if len(rets) >= lookback_days else rets

    weights: dict[str, float] = {}
    for ticker in closes.columns:
        p_start = p_t_minus_lookback.get(ticker)
        p_end   = p_t_minus_skip.get(ticker)
        if not (np.isfinite(p_start) and np.isfinite(p_end)) or p_start <= 0:
            continue
        ret_12_1 = (p_end / p_start) - 1.0
        if not np.isfinite(ret_12_1) or ret_12_1 == 0.0:
            continue
        sign = 1.0 if ret_12_1 > 0 else -1.0

        # Vol scaling
        vol_series = vol_window[ticker].dropna() if ticker in vol_window.columns else pd.Series(dtype=float)
        if len(vol_series) < 60:
            continue
        sigma = float(vol_series.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        weights[ticker] = sign * (target_vol_annual / sigma)

    if not weights:
        return {}

    # Normalize gross exposure to 1
    gross = sum(abs(w) for w in weights.values())
    if gross <= 0:
        return {}
    return {t: w / gross for t, w in weights.items()}


def _compute_csmom_weights(
    closes:           pd.DataFrame,
    asset_classes:    dict[str, str],
    lookback_days:    int   = _TSMOM_LOOKBACK_DAYS,
    skip_days:        int   = _TSMOM_SKIP_DAYS,
    tertile_pct:      float = _CSMOM_TERTILE_PCT,
    target_vol_annual:float = _TARGET_VOL_ANNUAL,
) -> dict[str, float]:
    """Within-asset-class Cross-Sectional Momentum
    (Asness, Moskowitz, Pedersen 2013 JF 68(3):929).

    Algorithm (per spec §2.1):
      1. Within each asset class, compute ret_12_1 per ETF
      2. Rank within class; long top tertile, short bottom tertile
      3. Vol-targeted scaling per leg

    Args:
        closes: DataFrame[date, ticker] of daily ETF closes.
        asset_classes: dict[ticker → asset_class] (e.g. "equity_sector", "fixed_income").
        lookback_days: spec locks 252.
        skip_days: spec locks 21.
        tertile_pct: spec locks 1/3.
        target_vol_annual: 10% (Moskowitz 2012 convention).

    Returns:
        dict[ticker → weight], summing to ≈ 0 within each asset class
        (long-short within-class), gross-normalized to 1 across all classes.
    """
    _validate_closes(closes, "_compute_csmom_weights")
    if not asset_classes:
        raise ValueError("asset_classes must be non-empty dict[ticker, class]")

    if len(closes) < lookback_days + 5:
        return {}

    p_t_minus_lookback = closes.iloc[-lookback_days]
    p_t_minus_skip     = closes.iloc[-skip_days - 1] if skip_days > 0 else closes.iloc[-1]
    rets = _daily_returns(closes)
    vol_window = rets.iloc[-lookback_days:] if len(rets) >= lookback_days else rets

    # Compute ret_12_1 + σ per ticker
    momentum: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    for ticker in closes.columns:
        p_start = p_t_minus_lookback.get(ticker)
        p_end   = p_t_minus_skip.get(ticker)
        if not (np.isfinite(p_start) and np.isfinite(p_end)) or p_start <= 0:
            continue
        m = (p_end / p_start) - 1.0
        if not np.isfinite(m):
            continue
        if ticker not in vol_window.columns:
            continue
        s = vol_window[ticker].dropna()
        if len(s) < 60:
            continue
        sig = float(s.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if not np.isfinite(sig) or sig <= 0:
            continue
        momentum[ticker] = m
        sigmas[ticker]   = sig

    # Group by asset class
    class_to_tickers: dict[str, list[str]] = {}
    for t in momentum:
        cls = asset_classes.get(t)
        if cls is None:
            continue
        class_to_tickers.setdefault(cls, []).append(t)

    # Within each class: rank, take top/bottom tertile, vol-target each leg
    weights: dict[str, float] = {}
    for cls, tickers in class_to_tickers.items():
        if len(tickers) < 3:  # need ≥ 3 for tertile split
            continue
        sorted_in_class = sorted(tickers, key=lambda t: momentum[t])
        n = len(sorted_in_class)
        n_leg = max(1, int(n * tertile_pct))
        short_tickers = sorted_in_class[:n_leg]            # low momentum
        long_tickers  = sorted_in_class[-n_leg:]           # high momentum

        for t in long_tickers:
            weights[t] = +(target_vol_annual / sigmas[t])
        for t in short_tickers:
            weights[t] = -(target_vol_annual / sigmas[t])

    if not weights:
        return {}

    gross = sum(abs(w) for w in weights.values())
    if gross <= 0:
        return {}
    return {t: w / gross for t, w in weights.items()}


def _compute_donchian_trend_weights(
    closes:           pd.DataFrame,
    horizons:         Sequence[int] = _DONCHIAN_HORIZONS,
    target_vol_annual:float = _TARGET_VOL_ANNUAL,
) -> dict[str, float]:
    """Donchian breakout trend ensemble
    (Hurst, Ooi, Pedersen 2017 JPM 44(1):15-29).

    Algorithm (per spec §2.1):
      For each ticker, for each horizon h in (21, 63, 252):
        signal_h = +1 if P_t > max(P_{t-h:t-1})
                  -1 if P_t < min(P_{t-h:t-1})
                   0 otherwise
      Ensemble = mean of signals across horizons → range [-1, +1]
      Position weight = ensemble × (target_vol / σ)

    Args:
        closes: DataFrame[date, ticker].
        horizons: lookback horizons; spec locks (21, 63, 252) days.
        target_vol_annual: 10% convention.

    Returns:
        dict[ticker → weight]; long if ensemble > 0, short if < 0.
        Gross-normalized so |weights|.sum() = 1.
    """
    _validate_closes(closes, "_compute_donchian_trend_weights")

    max_h = max(horizons)
    if len(closes) < max_h + 2:
        return {}

    rets = _daily_returns(closes)
    vol_window = rets.iloc[-_VOL_WINDOW_DAYS:] if len(rets) >= _VOL_WINDOW_DAYS else rets

    weights: dict[str, float] = {}
    last_prices = closes.iloc[-1]

    for ticker in closes.columns:
        p_today = last_prices.get(ticker)
        if not np.isfinite(p_today):
            continue

        # Vol scaling
        if ticker not in vol_window.columns:
            continue
        vs = vol_window[ticker].dropna()
        if len(vs) < 60:
            continue
        sigma = float(vs.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
        if not np.isfinite(sigma) or sigma <= 0:
            continue

        # Compute ensemble signal across horizons
        signals: list[float] = []
        for h in horizons:
            if len(closes) < h + 1:
                continue
            window = closes[ticker].iloc[-h - 1:-1].dropna()  # last h obs, exclude today
            if len(window) < max(5, h // 4):
                continue
            hi = float(window.max())
            lo = float(window.min())
            if not (np.isfinite(hi) and np.isfinite(lo)):
                continue
            if p_today > hi:
                signals.append(+1.0)
            elif p_today < lo:
                signals.append(-1.0)
            else:
                signals.append(0.0)

        if not signals:
            continue
        ensemble = float(np.mean(signals))
        if ensemble == 0.0:
            continue
        weights[ticker] = ensemble * (target_vol_annual / sigma)

    if not weights:
        return {}

    gross = sum(abs(w) for w in weights.values())
    if gross <= 0:
        return {}
    return {t: w / gross for t, w in weights.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Production signal_fn wrappers — fetch real prices + call pure compute helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_universe_closes(
    as_of:             datetime.date,
    lookback_days:     int,
    extra_tickers:     Iterable[str] = (),
    min_history_years: int = 3,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch daily closes for the as-of universe + optional extras (e.g. SPY benchmark).

    Returns:
        (closes_df, asset_classes_dict)
        closes_df: DataFrame[date, ticker]; date index is DatetimeIndex.
        asset_classes_dict: dict[ticker → asset_class] (for CSMOM).
    """
    from engine.signal import _fetch_closes
    from engine.universe_manager import get_universe_as_of, get_universe_by_class

    sector_to_ticker = get_universe_as_of(as_of, min_history_years=min_history_years)
    tickers = sorted(set(sector_to_ticker.values()) | set(extra_tickers))
    if not tickers:
        return pd.DataFrame(), {}

    # Fetch ~lookback_days + buffer (10% extra) trading-day history
    days_to_fetch = int(lookback_days * 1.5) + 30  # buffer for weekends/holidays
    start = as_of - datetime.timedelta(days=int(days_to_fetch * 1.5))
    closes = _fetch_closes(tickers, start, as_of)
    if closes.empty:
        return pd.DataFrame(), {}

    # Build asset_classes map (ticker → class) by inverting universe_by_class
    universe_by_class = get_universe_by_class()
    asset_classes: dict[str, str] = {}
    for cls, sector_to_t in universe_by_class.items():
        for sector, ticker in sector_to_t.items():
            if ticker in closes.columns:
                asset_classes[ticker] = cls

    return closes, asset_classes


def _bab_signal_fn(as_of: datetime.date) -> dict[str, float]:
    """Frazzini-Pedersen 2014 BAB. Fetches yfinance daily closes for as-of universe
    + SPY benchmark, computes 252d β, returns β-neutralized long-short tertile weights."""
    closes, _ = _fetch_universe_closes(as_of, _BETA_WINDOW_DAYS, extra_tickers=["SPY"])
    if closes.empty or "SPY" not in closes.columns:
        logger.warning("_bab_signal_fn: insufficient data for as_of=%s", as_of)
        return {}
    bench = closes["SPY"]
    universe_closes = closes.drop(columns=["SPY"])
    return _compute_bab_weights(universe_closes, bench)


def _low_vol_signal_fn(as_of: datetime.date) -> dict[str, float]:
    """Baker-Bradley-Wurgler 2011 low-vol. Fetches yfinance daily closes,
    computes 252d realized vol, returns equal-weight long-short quintile weights."""
    closes, _ = _fetch_universe_closes(as_of, _VOL_WINDOW_DAYS)
    if closes.empty:
        logger.warning("_low_vol_signal_fn: insufficient data for as_of=%s", as_of)
        return {}
    return _compute_low_vol_weights(closes)


def _tsmom_12_1_signal_fn(as_of: datetime.date) -> dict[str, float]:
    """Moskowitz-Ooi-Pedersen 2012 TSMOM 12-1. Vol-targeted sign-of-12m-1m return."""
    closes, _ = _fetch_universe_closes(as_of, _TSMOM_LOOKBACK_DAYS)
    if closes.empty:
        logger.warning("_tsmom_12_1_signal_fn: insufficient data for as_of=%s", as_of)
        return {}
    return _compute_tsmom_weights(closes)


def _csmom_signal_fn(as_of: datetime.date) -> dict[str, float]:
    """Asness-Moskowitz-Pedersen 2013 CSMOM (within-class). Within-asset-class
    rank by 12-1 return; long top tertile, short bottom, vol-targeted."""
    closes, asset_classes = _fetch_universe_closes(as_of, _TSMOM_LOOKBACK_DAYS)
    if closes.empty or not asset_classes:
        logger.warning("_csmom_signal_fn: insufficient data for as_of=%s", as_of)
        return {}
    return _compute_csmom_weights(closes, asset_classes)


def _donchian_trend_signal_fn(as_of: datetime.date) -> dict[str, float]:
    """Hurst-Ooi-Pedersen 2017 Donchian trend ensemble across 1m / 3m / 12m horizons."""
    closes, _ = _fetch_universe_closes(as_of, max(_DONCHIAN_HORIZONS))
    if closes.empty:
        logger.warning("_donchian_trend_signal_fn: insufficient data for as_of=%s", as_of)
        return {}
    return _compute_donchian_trend_weights(closes)


# ─────────────────────────────────────────────────────────────────────────────
# FACTOR_REGISTRY (locked v1 per spec §2.1)
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_REGISTRY: dict[str, FactorSpec] = {
    "bab": FactorSpec(
        factor_id="bab",
        citation="Frazzini & Pedersen (2014) Journal of Financial Economics 111(1):1-25",
        asset_class="equity_etf",
        formula_summary=(
            "rank ETFs by 252d β to SPY; long bottom tertile (low β), "
            "short top tertile (high β); β-neutralize"
        ),
        signal_fn=_bab_signal_fn,
    ),
    "low_vol": FactorSpec(
        factor_id="low_vol",
        citation="Baker, Bradley, Wurgler (2011) Financial Analysts Journal 67(1):40-54",
        asset_class="equity_etf",
        formula_summary=(
            "rank ETFs by 252d realized vol; long bottom quintile, short top quintile"
        ),
        signal_fn=_low_vol_signal_fn,
    ),
    "tsmom_12_1": FactorSpec(
        factor_id="tsmom_12_1",
        citation="Moskowitz, Ooi, Pedersen (2012) Journal of Financial Economics 104(2):228 eq.(3)",
        asset_class="cross_asset",
        formula_summary=(
            "sign(12-month return excluding most recent month); position size = 1/σ_12mo"
        ),
        signal_fn=_tsmom_12_1_signal_fn,
    ),
    "csmom": FactorSpec(
        factor_id="csmom",
        citation="Asness, Moskowitz, Pedersen (2013) Journal of Finance 68(3):929",
        asset_class="equity_etf",
        formula_summary=(
            "within asset class, rank by 12-1 return; long top tertile, "
            "short bottom tertile, vol-targeted"
        ),
        signal_fn=_csmom_signal_fn,
    ),
    "donchian_trend": FactorSpec(
        factor_id="donchian_trend",
        citation="Hurst, Ooi, Pedersen (2017) Journal of Portfolio Management 44(1):15-29",
        asset_class="cross_asset",
        formula_summary=(
            "binary signal: +1 if price > 20d high, -1 if price < 20d low; "
            "ensemble 1m / 3m / 12m breakout horizons"
        ),
        signal_fn=_donchian_trend_signal_fn,
    ),
}
# inverse_vol DROPPED v1 — concept overlap with §2.4 risk-parity weighting.

# Spec §2.2 selection: pre-2010 in-sample retained list.
#
# LOCKED EMPTY VIA STAGE_1_FAIL VERDICT (2026-05-08):
#   In-sample analysis 2026-05-08 (script: run_factor_library_d4a.py) produced
#   STAGE_1_FAIL per spec_factor_library_v1.md §3.3 — 5/5 factors fail BHY-FDR
#   at α=0.05, N=5 (smallest p=0.2147 vs rank-1 threshold 0.00438; even raw 5%
#   one-sided gate fails by 4.3×).
#
#   Per spec §3.3 + §6 forbidden mods: no retesting, no threshold adjustment.
#   This empty tuple is the PERMANENT v1 result; future v2 amendment with a
#   different universe/factor list would need a new spec (new spec_id, new
#   EFFECTIVE_N_TRIALS).
#
#   Verdict file: docs/decisions/factor_library_v1_2026-05-08_STAGE_1_FAIL.md
#   Spec status: superseded (per spec §11 trigger b)
#   Production unchanged: PRODUCTION_SIGNAL = "ql01_bab" (single-factor BAB
#   from B++ Mass FDR 2010-2024 OOS test, Sharpe +0.985, t=+2.31).
SELECTED_FACTORS_V1: tuple[str, ...] = ()

# Spec §2.4 regime weighting: locked scalar table.
#
# NEVER DERIVED — STAGE_1_FAIL closed the ensemble path before regime scalar
# derivation (W1 D4b) was scheduled. Permanently empty for this spec.
REGIME_SCALAR_LOCKED: dict[str, dict[str, float]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Reusable: compute factor returns time series from cached prices
# ─────────────────────────────────────────────────────────────────────────────

# Map factor_id → (compute_fn, kwargs_builder) for compute_factor_returns_series.
# kwargs_builder receives (closes_window, asset_classes, benchmark_close) and
# returns the keyword-argument dict for the pure compute function.
def _bab_kwargs_builder(closes_window, asset_classes, benchmark_close):
    return {"closes": closes_window, "benchmark_close": benchmark_close}


def _low_vol_kwargs_builder(closes_window, asset_classes, benchmark_close):
    return {"closes": closes_window}


def _tsmom_kwargs_builder(closes_window, asset_classes, benchmark_close):
    return {"closes": closes_window}


def _csmom_kwargs_builder(closes_window, asset_classes, benchmark_close):
    return {"closes": closes_window, "asset_classes": asset_classes}


def _donchian_kwargs_builder(closes_window, asset_classes, benchmark_close):
    return {"closes": closes_window}


_FACTOR_COMPUTE_DISPATCH = {
    "bab":            (_compute_bab_weights,            _bab_kwargs_builder),
    "low_vol":        (_compute_low_vol_weights,        _low_vol_kwargs_builder),
    "tsmom_12_1":     (_compute_tsmom_weights,          _tsmom_kwargs_builder),
    "csmom":          (_compute_csmom_weights,          _csmom_kwargs_builder),
    "donchian_trend": (_compute_donchian_trend_weights, _donchian_kwargs_builder),
}


def compute_factor_returns_series(
    factor_id:       str,
    closes:          pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    asset_classes:   dict[str, str] | None = None,
    benchmark_close: pd.Series | None = None,
) -> pd.Series:
    """Walk-forward factor returns time series from a closes DataFrame.

    For each rebalance date t:
        1. Slice closes to all observations on or before t (no look-ahead)
        2. Call factor's pure compute_*_weights on the slice → weights at t
        3. Compute next-period factor return = Σ w_i × (P_{t+1} / P_t - 1) per ticker
           where P_{t+1} is the close on the next rebalance date

    Args:
        factor_id: FACTOR_REGISTRY key (one of the 5 v1 factors).
        closes: DataFrame[date, ticker] daily closes spanning all rebalance dates
            plus enough lookback (max 252 days) before the first.
        rebalance_dates: monthly rebalance timestamps (sorted ascending). Returns
            are computed BETWEEN consecutive dates; result Series index = dates[1:].
        asset_classes: dict[ticker → asset_class] (required for csmom; ignored for
            others).
        benchmark_close: Series[date] of benchmark daily closes (required for bab;
            ignored for others).

    Returns:
        pd.Series indexed by rebalance_dates[1:], values = factor returns.
        NaN where signal compute returned empty dict (insufficient data) or where
        no overlap with closes.
    """
    if factor_id not in _FACTOR_COMPUTE_DISPATCH:
        raise ValueError(
            f"unknown factor_id {factor_id!r}; expected one of "
            f"{list(_FACTOR_COMPUTE_DISPATCH)}"
        )
    if len(rebalance_dates) < 2:
        raise ValueError("need at least 2 rebalance_dates to compute returns")
    if not isinstance(closes.index, pd.DatetimeIndex):
        raise ValueError("closes.index must be DatetimeIndex")

    compute_fn, kwargs_builder = _FACTOR_COMPUTE_DISPATCH[factor_id]
    rebalance_dates = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in rebalance_dates))

    out_idx = rebalance_dates[1:]
    out_vals = np.full(len(out_idx), np.nan)

    for i, t_next in enumerate(rebalance_dates[1:]):
        t_now = rebalance_dates[i]

        # Closes on or before t_now (point-in-time slice, no look-ahead)
        window_mask = closes.index <= t_now
        closes_window = closes.loc[window_mask]
        if closes_window.empty:
            continue

        bench_window = None
        if benchmark_close is not None:
            bench_mask = benchmark_close.index <= t_now
            bench_window = benchmark_close.loc[bench_mask]
            if bench_window.empty:
                bench_window = None

        try:
            kwargs = kwargs_builder(closes_window, asset_classes, bench_window)
            weights = compute_fn(**kwargs)
        except Exception as exc:
            logger.warning(
                "compute_factor_returns_series[%s] @ %s: compute failed: %s",
                factor_id, t_now.date(), exc,
            )
            continue

        if not weights:
            continue

        # Next-period returns: P at or before t_next vs P at or before t_now per ticker
        try:
            p_now = closes.loc[closes.index <= t_now].iloc[-1]
            p_next_mask = (closes.index > t_now) & (closes.index <= t_next)
            p_next_slice = closes.loc[p_next_mask]
            if p_next_slice.empty:
                continue
            p_next = p_next_slice.iloc[-1]
        except (IndexError, KeyError):
            continue

        period_ret = 0.0
        used = 0
        for ticker, w in weights.items():
            if ticker not in p_now.index or ticker not in p_next.index:
                continue
            pn = p_now.get(ticker)
            pf = p_next.get(ticker)
            if not (np.isfinite(pn) and np.isfinite(pf)) or pn <= 0:
                continue
            r = (pf / pn) - 1.0
            if np.isfinite(r):
                period_ret += w * r
                used += 1

        if used > 0:
            out_vals[i] = period_ret

    return pd.Series(out_vals, index=out_idx, name=factor_id)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 BHY-FDR per-factor inclusion test (spec §3.2 Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def bhy_fdr_filter(
    p_values: dict[str, float],
    alpha:    float = 0.05,
) -> dict[str, bool]:
    """Benjamini-Hochberg-Yekutieli (2001) FDR correction for N candidates.

    Per spec §3.2 Stage 1 — applied to per-factor in-sample raw NW t-stat
    p-values. Uses the BY (BHY) variant which is more conservative than BH but
    valid under arbitrary dependence (BAB / Low-Vol / TSMOM are correlated).

    Procedure:
        1. Sort p-values ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(N)
        2. Compute c(N) = Σ_{k=1}^{N} (1/k)   (harmonic; BHY's correction factor)
        3. Reject H_(i) if p_(i) ≤ (i/N) · α / c(N)
        4. Once accepted at some rank, accept all earlier ranks too (step-up)

    Args:
        p_values: dict[factor_id → raw p-value]; values in [0, 1] or NaN.
            NaN p-values are treated as 1.0 (definitely not significant).
        alpha: FDR target; spec §3.2 locks 0.05.

    Returns:
        dict[factor_id → bool]; True iff factor passes BHY at α.

    Raises:
        ValueError: if p_values is empty or alpha out of (0, 1).
    """
    if not p_values:
        raise ValueError("p_values is empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    # Treat NaN as 1.0 (definitely not significant)
    cleaned = {f: (1.0 if (p is None or not np.isfinite(p)) else float(p))
               for f, p in p_values.items()}
    # Clip to [0, 1] for numerical safety
    cleaned = {f: float(np.clip(p, 0.0, 1.0)) for f, p in cleaned.items()}

    n = len(cleaned)
    cN = sum(1.0 / k for k in range(1, n + 1))

    # Sort ascending by p-value
    ranked = sorted(cleaned.items(), key=lambda kv: kv[1])
    # Step-up: find the largest rank r where p_(r) ≤ (r/N)·α/c(N); accept ranks 1..r
    threshold_at_rank = lambda r: (r / n) * alpha / cN
    last_pass_rank = 0
    for i, (_, p) in enumerate(ranked, start=1):
        if p <= threshold_at_rank(i):
            last_pass_rank = i

    accepted_ids = {f for f, _ in ranked[:last_pass_rank]}
    return {f: (f in accepted_ids) for f in cleaned}


# ─────────────────────────────────────────────────────────────────────────────
# Selection: greedy correlation filter (spec §2.2)
# ─────────────────────────────────────────────────────────────────────────────

def select_independent_factors(
    in_sample_returns: pd.DataFrame,
    candidates:        Iterable[str] | None = None,
    corr_threshold:    float = 0.7,
) -> list[str]:
    """Greedy selection: rank by in-sample Sharpe descending, drop if max
    Spearman corr to retained ≥ corr_threshold.

    Per spec §2.2 selection procedure (locked v1):
        1. Compute in-sample Sharpe ratio per candidate (1996-2009)
        2. Sort descending
        3. Greedy add: include factor if max(|Spearman corr| to retained) < threshold
        4. Lock retained list — no re-running on OOS

    Args:
        in_sample_returns: DataFrame indexed by date, columns = factor_id, values
            = monthly returns over in-sample period (1996-01 to 2009-12).
        candidates: Subset of FACTOR_REGISTRY keys to consider. Default: all 5.
        corr_threshold: Spearman correlation threshold above which a candidate
            is dropped (per spec §2.2 lock = 0.7).

    Returns:
        Ordered list of retained factor_ids (descending Sharpe at retention time).

    Raises:
        ValueError: if any candidate is not in FACTOR_REGISTRY.
        ValueError: if in_sample_returns is empty or has unrecognized columns.
    """
    cands = list(candidates) if candidates is not None else list(FACTOR_REGISTRY.keys())

    unknown = [c for c in cands if c not in FACTOR_REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown candidates {unknown}; FACTOR_REGISTRY keys = {list(FACTOR_REGISTRY)}"
        )
    if in_sample_returns.empty:
        raise ValueError("in_sample_returns is empty; cannot compute Sharpe")
    missing_cols = [c for c in cands if c not in in_sample_returns.columns]
    if missing_cols:
        raise ValueError(
            f"in_sample_returns missing columns for candidates {missing_cols}"
        )
    if not (0.0 < corr_threshold < 1.0):
        raise ValueError(
            f"corr_threshold must be in (0, 1); got {corr_threshold}"
        )

    # Per-candidate in-sample annualized Sharpe (Lo 2002 frequency convention).
    sharpe = {}
    for c in cands:
        r = in_sample_returns[c].dropna()
        if len(r) < 12:  # need ≥ 1 year of monthly data for stable Sharpe
            sharpe[c] = float("-inf")
            continue
        mean_ret = r.mean()
        std_ret = r.std(ddof=1)
        if std_ret <= 0 or not np.isfinite(std_ret):
            sharpe[c] = float("-inf")
        else:
            # Assume monthly returns → annualize by √12
            sharpe[c] = (mean_ret / std_ret) * np.sqrt(12)

    # Spearman rank correlation matrix on overlapping in-sample observations.
    corr = in_sample_returns[cands].corr(method="spearman")

    # Greedy: sort by Sharpe descending, retain if max |corr| to already-retained < threshold.
    ranked = sorted(cands, key=lambda c: sharpe[c], reverse=True)
    retained: list[str] = []
    for c in ranked:
        if sharpe[c] == float("-inf"):
            continue   # skip degenerate Sharpe
        if not retained:
            retained.append(c)
            continue
        max_abs_corr = max(abs(corr.loc[c, r]) for r in retained)
        if max_abs_corr < corr_threshold:
            retained.append(c)

    return retained


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble: ERC + regime-aware weighting (spec §2.3 + §2.4)
# ─────────────────────────────────────────────────────────────────────────────

def build_ensemble_weights(
    factor_signals:    dict[str, pd.Series],
    regime_label:      str,
    regime_scalars:    dict[str, dict[str, float]] | None = None,
    rolling_vol_window_months: int = 12,
) -> pd.Series:
    """Risk-Parity (ERC) base weights × regime scalar; project to spec constraints.

    Per spec §2.3 + §2.4:
        1. Equal Risk Contribution (ERC): w_i = (1/σ_i) / Σⱼ (1/σⱼ),
           σ_i computed from monthly factor returns via 12-month rolling window
           (no look-ahead).
        2. Multiply by regime_scalars[regime_label][factor_id] ∈ [0.5, 1.5]
           (locked from in-sample regime-conditional Sharpe quintiles, per §2.4).
        3. Renormalize to Σw = 1.
        4. Apply constraints: max single weight ≤ 25%, vol target ≤ 10% annualized;
           simplex-project if violated.

    Args:
        factor_signals: dict[factor_id → pd.Series of monthly factor returns
            (most recent rolling_vol_window_months observations sufficient)].
        regime_label: current regime label (e.g., "risk-on" / "transition" / "risk-off").
        regime_scalars: nested dict[regime_label][factor_id → scalar in [0.5, 1.5]].
            Default: REGIME_SCALAR_LOCKED (populated W1 D4).
        rolling_vol_window_months: window for σ_i computation; spec §2.3 locks 12.

    Returns:
        pd.Series[factor_id → weight], summing to 1, each ≤ 0.25.

    Status: skeleton — full implementation pending W2 sprint (constraint
    projection + regime scalar application). Currently returns ERC-only
    placeholder; raises NotImplementedError for non-trivial use.
    """
    raise NotImplementedError(
        "build_ensemble_weights pending W2 sprint per spec §2.3 + §2.4 "
        "(ERC + regime scalar + simplex projection on max-weight + vol-target constraints)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API surface
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "FactorSpec",
    "SignalFn",
    "FACTOR_REGISTRY",
    "SELECTED_FACTORS_V1",
    "REGIME_SCALAR_LOCKED",
    "select_independent_factors",
    "build_ensemble_weights",
    "compute_factor_returns_series",
    "bhy_fdr_filter",
    # Pure compute helpers (testable without yfinance)
    "_compute_bab_weights",
    "_compute_low_vol_weights",
    "_compute_tsmom_weights",
    "_compute_csmom_weights",
    "_compute_donchian_trend_weights",
]
