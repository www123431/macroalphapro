"""
Signal-agnostic backtest engine for self-built CTA horse race (Path P/Q/S/T/U).

Design contract — strict signal/framework decoupling:
  - signal_fn returns FINAL intra-sleeve weights (already sized)
  - Framework just applies them: monthly rebalance · 1-week execution lag · TC
  - Vol-targeting helpers exposed for specs to call; framework does NOT auto-vol-target

This decoupling is essential because Path U (Vol-Scaled Risk Parity) has its own
gross-scaling overlay that would conflict with framework-level vol-targeting.
Path P/Q/S/T also do their own vol-target via the helper.

Per spec §2.4: monthly rebalance, 1-day execution lag (with weekly bars, 1-week
lag at rebalance is the closest analog). TC = 10bp per side applied as return
haircut at rebalance bar.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers exposed to spec signal_fn implementations
# ─────────────────────────────────────────────────────────────────────────────

def ewma_volatility(returns: pd.DataFrame,
                    lambda_: float = 0.94,
                    lookback: int = 60,
                    annualize_factor: float = 52.0) -> pd.Series:
    """EWMA annualized volatility per ticker. Uses last `lookback` weekly returns.

    Standard RiskMetrics λ=0.94. Default annualization 52 (weekly data).
    """
    recent = returns.tail(lookback).dropna(how="all")
    if len(recent) < 4:
        return pd.Series(np.nan, index=returns.columns)
    weights = np.array([lambda_ ** k for k in range(len(recent) - 1, -1, -1)])
    weights /= weights.sum()
    weighted = recent.fillna(0).values * weights[:, None]
    ewma_var = (weighted * recent.fillna(0).values).sum(axis=0)
    ewma_vol = np.sqrt(np.maximum(ewma_var, 0) * annualize_factor)
    return pd.Series(ewma_vol, index=recent.columns)


def vol_target_weights(raw_weights: pd.Series,
                       recent_returns: pd.DataFrame,
                       target_vol_annualized: float = 0.10) -> pd.Series:
    """Scale raw_weights so realized portfolio vol equals target.

    Computes sample annualized cov from recent_returns (weekly · annualization 52).
    """
    cov = recent_returns.dropna().cov() * 52.0
    # Align weights to cov tickers
    w = raw_weights.reindex(cov.index).fillna(0).values
    port_var = float(w @ cov.values @ w)
    if port_var <= 1e-12:
        return raw_weights * 0
    port_vol = port_var ** 0.5
    scale = target_vol_annualized / port_vol
    return raw_weights * scale


# ─────────────────────────────────────────────────────────────────────────────
# Backtest result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class BacktestResult:
    """Output of run_backtest()."""
    weekly_returns: pd.Series           # post-TC weekly returns
    nav:            pd.Series           # cumulative NAV starting from 1.0
    holdings:       pd.DataFrame        # per-rebalance-date weights × ticker
    turnover:       pd.Series           # per-rebalance turnover (sum |Δw|)
    n_rebalances:   int
    n_weeks:        int
    spec_label:     str = ""

    @property
    def sharpe(self) -> float:
        """Annualized Sharpe (RFR=4%)."""
        if len(self.weekly_returns) < 4:
            return float("nan")
        ann_ret = float(self.weekly_returns.mean() * 52)
        ann_vol = float(self.weekly_returns.std() * np.sqrt(52))
        if ann_vol <= 1e-9:
            return float("nan")
        return (ann_ret - 0.04) / ann_vol

    @property
    def max_drawdown(self) -> float:
        if self.nav.empty:
            return float("nan")
        running_peak = self.nav.cummax()
        dd = (self.nav / running_peak) - 1
        return float(dd.min())

    @property
    def ann_return(self) -> float:
        return float(self.weekly_returns.mean() * 52) if len(self.weekly_returns) else float("nan")

    @property
    def ann_vol(self) -> float:
        return float(self.weekly_returns.std() * np.sqrt(52)) if len(self.weekly_returns) else float("nan")

    @property
    def avg_turnover_per_rebalance(self) -> float:
        return float(self.turnover.mean()) if len(self.turnover) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest engine — signal-agnostic
# ─────────────────────────────────────────────────────────────────────────────

# Signal function signature:
#   signal_fn(prices_up_to_t: pd.DataFrame, as_of: pd.Timestamp,
#             extras: dict | None) -> pd.Series[ticker → weight]
SignalFn = Callable[[pd.DataFrame, pd.Timestamp, Optional[dict]], pd.Series]


def _is_first_friday_of_month(date: pd.Timestamp, prev_date: pd.Timestamp | None) -> bool:
    """True if date is the first Friday of a new month relative to prev_date."""
    if prev_date is None:
        return True
    return date.month != prev_date.month


def run_backtest(
    signal_fn:          SignalFn,
    prices:             pd.DataFrame,
    *,
    universe:           tuple[str, ...],
    tc_bps_per_side:    float = 10.0,
    warmup_weeks:       int   = 52,
    extras:             Optional[dict] = None,
    spec_label:         str = "",
) -> BacktestResult:
    """Monthly-rebalanced backtest.

    Per spec §2.4 invariants:
      - Rebalance on first Friday of each month (monthly cadence)
      - Execution lag: signal computed at end of week t → applied at t+1
      - TC = tc_bps_per_side × |Δweights| per rebalance, applied as return haircut
      - Within month: weights held constant (drift-adjusted return computed)

    Args:
      signal_fn:   spec-provided callable computing target weights
      prices:      weekly Friday-close prices · columns include universe + extras
      universe:    tickers the signal operates on (subset of prices.columns)
      tc_bps_per_side: locked at 10 per spec §2.5
      warmup_weeks: bars to skip before signal can be computed (52w for 12m signals)
      extras:      passed to signal_fn (e.g., {"vix": vix_series, "spy": spy_series})

    Returns:
      BacktestResult with weekly_returns (net of TC), NAV, holdings, turnover.
    """
    if extras is None:
        extras = {}

    px = prices[list(universe)].copy()
    n = len(px)
    if n <= warmup_weeks + 4:
        raise ValueError(f"insufficient data: {n} weeks ≤ warmup {warmup_weeks}+4")

    # Holdings track t-1 → t period: weight applied at start of bar produces P&L for that bar
    current_weights = pd.Series(0.0, index=universe)

    weekly_returns: list[float] = []
    weekly_dates:   list[pd.Timestamp] = []
    holdings_log:   list[dict] = []
    turnover_log:   list[tuple[pd.Timestamp, float]] = []
    prev_date:      pd.Timestamp | None = None

    for i, t in enumerate(px.index):
        # ─── Compute period return BEFORE potentially rebalancing at t ──────
        if i > 0 and i > warmup_weeks:
            prev_t = px.index[i - 1]
            # Per-ticker return for the bar
            bar_returns = (px.loc[t] / px.loc[prev_t]) - 1
            # Portfolio return at this bar using weights held during bar
            port_return = float((current_weights * bar_returns).sum())
            weekly_returns.append(port_return)
            weekly_dates.append(t)

        # ─── Rebalance check (first Friday of new month, after warmup) ──────
        if i >= warmup_weeks and _is_first_friday_of_month(t, prev_date):
            prices_so_far = px.iloc[: i + 1]
            try:
                new_weights = signal_fn(prices_so_far, t, extras)
                # Coerce to Series indexed by universe; missing = 0
                new_weights = pd.Series(new_weights).reindex(universe).fillna(0)
            except Exception as exc:
                logger.warning("signal_fn failed at %s for %s: %s", t, spec_label, exc)
                new_weights = current_weights.copy()

            # Apply TC at the rebalance bar (subtract from this bar's return)
            turnover = float((new_weights - current_weights).abs().sum())
            tc_drag = turnover * tc_bps_per_side / 10_000.0
            if weekly_returns:
                weekly_returns[-1] -= tc_drag

            current_weights = new_weights
            holdings_log.append({"date": t, **new_weights.to_dict()})
            turnover_log.append((t, turnover))

        prev_date = t

    # Build result
    weekly_ret_s = pd.Series(weekly_returns, index=pd.DatetimeIndex(weekly_dates), name="ret")
    nav = (1.0 + weekly_ret_s).cumprod()
    holdings_df = pd.DataFrame(holdings_log).set_index("date") if holdings_log else pd.DataFrame()
    turnover_s = (pd.Series([v for _, v in turnover_log],
                             index=pd.DatetimeIndex([d for d, _ in turnover_log]),
                             name="turnover")
                  if turnover_log else pd.Series(dtype=float, name="turnover"))

    return BacktestResult(
        weekly_returns = weekly_ret_s,
        nav            = nav,
        holdings       = holdings_df,
        turnover       = turnover_s,
        n_rebalances   = len(turnover_log),
        n_weeks        = len(weekly_ret_s),
        spec_label     = spec_label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test helper: equal-weight buy-and-hold baseline (NOT a spec)
# Used to verify the framework pipeline before Phase 3 spec signal_fn impls.
# ─────────────────────────────────────────────────────────────────────────────

def _equal_weight_signal(prices_so_far, as_of, extras=None) -> pd.Series:
    """Buy-and-hold equal-weighted (no vol-target). Smoke test only."""
    n = prices_so_far.shape[1]
    return pd.Series(1.0 / n, index=prices_so_far.columns)
