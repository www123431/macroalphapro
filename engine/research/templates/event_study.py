"""engine/research/templates/event_study.py — generic event-drift template.

CRITICAL DESIGN PER [[project-senior-pipeline-roadmap-2026-05-30]]:
  Event-study portfolios trip up naive implementations because OVERLAPPING
  EVENTS (same firm having multiple events within window, or multiple
  firms having events in same month) produce **dependent** observations.
  Plain monthly L/S t-stats overstate significance 2-3x in this regime.
  This template handles it by:
    1. Holding portfolio is *event-time aligned* — each firm contributes
       returns only for months it's within the holding window
    2. Returns aggregated at month-level (not event-level) so the
       gate's existing HAC-aware machinery sees a clean monthly series
    3. Equal-weight across all firms currently "active" (within window)
       so the implicit firm-month clustering happens at construction
  This is the Daniel-Hirshleifer-Subrahmanyam 1998 / Bernard-Thomas 1989
  post-earnings-drift architecture.

USE CASES (covered by this template via different event_panel sources):
  - Post-earnings-announcement drift (PEAD)
  - Insider-purchase drift
  - Analyst upgrade/downgrade drift
  - Pre-FOMC drift
  - Index-reconstitution drift

Binding schema:
  hold_months           — int, holding window length M (e.g. 3 for PEAD)
  skip_first_month      — bool, drop month 0 (typical to avoid micro-structure)
  cost_bps_per_side     — float (e.g. 25.0 for high-turnover event book)
  vol_target            — float | null (e.g. 0.10)
  vol_target_lookback   — int (default 36)

Inputs (passed via run_gate data_kwargs):
  return_panel: monthly returns wide DataFrame (dates × tickers)
  event_panel:  bool DataFrame (dates × tickers); True = event happens
                 in that month for that ticker
  benchmark_returns: optional Series; if provided, output is L/S
                     (event_basket - benchmark); else long-only abnormal

Returns:
  pd.Series of monthly L/S net-of-cost returns
"""
from __future__ import annotations

import pandas as pd

from engine.research import primitives as P

# Per [[project-gate-production-redesign-2026-05-30]]:
# Event-driven holds 3+ months → residual autocorrelation → HAC lags=12.
# Equity universe → pead_control=True. 2D grid (hold × skip_first) → 8 trials.
# Sparse events → OOS must split by event count, not time-bisect. Caller
# of run_gate must pass event_density Series alongside this profile.
GATE_PROFILE = {
    "hac_lags":         12,
    "cost_bps_default": 25,
    "pead_control":     True,
    "n_trials_base":    8,
    "oos_split":        "event_count",
}


def event_density_from_panel(event_panel: pd.DataFrame) -> pd.Series:
    """Compute per-month event count from event_panel for run_gate's
    event_count OOS split."""
    return event_panel.astype(int).sum(axis=1)


def warmup_months(binding: dict) -> int:
    """Event template has no rolling-window construction → minimal warmup."""
    b = binding or {}
    if b.get("vol_target") is not None:
        return int(b.get("vol_target_lookback", 36)) + 1
    return int(b.get("hold_months", 3)) + 1


def _active_basket_at_each_month(
    event_panel: pd.DataFrame,
    *,
    hold_months: int,
    skip_first_month: bool = True,
) -> pd.DataFrame:
    """For each (date, ticker), True iff ticker is within the holding
    window of any prior event.

    Returns: boolean DataFrame same shape as event_panel.
    """
    # Shift event by 1 (skip event month itself if skip_first_month) or 0
    start_offset = 1 if skip_first_month else 0
    # active = any event in [t - hold_months, t - start_offset]
    # Use rolling-OR over the appropriate window
    shifted = event_panel.shift(start_offset)
    # Rolling-max with window=hold_months over boolean → "any True in window"
    active = (shifted
                .rolling(window=hold_months, min_periods=1)
                .max()
                .fillna(0).astype(bool))
    return active


def run_event_study(
    *,
    return_panel: pd.DataFrame,
    event_panel: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    hold_months: int = 3,
    skip_first_month: bool = True,
    cost_bps_per_side: float = 25.0,
    vol_target: float | None = 0.10,
    vol_target_lookback: int = 36,
) -> pd.Series:
    """Compose primitives into monthly event-drift L/S series.

    Pipeline:
      1. Derive active basket at each month (rolling-OR of events)
      2. Equal-weight returns across active basket members
      3. Subtract benchmark if provided (else long-only abnormal)
      4. Apply turnover-based cost (active set churn × bps)
      5. vol_target_normalize (if specified)
    """
    # 1. Align panels
    common_idx = return_panel.index.intersection(event_panel.index)
    common_cols = return_panel.columns.intersection(event_panel.columns)
    rp = return_panel.loc[common_idx, common_cols]
    ep = event_panel.loc[common_idx, common_cols].astype(bool)

    # 2. Active basket each month
    active = _active_basket_at_each_month(
        ep, hold_months=hold_months, skip_first_month=skip_first_month,
    )

    # 3. Equal-weight returns across active basket
    n_active = active.sum(axis=1)
    basket_returns = (rp.where(active, 0.0).sum(axis=1)
                         / n_active.where(n_active > 0))
    basket_returns = basket_returns.dropna().astype(float)

    # 4. Subtract benchmark if available
    if benchmark_returns is not None:
        bench_aligned = benchmark_returns.reindex(basket_returns.index).fillna(0)
        out = basket_returns - bench_aligned
    else:
        out = basket_returns

    # 5. Turnover cost — basket composition changes each month
    # Turnover proxy: (active.diff().abs().sum(axis=1) / n_active * 2)
    # Cost = turnover × bps_per_side
    churn = active.astype(int).diff().abs().sum(axis=1)
    turnover = (churn / n_active.where(n_active > 0)) * 2.0
    turnover = turnover.astype(float).reindex(out.index).fillna(0.0)
    cost = turnover * (cost_bps_per_side / 10000.0)
    out = out.astype(float) - cost

    # 6. Vol target
    if vol_target is not None:
        out = P.vol_target_normalize(
            out, target_vol=vol_target, lookback_months=vol_target_lookback,
        )

    return out.dropna()
