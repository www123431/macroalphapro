"""
engine/factor_ensemble_v2/multi_baseline.py — 4-baseline runner.

Pre-registration: docs/spec_factor_ensemble_v2_robust.md §2.4

4 locked baselines:
  1. bab_only       — production-equivalent (same as v1 baseline)
  2. sixty_forty    — 60% SPY + 40% AGG, monthly rebalance
  3. equal_weight   — 1/N across equity_sector + equity_factor (24 tickers)
  4. spy_buyhold    — pure SPY buy & hold from 2011-01-31

All run through SAME walk-forward harness with TC modeling enabled.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from engine.factor_ensemble_v2.tc import compute_tc_drag, TC_BPS_ROUNDTRIP_LOCKED

logger = logging.getLogger(__name__)

# Locked per spec §2.4
BASELINE_DEFINITIONS_LOCKED: tuple[str, ...] = (
    "bab_only",
    "sixty_forty",
    "equal_weight",
    "spy_buyhold",
)


def _compute_baseline_weights(
    baseline_id: str,
    as_of:       datetime.date,
    universe:    list[str],
    asset_classes: dict[str, str],
    is_first_period: bool,
    panel:       Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Compute weights for one baseline at one rebalance date."""
    if baseline_id == "bab_only":
        # Same as v1 baseline — uses the existing factor_ensemble_walk_forward path
        # We re-implement here as direct call to compute_bab_signal + inv-vol weighting
        from engine.factors import compute_bab_signal
        from engine.factor_ensemble_walk_forward import _compute_weights_from_signal
        sig = compute_bab_signal(as_of=as_of, universe=universe, asset_classes=asset_classes, use_cache=True)
        return _compute_weights_from_signal(sig, as_of, panel=panel)

    if baseline_id == "sixty_forty":
        # Static 60% SPY + 40% AGG
        out = pd.Series(0.0, index=universe, dtype=float)
        if "SPY" in universe:
            out["SPY"] = 0.60
        if "AGG" in universe:
            out["AGG"] = 0.40
        return out[out != 0]

    if baseline_id == "equal_weight":
        # 1/N across equity_sector + equity_factor
        eq = [t for t, c in asset_classes.items() if c in {"equity_sector", "equity_factor"}]
        eq = [t for t in eq if t in universe]
        if not eq:
            return pd.Series(dtype=float)
        w = 1.0 / len(eq)
        return pd.Series({t: w for t in eq}, dtype=float)

    if baseline_id == "spy_buyhold":
        # First period only: 100% SPY. Subsequent periods: 0 turnover (positions roll).
        if not is_first_period:
            return pd.Series(dtype=float)  # signaling "no rebalance"
        if "SPY" in universe:
            return pd.Series({"SPY": 1.0}, dtype=float)
        return pd.Series(dtype=float)

    raise ValueError(f"Unknown baseline_id: {baseline_id}")


def _run_spy_buyhold_special(
    rebalance_dates: list[datetime.date],
    panel: Optional[pd.DataFrame],
    bps_roundtrip: float = TC_BPS_ROUNDTRIP_LOCKED,
) -> dict:
    """Special-case SPY buy-and-hold: 1 establishment cost at period 0, then
    monthly returns are SPY's actual realized return between rebalance dates.

    Walk-forward harness loop has rebalance-or-skip semantics that don't fit
    'buy once, hold forever' — bypass it by computing directly from panel.
    """
    if panel is None or panel.empty or "SPY" not in panel.columns:
        return {
            "baseline_id":           "spy_buyhold",
            "monthly_returns_gross": pd.Series(dtype=float),
            "monthly_returns_net":   pd.Series(dtype=float),
            "turnover_per_period":   pd.Series(dtype=float),
            "n_successful_periods":  0,
        }

    spy = panel["SPY"].dropna()
    if spy.empty:
        return {
            "baseline_id":           "spy_buyhold",
            "monthly_returns_gross": pd.Series(dtype=float),
            "monthly_returns_net":   pd.Series(dtype=float),
            "turnover_per_period":   pd.Series(dtype=float),
            "n_successful_periods":  0,
        }

    # First period: establishment TC (full position from cash)
    establishment_tc = 0.5 * (bps_roundtrip / 10000.0)  # turnover=0.5 since starting from cash to 100% SPY

    monthly_records = []
    for i in range(len(rebalance_dates) - 1):
        period_start = rebalance_dates[i]
        period_end = rebalance_dates[i + 1]
        # Get SPY price at-or-before period_start and period_end
        before_start = spy.loc[spy.index <= pd.Timestamp(period_start)]
        before_end = spy.loc[spy.index <= pd.Timestamp(period_end)]
        if before_start.empty or before_end.empty:
            continue
        p_start = float(before_start.iloc[-1])
        p_end = float(before_end.iloc[-1])
        if p_start <= 0:
            continue
        gross_return = p_end / p_start - 1
        # TC: only the very first period bears establishment cost; subsequent periods 0
        tc = establishment_tc if i == 0 else 0.0
        monthly_records.append({
            "rebal_date":           period_start,
            "monthly_return_gross": gross_return,
            "tc_drag":              tc,
            "monthly_return_net":   gross_return - tc,
            "turnover":             0.5 if i == 0 else 0.0,
            "n_positions":          1,
        })

    if not monthly_records:
        return {
            "baseline_id":           "spy_buyhold",
            "monthly_returns_gross": pd.Series(dtype=float),
            "monthly_returns_net":   pd.Series(dtype=float),
            "turnover_per_period":   pd.Series(dtype=float),
            "n_successful_periods":  0,
        }
    df = pd.DataFrame(monthly_records).set_index("rebal_date")
    return {
        "baseline_id":           "spy_buyhold",
        "monthly_returns_gross": df["monthly_return_gross"],
        "monthly_returns_net":   df["monthly_return_net"],
        "turnover_per_period":   df["turnover"],
        "n_successful_periods":  len(df),
        "diagnostics_df":        df,
    }


def run_baseline(
    baseline_id: str,
    rebalance_dates: list[datetime.date],
    universe_at_date_fn,
    asset_classes_fn,
    panel:       Optional[pd.DataFrame],
    bps_roundtrip: float = TC_BPS_ROUNDTRIP_LOCKED,
) -> dict:
    """Run a baseline through the walk-forward harness.

    Args:
        baseline_id: one of BASELINE_DEFINITIONS_LOCKED
        rebalance_dates: list of month-end dates (same as v1 harness)
        universe_at_date_fn(date): callable returning {sector: ticker} dict
        asset_classes_fn(universe): callable returning {ticker: asset_class}
        panel: pre-fetched price panel (passed to compute_realized_return)

    Returns:
        dict with keys:
          monthly_returns_gross: pd.Series indexed by rebalance_date
          monthly_returns_net:   pd.Series (gross - tc_drag)
          turnover_per_period:   pd.Series
          n_successful_periods:  int
    """
    if baseline_id not in BASELINE_DEFINITIONS_LOCKED:
        raise ValueError(f"baseline_id {baseline_id!r} not in locked set {BASELINE_DEFINITIONS_LOCKED}")

    # Special-case spy_buyhold: walk-forward harness loop doesn't fit "buy once
    # hold forever". Compute directly from SPY panel.
    if baseline_id == "spy_buyhold":
        return _run_spy_buyhold_special(
            rebalance_dates=rebalance_dates,
            panel=panel,
            bps_roundtrip=bps_roundtrip,
        )

    from engine.factor_ensemble_walk_forward import _compute_realized_return

    monthly_records: list[dict] = []
    prev_weights: Optional[pd.Series] = None

    for i, rebal_date in enumerate(rebalance_dates):
        u_dict = universe_at_date_fn(rebal_date)
        if not u_dict:
            continue
        universe = list(u_dict.values())
        ac = asset_classes_fn(universe)

        try:
            weights = _compute_baseline_weights(
                baseline_id=baseline_id,
                as_of=rebal_date,
                universe=universe,
                asset_classes=ac,
                is_first_period=(prev_weights is None),
                panel=panel,
            )
        except Exception as exc:
            logger.warning("baseline %s @ %s: weight compute failed: %s", baseline_id, rebal_date, exc)
            continue

        # spy_buyhold special: subsequent periods have no rebalance, but realized
        # return must be computed against PREVIOUS period's positions (drift OK)
        if baseline_id == "spy_buyhold" and i > 0:
            effective_weights = prev_weights if prev_weights is not None else weights
        else:
            effective_weights = weights if not weights.empty else (prev_weights if prev_weights is not None else weights)

        if effective_weights.empty:
            continue

        next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        if next_rebal is None:
            break

        try:
            gross_return = _compute_realized_return(
                weights=effective_weights,
                period_start=rebal_date,
                period_end=next_rebal,
                panel=panel,
            )
        except Exception as exc:
            logger.warning("baseline %s @ %s: realized return failed: %s", baseline_id, rebal_date, exc)
            continue

        # TC: spy_buyhold has 0 turnover after first period (drift, no rebalance)
        if baseline_id == "spy_buyhold" and i > 0:
            tc = 0.0
        else:
            tc = compute_tc_drag(weights_new=effective_weights, weights_prev=prev_weights, bps_roundtrip=bps_roundtrip)

        monthly_records.append({
            "rebal_date":       rebal_date,
            "monthly_return_gross": gross_return,
            "tc_drag":          tc,
            "monthly_return_net":   gross_return - tc,
            "turnover":         tc / (bps_roundtrip / 10000.0) if bps_roundtrip > 0 else 0.0,
            "n_positions":      int((effective_weights != 0).sum()),
        })

        # Update prev_weights for TC accounting next period
        if baseline_id == "spy_buyhold" and i > 0:
            pass  # prev_weights stays
        else:
            prev_weights = effective_weights

    if not monthly_records:
        return {
            "baseline_id": baseline_id,
            "monthly_returns_gross": pd.Series(dtype=float),
            "monthly_returns_net":   pd.Series(dtype=float),
            "turnover_per_period":   pd.Series(dtype=float),
            "n_successful_periods":  0,
        }

    df = pd.DataFrame(monthly_records).set_index("rebal_date")
    return {
        "baseline_id":           baseline_id,
        "monthly_returns_gross": df["monthly_return_gross"],
        "monthly_returns_net":   df["monthly_return_net"],
        "turnover_per_period":   df["turnover"],
        "n_successful_periods":  len(df),
        "diagnostics_df":        df,
    }
