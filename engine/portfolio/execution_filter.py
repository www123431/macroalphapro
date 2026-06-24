"""engine/portfolio/execution_filter.py — Phase 5.7: Cost-Aware
Execution Filter (CA filter) at the signal-to-trade boundary.

⚠ SCOPE NOTE (2026-06-01 fitness review): only cross_asset_carry among
deployed sleeves fits the CA filter semantics. Other deployed sleeves
declare ca_filter_k_method: not_applicable in their YAML for sleeve-
shape reasons (event-driven, regime-trigger, continuous-weight, etc).
should_trade() and apply_ca_filter_to_panel() remain valid building
blocks for any future high-turnover rank-based L/S sleeve. See
[[project-multi-asset-ca-filter-gap-2026-06-01]] +
[[feedback-pre-implementation-fitness-check-2026-06-01]].


Implements the BTC paper's CA gate
    |expected_return| > k × |position_change| × tcost
adapted for OUR heterogeneous signal types via the 5.6 taxonomy.

Two surfaces:
  (1) Per-period gate: should_trade() → bool + diagnostic
  (2) Series-level helper: apply_ca_filter_to_returns() →
      counterfactual filtered series, ready to feed into 5.5 scaffold
      for PBB-validated A/B testing.

Per-sleeve k:
  - Default 2.0 (paper's value, validated empirically there)
  - Per-sleeve override via library YAML `cost_model.ca_filter_k`
  - Selection method: run 5.5 evaluate_k_sweep on historical data,
    pick the SMALLEST k that ships DEPLOY verdict under Hochberg
    correction (minimum intervention while staying significant)

Companion: [[project-paper-borrow-ml-btc-costs-2026-06-01]] item 5.7.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from engine.portfolio.signal_taxonomy import (
    SignalType, calibrate, get_sleeve_spec,
)

logger = logging.getLogger(__name__)

DEFAULT_K = 2.0


@dataclass(frozen=True)
class CADecision:
    """Output of should_trade — boolean + diagnostic."""
    trade:                  bool
    reason:                 str
    calibrated_er:          Optional[float]
    method:                 str
    confident_calibration:  bool
    cost_threshold:         float
    k:                      float
    tcost_round_trip:       float
    position_change:        float
    signal_type:            Optional[str]


def should_trade(
    sleeve_id:             str,
    raw_signal:            float,
    current_position:      float,
    target_position:       float,
    tcost_round_trip:      float,
    k:                     float = DEFAULT_K,
    historical_panel:      Optional[pd.DataFrame] = None,
) -> CADecision:
    """Apply Cost-Aware Execution Filter to one trade decision.

    Returns CADecision { trade: bool, ...diagnostic }. Caller is
    expected to use .trade to gate execution; the rest of the fields
    are for trace_log / Cockpit visibility.

    Conservative defaults:
      - Unknown sleeve_id → trade (don't silently suppress; surface
        in reason for senior review)
      - Zero position change → don't trade (no cost to pay either way)
      - calibration not confident → still apply gate but flag in diag
    """
    position_change = abs(target_position - current_position)
    if position_change == 0:
        return CADecision(
            trade=False, reason="zero position change",
            calibrated_er=None, method="n/a",
            confident_calibration=True,
            cost_threshold=0.0, k=k,
            tcost_round_trip=tcost_round_trip,
            position_change=0.0,
            signal_type=None,
        )

    spec = get_sleeve_spec(sleeve_id)
    if spec is None:
        # No taxonomy → conservative: trade. Surface for senior fix.
        return CADecision(
            trade=True, reason=f"no taxonomy for sleeve {sleeve_id!r}",
            calibrated_er=None, method="n/a",
            confident_calibration=False,
            cost_threshold=0.0, k=k,
            tcost_round_trip=tcost_round_trip,
            position_change=position_change,
            signal_type=None,
        )

    cal = calibrate(spec.signal_type, raw_signal, historical_panel)
    cost_threshold = k * position_change * tcost_round_trip
    trade = abs(cal.expected_return) > cost_threshold
    reason = (
        f"|ER {cal.expected_return:.4f}| > threshold {cost_threshold:.4f}"
        if trade else
        f"|ER {cal.expected_return:.4f}| <= threshold {cost_threshold:.4f}"
        " — hold position"
    )
    return CADecision(
        trade=trade, reason=reason,
        calibrated_er=cal.expected_return,
        method=cal.method,
        confident_calibration=cal.confident,
        cost_threshold=cost_threshold,
        k=k, tcost_round_trip=tcost_round_trip,
        position_change=position_change,
        signal_type=spec.signal_type.value,
    )


# ── Series-level counterfactual for 5.5 scaffold ───────────────────────


def apply_ca_filter_to_returns(
    sleeve_id:             str,
    raw_signal_series:     pd.Series,       # signal per period
    gross_returns_series:  pd.Series,       # what the strategy WOULD earn
    tcost_round_trip:      float,
    k:                     float = DEFAULT_K,
    historical_panel:      Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Build a counterfactual returns series WITH CA filter applied.

    Trade-or-hold per period; when held, the sleeve continues to earn
    yesterday's signed exposure on today's gross return (since position
    is unchanged); when trading, the sleeve incurs tcost.

    SIMPLIFICATION (5.7 v1): assumes a position-1 long-only sleeve for
    the cost accounting; binary trade/hold gate. More general N-asset
    position logic to ship with per-sleeve adapter (5.7b).
    """
    aligned = pd.concat([
        raw_signal_series.rename("signal"),
        gross_returns_series.rename("gross"),
    ], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    # current position state — start flat, target = sign(signal)
    cur = 0.0
    out_returns = np.zeros(len(aligned), dtype=float)
    signals = aligned["signal"].values
    grosses = aligned["gross"].values

    for i, (sig, gross) in enumerate(zip(signals, grosses)):
        target = float(np.sign(sig)) if abs(sig) > 0 else 0.0
        dec = should_trade(
            sleeve_id=sleeve_id,
            raw_signal=float(sig),
            current_position=cur,
            target_position=target,
            tcost_round_trip=tcost_round_trip,
            k=k,
            historical_panel=historical_panel,
        )
        if dec.trade:
            # Realize cost on the change in position
            cost = abs(target - cur) * tcost_round_trip
            cur = target
            out_returns[i] = cur * gross - cost
        else:
            # Hold — no cost, position earns on gross
            out_returns[i] = cur * gross

    return pd.Series(out_returns, index=aligned.index)


def apply_ca_filter_to_panel(
    sleeve_id:             str,
    signal_panel:          pd.DataFrame,    # [t × contract]
    returns_panel:         pd.DataFrame,    # [t × contract], realized next-period
    target_position_panel: pd.DataFrame,    # [t × contract] ∈ {-1, 0, +1}
    tcost_round_trip:      float,
    k:                     float = DEFAULT_K,
    historical_panel:      Optional[pd.DataFrame] = None,
    leg_q_map:             Optional[dict[str, float]] = None,
) -> tuple[pd.Series, dict]:
    """Multi-asset CA filter for risk-parity L/S sleeves.

    For each (month t, contract c), compute should_trade given:
      raw_signal       = signal_panel[t, c]    (contract's carry yield)
      current_position = previous month's filtered position for c
      target_position  = target_position_panel[t, c]  (rank-based intent)
      tcost            = tcost_round_trip
    Trade or hold per contract; aggregate to sleeve-level return:
      sleeve_ret[t+1] = mean(realized[c | new_pos[c]=+1]) -
                         mean(realized[c | new_pos[c]=-1]) -
                         sum(|Δpos[c]| × tcost for traded contracts)

    Returns (sleeve_returns, diagnostics) where diagnostics carries
    actual turnover statistics so the caller can validate
    monthly_turnover_estimate.

    NOTE: addresses the single-asset abstraction defect caught in
    2026-06-01 senior post-audit. See
    [[project-multi-asset-ca-filter-gap-2026-06-01]].
    """
    months = sorted(set(target_position_panel.index)
                     & set(returns_panel.index))
    current_pos = pd.Series(0.0, index=target_position_panel.columns)
    out_rows: list[tuple[pd.Timestamp, float]] = []
    n_trade_events = 0
    n_decision_events = 0
    cumulative_turnover = 0.0

    for t in months:
        target = target_position_panel.loc[t]
        signal = signal_panel.loc[t] if t in signal_panel.index else pd.Series(
            0.0, index=target.index,
        )

        new_pos = current_pos.copy()
        period_cost = 0.0
        period_turnover = 0.0

        for c in target.index:
            sig_c = signal.get(c, float("nan"))
            cur_c = float(current_pos.get(c, 0.0))
            tgt_c = float(target.get(c, 0.0))
            if pd.isna(sig_c) or pd.isna(tgt_c):
                continue
            n_decision_events += 1
            decision = should_trade(
                sleeve_id=sleeve_id,
                raw_signal=float(sig_c),
                current_position=cur_c,
                target_position=tgt_c,
                tcost_round_trip=tcost_round_trip,
                k=k,
                historical_panel=historical_panel,
            )
            if decision.trade:
                delta = abs(tgt_c - cur_c)
                period_cost += delta * tcost_round_trip
                period_turnover += delta
                new_pos[c] = tgt_c
                n_trade_events += 1
            # else: hold previous position

        # Realize next-period sleeve return
        rets = returns_panel.loc[t] if t in returns_panel.index else None
        if rets is not None:
            longs  = new_pos[new_pos > 0]
            shorts = new_pos[new_pos < 0]
            r_long  = (rets.reindex(longs.index).mean()
                        if len(longs) else 0.0)
            r_short = (rets.reindex(shorts.index).mean()
                        if len(shorts) else 0.0)
            r_long_clean  = 0.0 if pd.isna(r_long)  else float(r_long)
            r_short_clean = 0.0 if pd.isna(r_short) else float(r_short)
            period_ret = r_long_clean - r_short_clean - period_cost
            out_rows.append((t, period_ret))

        cumulative_turnover += period_turnover
        current_pos = new_pos

    sleeve_returns = pd.Series(dict(out_rows)).sort_index()
    n_months = max(1, len(out_rows))
    diagnostics = {
        "n_months":            n_months,
        "n_decision_events":   n_decision_events,
        "n_trade_events":      n_trade_events,
        "trade_rate_pct":      round(n_trade_events / max(1, n_decision_events) * 100, 2),
        "monthly_turnover":    round(cumulative_turnover / n_months, 4),
        "annual_turnover":     round(cumulative_turnover / n_months * 12, 4),
    }
    return sleeve_returns, diagnostics
