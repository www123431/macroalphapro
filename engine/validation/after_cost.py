"""engine/validation/after_cost.py — convert GROSS strategy returns to NET.

Phase 1 ran deflated Sharpe / factor attribution / decay on the per-
strategy weekly returns, which the gross-vs-reported reconciliation
showed are GROSS (book ~1.32 Sharpe vs reported ~0.54). This module
applies a defensible per-strategy transaction-cost drag and re-runs
deflated Sharpe on the NET series, so the deployment decision rests on
honest after-cost numbers.

Cost = annual_turnover x round_trip_cost_bps, subtracted uniformly per
week (annual_drag / 52). This preserves vol / skew / kurtosis and
shifts the mean down — the right first-order treatment when only the
return series (not positions) is available.

Round-trip cost bps are derived from the existing instrument-class TC
tiers in engine.execution.cost_model (base + half-spread, doubled for
enter+exit). Turnover is estimated from rebalance frequency; it is the
weakest input, so a BASE and a HIGH scenario are both reported.

THE turnover numbers are estimates (no position-level data). Treat the
output as a sensitivity band, not a point estimate. The robust takeaway
is the RELATIVE ranking + which strategies collapse under cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Per-strategy cost spec. round_trip_bps from engine.execution.cost_model
# instrument tiers (one-way base+half-spread, x2 for enter+exit).
# annual_turnover = book-fraction traded per year (base / high scenario).
@dataclass(frozen=True)
class CostSpec:
    instrument_note:   str
    round_trip_bps:    float
    turnover_base:     float
    turnover_high:     float


COST_SPECS: dict[str, CostSpec] = {
    # 43 ETFs (Tier1/Tier2 mix). ETF round-trip ~10-16bp. Monthly
    # rebalance, partial cross-sectional turnover.
    "K1_BAB": CostSpec("ETF Tier1/2", round_trip_bps=14.0,
                       turnover_base=5.0, turnover_high=8.0),
    # top-1500 single stocks, mostly large/mid. SS round-trip ~26-40bp.
    # 60d rebalance, high cross-sectional name rotation.
    "D_PEAD": CostSpec("SS large/mid", round_trip_bps=32.0,
                       turnover_base=5.0, turnover_high=8.0),
    # S&P reconstitution names, event-driven. Handled precisely by the
    # event-level cost_stress; here use a period-equivalent for the book.
    # 24 events/yr, full position each → high effective turnover.
    "PATH_N": CostSpec("SS large (event, crowded)", round_trip_bps=30.0,
                       turnover_base=12.0, turnover_high=24.0),
    # PQTIX mutual fund — trades at NAV, no bid-ask. Near-zero cost.
    "CTA_PQTIX": CostSpec("mutual fund (NAV)", round_trip_bps=0.0,
                          turnover_base=1.5, turnover_high=3.0),
    # TLT/GLD Tier1 ETFs, monthly 50/50 rebalance — very low turnover.
    "AC_proxy_AB_2014_23": CostSpec("ETF Tier1 insurance", round_trip_bps=10.0,
                                    turnover_base=1.5, turnover_high=3.0),
}


@dataclass(frozen=True)
class NetResult:
    strategy:          str
    gross_ann_return:  float
    gross_deflated_sr: float
    annual_drag_base:  float
    net_ann_return_base: float
    net_deflated_sr_base: float
    annual_drag_high:  float
    net_deflated_sr_high: float
    verdict:           str


def apply_cost(returns: pd.Series, annual_drag: float, ppy: int = 52) -> pd.Series:
    """Subtract a uniform per-period cost drag from a gross return series."""
    return returns - (annual_drag / ppy)


def net_audit(
    strat_returns: pd.DataFrame,
    n_trials:      int = 35,
    ppy:           int = 52,
) -> dict[str, NetResult]:
    """Re-run deflated Sharpe on NET-of-cost returns, base + high cost.

    Returns {strategy: NetResult}. The deflated Sharpe is recomputed on
    the cost-adjusted series so the multiple-testing correction applies
    to the honest after-cost edge.
    """
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio

    out: dict[str, NetResult] = {}
    for col in strat_returns.columns:
        gross = strat_returns[col].dropna()
        spec = COST_SPECS.get(col)
        if spec is None:
            continue

        gross_ann = float(gross.mean() * ppy)
        gross_dsr = deflated_sharpe_ratio(gross.values, n_trials=n_trials).deflated_sr

        drag_base = spec.turnover_base * spec.round_trip_bps / 10000.0
        drag_high = spec.turnover_high * spec.round_trip_bps / 10000.0

        net_base = apply_cost(gross, drag_base, ppy)
        net_high = apply_cost(gross, drag_high, ppy)
        dsr_base = deflated_sharpe_ratio(net_base.values, n_trials=n_trials).deflated_sr
        dsr_high = deflated_sharpe_ratio(net_high.values, n_trials=n_trials).deflated_sr

        # Verdict on after-cost survival of the multiple-testing bar.
        if dsr_base >= 0.90:
            verdict = "SURVIVES cost — net deflated SR still strong"
        elif dsr_base >= 0.70:
            verdict = "HOLDS — net deflated SR moderate"
        elif dsr_base >= 0.50:
            verdict = "WEAKENS materially under cost"
        else:
            verdict = "COLLAPSES under cost"

        out[col] = NetResult(
            strategy=col, gross_ann_return=gross_ann, gross_deflated_sr=gross_dsr,
            annual_drag_base=drag_base, net_ann_return_base=float(net_base.mean()*ppy),
            net_deflated_sr_base=dsr_base, annual_drag_high=drag_high,
            net_deflated_sr_high=dsr_high, verdict=verdict,
        )
    return out
