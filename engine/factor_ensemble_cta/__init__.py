"""
engine/factor_ensemble_cta — Path O CTA Defensive Overlay sleeve (spec id=73).

Per spec_path_o_cta_defensive_overlay_v1.md hash 9630c2bb:
  - 10% portfolio allocation to PQTIX (PIMCO TRENDS Managed Futures Strategy)
  - Annual rebalance + ±2% drift trigger
  - 25 bp roundtrip TC per rebalance event
  - 5 deployability gates (NOT alpha gates)
  - Verdict labels: SAA_DEPLOYABLE / SAA_MARGINAL / SAA_INFEASIBLE

Replaces deprecated crypto SAA sleeve (specs 71/72). Empirically-validated
crisis-positive behavior across 2018-Q4 / 2020-COVID / 2022-Inflation crises.

This is portfolio construction (manager outsourcing), NOT alpha hypothesis testing.
"""
from engine.factor_ensemble_cta.data_loader import (
    UNIVERSE_LOCKED,
    EQUITY_PROXY_TICKER,
    WINDOW_START_LOCKED,
    WINDOW_END_LOCKED,
    load_cta_panel,
)
from engine.factor_ensemble_cta.tc import TC_BPS_PER_EVENT_LOCKED
from engine.factor_ensemble_cta.saa import (
    SPEC_ID,
    SLEEVE_ID,
    CTA_WEIGHT_IN_PORTFOLIO,
    SAABacktestResult,
    run_saa_backtest,
    evaluate_saa_verdict,
)

__all__ = [
    "UNIVERSE_LOCKED",
    "EQUITY_PROXY_TICKER",
    "WINDOW_START_LOCKED",
    "WINDOW_END_LOCKED",
    "load_cta_panel",
    "TC_BPS_PER_EVENT_LOCKED",
    "SPEC_ID",
    "SLEEVE_ID",
    "CTA_WEIGHT_IN_PORTFOLIO",
    "SAABacktestResult",
    "run_saa_backtest",
    "evaluate_saa_verdict",
]
