"""
Macro CTA Research — self-built CTA horse race infrastructure (Phase 2).

5-spec pre-registered horse race vs PQTIX baseline. Public API:

  - load_universe_weekly:  yfinance loader + cache + dataset hash
  - run_backtest:          monthly-rebalanced backtest engine (signal-agnostic)
  - evaluate_gates:        4-gate evaluation vs PQTIX baseline
  - write_capability_evidence: per-spec MD output

Per `docs/spec_path_p_macro_trend_xsec_momentum_v1.md` §2.1 etc., all 5 specs
share locked universe (TLT/HYG/DBC/GLD), window (2014-09-12 → 2023-12-29),
sizing convention, TC model, gates, statistical framework.
"""
from engine.portfolio.macro_cta_research.universe import (
    load_universe_weekly,
    get_dataset_hash,
    HORSE_RACE_UNIVERSE,
    PQTIX_TICKER,
    REGIME_INDICATOR,
    VOL_INDICATOR,
    WINDOW_START,
    WINDOW_END,
)
from engine.portfolio.macro_cta_research.backtest import (
    run_backtest,
    ewma_volatility,
    vol_target_weights,
    BacktestResult,
)
from engine.portfolio.macro_cta_research.gate_eval import (
    evaluate_gates,
    load_other_sleeves_combined_weekly,
    GateResult,
)
from engine.portfolio.macro_cta_research.crisis_windows import (
    CRISIS_WINDOWS,
    crisis_positive_count,
)
from engine.portfolio.macro_cta_research.output import write_capability_evidence

__all__ = [
    # Universe
    "load_universe_weekly",
    "get_dataset_hash",
    "HORSE_RACE_UNIVERSE",
    "PQTIX_TICKER",
    "REGIME_INDICATOR",
    "VOL_INDICATOR",
    "WINDOW_START",
    "WINDOW_END",
    # Backtest
    "run_backtest",
    "ewma_volatility",
    "vol_target_weights",
    "BacktestResult",
    # Gates
    "evaluate_gates",
    "load_other_sleeves_combined_weekly",
    "GateResult",
    # Crisis
    "CRISIS_WINDOWS",
    "crisis_positive_count",
    # Output
    "write_capability_evidence",
]
