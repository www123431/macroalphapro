"""
engine.execution — transaction-cost and capacity modeling.

Tier-1 audit #3 (2026-05-14) — addresses "Execution model thin / TC=10bp
hardcoded" critique. Provides ADV-aware TC estimates and capacity warnings
for paper-trade pipeline and Positions page UI.

Public API:
  - cost_model.estimate_tc_bps(...)  → 3-component TC estimate per fill
  - cost_model.classify_instrument(ticker) → 'etf' / 'single_stock' / 'mutual_fund'
  - cost_model.compute_portfolio_tc(positions, ...) → portfolio-level TC + caps
"""
from engine.execution.cost_model import (
    estimate_tc_bps,
    classify_instrument,
    compute_portfolio_tc,
    CAPACITY_WARN_FRAC,
    InstrumentClass,
    KNOWN_ETFS,
    KNOWN_MUTUAL_FUNDS,
)

__all__ = [
    "estimate_tc_bps",
    "classify_instrument",
    "compute_portfolio_tc",
    "CAPACITY_WARN_FRAC",
    "InstrumentClass",
    "KNOWN_ETFS",
    "KNOWN_MUTUAL_FUNDS",
]
