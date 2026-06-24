"""
engine/factors/ — Multi-Strategy Factor Ensemble v1 factor modules.

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50, hash 1665945d2ca5)
Spec section: §2.2 Factor Definitions

Each factor exposes:
    compute_*_signal(as_of, universe, asset_classes, ...) -> pd.Series

Returns per-ticker signal value or np.nan per §2.3 NaN protocol:
  - Insufficient history → NaN
  - Non-applicable factor for asset class → NaN
  - Excluded (e.g., Carry on commodity per equity-only scope) → NaN

Boundary invariant (project rule "0-LLM-in-evaluation"):
  All factor signals are pure deterministic functions; no LLM in this directory.
"""
from engine.factors.tsmom import compute_tsmom_signal
from engine.factors.carry_equity import compute_carry_equity_signal
from engine.factors.quality import compute_quality_signal
from engine.factors.bab_compat import compute_bab_signal

__all__ = [
    "compute_tsmom_signal",
    "compute_carry_equity_signal",
    "compute_quality_signal",
    "compute_bab_signal",
]
