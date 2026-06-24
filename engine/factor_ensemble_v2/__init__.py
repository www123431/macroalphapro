"""
engine/factor_ensemble_v2 — Robust v2 spec implementation.

Pre-registration: docs/spec_factor_ensemble_v2_robust.md (id=51, hash c6d395ad0fb7)
Project axis: per project_reframe_quant_alpha_agentic_ops_2026-05-09.md

Modules:
  - tc.py:               TC modeling (8bps locked per FP 2014)
  - beta_neutral.py:     TSMOM β-neutralization (AFP 2014 pattern, TSMOM only)
  - regime.py:           4-regime classifier (Goyal-Welch + Moreira-Muir)
  - multi_baseline.py:   4-baseline runner (BAB / 60-40 / equal-weight / SPY-buy-hold)
  - verdict.py:          extended verdict (per-baseline + per-regime aggregation)

v1 (id=50) stays untouched. v2 is independent additive enhancement.
"""
from engine.factor_ensemble_v2.tc import (
    TC_BPS_ROUNDTRIP_LOCKED,
    compute_tc_drag,
    apply_tc_to_realized_returns,
)
from engine.factor_ensemble_v2.beta_neutral import (
    BETA_NEUTRAL_FACTORS_LOCKED,
    compute_beta_panel,
    beta_neutralize_tsmom,
)
from engine.factor_ensemble_v2.regime import (
    REGIMES_LOCKED,
    REGIME_VOL_THRESHOLD_LOCKED,
    REGIME_RETURN_THRESHOLD_LOCKED,
    classify_regime,
    classify_regime_series,
)
from engine.factor_ensemble_v2.multi_baseline import (
    BASELINE_DEFINITIONS_LOCKED,
    run_baseline,
)
from engine.factor_ensemble_v2.verdict import (
    compute_v2_verdict,
    V2VerdictResult,
)

__all__ = [
    "TC_BPS_ROUNDTRIP_LOCKED",
    "compute_tc_drag",
    "apply_tc_to_realized_returns",
    "BETA_NEUTRAL_FACTORS_LOCKED",
    "compute_beta_panel",
    "beta_neutralize_tsmom",
    "REGIMES_LOCKED",
    "REGIME_VOL_THRESHOLD_LOCKED",
    "REGIME_RETURN_THRESHOLD_LOCKED",
    "classify_regime",
    "classify_regime_series",
    "BASELINE_DEFINITIONS_LOCKED",
    "run_baseline",
    "compute_v2_verdict",
    "V2VerdictResult",
]
