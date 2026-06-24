"""
engine/factor_ensemble_singlename — Stage 2 single-stock walk-forward.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52)
Project axis: per project_reframe_quant_alpha_agentic_ops_2026-05-09.md (量化主线)

Modules:
  - walk_forward.py     — single-stock walk-forward harness (Wave A 3-factor / Wave B 4-factor)
  - panel_fetcher.py    — chunked yfinance bulk fetch for 500-stock universe
  - verdict.py          — PRELIMINARY_* / PASS/PARTIAL/FAIL aggregator (extends v2)

Reuses heavily from v1/v2:
  - engine.factor_ensemble_walk_forward._compute_realized_return + _panel_slice
  - engine.factor_ensemble_v2.tc.compute_tc_drag (12bps single-stock)
  - engine.factor_ensemble_v2.beta_neutral.beta_neutralize_tsmom
  - engine.factor_ensemble_v2.regime (4-regime classifier)
  - engine.factor_ensemble.compute_ensemble_signal (cross-section z-score + NaN-aware avg)
  - engine.multivariate_msm_verdict (bootstrap CI + Memmel Z)
"""
from engine.factor_ensemble_singlename.panel_fetcher import (
    CHUNK_SIZE_LOCKED,
    bulk_fetch_singlestock_panel,
)
from engine.factor_ensemble_singlename.walk_forward import (
    OOS_START_DATE_WAVE_A,
    OOS_END_DATE_WAVE_A,
    TC_BPS_LOCKED,
    VOL_TARGET_LOCKED,
    MAX_LEVERAGE_LOCKED,
    MAX_NAME_WEIGHT_LOCKED,
    run_singlestock_walk_forward,
    SinglestockWalkForwardResult,
)
from engine.factor_ensemble_singlename.verdict import (
    compute_singlestock_verdict,
    SinglestockVerdictResult,
)

__all__ = [
    "CHUNK_SIZE_LOCKED",
    "bulk_fetch_singlestock_panel",
    "OOS_START_DATE_WAVE_A",
    "OOS_END_DATE_WAVE_A",
    "TC_BPS_LOCKED",
    "VOL_TARGET_LOCKED",
    "MAX_LEVERAGE_LOCKED",
    "MAX_NAME_WEIGHT_LOCKED",
    "run_singlestock_walk_forward",
    "SinglestockWalkForwardResult",
    "compute_singlestock_verdict",
    "SinglestockVerdictResult",
]
