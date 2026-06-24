"""
Factor Lab (P-LAB) — production-grade factor proposal & validation pipeline.

Spec: docs/spec_factor_lab.md
Pre-registered: 2026-05-08 (retro=False, n_trials_contributed=0 — infrastructure spec)

Public API:
  factor_lab.power.required_sample_size_sharpe_diff(...)
  factor_lab.power.power_check(...)
  factor_lab.types.FactorState  (enum: DRAFT, PROPOSED, BLOCKED_UNDERPOWERED,
                                  REGISTERED, TESTING, PASS, MARGINAL, FAIL,
                                  FAIL_UNDERPOWERED)

Boundary invariant (per spec §4):
  Modules `power.py` and (Session 2) `runner.py` must NOT import any LLM
  client (genai / openai / google.generativeai). Tier R rule
  `rule_no_llm_in_factor_lab_evaluation` enforces this statically.
"""
from engine.factor_lab.types import (
    FactorState, PowerCheckResult, IllegalTransition, assert_legal_transition,
)
from engine.factor_lab.power import (
    required_sample_size_sharpe_diff,
    achieved_power_at_n,
    power_check,
)
from engine.factor_lab.registry import (
    list_active_candidates,
    list_legacy_specs,
    list_infrastructure_specs,
    get_candidate,
    state_counts,
    transition_state,
)
from engine.factor_lab.runner import (
    run_factor_lab_test,
)

__all__ = [
    "FactorState",
    "PowerCheckResult",
    "IllegalTransition",
    "assert_legal_transition",
    "required_sample_size_sharpe_diff",
    "achieved_power_at_n",
    "power_check",
    "list_active_candidates",
    "list_legacy_specs",
    "list_infrastructure_specs",
    "get_candidate",
    "state_counts",
    "transition_state",
    "run_factor_lab_test",
]
