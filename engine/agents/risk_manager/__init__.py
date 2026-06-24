"""
engine/agents/risk_manager/ — Risk Manager Agent v1.0 (spec id=69).
Current hash + amendment log live in SpecRegistry table; call
engine.preregistration.list_specs() or engine.agents.persona.tools.lookup_spec(69)
for the canonical state. 2026-05-19 §2.1a Q1a/Q1b two-tier cap amend recorded.

Public API exports for the agent. Internals are split across:
  - agent.py        : entry point + result dataclass + RiskManagerAgent class
  - gates.py        : 12 deterministic detectors (Phase 2)
  - thresholds.py   : Tier-3-locked frozen thresholds + BOOK_SINGLE_TICKER_ABS_CAP
                       + SLEEVE_CLASS_INTRA_CAPS (Phase 3 + 2026-05-19 §2.1a amend)
  - persist.py      : RiskManagerAlert SQLAlchemy persistence (Phase 4)
  - narrator.py     : Gemini Flash LLM narration layer (Phase 7)
  - advisory.py     : Engineer-PR sign-off API (Phase 8)

Reading guide for the Engineer agent / human reviewers:
  - The DECISION authority for halt-or-proceed lives in gates.py (pure
    deterministic functions). 0-LLM-in-DECISION preserved.
  - The LLM layer (narrator.py) only generates one-paragraph English
    summaries AFTER the halt decision is made.
  - All thresholds in thresholds.py are spec-amendment-locked.
"""
from engine.agents.risk_manager.agent import (
    RiskManagerAgent,
    RiskManagerRunResult,
    run_risk_manager_check,
)
from engine.agents.risk_manager.gates import (
    Breach,
    evaluate_all_modes,
    classify_severity,
    any_hard_halt,
)
from engine.agents.risk_manager.orchestrator_hook import (
    pre_trade_gate,
    post_trade_gate,
)
from engine.agents.risk_manager.cb_absorption import (
    unified_circuit_state,
    persist_risk_manager_severe,
    get_circuit_state_breakdown,
)
from engine.agents.risk_manager.narrator import (
    NarrationResult,
    PersonaContext,
    narrate_breach,
    narrate_run_result,
    contains_banned_phrase,
)
from engine.agents.risk_manager.advisory import (
    SignOffResult,
    sign_off,
    render_sign_off_markdown,
)

__all__ = [
    # Phase 1 agent shell
    "RiskManagerAgent",
    "RiskManagerRunResult",
    "run_risk_manager_check",
    # Phase 2 gates
    "Breach",
    "evaluate_all_modes",
    "classify_severity",
    "any_hard_halt",
    # Phase 5 circuit-breaker absorption
    "unified_circuit_state",
    "persist_risk_manager_severe",
    "get_circuit_state_breakdown",
    # Phase 6 orchestrator hooks
    "pre_trade_gate",
    "post_trade_gate",
    # Phase 7 narrator
    "NarrationResult",
    "PersonaContext",
    "narrate_breach",
    "narrate_run_result",
    "contains_banned_phrase",
    # Phase 8 advisory
    "SignOffResult",
    "sign_off",
    "render_sign_off_markdown",
]
