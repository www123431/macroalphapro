"""
engine/agents/dq_inspector/ — Data Quality Inspector Agent v1.0 (spec id=70).

Mirrors engine.agents.risk_manager structure. Files:
  - agent.py             : entry point + result dataclass + DQInspectorAgent
  - gates.py             : 10 deterministic detectors (Phase 2)
  - thresholds.py        : RISK_THRESHOLDS singleton + per-source dicts (Phase 3)
  - source_inspectors.py : per-source freshness/coverage helpers (Phase 4)
  - persist.py           : DataQualityAlert SQLAlchemy persistence (Phase 5)
  - narrator.py          : DQ-specific 11-template DeterministicNarrator (Phase 7,
                           mirrors RM narrator structure with DQ mode coverage)

DOCTRINE: same as Risk Manager (spec id=69) — see RM module docstring for
0-LLM-in-DECISION / spec-lock / risk-side invariants. DQ Inspector
inherits these without restating.
"""
from engine.agents.dq_inspector.agent import (
    DQInspectorAgent,
    DQInspectorRunResult,
    run_dq_check,
)
from engine.agents.dq_inspector.narrator import (
    DeterministicNarrator,
    GeminiFlashNarrator,
    NarrationResult,
    PersonaContext,
    contains_banned_phrase,
    narrate_breach,
    narrate_run_result,
)

__all__ = [
    "DQInspectorAgent",
    "DQInspectorRunResult",
    "run_dq_check",
    # Phase 7 narrator (deterministic backend default; gemini deferred)
    "DeterministicNarrator",
    "GeminiFlashNarrator",
    "NarrationResult",
    "PersonaContext",
    "contains_banned_phrase",
    "narrate_breach",
    "narrate_run_result",
]
