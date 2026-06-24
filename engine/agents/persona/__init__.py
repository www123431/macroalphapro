"""
engine/agents/persona/ — Persona Voice Layer (β.2 Chat UI).

Per [[project-agent-team-persona-locked-2026-05-18]] β.2 mini-sprint
and 施工 decision A (per-agent independent chat history, no shared
mega-chat). Multi-agent group chat UI is DEFERRED until ≥3 personas
built; Pattern 5 autonomous-debate is BANNED indefinitely.

Architecture (post-2026-05-19 refactor):
  base.py            — AgentPersona dataclass + generic chat_turn() loop
  tools.py           — shared read-only tool registry + select_tools(names)
                       per-agent subset helper
  session_store.py   — Phase A.3 SQLite ε memory persistence
  risk_manager.py    — RISK_MANAGER     (BUILD COMPLETE)
  dq_inspector.py    — DQ_INSPECTOR     (BUILD COMPLETE)
  devils_advocate.py — DEVILS_ADVOCATE  (BUILD COMPLETE)
  anomaly_sentinel.py    — ANOMALY_SENTINEL    (BUILD COMPLETE 2026-05-19)
  attribution_analyst.py — ATTRIBUTION_ANALYST (BUILD COMPLETE 2026-05-19)
  audit_recorder.py      — AUDIT_RECORDER      (BUILD COMPLETE 2026-05-19)
  chief_of_staff.py      — CHIEF_OF_STAFF      (BUILD COMPLETE 2026-05-19;
                            Supervisor pattern; routes to the 6 specialists;
                            spec id=74)

Each new agent = one AgentPersona instance + one Streamlit page wrapper
calling ui.components.chat_page.render_chat_page(persona). No new loop
code, no duplicated UI code.
"""
from engine.agents.persona.base import (
    AgentPersona,
    AgentTurnResult,
    chat_turn,
)
from engine.agents.persona.anomaly_sentinel import ANOMALY_SENTINEL
from engine.agents.persona.attribution_analyst import ATTRIBUTION_ANALYST
from engine.agents.persona.audit_recorder import AUDIT_RECORDER
from engine.agents.persona.chief_of_staff import CHIEF_OF_STAFF
from engine.agents.persona.decay_sentinel import DECAY_SENTINEL
from engine.agents.persona.devils_advocate import DEVILS_ADVOCATE
from engine.agents.persona.dq_inspector import DQ_INSPECTOR
from engine.agents.persona.risk_manager import RISK_MANAGER

__all__ = [
    # Generic API
    "AgentPersona",
    "AgentTurnResult",
    "chat_turn",
    # Per-agent persona instances
    "CHIEF_OF_STAFF",      # the user-facing supervisor (single entry point)
    "RISK_MANAGER",
    "DQ_INSPECTOR",
    "DEVILS_ADVOCATE",
    "ANOMALY_SENTINEL",
    "ATTRIBUTION_ANALYST",
    "AUDIT_RECORDER",
    "DECAY_SENTINEL",
]
