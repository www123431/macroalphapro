"""
engine/workflows/ — event-driven workflow layer (Wave A 2026-05-19).

Per the Wave-A "Morning Briefing" build:
  briefing.py  — generate a single one-page daily brief (deterministic
                 SQL queries + curated top-K + Markdown render)

Future workflows (deferred):
  case.py      — agent_cases table + open/close lifecycle (Wave B)
  predeploy.py — strategy pre-deploy review pipeline (Wave C)
  lineage.py   — case ↔ amendment ↔ commit chain (Wave D)

Doctrine: workflows are DETERMINISTIC by default. LLM is only invited
for narrative SUMMARY at the end, never for routing / triage / state
mutation. Pattern 5 ban + 0-LLM-in-DECISION both preserved.
"""
from engine.workflows.briefing import (
    Briefing,
    BriefingItem,
    generate_briefing,
    render_as_markdown,
    persist_briefing,
)

__all__ = [
    "Briefing",
    "BriefingItem",
    "generate_briefing",
    "render_as_markdown",
    "persist_briefing",
]
