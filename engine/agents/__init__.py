"""
engine.agents — Operations-architecture agent framework.

Spec: docs/spec_factor_mad_redesign.md §2 (locked 2026-05-02).

Public API:
    from engine.agents import Agent, Trigger, AgentResult, AgentEvent
    from engine.agents.event_bus import EventBus, get_event_bus

The framework is dataclass-light, SQLite-backed (via engine.memory), and
synchronous. FactorMAD is the first agent to migrate; ERA / UniverseReview /
FailureAttribution / SignalDecayPatrol are scheduled to follow in later sprints.
"""
from engine.agents.base import (
    Agent,
    AgentEvent,
    AgentResult,
    Trigger,
)

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentResult",
    "Trigger",
]
