"""engine.sessions.protocols — per-session_type pre-flight + exit checkers.

Each protocol module exports:
  preflight_required_fields(digest) -> list[str]  # which fields must be filled
  exit_check(session, events, commits) -> (satisfied: bool, missing: list[str])

The dispatcher below picks the right module per session_type.
"""
from __future__ import annotations

from engine.sessions.protocols import (
    research_new, audit, ops, doctrine, exploration,
)
from engine.sessions.schema import SessionType


_DISPATCH = {
    SessionType.research_new: research_new,
    SessionType.audit:        audit,
    SessionType.ops:          ops,
    SessionType.doctrine:     doctrine,
    SessionType.exploration:  exploration,
}


def for_type(session_type: SessionType):
    """Return the protocol module for a given session_type."""
    return _DISPATCH[session_type]
