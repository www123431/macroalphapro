"""doctrine protocol checker.

Doctrine sessions lock a lesson or amend memory. Exit requires at least one
memory_doctrine_locked event linked to this session.
"""
from __future__ import annotations

from engine.sessions.schema import PreflightDigest


SESSION_TYPE = "doctrine"
EXPECTED_DURATION = "15-45min"
DESCRIPTION = "Lock a lesson / amend memory file / capture a doctrine."


def preflight_required_fields(digest: PreflightDigest) -> list[str]:
    if len(digest.goal.strip()) < 30:
        return ["goal (≥ 30 chars describing the doctrine / lesson you're locking.)"]
    return []


def exit_check(events: list, commits: list[str]) -> tuple[bool, list[str]]:
    memory_events = [e for e in events if e.event_type.value == "memory_doctrine_locked"]
    if memory_events:
        return (True, [])
    return (False, [
        "Doctrine session needs ≥1 memory_doctrine_locked event. "
        "Write the memory file then call emit.memory_locked(...)."
    ])
