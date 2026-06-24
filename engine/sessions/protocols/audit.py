"""audit protocol checker.

Audit sessions investigate bugs / suspicious numbers / Layer 2 concerns.
Exit accepts either a git commit (code fixed) OR a new verdict event
(verdict revised after investigation).
"""
from __future__ import annotations

from engine.sessions.schema import PreflightDigest


SESSION_TYPE = "audit"
EXPECTED_DURATION = "30min-3h"
DESCRIPTION = "Investigate bug, suspicious number, or Layer 2 concern."


def preflight_required_fields(digest: PreflightDigest) -> list[str]:
    missing = []
    if len(digest.goal.strip()) < 30:
        missing.append(
            "goal (Must be ≥ 30 chars describing what looks wrong + what you suspect.)"
        )
    if not digest.cockpit_reviewed:
        missing.append("cockpit_reviewed (Did you check Cockpit / audit trail for the symptom?)")
    return missing


def exit_check(events: list, commits: list[str]) -> tuple[bool, list[str]]:
    """Either a code fix (commit) or a state-changing event satisfies exit."""
    if commits:
        return (True, [])
    state_change_events = [
        e for e in events
        if e.event_type.value in ("factor_verdict_filed", "spec_amended", "memory_doctrine_locked")
    ]
    if state_change_events:
        return (True, [])
    return (False, [
        "Audit session needs ≥1 git commit OR ≥1 state-changing event "
        "(factor_verdict_filed / spec_amended / memory_doctrine_locked). "
        "If audit found no issue, prefer closing as 'no-op' via abandon endpoint."
    ])
