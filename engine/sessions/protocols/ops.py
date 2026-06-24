"""ops protocol checker.

Ops sessions are monitoring / alert response. Often read-only — closing
with no emit is normal. Exit always satisfied (ops doesn't fail to close).
"""
from __future__ import annotations

from engine.sessions.schema import PreflightDigest


SESSION_TYPE = "ops"
EXPECTED_DURATION = "15min-1h"
DESCRIPTION = "Monitor / respond to alert. Usually read-only."


def preflight_required_fields(digest: PreflightDigest) -> list[str]:
    """Ops just needs a goal — what are you checking?"""
    if len(digest.goal.strip()) < 10:
        return ["goal (≥ 10 chars describing what you're monitoring or responding to.)"]
    return []


def exit_check(events: list, commits: list[str]) -> tuple[bool, list[str]]:
    """Ops always exits cleanly — no required artifacts."""
    return (True, [])
