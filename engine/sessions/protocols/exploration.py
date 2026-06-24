"""exploration protocol checker — the ESCAPE HATCH.

5th session type added 2026-06-02 to prevent over-bureaucratization.
Some sessions are "I haven't formed a hypothesis yet, just want to think"
— protocol-locking these would kill creativity.

NO exit emit required. NO pre-flight checks beyond goal.
Outputs (if any) are tagged 'tags=["exploration"]' and DO NOT count as
production-grade verdicts.

Doctrine: if you're using exploration to bypass research_new strict gate,
that's misuse. Use exploration for ideation; transition to research_new
the moment you have a testable hypothesis.
"""
from __future__ import annotations

from engine.sessions.schema import PreflightDigest


SESSION_TYPE = "exploration"
EXPECTED_DURATION = "open-ended"
DESCRIPTION = (
    "Open-ended thinking / ideation. No exit enforcement. "
    "Escape hatch — use when you don't have a testable hypothesis yet."
)


def preflight_required_fields(digest: PreflightDigest) -> list[str]:
    """Only goal — minimum viable preflight."""
    if len(digest.goal.strip()) < 10:
        return ["goal (≥ 10 chars describing what you want to think about.)"]
    return []


def exit_check(events: list, commits: list[str]) -> tuple[bool, list[str]]:
    """No exit verification — exploration always closes cleanly."""
    return (True, [])
