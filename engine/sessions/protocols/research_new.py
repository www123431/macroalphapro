"""research_new protocol checker.

Heaviest protocol — research_new MUST produce at minimum:
  - a factor_verdict_filed event (GREEN/MARGINAL/RED) linked to this session
  - a capability_evidence_filed event with parent → the verdict
"""
from __future__ import annotations

from engine.sessions.schema import PreflightDigest


SESSION_TYPE = "research_new"
EXPECTED_DURATION = "2-6h"
DESCRIPTION = "Test a new factor with strict gate enforcement."


def preflight_required_fields(digest: PreflightDigest) -> list[str]:
    """Return list of REQUIRED fields that are not yet filled.
    Empty list = preflight passes; non-empty = list of missing fields."""
    missing = []
    if not digest.cockpit_reviewed:
        missing.append("cockpit_reviewed (Did you check Cockpit state?)")
    if not digest.graveyard_search_query.strip():
        missing.append("graveyard_search_query (Search graveyard for related verdicts.)")
    if not digest.library_overlap_checked:
        missing.append("library_overlap_checked (Did you check /lab/library for sleeve overlap?)")
    if len(digest.goal.strip()) < 30:
        missing.append("goal (Must be ≥ 30 chars describing what you're testing.)")
    return missing


def exit_check(events: list, commits: list[str]) -> tuple[bool, list[str]]:
    """Verify research_new exit conditions are satisfied.

    Args:
        events: list of ResearchEvent objects linked to this session
        commits: list of git_sha strings produced during session
    Returns:
        (satisfied, missing_reasons)
    """
    missing = []
    verdict_events = [e for e in events if e.event_type.value == "factor_verdict_filed"]
    if not verdict_events:
        missing.append(
            "≥1 factor_verdict_filed event required; none linked to this session. "
            "Emit one via engine.research_store.emit.factor_verdict()."
        )
    evidence_events = [e for e in events if e.event_type.value == "capability_evidence_filed"]
    if not evidence_events:
        missing.append(
            "≥1 capability_evidence_filed event required (lineage to verdict). "
            "Write the evidence doc, then emit.capability_evidence_filed(parent_event_ids=(verdict_id,))."
        )
    return (len(missing) == 0, missing)
