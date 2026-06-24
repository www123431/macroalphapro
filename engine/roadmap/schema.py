"""engine.roadmap.schema — typed research-direction roadmap.

A ResearchAxis is a first-class typed intent object: "what direction am
I pushing, what's next, what's blocked." NOT derived from PFH or memory
scrape — explicit, slow-changing, doctrine-session-edited.

Replaces the hand-written "forward axes" header block in MEMORY.md by
giving it a queryable + UI-renderable shape. Memory still records
RATIONALE (why an axis is active/closed); roadmap records STATE
(currently active / queued next / paused / closed).

Per session-protocol audit Gap A (2026-06-02). Architectural decision:
- intent vs state separation (this = intent; library/events = state)
- typed tier (committed / candidate / scratchpad) so seriousness
  is encoded, not inferred from text
- forward decay attachment (Gap B integration) so the badge can show
  "this axis is in a HIGH-decay family — proceed knowing that"
- capacity placeholder for Gap C
"""
from __future__ import annotations

import dataclasses as _dc
from enum import Enum
from typing import Any, Optional


SCHEMA_VERSION = 1


class AxisState(str, Enum):
    """Current state in the lifecycle."""
    active   = "active"     # currently pushing
    queued   = "queued"     # next-up, will start when active slot frees
    paused   = "paused"     # was active, suspended; may resume
    closed   = "closed"     # done (GREEN deployed / RED killed / abandoned)


class AxisTier(str, Enum):
    """How seriously is this axis being treated."""
    committed   = "committed"    # formal, will be worked
    candidate   = "candidate"    # serious evaluation; not yet committed
    scratchpad  = "scratchpad"   # brainstorm only, no commitment


class AxisOutcome(str, Enum):
    """For closed axes — what happened."""
    GREEN     = "GREEN"        # deployed
    RED       = "RED"          # killed at strict gate
    MARGINAL  = "MARGINAL"     # logged but not deployed
    ABANDONED = "ABANDONED"    # gave up before verdict
    NONE      = "NONE"         # not closed yet


@_dc.dataclass(frozen=True)
class ResearchAxis:
    """One research direction with state + linked context.

    Identity:
        axis_id        — short stable slug (e.g. 'carry_gx_stir')
        name           — human-readable label

    State:
        state          — current lifecycle position
        tier           — seriousness encoding
        outcome        — for closed: what verdict was reached

    Linkage:
        parent_axis_id — DAG parent (e.g. carry_gx_stir's parent is carry)
        family         — factor family (links to decay forecast registry)
        related_subject_ids — research_store subjects this axis depends on
        related_memory_files — memory frontmatter slugs this axis references

    Content:
        rationale      — paragraph (why active / why closed / why queued)
        next_actions   — ordered list of concrete next steps
        blocking_notes — what's preventing progress (if any)

    Optional cached:
        decay_estimate    — Gap B integration; cached MP/LR forecast
        capacity_estimate — Gap C placeholder

    Metadata:
        created_ts / updated_ts / created_by / updated_by
    """
    axis_id:               str
    name:                  str
    state:                 AxisState
    tier:                  AxisTier
    outcome:               AxisOutcome

    parent_axis_id:        Optional[str]
    family:                Optional[str]
    related_subject_ids:   tuple[str, ...]
    related_memory_files:  tuple[str, ...]

    rationale:             str
    next_actions:          tuple[str, ...]
    blocking_notes:        str

    decay_estimate:        Optional[dict[str, Any]]
    capacity_estimate:     Optional[dict[str, Any]]

    created_ts:            str
    updated_ts:            str
    created_by:            str
    updated_by:            str

    schema_version:        int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = _dc.asdict(self)
        d["state"]   = self.state.value
        d["tier"]    = self.tier.value
        d["outcome"] = self.outcome.value
        d["related_subject_ids"]  = list(self.related_subject_ids)
        d["related_memory_files"] = list(self.related_memory_files)
        d["next_actions"]         = list(self.next_actions)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResearchAxis":
        return cls(
            axis_id               = d["axis_id"],
            name                  = d["name"],
            state                 = AxisState(d["state"]),
            tier                  = AxisTier(d["tier"]),
            outcome               = AxisOutcome(d.get("outcome", "NONE")),
            parent_axis_id        = d.get("parent_axis_id"),
            family                = d.get("family"),
            related_subject_ids   = tuple(d.get("related_subject_ids") or ()),
            related_memory_files  = tuple(d.get("related_memory_files") or ()),
            rationale             = d.get("rationale", ""),
            next_actions          = tuple(d.get("next_actions") or ()),
            blocking_notes        = d.get("blocking_notes", ""),
            decay_estimate        = d.get("decay_estimate"),
            capacity_estimate     = d.get("capacity_estimate"),
            created_ts            = d.get("created_ts", ""),
            updated_ts            = d.get("updated_ts", ""),
            created_by            = d.get("created_by", "unknown"),
            updated_by            = d.get("updated_by", "unknown"),
            schema_version        = int(d.get("schema_version", 1)),
        )
