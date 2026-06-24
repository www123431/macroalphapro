"""engine.research_store.forward_vectors.schema — ForwardVectorV2 dataclass.

A ForwardVectorV2 = "untested hypothesis from a specific paper that the
system suggests trying". The user sees a ranked list and picks one to
open a research session on.

Stable fields the UI / CLI can rank:
  - source_paper_id + paper_title
  - hypothesis_id + hypothesis claim
  - mechanism_family + mechanism_subtype
  - required_data (so user knows if we have it)
  - predicted_direction + predicted_magnitude
  - priority (high / med / low)
  - priority_signals (dict of WHY this is high/med/low)

Priority is computed deterministically by `generator.py`:
  - HIGH if the source paper is on DOCTRINE_METHOD or GREEN_MOTIVATION
    shelf (high reviewer relevance)
  - MED if source paper is on any other motivation shelf
  - LOW otherwise

Versioning: like all chain artifacts, append-only. To amend a forward
vector (mark as "tested" or "deprecated"), write a new version.
"""
from __future__ import annotations

import dataclasses as _dc
import uuid as _uuid
from enum import Enum
from typing import Any

from engine.research_store.red_lessons.mechanism_families import MechanismFamily


FORWARD_VECTOR_SCHEMA_VERSION = 1


class Priority(str, Enum):
    HIGH    = "high"
    MEDIUM  = "medium"
    LOW     = "low"


class ForwardVectorStatus(str, Enum):
    """Lifecycle of a forward vector."""
    PROPOSED       = "proposed"        # generator output, awaiting user pick
    USER_ACCEPTED  = "user_accepted"   # user opened a session for this
    TESTED         = "tested"          # has at least one paper_grounded
                                       #   lesson with this hypothesis_id
    DEPRECATED     = "deprecated"      # withdrawn (paper retracted etc.)


@_dc.dataclass(frozen=True)
class ForwardVectorV2:
    """A "what should we test next" suggestion grounded in a specific
    paper's untested hypothesis.

    Identity:
      forward_vector_id  — UUID4
      version            — int starting at 1
      parent_id          — for amendments

    Source (REQUIRED — load-bearing chain):
      source_paper_id    — papers_registry.paper_id (must INGESTED)
      paper_title        — copied for display
      source_hypothesis_id — hypothesis_id, MUST resolve in hypotheses store

    Claim (echoed from hypothesis for display):
      claim              — hypothesis.claim
      mechanism_family   — hypothesis.mechanism_family
      mechanism_subtype  — hypothesis.mechanism_subtype
      predicted_direction
      predicted_magnitude
      required_data
      test_methodology

    Priority + signals:
      priority           — Priority enum
      priority_signals   — dict explaining the priority (e.g.
                           {"paper_shelves": ["doctrine_method",
                                              "green_motivation"]})

    Metadata:
      status             — ForwardVectorStatus
      created_ts         — ISO-8601
      created_by         — actor
      tags               — labels

    Versioning:
      schema_version     — int
    """

    forward_vector_id:    str
    version:              int
    parent_id:            str | None

    # Source — chain back to paper + hypothesis
    source_paper_id:      str
    paper_title:          str
    source_hypothesis_id: str

    # Hypothesis content (denormalized for display)
    claim:                str
    mechanism_family:     MechanismFamily
    mechanism_subtype:    str
    predicted_direction:  str   # HypothesisDirection value
    predicted_magnitude:  str
    required_data:        tuple[str, ...]
    test_methodology:     str

    # Priority
    priority:             Priority
    priority_signals:     dict[str, Any]

    # Metadata
    status:               ForwardVectorStatus
    created_ts:           str
    created_by:           str
    tags:                 tuple[str, ...]

    schema_version:       int = FORWARD_VECTOR_SCHEMA_VERSION

    @staticmethod
    def new_id() -> str:
        return str(_uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_vector_id":    self.forward_vector_id,
            "version":              self.version,
            "parent_id":            self.parent_id,
            "schema_version":       self.schema_version,

            "source_paper_id":      self.source_paper_id,
            "paper_title":          self.paper_title,
            "source_hypothesis_id": self.source_hypothesis_id,

            "claim":                self.claim,
            "mechanism_family":     self.mechanism_family.value,
            "mechanism_subtype":    self.mechanism_subtype,
            "predicted_direction":  self.predicted_direction,
            "predicted_magnitude":  self.predicted_magnitude,
            "required_data":        list(self.required_data),
            "test_methodology":     self.test_methodology,

            "priority":             self.priority.value,
            "priority_signals":     dict(self.priority_signals),

            "status":               self.status.value,
            "created_ts":           self.created_ts,
            "created_by":           self.created_by,
            "tags":                 list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ForwardVectorV2":
        return cls(
            forward_vector_id    = d["forward_vector_id"],
            version              = int(d.get("version", 1)),
            parent_id            = d.get("parent_id"),
            schema_version       = int(d.get("schema_version", FORWARD_VECTOR_SCHEMA_VERSION)),

            source_paper_id      = d["source_paper_id"],
            paper_title          = d["paper_title"],
            source_hypothesis_id = d["source_hypothesis_id"],

            claim                = d["claim"],
            mechanism_family     = MechanismFamily(d["mechanism_family"]),
            mechanism_subtype    = d.get("mechanism_subtype", ""),
            predicted_direction  = d.get("predicted_direction", ""),
            predicted_magnitude  = d.get("predicted_magnitude", ""),
            required_data        = tuple(d.get("required_data") or ()),
            test_methodology     = d.get("test_methodology", ""),

            priority             = Priority(d.get("priority", "medium")),
            priority_signals     = dict(d.get("priority_signals") or {}),

            status               = ForwardVectorStatus(d.get("status", "proposed")),
            created_ts           = d["created_ts"],
            created_by           = d["created_by"],
            tags                 = tuple(d.get("tags") or ()),
        )

    def validate(self) -> list[str]:
        """Phase 2.1b (2026-06-06): relaxed source_paper_id requirement
        so brainstorm-track FVs (created from LLM_SYNTHESIS hypotheses)
        can validate. Track distinction lives in tags + priority_signals,
        not in this syntactic guard. Hard requirements:
          - source_hypothesis_id non-empty (always must trace to a hyp)
          - claim non-empty
          - required_data non-empty (you can't test what doesn't say
            what data it needs)
        """
        errs: list[str] = []
        if not self.source_hypothesis_id.strip():
            errs.append("source_hypothesis_id is empty")
        if not self.claim.strip():
            errs.append("claim is empty")
        if not self.required_data:
            errs.append("required_data empty — a forward vector must say what data is needed")
        return errs
