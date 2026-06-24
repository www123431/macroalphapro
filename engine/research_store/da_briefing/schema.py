"""engine.research_store.da_briefing.schema — DAVerdict + DAClaim dataclasses.

The DA verdict is the OUTPUT of a Devil's Advocate critique pass. Every
verdict must trace evidence back to specific papers_chroma chunks via
DAClaim atoms.

Schema rules (self-validate at dataclass layer):

  - DAClaim.chunk_id non-empty AND quote_text ≥ 20 chars AND argument
    non-empty
  - DAVerdict total claims (refutes + supports + conditional) > 0 OR
    overall_stance == "insufficient_evidence"
  - DAVerdict.overall_rationale ≥ 50 chars
  - DAVerdict.candidate_name + target_hypothesis_id non-empty

Cross-store rules (in `cross_validate.py`, not in `validate()`):

  - Each chunk_id MUST resolve in papers_chroma
  - Each quote_text MUST be a verbatim substring of the chunk text
  - Each paper_id MUST resolve in papers_registry with status INGESTED
"""
from __future__ import annotations

import dataclasses as _dc
import uuid as _uuid
from enum import Enum
from typing import Any


DA_VERDICT_SCHEMA_VERSION = 1


class DAStance(str, Enum):
    """Position of one DAClaim relative to the candidate hypothesis."""
    REFUTES        = "refutes"        # paper evidence contradicts the candidate
    SUPPORTS       = "supports"       # paper evidence corroborates
    CONDITIONAL    = "conditional"    # paper evidence applies under stated conditions
    INSUFFICIENT   = "insufficient"   # quote is relevant but inconclusive


class OverallStance(str, Enum):
    """The DA's bottom-line recommendation about the candidate."""
    REJECT                  = "reject"                   # strong refutation; do not test
    PROCEED_WITH_CAVEATS    = "proceed_with_caveats"     # test but watch the caveats
    NEEDS_MORE_DATA         = "needs_more_data"          # claim is testable but library
                                                          #   doesn't have what's needed
    INSUFFICIENT_EVIDENCE   = "insufficient_evidence"    # papers don't speak to this
                                                          #   candidate at all


@_dc.dataclass(frozen=True)
class DAClaim:
    """One critique-evidence atom.

    Fields:
      stance:        DAStance enum value
      chunk_id:      papers_chroma chunk_id (MUST resolve cross-store)
      paper_id:      papers_registry.paper_id (MUST resolve INGESTED)
      quote_text:    verbatim substring from the chunk (≥ 20 chars)
      section_ref:   "p.1467, §3.2" (or "" if not given)
      argument:      1-2 sentence DA reasoning re: stance
    """
    stance:          DAStance
    chunk_id:        str
    paper_id:        str
    quote_text:      str
    section_ref:     str
    argument:        str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stance":      self.stance.value,
            "chunk_id":    self.chunk_id,
            "paper_id":    self.paper_id,
            "quote_text":  self.quote_text,
            "section_ref": self.section_ref,
            "argument":    self.argument,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DAClaim":
        return cls(
            stance      = DAStance(d["stance"]),
            chunk_id    = d["chunk_id"],
            paper_id    = d["paper_id"],
            quote_text  = d["quote_text"],
            section_ref = d.get("section_ref", ""),
            argument    = d["argument"],
        )

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.chunk_id.strip():
            errs.append("DAClaim.chunk_id is empty")
        if not self.paper_id.strip():
            errs.append("DAClaim.paper_id is empty")
        if len(self.quote_text) < 20:
            errs.append(
                f"DAClaim.quote_text too short ({len(self.quote_text)} chars); "
                f"must be ≥ 20 chars (substantive substring of chunk)"
            )
        if not self.argument.strip():
            errs.append("DAClaim.argument is empty")
        return errs


@_dc.dataclass(frozen=True)
class DAVerdict:
    """The full DA output. One verdict per (candidate_name,
    target_hypothesis_id) pair.

    Identity:
      verdict_id            — UUID4
      candidate_name        — what we're testing
      target_hypothesis_id  — Hypothesis.hypothesis_id this candidate
                              proposes to test (MUST resolve)
      version               — int starting at 1
      parent_verdict_id     — for amendments

    Evidence:
      refutes               — DAClaim list with stance=REFUTES
      supports              — DAClaim list with stance=SUPPORTS
      conditional           — DAClaim list with stance=CONDITIONAL

    Bottom line:
      overall_stance        — OverallStance enum
      overall_rationale     — ≥ 50 chars synthesis

    Metadata:
      n_chunks_retrieved    — how many chunks the LLM was given as context
      papers_consulted      — paper_ids referenced
      created_ts            — ISO-8601
      created_by            — actor (LLM model name or human)
      tags                  — free-form
    """
    # Identity
    verdict_id:            str
    candidate_name:        str
    target_hypothesis_id:  str
    version:               int
    parent_verdict_id:     str | None

    # Evidence
    refutes:               tuple[DAClaim, ...]
    supports:              tuple[DAClaim, ...]
    conditional:           tuple[DAClaim, ...]

    # Bottom line
    overall_stance:        OverallStance
    overall_rationale:     str

    # Metadata
    n_chunks_retrieved:    int
    papers_consulted:      tuple[str, ...]
    created_ts:            str
    created_by:            str
    tags:                  tuple[str, ...]

    schema_version:        int = DA_VERDICT_SCHEMA_VERSION

    @staticmethod
    def new_id() -> str:
        return str(_uuid.uuid4())

    def all_claims(self) -> tuple[DAClaim, ...]:
        return self.refutes + self.supports + self.conditional

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict_id":           self.verdict_id,
            "candidate_name":       self.candidate_name,
            "target_hypothesis_id": self.target_hypothesis_id,
            "version":              self.version,
            "parent_verdict_id":    self.parent_verdict_id,
            "schema_version":       self.schema_version,

            "refutes":              [c.to_dict() for c in self.refutes],
            "supports":             [c.to_dict() for c in self.supports],
            "conditional":          [c.to_dict() for c in self.conditional],

            "overall_stance":       self.overall_stance.value,
            "overall_rationale":    self.overall_rationale,

            "n_chunks_retrieved":   self.n_chunks_retrieved,
            "papers_consulted":     list(self.papers_consulted),
            "created_ts":           self.created_ts,
            "created_by":           self.created_by,
            "tags":                 list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DAVerdict":
        return cls(
            verdict_id           = d["verdict_id"],
            candidate_name       = d["candidate_name"],
            target_hypothesis_id = d["target_hypothesis_id"],
            version              = int(d.get("version", 1)),
            parent_verdict_id    = d.get("parent_verdict_id"),
            schema_version       = int(d.get("schema_version", DA_VERDICT_SCHEMA_VERSION)),

            refutes              = tuple(DAClaim.from_dict(c) for c in (d.get("refutes") or ())),
            supports             = tuple(DAClaim.from_dict(c) for c in (d.get("supports") or ())),
            conditional          = tuple(DAClaim.from_dict(c) for c in (d.get("conditional") or ())),

            overall_stance       = OverallStance(d["overall_stance"]),
            overall_rationale    = d["overall_rationale"],

            n_chunks_retrieved   = int(d.get("n_chunks_retrieved", 0)),
            papers_consulted     = tuple(d.get("papers_consulted") or ()),
            created_ts           = d["created_ts"],
            created_by           = d["created_by"],
            tags                 = tuple(d.get("tags") or ()),
        )

    def validate(self) -> list[str]:
        errs: list[str] = []

        if not self.candidate_name.strip():
            errs.append("candidate_name is empty")
        if not self.target_hypothesis_id.strip():
            errs.append("target_hypothesis_id is empty")
        if len(self.overall_rationale) < 50:
            errs.append(
                f"overall_rationale too short ({len(self.overall_rationale)} "
                f"chars); must be ≥ 50 chars (a 1-sentence verdict isn't enough)"
            )

        # KEY INVARIANT: total claims > 0 OR overall_stance is
        # INSUFFICIENT_EVIDENCE. Anything else = silently empty DA pass
        # masquerading as a real verdict.
        total = len(self.all_claims())
        if total == 0 and self.overall_stance != OverallStance.INSUFFICIENT_EVIDENCE:
            errs.append(
                f"DAVerdict has 0 claims (refutes + supports + conditional) "
                f"but overall_stance={self.overall_stance.value}; either "
                f"add at least 1 claim or set "
                f"overall_stance=insufficient_evidence"
            )

        # REJECT requires ≥ 1 refute claim
        if self.overall_stance == OverallStance.REJECT and not self.refutes:
            errs.append(
                "overall_stance=reject requires ≥ 1 refutes claim "
                "(can't reject without cited refutation evidence)"
            )

        # Each claim self-validates
        for label, claims in [("refutes", self.refutes),
                              ("supports", self.supports),
                              ("conditional", self.conditional)]:
            for i, c in enumerate(claims):
                for cerr in c.validate():
                    errs.append(f"{label}[{i}]: {cerr}")

        return errs
