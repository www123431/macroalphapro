"""engine.research_store.red_lessons.schema — RED Lesson dataclasses.

A RED Lesson is the structured-learning representation of a research
verdict. Unlike `engine.research_store.schema.ResearchEvent` (immutable
event log), a lesson is REVIEWABLE and AMENDABLE:

  - lesson.review_state advances `proposed` → `claude_drafted` →
    `human_reviewed` → `locked`
  - to amend a locked lesson, write a NEW lesson with
    parent_lesson_id pointing at the prior and bump version

Lesson identity:
    lesson_id        — UUID4 string (auto-generated)
    candidate_name   — the candidate the lesson is ABOUT (e.g. "china_pead")
    version          — int starting at 1, bumped on amendment

Schema is intentionally rich: this is the load-bearing data structure for
the entire Layer-1/2/3 retrieval system; future RAG briefing quality is
bounded by the field discipline here.
"""
from __future__ import annotations

import dataclasses as _dc
import uuid as _uuid
from enum import Enum
from typing import Any

from engine.research_store.red_lessons.failure_modes import FailureMode
from engine.research_store.red_lessons.mechanism_families import MechanismFamily

# VerbatimQuote is defined in engine.research_store.hypothesis.schema.
# Top-level import here would create a circular import (hypothesis.schema
# imports red_lessons.mechanism_families, which triggers red_lessons
# package init, which imports this module). VerbatimQuote is therefore
# imported LAZILY inside from_dict() — the field is annotated by the
# class object directly via runtime resolution.
#
# At the dataclass-field level we declare `verbatim_quotes: tuple` (no
# generic parameter) to avoid annotation-eval-time import.
def _import_verbatim_quote_cls():
    from engine.research_store.hypothesis.schema import VerbatimQuote
    return VerbatimQuote


LESSON_SCHEMA_VERSION = 2     # bumped 2026-06-04: added grounding_method
                              #                   + tested_hypothesis_ids
                              #                   + verbatim_quotes

# Cutoff timestamp: lessons created after this MUST have grounding_method
# != pretrain_grounded (per locked PAPER→HYPOTHESIS→TEST→VERDICT chain).
# See [[feedback-paper-driven-research-chain-locked-2026-06-04]].
PRETRAIN_GROUNDED_FREEZE_TS = "2026-06-04T12:00:00Z"


# ────────────────────────── enums ─────────────────────────────────────


class ReviewState(str, Enum):
    """Lifecycle of a lesson record."""
    proposed         = "proposed"          # auto-extracted, not yet reviewed
    claude_drafted   = "claude_drafted"    # Claude wrote it, needs human OK
    human_reviewed   = "human_reviewed"    # human passed; can be queried
    locked           = "locked"            # immutable; amend via new version
    deprecated       = "deprecated"        # superseded; do not surface


class LessonStrength(str, Enum):
    """How strong is the evidence behind this lesson?"""
    strong  = "strong"     # multiple independent papers + our own RED record
    medium  = "medium"     # one anchor paper + our own RED record
    weak    = "weak"       # internal-only or weak academic support


class GroundingMethod(str, Enum):
    """How is this lesson grounded?

    LOAD-BEARING per locked PAPER→HYPOTHESIS→TEST→VERDICT chain
    (2026-06-04). Schema validate() enforces:

      - paper_grounded:    requires tested_hypothesis_ids ≠ () AND
                           verbatim_quotes ≠ () (the only valid
                           grounding for NEW lessons created after
                           PRETRAIN_GROUNDED_FREEZE_TS)
      - stat_only_grounded: lesson has real stat_evidence but no
                           hypothesis chain. Acceptable for legacy
                           re-describe AND for genuine
                           engineering-discovery lessons (rare).
      - pretrain_grounded:  Claude pretrain-knowledge motivated lesson
                           with no verifiable evidence chain. FROZEN —
                           NEW pretrain_grounded lessons are rejected
                           by validate() if created_ts > freeze TS.
                           Only the legacy 47 lessons retain this state.
    """
    paper_grounded     = "paper_grounded"
    stat_only_grounded = "stat_only_grounded"
    pretrain_grounded  = "pretrain_grounded"     # FROZEN — legacy only


# ──────────────────────── value objects ───────────────────────────────


@_dc.dataclass(frozen=True)
class PaperRef:
    """Reference to an academic / working paper.

    Fields:
      doi:           preferred; "10.xxxx/..." form. Empty string if unavailable.
      arxiv_id:      preferred for arXiv-only papers.
      ssrn_id:       preferred for SSRN preprints.
      year:          publication year (int).
      authors:       short author list (e.g. ("McLean", "Pontiff")).
      title:         paper title.
      venue:         journal / "Working Paper" / "SSRN" / "NBER WP".
      key_claim:     1-2 sentence paraphrase of the claim relevant to THIS lesson.
                     This is the most-load-bearing field — it's what gets fed
                     into the RAG briefing prompt.
      our_finding:   1 sentence — what did OUR test of this claim find?
                     ('confirmed' / 'partial' / 'failed because...').
      section_ref:   "p.2358, Table 4" — for verifiability.
    """
    title:        str
    year:         int
    authors:      tuple[str, ...]
    key_claim:    str
    our_finding:  str
    doi:          str = ""
    arxiv_id:     str = ""
    ssrn_id:      str = ""
    venue:        str = ""
    section_ref:  str = ""

    def to_dict(self) -> dict[str, Any]:
        d = _dc.asdict(self)
        d["authors"] = list(self.authors)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaperRef":
        return cls(
            title       = d["title"],
            year        = int(d["year"]),
            authors     = tuple(d.get("authors") or ()),
            key_claim   = d["key_claim"],
            our_finding = d["our_finding"],
            doi         = d.get("doi", ""),
            arxiv_id    = d.get("arxiv_id", ""),
            ssrn_id     = d.get("ssrn_id", ""),
            venue       = d.get("venue", ""),
            section_ref = d.get("section_ref", ""),
        )


@_dc.dataclass(frozen=True)
class ForwardVector:
    """A concrete research direction this RED lesson POINTS toward.

    The point of structured forward vectors: when a new candidate comes in,
    the system can match its description against `direction` fields across
    all lessons to surface "here are X directions our prior failures point
    toward; your candidate aligns with #3 — proceed with their caveats".

    Fields:
      direction:           "try same mechanism in non-US futures"
      rationale:           why this direction avoids the original RED's failure
      avoids_failures:     tuple of FailureMode codes the direction sidesteps
      new_required_data:   list of data sources / properties needed to test
                           this direction (used by dormant_revisit detector)
      priority:            high / med / low
      blocked_by:          free-form notes on what's still blocking this
                           (e.g. "needs WRDS CIQ access")
    """
    direction:           str
    rationale:           str
    avoids_failures:     tuple[FailureMode, ...]
    new_required_data:   tuple[str, ...]
    priority:            str  # "high" | "med" | "low"
    blocked_by:          str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction":         self.direction,
            "rationale":         self.rationale,
            "avoids_failures":   [f.value for f in self.avoids_failures],
            "new_required_data": list(self.new_required_data),
            "priority":          self.priority,
            "blocked_by":        self.blocked_by,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ForwardVector":
        return cls(
            direction         = d["direction"],
            rationale         = d["rationale"],
            avoids_failures   = tuple(FailureMode(f) for f in (d.get("avoids_failures") or ())),
            new_required_data = tuple(d.get("new_required_data") or ()),
            priority          = d["priority"],
            blocked_by        = d.get("blocked_by", ""),
        )


@_dc.dataclass(frozen=True)
class DormantRevisit:
    """A condition that, if MET in the future, would justify re-testing this RED.

    Example:
      condition_label:    "OptionMetrics extends pre-1990"
      condition_check:    "data.cache contains pre-1990 IV surfaces"
      reactivation_note:  "F1 PUBLICATION_DECAY argued the signal was real but
                          arbitraged. Pre-publication-window data would let us
                          re-test the un-arbitraged mechanism directly."
    """
    condition_label:    str
    condition_check:    str
    reactivation_note:  str

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DormantRevisit":
        return cls(
            condition_label    = d["condition_label"],
            condition_check    = d["condition_check"],
            reactivation_note  = d["reactivation_note"],
        )


# ──────────────────────── the lesson itself ───────────────────────────


@_dc.dataclass(frozen=True)
class REDLesson:
    """Structured RED Lesson.

    Identity:
      lesson_id        — UUID4
      candidate_name   — name of the candidate / hypothesis the lesson is about
      version          — int, starts at 1
      parent_lesson_id — if this lesson amends a prior, the prior's lesson_id

    Source / verdict:
      source_event_ids — research_store event_ids that informed this lesson
                         (e.g. the factor_verdict_filed event)
      verdict          — RED / YELLOW (always one of these; GREEN lessons aren't
                         RED Lessons by definition)
      stat_evidence    — dict of the key statistics that drove the verdict
                         (e.g. {"deflated_sr": 0.128, "alpha_t_ff5umd": -0.31,
                                "n_months": 119})

    Classification (LOAD-BEARING):
      mechanism_family    — controlled MechanismFamily enum
      mechanism_subtype   — free-form refinement (e.g. "post_earnings_drift_china")
      failure_modes       — 1-3 FailureMode enum values, ranked by importance
      failure_evidence    — {failure_mode_value → evidence string}

    Paper anchors:
      paper_motivation    — paper that motivated trying this candidate
                            (None if internal hypothesis)
      paper_critiques     — papers / claims that explain why it failed

    Relations:
      subsumed_by         — deployed-factor IDs whose presence kills this candidate
                            (only relevant if F3 in failure_modes)
      related_lesson_ids  — sibling lessons (same family OR same failure mode)

    Forward learning:
      forward_directions  — structured next-directions
      do_not_retry        — free-form "do not retry under these conditions" notes
      dormant_revisits    — conditions that would justify re-test

    Metadata:
      review_state        — lifecycle
      strength            — evidence strength
      created_ts          — ISO-8601 UTC of FIRST creation
      updated_ts          — ISO-8601 UTC of LATEST amendment
      created_by          — actor (e.g. "claude-opus-4-7" / "zhangxizhe")
      summary             — 1-2 sentence human-readable summary (≤ 400 chars)
      tags                — free-form labels for ad-hoc query filtering
    """

    # Identity
    lesson_id:           str
    candidate_name:      str
    version:             int

    # Source / verdict
    source_event_ids:    tuple[str, ...]
    verdict:             str    # "RED" | "YELLOW" (sub-HLZ marginal)
    stat_evidence:       dict[str, Any]

    # Classification — LOAD-BEARING
    mechanism_family:    MechanismFamily
    mechanism_subtype:   str
    failure_modes:       tuple[FailureMode, ...]
    failure_evidence:    dict[str, str]      # FailureMode.value → evidence

    # Paper anchors
    paper_motivation:    PaperRef | None
    paper_critiques:     tuple[PaperRef, ...]

    # Relations
    subsumed_by:         tuple[str, ...]     # factor / sleeve names
    related_lesson_ids:  tuple[str, ...]

    # Forward learning
    forward_directions:  tuple[ForwardVector, ...]
    do_not_retry:        tuple[str, ...]
    dormant_revisits:    tuple[DormantRevisit, ...]

    # ── NEW 2026-06-04: paper-driven research chain (LOAD-BEARING) ──
    # See [[feedback-paper-driven-research-chain-locked-2026-06-04]].
    # tested_hypothesis_ids: Hypothesis.hypothesis_id values this lesson
    #   tested. Required when grounding_method=paper_grounded.
    # verbatim_quotes: quotes from papers_chroma chunks supporting the
    #   verdict. Required when grounding_method=paper_grounded.
    # grounding_method: how this lesson is grounded. NEW lessons
    #   (created_ts > PRETRAIN_GROUNDED_FREEZE_TS) cannot be
    #   pretrain_grounded — validate() rejects them.
    tested_hypothesis_ids: tuple[str, ...]
    # field type is tuple of VerbatimQuote (from hypothesis.schema); kept
    # un-parameterized here to defeat circular import at field-eval time.
    verbatim_quotes:       tuple
    grounding_method:      GroundingMethod

    # Metadata
    review_state:        ReviewState
    strength:            LessonStrength
    created_ts:          str
    updated_ts:          str
    created_by:          str
    summary:             str
    tags:                tuple[str, ...]

    # Versioning / chain
    parent_lesson_id:    str | None
    schema_version:      int = LESSON_SCHEMA_VERSION

    # ─────────── factory ───────────

    @staticmethod
    def new_id() -> str:
        return str(_uuid.uuid4())

    # ─────────── (de)serialization ───────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson_id":          self.lesson_id,
            "candidate_name":     self.candidate_name,
            "version":            self.version,
            "parent_lesson_id":   self.parent_lesson_id,

            "source_event_ids":   list(self.source_event_ids),
            "verdict":            self.verdict,
            "stat_evidence":      dict(self.stat_evidence),

            "mechanism_family":   self.mechanism_family.value,
            "mechanism_subtype":  self.mechanism_subtype,
            "failure_modes":      [m.value for m in self.failure_modes],
            "failure_evidence":   dict(self.failure_evidence),

            "paper_motivation":   self.paper_motivation.to_dict() if self.paper_motivation else None,
            "paper_critiques":    [p.to_dict() for p in self.paper_critiques],

            "subsumed_by":        list(self.subsumed_by),
            "related_lesson_ids": list(self.related_lesson_ids),

            "forward_directions": [v.to_dict() for v in self.forward_directions],
            "do_not_retry":       list(self.do_not_retry),
            "dormant_revisits":   [r.to_dict() for r in self.dormant_revisits],

            # NEW 2026-06-04 paper-driven chain fields
            "tested_hypothesis_ids": list(self.tested_hypothesis_ids),
            "verbatim_quotes":       [q.to_dict() for q in self.verbatim_quotes],
            "grounding_method":      self.grounding_method.value,

            "review_state":       self.review_state.value,
            "strength":           self.strength.value,
            "created_ts":         self.created_ts,
            "updated_ts":         self.updated_ts,
            "created_by":         self.created_by,
            "summary":            self.summary,
            "tags":               list(self.tags),

            "schema_version":     self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "REDLesson":
        return cls(
            lesson_id          = d["lesson_id"],
            candidate_name     = d["candidate_name"],
            version            = int(d["version"]),
            parent_lesson_id   = d.get("parent_lesson_id"),

            source_event_ids   = tuple(d.get("source_event_ids") or ()),
            verdict            = d["verdict"],
            stat_evidence      = dict(d.get("stat_evidence") or {}),

            mechanism_family   = MechanismFamily(d["mechanism_family"]),
            mechanism_subtype  = d.get("mechanism_subtype", ""),
            failure_modes      = tuple(FailureMode(m) for m in (d.get("failure_modes") or ())),
            failure_evidence   = dict(d.get("failure_evidence") or {}),

            paper_motivation   = PaperRef.from_dict(d["paper_motivation"]) if d.get("paper_motivation") else None,
            paper_critiques    = tuple(PaperRef.from_dict(p) for p in (d.get("paper_critiques") or ())),

            subsumed_by        = tuple(d.get("subsumed_by") or ()),
            related_lesson_ids = tuple(d.get("related_lesson_ids") or ()),

            forward_directions = tuple(ForwardVector.from_dict(v) for v in (d.get("forward_directions") or ())),
            do_not_retry       = tuple(d.get("do_not_retry") or ()),
            dormant_revisits   = tuple(DormantRevisit.from_dict(r) for r in (d.get("dormant_revisits") or ())),

            # NEW 2026-06-04 paper-driven chain fields.
            # Default to pretrain_grounded ONLY if schema_version < 2
            # (legacy records being read back); else require explicit.
            tested_hypothesis_ids = tuple(d.get("tested_hypothesis_ids") or ()),
            verbatim_quotes       = tuple(
                _import_verbatim_quote_cls().from_dict(q)
                for q in (d.get("verbatim_quotes") or ())
            ),
            grounding_method      = GroundingMethod(
                d.get("grounding_method")
                or ("pretrain_grounded"
                    if int(d.get("schema_version", LESSON_SCHEMA_VERSION)) < 2
                    else "paper_grounded")
            ),

            review_state       = ReviewState(d["review_state"]),
            strength           = LessonStrength(d["strength"]),
            created_ts         = d["created_ts"],
            updated_ts         = d.get("updated_ts", d["created_ts"]),
            created_by         = d["created_by"],
            summary            = d.get("summary", ""),
            tags               = tuple(d.get("tags") or ()),

            schema_version     = int(d.get("schema_version", LESSON_SCHEMA_VERSION)),
        )

    # ─────────── validation ───────────

    def validate(self) -> list[str]:
        """Return list of validation error messages (empty list = valid).

        Lessons CAN exist in `proposed` state with weaker validation. As
        review_state advances toward `locked`, more fields must be filled.
        """
        errs: list[str] = []

        # Always-required
        if not self.candidate_name.strip():
            errs.append("candidate_name is empty")
        if len(self.summary) > 400:
            errs.append(f"summary > 400 chars ({len(self.summary)})")
        if not 1 <= len(self.failure_modes) <= 3:
            errs.append(f"failure_modes must be 1-3, got {len(self.failure_modes)}")
        if self.verdict not in ("RED", "YELLOW"):
            errs.append(f"verdict must be RED or YELLOW, got {self.verdict!r}")

        # Every failure_mode must have a corresponding failure_evidence entry
        missing = [m.value for m in self.failure_modes if m.value not in self.failure_evidence]
        if missing:
            errs.append(f"failure_evidence missing for: {missing}")

        # If F3 SUBSUMED_BY_EXISTING is in failure_modes, subsumed_by must be non-empty
        if FailureMode.F3_SUBSUMED_BY_EXISTING in self.failure_modes and not self.subsumed_by:
            errs.append("F3_SUBSUMED_BY_EXISTING requires non-empty subsumed_by list")

        # ── PAPER-DRIVEN CHAIN RULES (LOCKED 2026-06-04) ─────────────
        # See [[feedback-paper-driven-research-chain-locked-2026-06-04]].
        if self.grounding_method == GroundingMethod.paper_grounded:
            # paper_grounded REQUIRES the actual paper-trace fields.
            if not self.tested_hypothesis_ids:
                errs.append(
                    "grounding_method=paper_grounded requires "
                    "tested_hypothesis_ids (≥ 1)"
                )
            if not self.verbatim_quotes:
                errs.append(
                    "grounding_method=paper_grounded requires "
                    "verbatim_quotes (≥ 1)"
                )
            # Each verbatim_quote self-validates
            for i, q in enumerate(self.verbatim_quotes):
                for qerr in q.validate():
                    errs.append(f"verbatim_quotes[{i}]: {qerr}")

        elif self.grounding_method == GroundingMethod.pretrain_grounded:
            # FROZEN. Only legacy lessons (created before the freeze TS)
            # may carry this state. New lessons attempting to set
            # pretrain_grounded are REJECTED.
            if self.created_ts > PRETRAIN_GROUNDED_FREEZE_TS:
                errs.append(
                    f"grounding_method=pretrain_grounded is FROZEN as of "
                    f"{PRETRAIN_GROUNDED_FREEZE_TS}; lesson created_ts="
                    f"{self.created_ts} is past the freeze. NEW lessons "
                    f"MUST use paper_grounded or stat_only_grounded."
                )

        # stat_only_grounded: no extra requirements; must just have
        # real stat_evidence non-empty (already implied by usage but
        # let's enforce).
        if (self.grounding_method == GroundingMethod.stat_only_grounded
                and not self.stat_evidence):
            errs.append(
                "grounding_method=stat_only_grounded requires non-empty "
                "stat_evidence dict (otherwise it's not grounded in anything)"
            )

        # Stronger requirements for locked
        if self.review_state == ReviewState.locked:
            if not self.source_event_ids:
                errs.append("locked lesson must have source_event_ids")
            if not self.forward_directions:
                errs.append("locked lesson must have at least one forward_direction")

        return errs
