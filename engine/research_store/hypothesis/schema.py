"""engine.research_store.hypothesis.schema — Hypothesis dataclass + VerbatimQuote.

Hypothesis = a structured, testable claim extracted from a SPECIFIC
ingested paper. It is the bridge between PAPER and TEST in the locked
PAPER → HYPOTHESIS → TEST → VERDICT chain (2026-06-04).

Doctrine (load-bearing — read before changing):

  1. `source_paper_id` MUST refer to a paper in papers_registry whose
     `fulltext_status == INGESTED`. Cross-store check enforced in
     `save_hypothesis()` (not in dataclass validate(), because
     dataclass should not load the papers registry).

  2. `source_chunk_ids` MUST be non-empty AND each chunk_id must
     resolve in papers_chroma. Resolution check is done in
     `save_hypothesis()`.

  3. `verbatim_quotes` MUST have at least 2 entries. Each quote's
     `quote_text` must be a verbatim substring of the corresponding
     papers_chroma chunk text. Substring check is done in
     `save_hypothesis()`.

  4. `claim`, `predicted_direction`, `predicted_magnitude`,
     `required_data`, `test_methodology` are required. These are what
     makes the hypothesis TESTABLE — without them, it's a vague claim
     not a hypothesis.
"""
from __future__ import annotations

import dataclasses as _dc
import uuid as _uuid
from enum import Enum
from typing import Any

from engine.research_store.red_lessons.mechanism_families import MechanismFamily


HYPOTHESIS_SCHEMA_VERSION = 4     # bumped 2026-06-07 for Phase 2.2c
                                    # (+ citation_quality field carrying
                                    # the aggregate dict from
                                    # citation_verifier.aggregate_citation_quality
                                    # so B can downweight candidates with
                                    # weak / hallucinated citations).
                                    # Backward compat: old rows load with
                                    # citation_quality=None.


class HypothesisDirection(str, Enum):
    """Direction the hypothesis predicts."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ZERO     = "zero"


class ExtractionMethod(str, Enum):
    """How was the hypothesis extracted from the paper?"""
    LLM_EXTRACT       = "llm_extract"
    HUMAN_AUTHORED    = "human_authored"
    # Phase 2.0 step 1 (2026-06-06): cross-source synthesis emitted by
    # Employee A's papers_curator_synthesis call. The hypothesis is not
    # rooted in a single paper's claim field — it's a synthesis across
    # multiple papers + deployed sleeves + decay alerts + memory.
    # source_paper_id may be empty; synthesizes_event_ids carries the
    # triggering provenance. See
    # [[spec-research-session-orchestrator-2026-06-06]] §"Employee A".
    LLM_SYNTHESIS     = "llm_synthesis"


class HypothesisReviewState(str, Enum):
    """Lifecycle of a hypothesis record."""
    PROPOSED          = "proposed"          # LLM-extracted, no human review yet
    HUMAN_REVIEWED    = "human_reviewed"    # human pass; testable
    LOCKED            = "locked"            # final; can be cited by tests
    REJECTED          = "rejected"          # human rejected as not a testable claim


class HypothesisType(str, Enum):
    """What KIND of hypothesis this is — burn-1b-followup (2026-06-11).

    Pre-2026-06-11 we conflated four very different shapes under
    'mechanism_family'. The first authorized cron run exposed it: top-3
    candidates by score were all PROFITABILITY/VALUE/SIZE-tagged but
    claimed things like 'HML is redundant under FF5' / 'minimum t-ratio
    should be 3.0' — factor-ANALYSIS or methodology, not factor-PROPOSAL.
    LLM extractor correctly refused; cron burned ~$0.09 + 30s on
    candidates it could never dispatch.

    Type taxonomy + the burndown_ranker treatment:

      factor_proposal     — 'stocks with X have higher returns'.
                            Dispatchable via strict gate. CRON ELIGIBLE.
      factor_analysis     — 'HML becomes redundant under FF5'.
                            Analysis of existing factors. NOT cron eligible
                            (no spec to dispatch); belongs to literature
                            review queue.
      methodology         — 'minimum t-ratio should be 3.0'.
                            Meta-research. NOT cron eligible; doctrine
                            consumer or audit notebook.
      sleeve_improvement  — 'Add variance-swap to VXX short-vol sleeve'.
                            Improvement proposal targeting deployed sleeve.
                            NOT cron eligible; sleeve_fix_proposer consumer.
      unknown             — Pre-2026-06-11 rows that haven't been classified.
                            Conservative: NOT cron eligible until classified.

    Classification happens in engine.research_store.hypothesis.classifier
    by rules; backfill script normalizes existing 234 rows. New
    hypotheses created by hypothesis_extractor / papers_curator should
    populate this field at write time (deferred to a separate piece —
    extractor updates not in this commit)."""
    FACTOR_PROPOSAL    = "factor_proposal"
    FACTOR_ANALYSIS    = "factor_analysis"
    METHODOLOGY        = "methodology"
    SLEEVE_IMPROVEMENT = "sleeve_improvement"
    UNKNOWN            = "unknown"


@_dc.dataclass(frozen=True)
class VerbatimQuote:
    """A verbatim quote from a paper chunk.

    Schema-self-checks:
      - chunk_id non-empty
      - quote_text non-empty AND ≥ 20 chars (anything shorter is a
        citation, not a substantive quote)

    NOT checked at dataclass layer (deferred to save_hypothesis()):
      - chunk_id resolves in papers_chroma
      - quote_text is a verbatim substring of the chunk text
    """
    chunk_id:        str
    quote_text:      str
    section_ref:     str = ""
    relevance_note:  str = ""

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VerbatimQuote":
        return cls(
            chunk_id       = d["chunk_id"],
            quote_text     = d["quote_text"],
            section_ref    = d.get("section_ref", ""),
            relevance_note = d.get("relevance_note", ""),
        )

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.chunk_id.strip():
            errs.append("VerbatimQuote.chunk_id is empty")
        if not self.quote_text.strip():
            errs.append("VerbatimQuote.quote_text is empty")
        elif len(self.quote_text) < 20:
            errs.append(
                f"VerbatimQuote.quote_text too short ({len(self.quote_text)} "
                f"chars); a quote should be ≥ 20 chars to be substantive"
            )
        return errs


@_dc.dataclass(frozen=True)
class Hypothesis:
    """A testable claim extracted from a specific paper.

    Identity:
      hypothesis_id        — UUID4
      source_paper_id      — papers_registry.paper_id (MUST be INGESTED)
      version              — int starting at 1
      parent_hypothesis_id — for amendments

    Source:
      source_chunk_ids     — papers_chroma chunk IDs supporting the claim
      verbatim_quotes      — ≥ 2 quotes from those chunks
      claim                — 1-3 sentence paraphrase
      mechanism_family     — controlled MechanismFamily
      mechanism_subtype    — free-form refinement

    Testability (LOAD-BEARING):
      predicted_direction  — HypothesisDirection enum
      predicted_magnitude  — e.g. "Sharpe > 0.5", "alpha-t > 2"
      required_data        — e.g. ("US large-cap equity",
                                   "1990-2020 monthly returns")
      test_methodology     — method as described in paper

    Metadata:
      extraction_method    — ExtractionMethod enum
      review_state         — HypothesisReviewState enum
      created_ts           — ISO-8601 UTC
      updated_ts           — ISO-8601 UTC
      created_by           — actor
      tags                 — free-form labels

    Versioning:
      schema_version       — bumped on incompatible schema change
    """

    # Identity
    hypothesis_id:        str
    source_paper_id:      str
    version:              int
    parent_hypothesis_id: str | None

    # Source
    source_chunk_ids:     tuple[str, ...]
    verbatim_quotes:      tuple[VerbatimQuote, ...]
    claim:                str
    mechanism_family:     MechanismFamily
    mechanism_subtype:    str

    # Testability
    predicted_direction:  HypothesisDirection
    predicted_magnitude:  str
    required_data:        tuple[str, ...]
    test_methodology:     str

    # Metadata
    extraction_method:    ExtractionMethod
    review_state:         HypothesisReviewState
    created_ts:           str
    updated_ts:           str
    created_by:           str
    tags:                 tuple[str, ...]

    schema_version:       int = HYPOTHESIS_SCHEMA_VERSION

    # Phase 2.0 step 1 + 4a (2026-06-06): Employee A cross-source synthesis
    # carries multi-source provenance instead of a single source_paper_id.
    # All three fields default empty/None so pre-2.0 jsonl rows load
    # unchanged. See [[spec-research-session-orchestrator-2026-06-06]]
    # §"Employee A · Papers Curator — cross-source synthesis".
    #
    # synthesizes_paper_ids:  papers whose summaries fed the synthesis
    #   (multi-paper; a single paper-rooted hypothesis uses
    #   source_paper_id instead). Empty tuple for paper-extracted rows.
    # synthesizes_event_ids:  events that triggered this synthesis —
    #   factor_verdict_filed (RED that motivated this idea),
    #   doctrine_signal_detected (D's pattern), capability_evidence_filed
    #   (papers' summaries). Empty tuple for paper-extracted rows.
    # addresses_decay_in:     sleeve_id (e.g. "carry_g10") if this
    #   hypothesis is specifically aimed at a known decay; None
    #   otherwise. Lets B's downstream prioritizer rank decay-addressing
    #   candidates higher.
    synthesizes_paper_ids: tuple[str, ...] = ()
    synthesizes_event_ids: tuple[str, ...] = ()
    addresses_decay_in:    "str | None"    = None

    # Phase 2.2c (2026-06-07): citation_verifier aggregate.
    # Shape: {n_papers_cited, n_resolved, n_unresolved, mean_confidence,
    #         min_confidence, any_unresolved, low_confidence_flag}.
    # None = not yet verified (old rows / synthesis run before 2.2b).
    # B's review prompt reads this to weight candidates with weak or
    # hallucinated citations.
    citation_quality:      "dict | None"   = None

    # Stage C Phase E + Tier A (2026-06-07): orthogonality statements
    # A produced when proposing this candidate, against the canonical
    # anchor library (T1+T2 papers in papers_registry). Each entry:
    # {"anchor_paper_id": str, "why_orthogonal": str}. Empty for
    # pre-Phase-E rows AND for non-LLM_SYNTHESIS extraction methods
    # (paper-rooted hypotheses don't need explicit orthogonality —
    # the paper IS the anchor by construction).
    orthogonal_to_anchors: tuple = ()

    # burn-1b-followup (2026-06-11): hypothesis_type classifies what
    # KIND of claim this is — factor proposal vs analysis vs methodology
    # vs sleeve improvement. Pre-existing rows default to UNKNOWN; the
    # backfill script classifies via rules. burndown_ranker only sends
    # FACTOR_PROPOSAL through the cron path.
    #
    # Defaulting to UNKNOWN (not FACTOR_PROPOSAL) is deliberate: in the
    # absence of classification, the cron should NOT auto-treat a row
    # as dispatchable — conservatism prevents repeating the 2026-06-11
    # extractor cost waste.
    hypothesis_type:       HypothesisType = HypothesisType.UNKNOWN

    @staticmethod
    def new_id() -> str:
        return str(_uuid.uuid4())

    # ─────────── (de)serialization ───────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id":        self.hypothesis_id,
            "source_paper_id":      self.source_paper_id,
            "version":              self.version,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "schema_version":       self.schema_version,

            "source_chunk_ids":     list(self.source_chunk_ids),
            "verbatim_quotes":      [q.to_dict() for q in self.verbatim_quotes],
            "claim":                self.claim,
            "mechanism_family":     self.mechanism_family.value,
            "mechanism_subtype":    self.mechanism_subtype,

            "predicted_direction":  self.predicted_direction.value,
            "predicted_magnitude":  self.predicted_magnitude,
            "required_data":        list(self.required_data),
            "test_methodology":     self.test_methodology,

            "extraction_method":    self.extraction_method.value,
            "review_state":         self.review_state.value,
            "created_ts":           self.created_ts,
            "updated_ts":           self.updated_ts,
            "created_by":           self.created_by,
            "tags":                 list(self.tags),

            # Phase 2.0 step 1 + 4a — emit always (even when empty) so
            # the v3 shape is visible on disk. Forward compat: pre-2.0
            # readers ignore unknown keys.
            "synthesizes_paper_ids": list(self.synthesizes_paper_ids),
            "synthesizes_event_ids": list(self.synthesizes_event_ids),
            "addresses_decay_in":    self.addresses_decay_in,

            # Phase 2.2c
            "citation_quality":      self.citation_quality,

            # Stage C Phase E + Tier A
            "orthogonal_to_anchors": [dict(o) for o in (self.orthogonal_to_anchors or ())],

            # burn-1b-followup
            "hypothesis_type":       self.hypothesis_type.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Hypothesis":
        return cls(
            hypothesis_id        = d["hypothesis_id"],
            source_paper_id      = d["source_paper_id"],
            version              = int(d.get("version", 1)),
            parent_hypothesis_id = d.get("parent_hypothesis_id"),
            schema_version       = int(d.get("schema_version", HYPOTHESIS_SCHEMA_VERSION)),

            source_chunk_ids     = tuple(d.get("source_chunk_ids") or ()),
            verbatim_quotes      = tuple(
                VerbatimQuote.from_dict(q) for q in (d.get("verbatim_quotes") or ())
            ),
            claim                = d.get("claim", ""),
            mechanism_family     = MechanismFamily(d["mechanism_family"]),
            mechanism_subtype    = d.get("mechanism_subtype", ""),

            predicted_direction  = HypothesisDirection(d["predicted_direction"]),
            predicted_magnitude  = d.get("predicted_magnitude", ""),
            required_data        = tuple(d.get("required_data") or ()),
            test_methodology     = d.get("test_methodology", ""),

            extraction_method    = ExtractionMethod(d["extraction_method"]),
            review_state         = HypothesisReviewState(d["review_state"]),
            created_ts           = d["created_ts"],
            updated_ts           = d.get("updated_ts", d["created_ts"]),
            created_by           = d["created_by"],
            tags                 = tuple(d.get("tags") or ()),

            # Phase 2.0 step 1 + 4a — backward compat: pre-2.0 rows
            # missing these keys load with the empty/None defaults.
            synthesizes_paper_ids = tuple(d.get("synthesizes_paper_ids") or ()),
            synthesizes_event_ids = tuple(d.get("synthesizes_event_ids") or ()),
            addresses_decay_in    = d.get("addresses_decay_in"),

            # Phase 2.2c — pre-2.2c rows load with None (not yet verified)
            citation_quality      = d.get("citation_quality"),

            # Stage C Phase E + Tier A — pre-Phase-E rows load with ()
            orthogonal_to_anchors = tuple(
                {
                    "anchor_paper_id": str(o.get("anchor_paper_id", "")),
                    "why_orthogonal":  str(o.get("why_orthogonal", "")),
                }
                for o in (d.get("orthogonal_to_anchors") or [])
                if isinstance(o, dict)
            ),

            # burn-1b-followup — pre-existing rows without the field
            # load as UNKNOWN; classifier + backfill normalizes.
            hypothesis_type      = HypothesisType(d.get("hypothesis_type", "unknown")),
        )

    # ─────────── self-validation ───────────

    def validate(self) -> list[str]:
        """Self-check. Does NOT cross-validate against papers_registry /
        papers_chroma; that's `save_hypothesis()`'s job.

        Rule branches on extraction_method:

          - LLM_EXTRACT / HUMAN_AUTHORED (paper-rooted):  source_paper_id
            non-empty, ≥ 1 source_chunk_ids, ≥ 2 verbatim_quotes (the
            PAPER → HYPOTHESIS doctrine chain). LOCKED state requires
            ≥ 2 quotes + methodology.

          - LLM_SYNTHESIS (cross-source):  no single source paper —
            instead require synthesizes_paper_ids OR synthesizes_event_ids
            non-empty (carries multi-source provenance). chunks + quotes
            are not applicable (the hypothesis is synthesized from
            summaries + events, not extracted from a paper chunk).
            LOCKED state still allowed but cannot reach it through
            paper-quote requirements.
        """
        errs: list[str] = []
        is_synthesis = self.extraction_method == ExtractionMethod.LLM_SYNTHESIS

        if not self.claim.strip():
            errs.append("claim is empty")
        if len(self.claim) > 800:
            errs.append(f"claim too long ({len(self.claim)} chars; max 800)")

        if is_synthesis:
            # Synthesis rows carry multi-source provenance instead of a
            # single source_paper_id. Require at least one populated.
            if not (self.synthesizes_paper_ids or self.synthesizes_event_ids):
                errs.append(
                    "LLM_SYNTHESIS hypothesis requires synthesizes_paper_ids "
                    "OR synthesizes_event_ids non-empty"
                )
        else:
            if not self.source_paper_id.strip():
                errs.append("source_paper_id is empty")
            # Source chunks: ≥ 1 required (paper-rooted only)
            if not self.source_chunk_ids:
                errs.append("source_chunk_ids empty (must have ≥ 1)")
            # Quotes: ≥ 2 required per locked doctrine (paper-rooted only)
            if len(self.verbatim_quotes) < 2:
                errs.append(
                    f"verbatim_quotes has {len(self.verbatim_quotes)} entries; "
                    f"must have ≥ 2 per chain doctrine"
                )

        # Each quote self-validates (whatever the extraction method)
        for i, q in enumerate(self.verbatim_quotes):
            for qerr in q.validate():
                errs.append(f"verbatim_quotes[{i}]: {qerr}")

        # Testability required fields (apply to all)
        if not self.predicted_magnitude.strip():
            errs.append("predicted_magnitude is empty")
        if not self.required_data:
            errs.append("required_data empty (must specify ≥ 1 data requirement)")
        if not self.test_methodology.strip():
            errs.append("test_methodology is empty")

        # LOCKED state — paper-rooted hypotheses require both quotes +
        # methodology. Synthesis hypotheses can be LOCKED without paper
        # quotes (their provenance is the synthesizes_* fields), but
        # methodology is still required.
        if self.review_state == HypothesisReviewState.LOCKED:
            if not is_synthesis and len(self.verbatim_quotes) < 2:
                errs.append("LOCKED hypothesis requires ≥ 2 verbatim_quotes")
            if not self.test_methodology.strip():
                errs.append("LOCKED hypothesis requires test_methodology")

        return errs
