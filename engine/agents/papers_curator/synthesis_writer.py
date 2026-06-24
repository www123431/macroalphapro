"""engine.agents.papers_curator.synthesis_writer — Phase 2.0 step 4b.

Adapts a list[SynthesizedCandidate] from run_synthesis() to the
persisted Hypothesis dataclass and appends rows to hypotheses.jsonl.

Per [[spec-research-session-orchestrator-2026-06-06]] §"Employee A":
synthesis hypotheses land in the SAME store as paper-extracted
hypotheses — no parallel store. Downstream consumers (B's strengthener,
forward vectors, F14b) read the union via the existing load layer.

Field mapping (SynthesizedCandidate → Hypothesis):

  claim                  → claim
  mechanism_family       → mechanism_family   (case-coerced; OTHER fallback)
  mechanism_subtype      → mechanism_subtype
  predicted_direction    → predicted_direction
  predicted_magnitude    → predicted_magnitude
  required_data          → required_data
  test_methodology       → test_methodology
  synthesizes_paper_ids  → synthesizes_paper_ids
  synthesizes_event_ids  → synthesizes_event_ids
  addresses_decay_in     → addresses_decay_in

Fields set by the writer (not on candidate):

  hypothesis_id          = new UUID4
  source_paper_id        = ""             (synthesis has no single source)
  version                = 1
  parent_hypothesis_id   = None
  source_chunk_ids       = ()             (no paper chunks)
  verbatim_quotes        = ()             (no paper quotes)
  extraction_method      = LLM_SYNTHESIS
  review_state           = PROPOSED       (human review pending)
  created_ts/updated_ts  = now (UTC ISO)
  created_by             = passed-in actor (default "papers_curator_synthesis")
  tags                   = ("synthesis",) plus optional caller-passed tags

Fields DROPPED on write (kept on event payload, not on registry row):

  cochrane_frame, novelty_vs_known, estimated_n_trials_in_family,
  graveyard_conflicts, doctrine_conflicts, expected_outcome_prior,
  generation_ts, model.

These are generation-time metadata — important for audit + the eventual
emit event (step 4c), but NOT part of the testable hypothesis record.
Keeping the Hypothesis schema small means downstream consumers don't
have to special-case synthesis vs paper rows.

Save path uses save_hypothesis(..., skip_cross_checks=True) because:
  - source_paper_id is "" → registry lookup would always fail
  - source_chunk_ids is () → chroma lookup is a no-op
  - verbatim_quotes is () → substring check is a no-op
Cross-checks for synthesizes_paper_ids / synthesizes_event_ids
resolving in their stores are a future step (4c).
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

from engine.agents.papers_curator.synthesis import SynthesizedCandidate
from engine.research_store.hypothesis import Hypothesis, VerbatimQuote
from engine.research_store.hypothesis.schema import (
    ExtractionMethod,
    HypothesisDirection,
    HypothesisReviewState,
)
from engine.research_store.hypothesis.store import save_hypothesis
from engine.research_store.red_lessons.mechanism_families import MechanismFamily

logger = logging.getLogger(__name__)


_DEFAULT_CREATED_BY = "papers_curator_synthesis"


# ────────────────────────────────────────────────────────────────────
# Field coercion helpers
# ────────────────────────────────────────────────────────────────────
# 2026-06-22 alias map (W6-rigor-A end-to-end pipeline fix). When the
# synthesizer LLM uses common-vernacular family names that don't match the
# enum exactly, map to the canonical enum value rather than fall back to
# OTHER. OTHER causes downstream burndown to filter the candidate as
# NON_FACTOR_FAMILY → verdict-unreachable. Lossless: mechanism_subtype
# still carries the LLM's original intent for human review.
_FAMILY_ALIASES: dict[str, str] = {
    "VRP":                       "VOL_RISK_PREMIUM",
    "VARIANCE_RISK_PREMIUM":     "VOL_RISK_PREMIUM",
    "FIXED_INCOME_TERM_PREMIA":  "TERM_STRUCTURE",
    "BOND_TERM_STRUCTURE":       "TERM_STRUCTURE",
    "EQUITY_VRP":                "VOL_RISK_PREMIUM",
    "VOLATILITY_RISK_PREMIUM":   "VOL_RISK_PREMIUM",
}


def _coerce_mechanism_family(s: str) -> MechanismFamily:
    """LLM emits lowercase ('carry'); enum values are UPPERCASE.
    Unknown families fall back to OTHER rather than raise, so a single
    weird candidate doesn't kill the batch. The candidate's
    mechanism_subtype carries the LLM's intent for human review."""
    raw = str(s or "").upper().strip()
    # Alias map first — LLM-vernacular names → canonical enum values
    raw = _FAMILY_ALIASES.get(raw, raw)
    try:
        return MechanismFamily(raw)
    except (ValueError, AttributeError):
        logger.warning(
            "synthesis_writer: unknown mechanism_family %r, falling back to OTHER",
            s,
        )
        return MechanismFamily.OTHER


def _coerce_direction(s: str) -> HypothesisDirection:
    """LLM emits 'positive' / 'negative' / 'zero' (already lowercase).
    Unknown → ZERO (most conservative)."""
    try:
        return HypothesisDirection(str(s).lower().strip())
    except (ValueError, AttributeError):
        logger.warning(
            "synthesis_writer: unknown predicted_direction %r, falling back to ZERO",
            s,
        )
        return HypothesisDirection.ZERO


# ────────────────────────────────────────────────────────────────────
# Adapter — SynthesizedCandidate → Hypothesis
# ────────────────────────────────────────────────────────────────────
def candidate_to_hypothesis(
    cand: SynthesizedCandidate,
    *,
    created_by: str = _DEFAULT_CREATED_BY,
    extra_tags: tuple[str, ...] = (),
    now_iso: str | None = None,
) -> Hypothesis:
    """Pure transform — no I/O. Useful for tests + dry-run inspection.

    `now_iso` is overridable for deterministic testing; default is utcnow."""
    ts = now_iso or _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return Hypothesis(
        hypothesis_id        = Hypothesis.new_id(),
        source_paper_id      = "",
        version              = 1,
        parent_hypothesis_id = None,
        source_chunk_ids     = (),
        verbatim_quotes      = (),
        claim                = cand.claim,
        mechanism_family     = _coerce_mechanism_family(cand.mechanism_family),
        mechanism_subtype    = cand.mechanism_subtype,
        predicted_direction  = _coerce_direction(cand.predicted_direction),
        predicted_magnitude  = cand.predicted_magnitude,
        required_data        = tuple(cand.required_data),
        test_methodology     = cand.test_methodology,
        extraction_method    = ExtractionMethod.LLM_SYNTHESIS,
        review_state         = HypothesisReviewState.PROPOSED,
        created_ts           = ts,
        updated_ts           = ts,
        created_by           = created_by,
        tags                 = ("synthesis",) + tuple(extra_tags),
        synthesizes_paper_ids = tuple(cand.synthesizes_paper_ids),
        synthesizes_event_ids = tuple(cand.synthesizes_event_ids),
        addresses_decay_in    = cand.addresses_decay_in,
        # Phase 2.2c: persist citation quality so B can read it at
        # review time. dict copied so the persisted row owns its
        # own state independent of the source candidate.
        citation_quality      = (dict(cand.citation_quality)
                                  if cand.citation_quality else None),
        # Stage C Phase E + Tier A (2026-06-07): persist A's
        # orthogonality statements so they flow to /research/forward,
        # B's review context, and attribution rollups. Each entry is
        # a plain dict with anchor_paper_id + why_orthogonal —
        # owned-by-this-row copy via dict(o).
        orthogonal_to_anchors = tuple(
            dict(o) for o in (cand.orthogonal_to_anchors or ())
            if isinstance(o, dict)
        ),
    )


# ────────────────────────────────────────────────────────────────────
# Batch writer
# ────────────────────────────────────────────────────────────────────
def write_synthesized_candidates(
    candidates: list[SynthesizedCandidate],
    *,
    created_by: str = _DEFAULT_CREATED_BY,
    extra_tags: tuple[str, ...] = (),
    path: Path | None = None,
    validate_strict: bool = True,
    now_iso: str | None = None,
) -> list[Hypothesis]:
    """Adapt + persist each candidate; return the list of written
    Hypothesis records.

    Args:
      candidates:        run_synthesis() output (may be []).
      created_by:        actor recorded on the hypothesis row.
      extra_tags:        appended to ("synthesis",); use for session
                         tagging (e.g. ("session:cos-2026-06-06",)).
      path:              override hypotheses.jsonl path (tests).
      validate_strict:   pass-through to save_hypothesis. True (default)
                         raises ValueError on self-validation failure;
                         False logs + continues (useful for batch
                         resilience).
      now_iso:           override timestamp for deterministic tests.

    Behavior:
      - Empty input → returns [] without touching disk (no empty
        files created).
      - One malformed candidate (e.g. claim too long) raises if
        validate_strict; with validate_strict=False the call logs +
        the row gets written anyway (Hypothesis remains in the return
        list).
      - skip_cross_checks=True is hard-coded: synthesis rows are
        intentionally chunk/paper-free; running the resolution checks
        would always fail. Future step (4c) will add synthesis-specific
        cross-checks (do synthesizes_paper_ids resolve in registry, etc).
    """
    if not candidates:
        return []

    written: list[Hypothesis] = []
    for i, cand in enumerate(candidates):
        try:
            hyp = candidate_to_hypothesis(
                cand,
                created_by = created_by,
                extra_tags = extra_tags,
                now_iso    = now_iso,
            )
            save_hypothesis(
                hyp,
                path              = path,
                validate_strict   = validate_strict,
                skip_cross_checks = True,
            )
            written.append(hyp)
        except Exception as exc:
            logger.exception(
                "synthesis_writer: candidate %d failed to persist (%s); "
                "continuing with remaining batch",
                i, exc,
            )
            if validate_strict:
                # Re-raise in strict mode so the caller knows the batch
                # is incomplete and can decide whether to retry.
                raise
    return written
