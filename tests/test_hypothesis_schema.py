"""tests/test_hypothesis_schema.py — T1 Hypothesis + chain-rule tests.

Covers:
  - Hypothesis dataclass round-trip + self-validate
  - VerbatimQuote self-validate
  - REDLesson chain-rule enforcement (paper_grounded / pretrain freeze)
  - Cross-chain integration (lesson cites hypothesis_id)
"""
from __future__ import annotations

import json

import pytest

from engine.research_store.hypothesis import (
    Hypothesis, HypothesisDirection, VerbatimQuote,
)
from engine.research_store.hypothesis.schema import (
    ExtractionMethod, HypothesisReviewState,
)
from engine.research_store.hypothesis.store import save_hypothesis
from engine.research_store.red_lessons import (
    DormantRevisit, FailureMode, ForwardVector, GroundingMethod,
    LessonStrength, MechanismFamily, REDLesson, ReviewState,
    PRETRAIN_GROUNDED_FREEZE_TS,
)


# ─────────────────────── VerbatimQuote ────────────────────────────────


def _quote(chunk_id: str = "doi/x::p0001",
           text: str = "This is a verbatim quote from the paper section.") -> VerbatimQuote:
    return VerbatimQuote(
        chunk_id      = chunk_id,
        quote_text    = text,
        section_ref   = "p.10, §3.2",
        relevance_note= "supports the claim about momentum decay",
    )


def test_verbatim_quote_round_trip():
    q = _quote()
    q2 = VerbatimQuote.from_dict(q.to_dict())
    assert q == q2


def test_verbatim_quote_rejects_empty_chunk_id():
    q = VerbatimQuote(
        chunk_id="", quote_text="a valid 30-char quote here xxx", section_ref="")
    assert any("chunk_id" in e for e in q.validate())


def test_verbatim_quote_rejects_too_short():
    q = VerbatimQuote(chunk_id="x", quote_text="too short", section_ref="")
    assert any("too short" in e for e in q.validate())


def test_verbatim_quote_valid_passes():
    assert _quote().validate() == []


# ─────────────────────── Hypothesis dataclass ─────────────────────────


def _minimal_hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis_id        = Hypothesis.new_id(),
        source_paper_id      = "paper-uuid-001",
        version              = 1,
        parent_hypothesis_id = None,
        source_chunk_ids     = ("doi/abc::p0001", "doi/abc::p0002"),
        verbatim_quotes      = (
            _quote("doi/abc::p0001",
                   "Post-publication factor alpha decays by 26-58 percent on average."),
            _quote("doi/abc::p0002",
                   "Decay is sharper for predictors with higher in-sample returns."),
        ),
        claim                = "Post-publication factor returns decay 26-58% on average; "
                               "decay sharper for high in-sample returns.",
        mechanism_family     = MechanismFamily.MOMENTUM,
        mechanism_subtype    = "post_publication_decay",
        predicted_direction  = HypothesisDirection.NEGATIVE,
        predicted_magnitude  = "alpha-t falls by 30-60% post-publication",
        required_data        = ("US cross-section monthly returns 1990-2020",),
        test_methodology     = "Pre vs post publication-date alpha-t comparison",
        extraction_method    = ExtractionMethod.HUMAN_AUTHORED,
        review_state         = HypothesisReviewState.HUMAN_REVIEWED,
        created_ts           = "2026-06-04T14:00:00Z",
        updated_ts           = "2026-06-04T14:00:00Z",
        created_by           = "test",
        tags                 = ("test", "decay_doctrine"),
    )


def test_hypothesis_round_trip():
    h = _minimal_hypothesis()
    h2 = Hypothesis.from_dict(h.to_dict())
    assert h == h2


def test_hypothesis_round_trip_via_json():
    h = _minimal_hypothesis()
    s = json.dumps(h.to_dict(), ensure_ascii=False)
    h2 = Hypothesis.from_dict(json.loads(s))
    assert h == h2


def test_hypothesis_validate_requires_two_quotes():
    h = _minimal_hypothesis()
    bad = Hypothesis(**{**h.__dict__,
                        "verbatim_quotes": (h.verbatim_quotes[0],)})
    errs = bad.validate()
    assert any("≥ 2" in e for e in errs)


def test_hypothesis_validate_requires_source_chunks():
    h = _minimal_hypothesis()
    bad = Hypothesis(**{**h.__dict__, "source_chunk_ids": ()})
    assert any("source_chunk_ids empty" in e for e in bad.validate())


def test_hypothesis_validate_requires_testability_fields():
    h = _minimal_hypothesis()
    bad = Hypothesis(**{**h.__dict__,
                        "predicted_magnitude": "",
                        "test_methodology":    "",
                        "required_data":       ()})
    errs = bad.validate()
    assert any("predicted_magnitude" in e for e in errs)
    assert any("required_data" in e for e in errs)
    assert any("test_methodology" in e for e in errs)


def test_hypothesis_locked_requires_two_quotes_AND_methodology():
    h = _minimal_hypothesis()
    bad = Hypothesis(**{**h.__dict__,
                        "review_state":      HypothesisReviewState.LOCKED,
                        "verbatim_quotes":   (h.verbatim_quotes[0],),
                        "test_methodology":  ""})
    errs = bad.validate()
    assert any("LOCKED hypothesis requires ≥ 2 verbatim_quotes" in e for e in errs)
    assert any("LOCKED hypothesis requires test_methodology" in e for e in errs)


def test_hypothesis_minimal_well_formed_passes():
    assert _minimal_hypothesis().validate() == []


# ─────────────────────── REDLesson chain rules ────────────────────────


def _legacy_lesson(*,
                   grounding_method: GroundingMethod = GroundingMethod.pretrain_grounded,
                   created_ts: str = "2026-06-03T12:00:00Z",
                   tested_hypothesis_ids: tuple[str, ...] = (),
                   verbatim_quotes: tuple[VerbatimQuote, ...] = (),
                   stat_evidence: dict | None = None) -> REDLesson:
    return REDLesson(
        lesson_id           = REDLesson.new_id(),
        candidate_name      = "test_candidate",
        version             = 1,
        parent_lesson_id    = None,
        source_event_ids    = (),
        verdict             = "RED",
        stat_evidence       = stat_evidence if stat_evidence is not None
                              else {"deflated_sr": 0.21, "n_months": 100},
        mechanism_family    = MechanismFamily.MOMENTUM,
        mechanism_subtype   = "test_subtype",
        failure_modes       = (FailureMode.F8_OVERFIT_INDUCED,),
        failure_evidence    = {"F8_OVERFIT_INDUCED": "DSR 0.21 < 0.9 bar"},
        paper_motivation    = None,
        paper_critiques     = (),
        subsumed_by         = (),
        related_lesson_ids  = (),
        forward_directions  = (),
        do_not_retry        = (),
        dormant_revisits    = (),
        tested_hypothesis_ids = tested_hypothesis_ids,
        verbatim_quotes       = verbatim_quotes,
        grounding_method      = grounding_method,
        review_state        = ReviewState.claude_drafted,
        strength            = LessonStrength.medium,
        created_ts          = created_ts,
        updated_ts          = created_ts,
        created_by          = "test",
        summary             = "test lesson",
        tags                = ("test",),
    )


def test_lesson_paper_grounded_requires_hypothesis_ids():
    """paper_grounded with empty tested_hypothesis_ids → validate fails."""
    L = _legacy_lesson(
        grounding_method   = GroundingMethod.paper_grounded,
        tested_hypothesis_ids = (),
        verbatim_quotes       = (_quote(), _quote()),
        created_ts            = "2026-06-04T14:00:00Z",
    )
    errs = L.validate()
    assert any("tested_hypothesis_ids" in e for e in errs)


def test_lesson_paper_grounded_requires_verbatim_quotes():
    L = _legacy_lesson(
        grounding_method   = GroundingMethod.paper_grounded,
        tested_hypothesis_ids = ("hyp-001",),
        verbatim_quotes       = (),
        created_ts            = "2026-06-04T14:00:00Z",
    )
    assert any("verbatim_quotes" in e for e in L.validate())


def test_lesson_paper_grounded_passes_with_both():
    L = _legacy_lesson(
        grounding_method      = GroundingMethod.paper_grounded,
        tested_hypothesis_ids = ("hyp-001",),
        verbatim_quotes       = (
            _quote("doi/x::p0001",
                   "Verbatim quote with sufficient length here."),
            _quote("doi/x::p0002",
                   "Second quote also sufficient and verbatim."),
        ),
        created_ts            = "2026-06-04T14:00:00Z",
    )
    assert L.validate() == []


def test_lesson_pretrain_grounded_FROZEN_after_cutoff():
    """NEW pretrain_grounded lessons (after freeze TS) are rejected."""
    L = _legacy_lesson(
        grounding_method = GroundingMethod.pretrain_grounded,
        created_ts       = "2026-06-05T00:00:00Z",  # past freeze
    )
    errs = L.validate()
    assert any("FROZEN" in e for e in errs)


def test_lesson_pretrain_grounded_allowed_before_cutoff():
    """Legacy pretrain_grounded lessons (before freeze) are accepted."""
    L = _legacy_lesson(
        grounding_method = GroundingMethod.pretrain_grounded,
        created_ts       = "2026-06-03T12:00:00Z",  # before freeze
    )
    assert L.validate() == []


def test_lesson_stat_only_grounded_requires_stat_evidence():
    L = _legacy_lesson(
        grounding_method = GroundingMethod.stat_only_grounded,
        stat_evidence    = {},
        created_ts       = "2026-06-04T14:00:00Z",
    )
    assert any("stat_evidence" in e for e in L.validate())


def test_lesson_stat_only_grounded_passes_with_stats():
    L = _legacy_lesson(
        grounding_method = GroundingMethod.stat_only_grounded,
        stat_evidence    = {"deflated_sr": 0.5, "n_months": 80},
        created_ts       = "2026-06-04T14:00:00Z",
    )
    assert L.validate() == []


def test_lesson_grounding_method_enum_complete():
    """The 3 grounding methods are the ONLY ones available."""
    assert set(g.value for g in GroundingMethod) == {
        "paper_grounded", "stat_only_grounded", "pretrain_grounded"
    }


# ─────────────────── Phase 2.0 step 1 + 4a (2026-06-06) ────────────────
# ExtractionMethod.LLM_SYNTHESIS + synthesizes_paper_ids +
# synthesizes_event_ids + addresses_decay_in.
# Plus relaxed validate() rules when extraction_method == LLM_SYNTHESIS.
# See [[spec-research-session-orchestrator-2026-06-06]].


def test_schema_version_bumped_to_4():
    from engine.research_store.hypothesis.schema import HYPOTHESIS_SCHEMA_VERSION
    assert HYPOTHESIS_SCHEMA_VERSION == 4


def test_extraction_method_llm_synthesis_present():
    """The new value must exist and serialize as 'llm_synthesis'."""
    assert ExtractionMethod.LLM_SYNTHESIS.value == "llm_synthesis"


def test_synthesis_hypothesis_round_trip():
    """A synthesis hypothesis carries multi-source provenance instead
    of a single source_paper_id. All three new fields round-trip cleanly."""
    h = _minimal_hypothesis()
    import dataclasses
    syn = dataclasses.replace(h,
        extraction_method     = ExtractionMethod.LLM_SYNTHESIS,
        synthesizes_paper_ids = ("arxiv/2606.11111", "arxiv/2606.22222"),
        synthesizes_event_ids = ("ev_aaa", "ev_bbb", "ev_ccc"),
        addresses_decay_in    = "carry_g10",
    )
    h2 = Hypothesis.from_dict(syn.to_dict())
    assert h2.extraction_method == ExtractionMethod.LLM_SYNTHESIS
    assert h2.synthesizes_paper_ids == ("arxiv/2606.11111", "arxiv/2606.22222")
    assert h2.synthesizes_event_ids == ("ev_aaa", "ev_bbb", "ev_ccc")
    assert h2.addresses_decay_in == "carry_g10"


def test_backward_compat_pre_2_0_loads_with_defaults():
    """Pre-2.0 jsonl rows (missing the 3 new keys) MUST load with the
    empty/None defaults. Old data on disk must not break."""
    h = _minimal_hypothesis()
    d = h.to_dict()
    # Simulate a pre-2.0 row by dropping the new keys
    d.pop("synthesizes_paper_ids", None)
    d.pop("synthesizes_event_ids", None)
    d.pop("addresses_decay_in", None)
    h2 = Hypothesis.from_dict(d)
    assert h2.synthesizes_paper_ids == ()
    assert h2.synthesizes_event_ids == ()
    assert h2.addresses_decay_in is None


def test_to_dict_emits_v3_fields_even_when_empty():
    """to_dict must always emit the v3 fields (as empty/null) so disk
    rows have the v3 shape — downstream consumers can rely on the keys
    being present."""
    h = _minimal_hypothesis()
    d = h.to_dict()
    assert "synthesizes_paper_ids" in d
    assert "synthesizes_event_ids" in d
    assert "addresses_decay_in" in d
    assert d["synthesizes_paper_ids"] == []
    assert d["synthesizes_event_ids"] == []
    assert d["addresses_decay_in"] is None


def test_unknown_extraction_method_value_rejects():
    """Forward compat NOT applied to extraction_method — unknown
    values should hard-fail because they indicate a real data error
    (an extraction pipeline emitted an undeclared method)."""
    h = _minimal_hypothesis()
    d = h.to_dict()
    d["extraction_method"] = "some_future_method_we_never_added"
    import pytest
    with pytest.raises(ValueError):
        Hypothesis.from_dict(d)


# ─────────────────── Phase 2.0 step 4a: relaxed validate ──────────────


def _minimal_synthesis_hypothesis() -> Hypothesis:
    """A synthesis hypothesis: empty source_paper_id + empty chunks +
    empty quotes, but synthesizes_paper_ids + synthesizes_event_ids
    populated. Should validate cleanly under the relaxed LLM_SYNTHESIS
    rules."""
    import dataclasses
    h = _minimal_hypothesis()
    return dataclasses.replace(h,
        extraction_method     = ExtractionMethod.LLM_SYNTHESIS,
        source_paper_id       = "",
        source_chunk_ids      = (),
        verbatim_quotes       = (),
        synthesizes_paper_ids = ("arxiv/2606.11111",),
        synthesizes_event_ids = ("ev_xyz",),
        addresses_decay_in    = "carry_g10",
    )


def test_synthesis_skips_paper_rooted_checks():
    """Synthesis hypothesis MUST validate cleanly even though it has
    no source_paper_id / source_chunk_ids / verbatim_quotes — that's
    the whole point of the LLM_SYNTHESIS extraction method."""
    h = _minimal_synthesis_hypothesis()
    errs = h.validate()
    assert errs == [], f"unexpected errors: {errs}"


def test_synthesis_requires_at_least_one_synthesizes_field():
    """A synthesis with empty paper_ids AND empty event_ids has no
    provenance — must be rejected."""
    import dataclasses
    h = dataclasses.replace(_minimal_synthesis_hypothesis(),
        synthesizes_paper_ids = (),
        synthesizes_event_ids = (),
    )
    errs = h.validate()
    assert any("synthesizes_paper_ids" in e or "synthesizes_event_ids" in e
                for e in errs)


def test_synthesis_with_only_paper_ids_is_valid():
    import dataclasses
    h = dataclasses.replace(_minimal_synthesis_hypothesis(),
        synthesizes_paper_ids = ("arxiv/p1",),
        synthesizes_event_ids = (),
    )
    assert h.validate() == []


def test_synthesis_with_only_event_ids_is_valid():
    """A pure event-driven synthesis (e.g. addresses a decay alert
    without any new paper) is a legitimate synthesis shape."""
    import dataclasses
    h = dataclasses.replace(_minimal_synthesis_hypothesis(),
        synthesizes_paper_ids = (),
        synthesizes_event_ids = ("ev_decay_alert_1",),
    )
    assert h.validate() == []


def test_synthesis_still_requires_testability_fields():
    """Relaxing paper-rooted checks does NOT relax testability — a
    synthesis hypothesis without predicted_magnitude / required_data
    / test_methodology is still useless."""
    import dataclasses
    h = dataclasses.replace(_minimal_synthesis_hypothesis(),
        predicted_magnitude = "",
        required_data       = (),
        test_methodology    = "",
    )
    errs = h.validate()
    assert any("predicted_magnitude" in e for e in errs)
    assert any("required_data" in e for e in errs)
    assert any("test_methodology" in e for e in errs)


def test_synthesis_can_reach_locked_without_paper_quotes():
    """LOCKED on a synthesis hypothesis should NOT require ≥2 verbatim
    quotes — synthesis provenance lives in synthesizes_* fields.
    methodology still required."""
    import dataclasses
    h = dataclasses.replace(_minimal_synthesis_hypothesis(),
        review_state = HypothesisReviewState.LOCKED,
    )
    assert h.validate() == []


def test_paper_rooted_still_enforces_old_rules():
    """The paper-rooted path (LLM_EXTRACT / HUMAN_AUTHORED) MUST still
    enforce source_paper_id / chunks / ≥2 quotes — relaxing only
    applies to LLM_SYNTHESIS."""
    import dataclasses
    bad = dataclasses.replace(_minimal_hypothesis(),
        extraction_method = ExtractionMethod.LLM_EXTRACT,
        source_paper_id   = "",
        source_chunk_ids  = (),
        verbatim_quotes   = (),
    )
    errs = bad.validate()
    assert any("source_paper_id" in e for e in errs)
    assert any("source_chunk_ids" in e for e in errs)
    assert any("≥ 2" in e for e in errs)
