"""tests/test_red_lessons_retrieval.py — P3 3-layer retrieval API tests.

Synthetic lesson fixtures to exercise the L1/L2/L3/dormant logic without
depending on the live store. Edge cases: empty inputs, no-overlap candidates,
multi-failure-mode candidates, inferred failure modes.
"""
from __future__ import annotations

import pytest

from engine.research_store.red_lessons import (
    DormantRevisit, FailureMode, ForwardVector, GroundingMethod,
    LessonStrength, MechanismFamily, PaperRef, REDLesson, ReviewState,
)
from engine.research_store.red_lessons.retrieval import (
    Candidate, get_briefing,
    infer_failure_modes_from_family,
    query_dormant_reactivations,
    query_layer1_hard_stop,
    query_layer2_reconsider,
    query_layer3_structural,
)


def _lesson(name: str, family: MechanismFamily,
            failure_modes: tuple[FailureMode, ...],
            *, dsr: float = 0.5, n_months: int = 100,
            dormant: tuple[DormantRevisit, ...] = (),
            review_state: ReviewState = ReviewState.claude_drafted) -> REDLesson:
    return REDLesson(
        lesson_id           = REDLesson.new_id(),
        candidate_name      = name,
        version             = 1,
        parent_lesson_id    = None,
        source_event_ids    = (),
        verdict             = "RED",
        stat_evidence       = {"deflated_sr": dsr, "n_months": n_months},
        mechanism_family    = family,
        mechanism_subtype   = "synthetic",
        failure_modes       = failure_modes,
        failure_evidence    = {fm.value: f"evidence for {fm.value}" for fm in failure_modes},
        paper_motivation    = None,
        paper_critiques     = (),
        subsumed_by         = (),
        related_lesson_ids  = (),
        forward_directions  = (),
        do_not_retry        = (),
        dormant_revisits    = dormant,
        # 2026-06-04 paper-driven chain — synthetic lessons carry real stat
        # evidence (DSR + n_months) but no hypothesis chain, so
        # stat_only_grounded fits and they participate in retrieval by default.
        tested_hypothesis_ids = (),
        verbatim_quotes       = (),
        grounding_method      = GroundingMethod.stat_only_grounded,
        review_state        = review_state,
        strength            = LessonStrength.medium,
        created_ts          = "2026-06-03T12:00:00Z",
        updated_ts          = "2026-06-03T12:00:00Z",
        created_by          = "test",
        summary             = f"{name} test lesson",
        tags                = ("test",),
    )


# ─────────────────────── infer_failure_modes ──────────────────────────


def test_infer_failure_modes_union():
    """Union of all failure modes seen in the family's lessons."""
    lessons = [
        _lesson("a", MechanismFamily.CARRY, (FailureMode.F1_PUBLICATION_DECAY,)),
        _lesson("b", MechanismFamily.CARRY, (FailureMode.F3_SUBSUMED_BY_EXISTING,
                                             FailureMode.F8_OVERFIT_INDUCED)),
        _lesson("c", MechanismFamily.ATTENTION, (FailureMode.F1_PUBLICATION_DECAY,)),
    ]
    out = infer_failure_modes_from_family(MechanismFamily.CARRY, lessons)
    assert set(out) == {FailureMode.F1_PUBLICATION_DECAY,
                        FailureMode.F3_SUBSUMED_BY_EXISTING,
                        FailureMode.F8_OVERFIT_INDUCED}


def test_infer_failure_modes_empty_family():
    """Family with no historical lessons returns empty."""
    lessons = [_lesson("a", MechanismFamily.CARRY, (FailureMode.F1_PUBLICATION_DECAY,))]
    out = infer_failure_modes_from_family(MechanismFamily.MOMENTUM, lessons)
    assert out == ()


# ─────────────────────── Layer 1 — hard STOP ──────────────────────────


def test_l1_basic_match():
    """Same family + same failure mode → L1 hit."""
    lessons = [
        _lesson("prior_carry_decay", MechanismFamily.CARRY,
                (FailureMode.F1_PUBLICATION_DECAY,)),
    ]
    cand = Candidate(
        name="new_carry_test",
        mechanism_family=MechanismFamily.CARRY,
        likely_failure_modes=(FailureMode.F1_PUBLICATION_DECAY,),
    )
    hits = query_layer1_hard_stop(cand, lessons)
    assert len(hits) == 1
    assert hits[0].layer == "L1"
    assert "F1_PUBLICATION_DECAY" in hits[0].relevance_reason


def test_l1_different_family_no_match():
    lessons = [
        _lesson("prior_mom_decay", MechanismFamily.MOMENTUM,
                (FailureMode.F1_PUBLICATION_DECAY,)),
    ]
    cand = Candidate(
        name="new_carry",
        mechanism_family=MechanismFamily.CARRY,
        likely_failure_modes=(FailureMode.F1_PUBLICATION_DECAY,),
    )
    assert query_layer1_hard_stop(cand, lessons) == []


def test_l1_disjoint_failure_modes_no_match():
    lessons = [
        _lesson("prior_carry_subsumed", MechanismFamily.CARRY,
                (FailureMode.F3_SUBSUMED_BY_EXISTING,)),
    ]
    cand = Candidate(
        name="new_carry",
        mechanism_family=MechanismFamily.CARRY,
        likely_failure_modes=(FailureMode.F1_PUBLICATION_DECAY,),
    )
    assert query_layer1_hard_stop(cand, lessons) == []


def test_l1_inferred_failure_modes_from_family():
    """When caller doesn't supply likely_failure_modes, infer from family."""
    lessons = [
        _lesson("a", MechanismFamily.CARRY, (FailureMode.F1_PUBLICATION_DECAY,)),
        _lesson("b", MechanismFamily.CARRY, (FailureMode.F4_IMPLEMENTATION_COST,)),
    ]
    cand = Candidate(name="new", mechanism_family=MechanismFamily.CARRY)
    hits = query_layer1_hard_stop(cand, lessons)
    # Both should match — inferred = {F1, F4} which overlaps both
    assert len(hits) == 2


def test_l1_sorted_by_score():
    """Hits sorted by score descending."""
    strong = _lesson("strong", MechanismFamily.CARRY,
                     (FailureMode.F1_PUBLICATION_DECAY,),
                     dsr=0.1, n_months=200,
                     review_state=ReviewState.human_reviewed)
    weak   = _lesson("weak",   MechanismFamily.CARRY,
                     (FailureMode.F1_PUBLICATION_DECAY,),
                     dsr=0.85, n_months=30)
    cand = Candidate(
        name="new", mechanism_family=MechanismFamily.CARRY,
        likely_failure_modes=(FailureMode.F1_PUBLICATION_DECAY,),
    )
    hits = query_layer1_hard_stop(cand, [weak, strong])
    assert hits[0].lesson.candidate_name == "strong"
    assert hits[1].lesson.candidate_name == "weak"


# ─────────────────────── Layer 2 — reconsider ─────────────────────────


def test_l2_same_family_different_failure():
    """Same family but DIFFERENT failure modes → L2 hit."""
    lessons = [
        _lesson("subsumed_carry", MechanismFamily.CARRY,
                (FailureMode.F3_SUBSUMED_BY_EXISTING,)),
    ]
    cand = Candidate(
        name="new_carry",
        mechanism_family=MechanismFamily.CARRY,
        likely_failure_modes=(FailureMode.F1_PUBLICATION_DECAY,),
    )
    hits = query_layer2_reconsider(cand, lessons)
    assert len(hits) == 1
    assert hits[0].layer == "L2"
    assert "DIFFERENT" in hits[0].relevance_reason


def test_l2_overlap_excluded():
    """A lesson that overlaps goes to L1 not L2."""
    lessons = [
        _lesson("a", MechanismFamily.CARRY,
                (FailureMode.F1_PUBLICATION_DECAY, FailureMode.F3_SUBSUMED_BY_EXISTING)),
    ]
    cand = Candidate(
        name="new", mechanism_family=MechanismFamily.CARRY,
        likely_failure_modes=(FailureMode.F1_PUBLICATION_DECAY,),
    )
    assert query_layer2_reconsider(cand, lessons) == []  # overlap → L1


# ─────────────────────── Layer 3 — structural ─────────────────────────


def test_l3_cross_family_same_failure():
    """Cross-family but same failure mode → L3 hit."""
    lessons = [
        _lesson("mom_decay", MechanismFamily.MOMENTUM,
                (FailureMode.F1_PUBLICATION_DECAY,)),
        _lesson("attention_decay", MechanismFamily.ATTENTION,
                (FailureMode.F1_PUBLICATION_DECAY,)),
        _lesson("carry_subsumed", MechanismFamily.CARRY,
                (FailureMode.F3_SUBSUMED_BY_EXISTING,)),
    ]
    cand = Candidate(
        name="new_carry",
        mechanism_family=MechanismFamily.CARRY,
        likely_failure_modes=(FailureMode.F1_PUBLICATION_DECAY,),
    )
    hits = query_layer3_structural(cand, lessons)
    assert len(hits) == 2   # mom + attention
    assert all("structural risk" in h.relevance_reason for h in hits)


def test_l3_excludes_same_family():
    """L3 must be cross-family — same-family hits go to L1/L2."""
    lessons = [
        _lesson("a", MechanismFamily.CARRY, (FailureMode.F1_PUBLICATION_DECAY,)),
    ]
    cand = Candidate(name="new", mechanism_family=MechanismFamily.CARRY,
                     likely_failure_modes=(FailureMode.F1_PUBLICATION_DECAY,))
    assert query_layer3_structural(cand, lessons) == []


# ─────────────────────── dormant ──────────────────────────────────────


def test_dormant_matches_signal():
    """Available signal triggers dormant_revisit."""
    dr = DormantRevisit(
        condition_label="CN A-share Wiki pageview panel",
        condition_check="data/cache/cn_wiki_views.parquet exists",
        reactivation_note="Would let us test attention in retail-dominated CN.",
    )
    lessons = [
        _lesson("news_attention", MechanismFamily.ATTENTION,
                (FailureMode.F3_SUBSUMED_BY_EXISTING,),
                dormant=(dr,)),
    ]
    cand = Candidate(name="cn_attention", mechanism_family=MechanismFamily.ATTENTION)
    # Signal substring must match the condition_label / condition_check text
    hits = query_dormant_reactivations(cand, lessons,
                                       available_data_signals=("CN A-share Wiki",))
    assert len(hits) == 1
    assert hits[0].layer == "dormant"


def test_dormant_no_signal_no_match():
    dr = DormantRevisit("X", "Y", "Z")
    lessons = [_lesson("a", MechanismFamily.ATTENTION,
                       (FailureMode.F1_PUBLICATION_DECAY,), dormant=(dr,))]
    cand = Candidate(name="t", mechanism_family=MechanismFamily.ATTENTION)
    assert query_dormant_reactivations(cand, lessons, available_data_signals=()) == []


# ─────────────────────── top-level briefing ───────────────────────────


def test_briefing_full_chain():
    """End-to-end with all 4 layers having content."""
    lessons = [
        _lesson("l1_hit", MechanismFamily.CARRY, (FailureMode.F1_PUBLICATION_DECAY,)),
        _lesson("l2_hit", MechanismFamily.CARRY, (FailureMode.F4_IMPLEMENTATION_COST,)),
        _lesson("l3_hit", MechanismFamily.MOMENTUM, (FailureMode.F1_PUBLICATION_DECAY,)),
        _lesson(
            "dormant_hit", MechanismFamily.ATTENTION,
            (FailureMode.F3_SUBSUMED_BY_EXISTING,),
            dormant=(DormantRevisit("CN data", "panel exists", "reactivate"),),
        ),
    ]
    cand = Candidate(
        name="new_carry",
        mechanism_family=MechanismFamily.CARRY,
        likely_failure_modes=(FailureMode.F1_PUBLICATION_DECAY,),
    )
    briefing = get_briefing(cand, available_data_signals=("CN data",),
                            lessons=lessons, registry=[])
    assert len(briefing.layer1_hard_stop) == 1
    assert briefing.layer1_hard_stop[0].lesson.candidate_name == "l1_hit"
    assert len(briefing.layer2_reconsider) == 1
    assert briefing.layer2_reconsider[0].lesson.candidate_name == "l2_hit"
    assert len(briefing.layer3_structural) == 1
    assert briefing.layer3_structural[0].lesson.candidate_name == "l3_hit"
    assert len(briefing.dormant_reactivate) == 1
    assert briefing.total_hits() == 4


def test_briefing_empty_lessons():
    cand = Candidate(name="x", mechanism_family=MechanismFamily.CARRY)
    briefing = get_briefing(cand, lessons=[], registry=[])
    assert briefing.total_hits() == 0
    assert briefing.candidate == cand


def test_briefing_inferred_failure_modes_note():
    """When caller doesn't supply failure modes, briefing.note explains the inference."""
    lessons = [_lesson("a", MechanismFamily.CARRY, (FailureMode.F1_PUBLICATION_DECAY,))]
    cand = Candidate(name="x", mechanism_family=MechanismFamily.CARRY)
    briefing = get_briefing(cand, lessons=lessons, registry=[])
    assert "inferred from" in briefing.note.lower()


# ─────────────────────── paper-side queries ───────────────────────────


def test_paper_by_shelf_filter():
    """query_papers_by_shelf returns only papers carrying ANY of requested shelves."""
    from engine.research_store.papers import (
        FulltextStatus, PaperRegistryEntry, Shelf,
    )
    e1 = PaperRegistryEntry(
        paper_id="p1", version=1, parent_paper_id=None,
        doi="10.1/x", title="Doctrine paper", year=2014,
        authors=("A",), venue="JF", abstract="",
        fulltext_status=FulltextStatus.METADATA_ONLY,
        pdf_source_kind="", pdf_source_url="", n_chunks=0, ingested_ts="",
        referenced_by_lessons=(), referenced_by_factors=(),
        referenced_by_sleeves=(), referenced_by_doctrines=(),
        shelves=(Shelf.DOCTRINE_METHOD,), shelf_notes={},
        created_ts="2026-06-03T12:00:00Z", updated_ts="2026-06-03T12:00:00Z",
        created_by="t", tags=(), note="",
    )
    e2 = PaperRegistryEntry(**{**e1.__dict__,
                               "paper_id": "p2", "doi": "10.1/y",
                               "shelves": (Shelf.RED_MOTIVATION,)})
    from engine.research_store.red_lessons.retrieval import query_papers_by_shelf
    hits = query_papers_by_shelf((Shelf.RED_MOTIVATION,), [e1, e2])
    assert len(hits) == 1
    assert hits[0].entry.paper_id == "p2"


def test_paper_semantic_query_empty_collection_returns_empty():
    """Empty / unreachable chroma collection → returns [] not raises."""
    from engine.research_store.red_lessons.retrieval import query_papers_semantic
    # No exception even if collection is empty
    result = query_papers_semantic("test query", top_k=5)
    assert isinstance(result, list)   # may be [] or have results from live system


# ─────────────────────── T4 hypothesis-side queries ───────────────────


def _legacy_lesson_with_hypothesis(name: str, hyp_ids: tuple[str, ...],
                                    *, gm: GroundingMethod = GroundingMethod.stat_only_grounded
                                    ) -> REDLesson:
    """Synthetic lesson with explicit tested_hypothesis_ids for T4 tests."""
    return REDLesson(
        lesson_id           = REDLesson.new_id(),
        candidate_name      = name,
        version             = 1,
        parent_lesson_id    = None,
        source_event_ids    = (),
        verdict             = "RED",
        stat_evidence       = {"deflated_sr": 0.5, "n_months": 100},
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
        tested_hypothesis_ids = hyp_ids,
        verbatim_quotes       = (),
        grounding_method      = gm,
        review_state        = ReviewState.claude_drafted,
        strength            = LessonStrength.medium,
        created_ts          = "2026-06-04T16:00:00Z",
        updated_ts          = "2026-06-04T16:00:00Z",
        created_by          = "test",
        summary             = f"{name} synthetic lesson",
        tags                = ("test_t4",),
    )


def test_query_lessons_for_hypothesis_finds_match():
    from engine.research_store.red_lessons.retrieval import (
        query_lessons_for_hypothesis,
    )
    L_match = _legacy_lesson_with_hypothesis("matches", ("hyp-001",))
    L_other = _legacy_lesson_with_hypothesis("other", ("hyp-999",))
    hits = query_lessons_for_hypothesis("hyp-001", lessons=[L_match, L_other])
    assert len(hits) == 1
    assert hits[0].candidate_name == "matches"


def test_query_lessons_for_hypothesis_excludes_legacy_by_default():
    """Legacy (pretrain_grounded) lessons are filtered out even if they
    have hypothesis_ids (those would be dangling refs anyway)."""
    from engine.research_store.red_lessons.retrieval import (
        query_lessons_for_hypothesis,
    )
    L_legacy = _legacy_lesson_with_hypothesis(
        "legacy", ("hyp-001",),
        gm=GroundingMethod.pretrain_grounded,
    )
    # Have to override created_ts to before-freeze for legacy validate
    L_legacy = REDLesson(**{**L_legacy.__dict__,
                            "created_ts": "2026-06-03T12:00:00Z",
                            "updated_ts": "2026-06-03T12:00:00Z"})
    assert query_lessons_for_hypothesis("hyp-001", lessons=[L_legacy]) == []
    # opt-in finds it
    assert len(query_lessons_for_hypothesis("hyp-001", lessons=[L_legacy],
                                            include_legacy=True)) == 1


def test_tested_hypothesis_ids_returns_union():
    from engine.research_store.red_lessons.retrieval import (
        tested_hypothesis_ids,
    )
    L1 = _legacy_lesson_with_hypothesis("a", ("hyp-001", "hyp-002"))
    L2 = _legacy_lesson_with_hypothesis("b", ("hyp-002", "hyp-003"))
    L3 = _legacy_lesson_with_hypothesis("c", ())
    out = tested_hypothesis_ids(lessons=[L1, L2, L3])
    assert out == {"hyp-001", "hyp-002", "hyp-003"}


def test_tested_hypothesis_ids_excludes_legacy_by_default():
    from engine.research_store.red_lessons.retrieval import (
        tested_hypothesis_ids,
    )
    L_legacy = _legacy_lesson_with_hypothesis(
        "legacy", ("hyp-leg",),
        gm=GroundingMethod.pretrain_grounded,
    )
    L_legacy = REDLesson(**{**L_legacy.__dict__,
                            "created_ts": "2026-06-03T12:00:00Z",
                            "updated_ts": "2026-06-03T12:00:00Z"})
    L_real   = _legacy_lesson_with_hypothesis("real", ("hyp-real",))
    out = tested_hypothesis_ids(lessons=[L_legacy, L_real])
    assert out == {"hyp-real"}
    # opt-in includes legacy
    out2 = tested_hypothesis_ids(lessons=[L_legacy, L_real], include_legacy=True)
    assert out2 == {"hyp-leg", "hyp-real"}


def test_briefing_include_legacy_flag():
    """get_briefing default excludes legacy, opt-in finds them."""
    from engine.research_store.red_lessons.retrieval import get_briefing
    L_legacy = _legacy_lesson_with_hypothesis(
        "legacy", (),
        gm=GroundingMethod.pretrain_grounded,
    )
    L_legacy = REDLesson(**{**L_legacy.__dict__,
                            "created_ts": "2026-06-03T12:00:00Z",
                            "updated_ts": "2026-06-03T12:00:00Z"})
    cand = Candidate(
        name="x", mechanism_family=MechanismFamily.MOMENTUM,
        likely_failure_modes=(FailureMode.F8_OVERFIT_INDUCED,),
    )
    # Default: filtered out
    b = get_briefing(cand, lessons=[L_legacy], registry=[])
    assert b.total_hits() == 0
    # include_legacy=True: finds it
    b2 = get_briefing(cand, lessons=[L_legacy], registry=[], include_legacy=True)
    assert b2.total_hits() >= 1
