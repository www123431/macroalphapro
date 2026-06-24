"""tests/test_red_lessons_schema.py — P0 schema integrity tests.

Round-trip + validation + controlled-vocab sanity. These tests are the
gate that the schema is stable enough to start P1 (backfill) on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research_store.red_lessons import (
    DormantRevisit,
    FailureMode,
    ForwardVector,
    GroundingMethod,
    LESSON_SCHEMA_VERSION,
    LessonStrength,
    MechanismFamily,
    PaperRef,
    REDLesson,
    ReviewState,
    VerbatimQuote,
    FAILURE_MODE_DOCS,
    MECHANISM_FAMILY_DOCS,
)
from engine.research_store.red_lessons.store import (
    load_lessons,
    save_lesson,
    latest_per_candidate,
)


# ───────────────────── controlled vocab sanity ────────────────────────


def test_every_failure_mode_has_docs():
    """Each FailureMode enum must have a full docs entry with required keys."""
    required = {"label", "definition", "diagnostic", "forward_implication"}
    for fm in FailureMode:
        assert fm in FAILURE_MODE_DOCS, f"missing docs for {fm}"
        keys = set(FAILURE_MODE_DOCS[fm].keys())
        assert required.issubset(keys), f"{fm} missing keys: {required - keys}"


def test_every_mechanism_family_has_docs():
    """Each MechanismFamily enum must have a full docs entry."""
    required = {"definition", "anchor_paper"}
    for mf in MechanismFamily:
        assert mf in MECHANISM_FAMILY_DOCS, f"missing docs for {mf}"
        keys = set(MECHANISM_FAMILY_DOCS[mf].keys())
        assert required.issubset(keys), f"{mf} missing keys: {required - keys}"


def test_failure_mode_codes_are_F1_through_F9():
    """We intentionally have 9 modes; new ones must be added deliberately."""
    codes = sorted(fm.value for fm in FailureMode)
    expected = {f"F{i}_" for i in range(1, 10)}
    assert len(codes) == 9, f"expected 9 failure modes, got {len(codes)}: {codes}"
    for code in codes:
        prefix = code[:3]
        assert any(code.startswith(e) for e in expected), \
            f"code {code} doesn't match F1..F9 convention"


# ──────────────────────── round-trip ──────────────────────────────────


def _make_minimal_lesson() -> REDLesson:
    return REDLesson(
        lesson_id           = REDLesson.new_id(),
        candidate_name      = "test_candidate",
        version             = 1,
        parent_lesson_id    = None,
        source_event_ids    = ("evt-001",),
        verdict             = "RED",
        stat_evidence       = {"deflated_sr": 0.21, "n_months": 119, "alpha_t": -0.31},
        mechanism_family    = MechanismFamily.ATTENTION,
        mechanism_subtype   = "news_attention_shock",
        failure_modes       = (FailureMode.F3_SUBSUMED_BY_EXISTING,),
        failure_evidence    = {
            "F3_SUBSUMED_BY_EXISTING":
                "FF5+UMD spanning residual alpha-t = -0.31; corr with PEAD = 0.42"
        },
        paper_motivation    = PaperRef(
            title       = "In Search of Attention",
            year        = 2011,
            authors     = ("Da", "Engelberg", "Gao"),
            key_claim   = "Google Trends search predicts retail attention; "
                          "attention-shock stocks revert.",
            our_finding = "Failed: signal fully subsumed by PEAD-momentum factor.",
            doi         = "10.1111/j.1540-6261.2011.01679.x",
            venue       = "JF",
            section_ref = "Table 4, p.1480",
        ),
        paper_critiques     = (),
        subsumed_by         = ("D_PEAD",),
        related_lesson_ids  = (),
        forward_directions  = (
            ForwardVector(
                direction         = "Try attention proxy on non-US retail markets "
                                    "(e.g. China A-share Wikipedia views)",
                rationale         = "Subsumption by PEAD is US-specific; CN retail "
                                    "structure may have un-subsumed signal.",
                avoids_failures   = (FailureMode.F3_SUBSUMED_BY_EXISTING,),
                new_required_data = ("CN A-share Wikipedia pageview panel",),
                priority          = "low",
                blocked_by        = "no PIT Wikipedia archive for CN tickers",
            ),
        ),
        do_not_retry        = (
            "Don't retry US large-cap attention signals — exhausted (HXZ catalog).",
        ),
        dormant_revisits    = (
            DormantRevisit(
                condition_label   = "CN Wikipedia pageview PIT panel",
                condition_check   = "data/cache/cn_wiki_views.parquet exists",
                reactivation_note = "Would let us test attention in retail-dominated CN.",
            ),
        ),
        # 2026-06-04 paper-driven chain fields — fixture is legacy-style
        # (pretrain_grounded with empty hypothesis/quote sets). Created_ts
        # is 2026-06-03, BEFORE the freeze TS, so allowed.
        tested_hypothesis_ids = (),
        verbatim_quotes       = (),
        grounding_method      = GroundingMethod.pretrain_grounded,
        review_state        = ReviewState.claude_drafted,
        strength            = LessonStrength.medium,
        created_ts          = "2026-06-03T12:00:00Z",
        updated_ts          = "2026-06-03T12:00:00Z",
        created_by          = "claude-opus-4-7",
        summary             = "news_attention_shock RED: F3 subsumed by D_PEAD; "
                              "Da-Engelberg-Gao 2011 motivation invalidated by spanning test.",
        tags                = ("attention", "us_equity", "exhausted"),
    )


def test_round_trip_dict():
    """to_dict → from_dict → equality."""
    L = _make_minimal_lesson()
    d = L.to_dict()
    L2 = REDLesson.from_dict(d)
    assert L == L2, "round-trip dict serialization failed"


def test_round_trip_json():
    """JSON string round trip."""
    L = _make_minimal_lesson()
    s = json.dumps(L.to_dict(), ensure_ascii=False)
    L2 = REDLesson.from_dict(json.loads(s))
    assert L == L2


def test_paper_ref_round_trip_without_doi():
    """PaperRef with only arxiv/SSRN id (no DOI) round-trips."""
    p = PaperRef(
        title       = "Some Working Paper",
        year        = 2024,
        authors     = ("Smith",),
        key_claim   = "X predicts Y",
        our_finding = "untestable in our window",
        ssrn_id     = "5012345",
        venue       = "SSRN",
    )
    p2 = PaperRef.from_dict(p.to_dict())
    assert p == p2


def test_dormant_revisit_round_trip():
    r = DormantRevisit(
        condition_label   = "extended_data",
        condition_check   = "n_months > 200",
        reactivation_note = "would allow F7 to be retired",
    )
    r2 = DormantRevisit.from_dict(r.to_dict())
    assert r == r2


# ──────────────────────── validation rules ────────────────────────────


def test_validate_rejects_empty_candidate_name():
    L = _make_minimal_lesson()
    L2 = REDLesson(**{**L.__dict__, "candidate_name": "   "})
    errs = L2.validate()
    assert any("candidate_name" in e for e in errs)


def test_validate_rejects_oversized_summary():
    L = _make_minimal_lesson()
    long = "x" * 401
    L2 = REDLesson(**{**L.__dict__, "summary": long})
    assert any("summary" in e for e in L2.validate())


def test_validate_rejects_zero_failure_modes():
    L = _make_minimal_lesson()
    L2 = REDLesson(**{**L.__dict__, "failure_modes": ()})
    assert any("failure_modes" in e for e in L2.validate())


def test_validate_rejects_four_failure_modes():
    L = _make_minimal_lesson()
    L2 = REDLesson(**{**L.__dict__,
                      "failure_modes": (FailureMode.F1_PUBLICATION_DECAY,
                                        FailureMode.F2_MECHANISM_MISMATCH,
                                        FailureMode.F3_SUBSUMED_BY_EXISTING,
                                        FailureMode.F4_IMPLEMENTATION_COST)})
    assert any("failure_modes" in e for e in L2.validate())


def test_validate_rejects_invalid_verdict():
    L = _make_minimal_lesson()
    L2 = REDLesson(**{**L.__dict__, "verdict": "GREEN"})
    assert any("verdict" in e for e in L2.validate())


def test_validate_requires_evidence_for_each_mode():
    L = _make_minimal_lesson()
    L2 = REDLesson(**{**L.__dict__,
                      "failure_modes": (FailureMode.F1_PUBLICATION_DECAY,
                                        FailureMode.F3_SUBSUMED_BY_EXISTING),
                      "failure_evidence": {"F3_SUBSUMED_BY_EXISTING": "..."}})
    errs = L2.validate()
    assert any("F1_PUBLICATION_DECAY" in e for e in errs)


def test_validate_requires_subsumed_by_for_F3():
    L = _make_minimal_lesson()
    L2 = REDLesson(**{**L.__dict__,
                      "subsumed_by": ()})
    errs = L2.validate()
    assert any("F3_SUBSUMED_BY_EXISTING requires" in e for e in errs)


def test_validate_locked_state_requires_forward_directions():
    L = _make_minimal_lesson()
    L2 = REDLesson(**{**L.__dict__,
                      "review_state": ReviewState.locked,
                      "forward_directions": ()})
    errs = L2.validate()
    assert any("forward_direction" in e for e in errs)


def test_validate_passes_for_minimal_well_formed_lesson():
    L = _make_minimal_lesson()
    assert L.validate() == []


# ──────────────────────── store I/O ───────────────────────────────────


def test_store_round_trip(tmp_path: Path):
    p = tmp_path / "lessons.jsonl"
    L1 = _make_minimal_lesson()
    save_lesson(L1, path=p, validate_strict=True)
    loaded = load_lessons(p)
    assert len(loaded) == 1
    assert loaded[0] == L1


def test_store_append_multiple(tmp_path: Path):
    p = tmp_path / "lessons.jsonl"
    L1 = _make_minimal_lesson()
    L2 = REDLesson(**{**L1.__dict__,
                      "lesson_id": REDLesson.new_id(),
                      "candidate_name": "test_candidate_2"})
    save_lesson(L1, path=p)
    save_lesson(L2, path=p)
    loaded = load_lessons(p)
    assert len(loaded) == 2
    assert {L.candidate_name for L in loaded} == {"test_candidate", "test_candidate_2"}


def test_store_validation_strict_blocks_save(tmp_path: Path):
    p = tmp_path / "lessons.jsonl"
    L = _make_minimal_lesson()
    bad = REDLesson(**{**L.__dict__, "candidate_name": ""})
    with pytest.raises(ValueError):
        save_lesson(bad, path=p, validate_strict=True)
    assert not p.exists() or p.stat().st_size == 0


def test_store_validation_non_strict_warns_and_saves(tmp_path: Path):
    p = tmp_path / "lessons.jsonl"
    L = _make_minimal_lesson()
    bad = REDLesson(**{**L.__dict__, "candidate_name": ""})
    save_lesson(bad, path=p, validate_strict=False)
    assert p.exists() and p.stat().st_size > 0


def test_latest_per_candidate_picks_highest_version():
    L1 = _make_minimal_lesson()
    L2 = REDLesson(**{**L1.__dict__,
                      "lesson_id": REDLesson.new_id(),
                      "version": 2,
                      "parent_lesson_id": L1.lesson_id})
    latest = latest_per_candidate([L1, L2])
    assert latest[L1.candidate_name] == L2


def test_load_lessons_missing_file_returns_empty(tmp_path: Path):
    p = tmp_path / "does_not_exist.jsonl"
    assert load_lessons(p) == []


def test_load_lessons_skips_corrupt_lines(tmp_path: Path, caplog):
    p = tmp_path / "mixed.jsonl"
    L = _make_minimal_lesson()
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(L.to_dict()) + "\n")
        f.write("{not valid json\n")
        f.write("\n")  # blank line OK
        f.write(json.dumps(L.to_dict()) + "\n")
    loaded = load_lessons(p)
    assert len(loaded) == 2  # corrupt line skipped, blank line skipped
