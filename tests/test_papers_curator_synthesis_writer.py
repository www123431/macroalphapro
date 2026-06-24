"""tests/test_papers_curator_synthesis_writer.py — Phase 2.0 step 4b.

Writer tests. Uses tmp_path so no real hypotheses.jsonl is touched.

Verifies:
  - candidate_to_hypothesis pure adapter maps fields correctly
  - mechanism_family case + unknown fallback to OTHER
  - direction unknown fallback to ZERO
  - synthesis fields propagate (paper_ids, event_ids, decay_in)
  - written record validates clean under relaxed LLM_SYNTHESIS rules
  - empty input → no file created, returns []
  - batch writes append rows preserving order
  - strict mode raises on malformed candidate; non-strict logs + skips
  - written tags include 'synthesis' + extra_tags
  - timestamps deterministic when now_iso provided
"""
from __future__ import annotations

import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────
def _candidate(*, claim: str = "EM sovereign carry refresh delivers Sharpe 0.6 OOS",
                 mechanism_family: str = "carry",
                 mechanism_subtype: str = "qmj_em_sovereign",
                 predicted_direction: str = "positive",
                 predicted_magnitude: str = "Sharpe 0.5+ OOS",
                 required_data: tuple[str, ...] = ("EM sovereign bond returns",),
                 test_methodology: str = "long-short decile sort on composite quality",
                 synthesizes_paper_ids: tuple[str, ...] = ("arxiv/2606.11111",),
                 synthesizes_event_ids: tuple[str, ...] = ("ev_abc",),
                 addresses_decay_in=None):
    from engine.agents.papers_curator.synthesis import SynthesizedCandidate
    return SynthesizedCandidate(
        claim                       = claim,
        mechanism_family            = mechanism_family,
        mechanism_subtype           = mechanism_subtype,
        predicted_direction         = predicted_direction,
        predicted_magnitude         = predicted_magnitude,
        required_data               = required_data,
        test_methodology            = test_methodology,
        synthesizes_paper_ids       = synthesizes_paper_ids,
        synthesizes_event_ids       = synthesizes_event_ids,
        addresses_decay_in          = addresses_decay_in,
        cochrane_frame              = "risk",
        novelty_vs_known            = "extension_to_em_sov",
        estimated_n_trials_in_family= 5,
        graveyard_conflicts         = (),
        doctrine_conflicts          = (),
        expected_outcome_prior      = "marginal_per_HXZ_with_some_replication",
        generation_ts               = "2026-06-06T13:00:00Z",
        model                       = "claude-sonnet-4-6",
    )


# ─────────────────────────────────────────────────────────────────────
# Pure adapter — candidate_to_hypothesis
# ─────────────────────────────────────────────────────────────────────
def test_adapter_maps_core_fields():
    from engine.agents.papers_curator.synthesis_writer import candidate_to_hypothesis
    from engine.research_store.hypothesis.schema import (
        ExtractionMethod, HypothesisDirection, HypothesisReviewState,
    )
    h = candidate_to_hypothesis(_candidate(), now_iso="2026-06-06T13:00:00Z")
    assert h.claim.startswith("EM sovereign carry")
    assert h.predicted_direction == HypothesisDirection.POSITIVE
    assert h.predicted_magnitude == "Sharpe 0.5+ OOS"
    assert h.required_data == ("EM sovereign bond returns",)
    assert h.test_methodology.startswith("long-short")
    assert h.extraction_method == ExtractionMethod.LLM_SYNTHESIS
    assert h.review_state == HypothesisReviewState.PROPOSED
    assert h.created_ts == "2026-06-06T13:00:00Z"
    assert h.updated_ts == "2026-06-06T13:00:00Z"
    assert h.source_paper_id == ""
    assert h.source_chunk_ids == ()
    assert h.verbatim_quotes == ()
    assert h.version == 1
    assert h.parent_hypothesis_id is None


def test_adapter_propagates_synthesis_provenance():
    from engine.agents.papers_curator.synthesis_writer import candidate_to_hypothesis
    h = candidate_to_hypothesis(_candidate(
        synthesizes_paper_ids = ("arxiv/p1", "arxiv/p2"),
        synthesizes_event_ids = ("ev_a", "ev_b", "ev_c"),
        addresses_decay_in    = "carry_g10",
    ))
    assert h.synthesizes_paper_ids == ("arxiv/p1", "arxiv/p2")
    assert h.synthesizes_event_ids == ("ev_a", "ev_b", "ev_c")
    assert h.addresses_decay_in == "carry_g10"


def test_adapter_mechanism_family_case_coerced():
    from engine.agents.papers_curator.synthesis_writer import candidate_to_hypothesis
    from engine.research_store.red_lessons.mechanism_families import MechanismFamily
    h = candidate_to_hypothesis(_candidate(mechanism_family="carry"))
    assert h.mechanism_family == MechanismFamily.CARRY


def test_adapter_mechanism_family_unknown_falls_back_to_other():
    """A weird LLM-emitted family should NOT raise — it should land
    as OTHER so the principal can re-classify on review."""
    from engine.agents.papers_curator.synthesis_writer import candidate_to_hypothesis
    from engine.research_store.red_lessons.mechanism_families import MechanismFamily
    h = candidate_to_hypothesis(_candidate(
        mechanism_family="some_novel_family_that_does_not_exist"))
    assert h.mechanism_family == MechanismFamily.OTHER


def test_adapter_direction_unknown_falls_back_to_zero():
    from engine.agents.papers_curator.synthesis_writer import candidate_to_hypothesis
    from engine.research_store.hypothesis.schema import HypothesisDirection
    h = candidate_to_hypothesis(_candidate(predicted_direction="bullish"))
    assert h.predicted_direction == HypothesisDirection.ZERO


def test_adapter_tags_include_synthesis_plus_extra():
    from engine.agents.papers_curator.synthesis_writer import candidate_to_hypothesis
    h = candidate_to_hypothesis(_candidate(),
                                  extra_tags=("session:cos-2026-06-06",))
    assert "synthesis" in h.tags
    assert "session:cos-2026-06-06" in h.tags


def test_written_record_validates_clean():
    """The Hypothesis produced by the adapter must satisfy the
    relaxed LLM_SYNTHESIS validation rules (step 4a). If it didn't,
    save_hypothesis would raise on every write."""
    from engine.agents.papers_curator.synthesis_writer import candidate_to_hypothesis
    h = candidate_to_hypothesis(_candidate())
    assert h.validate() == []


# ─────────────────────────────────────────────────────────────────────
# Batch writer — write_synthesized_candidates
# ─────────────────────────────────────────────────────────────────────
def test_write_empty_input_returns_empty_no_file_created(tmp_path):
    from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates
    p = tmp_path / "hypotheses.jsonl"
    out = write_synthesized_candidates([], path=p)
    assert out == []
    assert not p.exists()


def test_write_appends_rows_preserving_order(tmp_path):
    from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates
    p = tmp_path / "hypotheses.jsonl"
    cands = [
        _candidate(claim="claim one"),
        _candidate(claim="claim two"),
        _candidate(claim="claim three"),
    ]
    written = write_synthesized_candidates(cands, path=p)
    assert len(written) == 3
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    rows = [json.loads(ln) for ln in lines]
    assert rows[0]["claim"] == "claim one"
    assert rows[1]["claim"] == "claim two"
    assert rows[2]["claim"] == "claim three"


def test_write_appends_to_existing_file(tmp_path):
    """Second batch should append, not truncate."""
    from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates
    p = tmp_path / "hypotheses.jsonl"
    write_synthesized_candidates([_candidate(claim="first batch")], path=p)
    write_synthesized_candidates([_candidate(claim="second batch")], path=p)
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").strip().split("\n")]
    assert len(rows) == 2
    assert rows[0]["claim"] == "first batch"
    assert rows[1]["claim"] == "second batch"


def test_write_rows_are_synthesis_shape_on_disk(tmp_path):
    """Disk rows must carry the v3 synthesis fields, the LLM_SYNTHESIS
    extraction_method, and empty paper-rooted fields."""
    from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates
    p = tmp_path / "hypotheses.jsonl"
    write_synthesized_candidates([_candidate(
        synthesizes_paper_ids=("arxiv/p1", "arxiv/p2"),
        synthesizes_event_ids=("ev1",),
        addresses_decay_in="carry_g10",
    )], path=p)
    row = json.loads(p.read_text(encoding="utf-8").strip())
    assert row["extraction_method"] == "llm_synthesis"
    assert row["source_paper_id"] == ""
    assert row["source_chunk_ids"] == []
    assert row["verbatim_quotes"] == []
    assert row["synthesizes_paper_ids"] == ["arxiv/p1", "arxiv/p2"]
    assert row["synthesizes_event_ids"] == ["ev1"]
    assert row["addresses_decay_in"] == "carry_g10"
    assert row["schema_version"] == 4   # bumped Phase 2.2c


def test_write_roundtrip_via_load_hypotheses(tmp_path):
    """Written rows must be readable back into Hypothesis objects."""
    from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates
    from engine.research_store.hypothesis.store import load_hypotheses
    from engine.research_store.hypothesis.schema import ExtractionMethod
    p = tmp_path / "hypotheses.jsonl"
    write_synthesized_candidates([_candidate()], path=p)
    loaded = load_hypotheses(path=p)
    assert len(loaded) == 1
    assert loaded[0].extraction_method == ExtractionMethod.LLM_SYNTHESIS
    assert loaded[0].synthesizes_paper_ids == ("arxiv/2606.11111",)


def test_write_strict_mode_raises_on_malformed(tmp_path):
    """validate_strict=True (default) must surface validation failures
    so the caller knows the batch is incomplete."""
    from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates
    import pytest
    p = tmp_path / "hypotheses.jsonl"
    bad = _candidate(predicted_magnitude="",   # required field empty
                      required_data=(),
                      test_methodology="")
    with pytest.raises(ValueError):
        write_synthesized_candidates([bad], path=p)


def test_write_strict_mode_writes_valid_before_failure(tmp_path):
    """First candidate valid, second malformed: writer raises but the
    first is already on disk (each row is independent fsync)."""
    from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates
    import pytest
    p = tmp_path / "hypotheses.jsonl"
    good = _candidate(claim="good one")
    bad  = _candidate(claim="bad one", predicted_magnitude="",
                       required_data=(), test_methodology="")
    with pytest.raises(ValueError):
        write_synthesized_candidates([good, bad], path=p)
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").strip().split("\n")]
    assert len(rows) == 1
    assert rows[0]["claim"] == "good one"


def test_write_non_strict_logs_and_skips_malformed(tmp_path):
    """validate_strict=False keeps the batch going past one bad row;
    the malformed row is logged + persisted (validate_strict=False on
    save_hypothesis = warn-only). Valid siblings are also persisted."""
    from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates
    p = tmp_path / "hypotheses.jsonl"
    good = _candidate(claim="good one")
    bad  = _candidate(claim="bad one", predicted_magnitude="",
                      required_data=(), test_methodology="")
    written = write_synthesized_candidates(
        [good, bad, _candidate(claim="another good")],
        path=p, validate_strict=False,
    )
    assert len(written) == 3
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").strip().split("\n")]
    assert {r["claim"] for r in rows} == {"good one", "bad one", "another good"}


def test_write_uses_default_actor_when_not_provided(tmp_path):
    from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates
    p = tmp_path / "hypotheses.jsonl"
    write_synthesized_candidates([_candidate()], path=p)
    row = json.loads(p.read_text(encoding="utf-8").strip())
    assert row["created_by"] == "papers_curator_synthesis"


def test_write_custom_actor_and_tags(tmp_path):
    from engine.agents.papers_curator.synthesis_writer import write_synthesized_candidates
    p = tmp_path / "hypotheses.jsonl"
    write_synthesized_candidates(
        [_candidate()], path=p,
        created_by="cos_weekly_session",
        extra_tags=("session:cos-2026-06-06", "weekly"),
    )
    row = json.loads(p.read_text(encoding="utf-8").strip())
    assert row["created_by"] == "cos_weekly_session"
    assert "synthesis" in row["tags"]
    assert "session:cos-2026-06-06" in row["tags"]
    assert "weekly" in row["tags"]
