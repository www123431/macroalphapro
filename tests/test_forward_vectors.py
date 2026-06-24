"""tests/test_forward_vectors.py — T5 forward vector schema + generator tests."""
from __future__ import annotations

import json

import pytest

from engine.research_store.forward_vectors import (
    ForwardVectorV2, Priority,
)
from engine.research_store.forward_vectors.generator import _compute_priority
from engine.research_store.forward_vectors.schema import ForwardVectorStatus
from engine.research_store.red_lessons.mechanism_families import MechanismFamily


# ─────────────────────── ForwardVectorV2 round-trip ──────────────────


def _minimal_fv() -> ForwardVectorV2:
    return ForwardVectorV2(
        forward_vector_id    = ForwardVectorV2.new_id(),
        version              = 1,
        parent_id            = None,
        source_paper_id      = "paper-uuid-001",
        paper_title          = "Test Paper",
        source_hypothesis_id = "hyp-uuid-001",
        claim                = "Cross-asset carry generates positive Sharpe across futures markets.",
        mechanism_family     = MechanismFamily.CARRY,
        mechanism_subtype    = "cross_asset_carry",
        predicted_direction  = "positive",
        predicted_magnitude  = "Sharpe > 0.5 across G10 FX + 19 commodities",
        required_data        = ("G10 FX forwards 1990-2020", "commodity futures roll yields"),
        test_methodology     = "Equal-weight quintile L/S on carry signal; monthly rebalance",
        priority             = Priority.HIGH,
        priority_signals     = {"paper_shelves": ["doctrine_method", "green_motivation"]},
        status               = ForwardVectorStatus.PROPOSED,
        created_ts           = "2026-06-04T16:00:00Z",
        created_by           = "test",
        tags                 = ("test",),
    )


def test_forward_vector_round_trip_dict():
    fv = _minimal_fv()
    fv2 = ForwardVectorV2.from_dict(fv.to_dict())
    assert fv == fv2


def test_forward_vector_round_trip_json():
    fv = _minimal_fv()
    s = json.dumps(fv.to_dict(), ensure_ascii=False)
    fv2 = ForwardVectorV2.from_dict(json.loads(s))
    assert fv == fv2


def test_forward_vector_validate_allows_empty_paper_id():
    """Phase 2.1b: brainstorm-track FVs (from LLM_SYNTHESIS hypotheses)
    have empty source_paper_id by design. validate() must allow it;
    the track distinction lives in priority_signals + tags, not as a
    syntactic guard."""
    fv = _minimal_fv()
    syn = ForwardVectorV2(**{**fv.__dict__, "source_paper_id": ""})
    assert syn.validate() == []   # no errors


def test_forward_vector_validate_rejects_empty_hypothesis_id():
    fv = _minimal_fv()
    bad = ForwardVectorV2(**{**fv.__dict__, "source_hypothesis_id": ""})
    assert any("source_hypothesis_id" in e for e in bad.validate())


def test_forward_vector_validate_rejects_empty_claim():
    fv = _minimal_fv()
    bad = ForwardVectorV2(**{**fv.__dict__, "claim": "  "})
    assert any("claim" in e for e in bad.validate())


def test_forward_vector_validate_rejects_empty_required_data():
    fv = _minimal_fv()
    bad = ForwardVectorV2(**{**fv.__dict__, "required_data": ()})
    assert any("required_data" in e for e in bad.validate())


def test_forward_vector_minimal_valid_passes():
    assert _minimal_fv().validate() == []


# ─────────────────────── priority rules ──────────────────────────────


def test_priority_HIGH_for_doctrine_method():
    pri, sig = _compute_priority({"doctrine_method"})
    assert pri == Priority.HIGH
    assert "doctrine_method" in sig["paper_shelves"]


def test_priority_HIGH_for_green_motivation():
    pri, _ = _compute_priority({"green_motivation"})
    assert pri == Priority.HIGH


def test_priority_HIGH_for_both_doctrine_and_green():
    pri, _ = _compute_priority({"doctrine_method", "green_motivation"})
    assert pri == Priority.HIGH


def test_priority_MEDIUM_for_yellow_or_red_motivation():
    assert _compute_priority({"yellow_motivation"})[0] == Priority.MEDIUM
    assert _compute_priority({"red_motivation"})[0] == Priority.MEDIUM
    assert _compute_priority({"green_critique"})[0] == Priority.MEDIUM


def test_priority_LOW_for_other_only():
    pri, _ = _compute_priority({"other"})
    assert pri == Priority.LOW


def test_priority_LOW_for_no_shelves():
    pri, _ = _compute_priority(set())
    assert pri == Priority.LOW


# ───────────────────────────────────────────────────────────────────────
# Phase 2.1b: dual-track priority + generator routing
# ───────────────────────────────────────────────────────────────────────
from engine.research_store.forward_vectors.generator import (
    _priority_from_b_verdict,
    _synthesis_title_from,
    generate_forward_vectors,
)


def test_brainstorm_priority_HIGH_at_conf_075():
    pri, sig = _priority_from_b_verdict({"confidence": 0.78})
    assert pri == Priority.HIGH
    assert sig["b_confidence"] == 0.78


def test_brainstorm_priority_HIGH_when_addresses_decay():
    """addresses_decay_in non-empty → HIGH regardless of confidence."""
    pri, sig = _priority_from_b_verdict({
        "confidence": 0.3,
        "addresses_decay_in": "carry_g10",
    })
    assert pri == Priority.HIGH
    assert sig["addresses_decay"] is True


def test_brainstorm_priority_MEDIUM_at_conf_055():
    pri, _ = _priority_from_b_verdict({"confidence": 0.6})
    assert pri == Priority.MEDIUM


def test_brainstorm_priority_LOW_at_low_conf():
    pri, _ = _priority_from_b_verdict({"confidence": 0.3})
    assert pri == Priority.LOW


def test_synthesis_title_includes_paper_count():
    class _FakeHyp:
        synthesizes_paper_ids = ("a", "b", "c")
    assert "3 papers" in _synthesis_title_from(_FakeHyp())


def test_synthesis_title_zero_papers_event_driven():
    class _FakeHyp:
        synthesizes_paper_ids = ()
    assert "event-driven" in _synthesis_title_from(_FakeHyp())


# ───────────────────────────────────────────────────────────────────────
# Generator routing — mocked stores
# ───────────────────────────────────────────────────────────────────────
def _llm_extract_hyp(hyp_id: str, paper_id: str):
    from engine.research_store.hypothesis import Hypothesis, VerbatimQuote
    from engine.research_store.hypothesis.schema import (
        ExtractionMethod, HypothesisDirection, HypothesisReviewState,
    )
    return Hypothesis(
        hypothesis_id        = hyp_id,
        source_paper_id      = paper_id,
        version              = 1,
        parent_hypothesis_id = None,
        source_chunk_ids     = ("c1",),
        verbatim_quotes      = (
            VerbatimQuote(chunk_id="c1", quote_text="enough chars for quote ok"),
            VerbatimQuote(chunk_id="c1", quote_text="another quote sufficient yes"),
        ),
        claim                = f"paper-stated claim {hyp_id}",
        mechanism_family     = MechanismFamily.CARRY,
        mechanism_subtype    = "g10_carry",
        predicted_direction  = HypothesisDirection.POSITIVE,
        predicted_magnitude  = "Sharpe 0.5+",
        required_data        = ("G10 FX",),
        test_methodology     = "decile sort",
        extraction_method    = ExtractionMethod.LLM_EXTRACT,
        review_state         = HypothesisReviewState.HUMAN_REVIEWED,
        created_ts           = "2026-06-06T10:00:00Z",
        updated_ts           = "2026-06-06T10:00:00Z",
        created_by           = "test",
        tags                 = (),
    )


def _llm_synthesis_hyp(hyp_id: str, *, synthesizes_papers=("p1", "p2")):
    from engine.research_store.hypothesis import Hypothesis
    from engine.research_store.hypothesis.schema import (
        ExtractionMethod, HypothesisDirection, HypothesisReviewState,
    )
    return Hypothesis(
        hypothesis_id        = hyp_id,
        source_paper_id      = "",
        version              = 1,
        parent_hypothesis_id = None,
        source_chunk_ids     = (),
        verbatim_quotes      = (),
        claim                = f"synthesized claim {hyp_id}",
        mechanism_family     = MechanismFamily.VOL_RISK_PREMIUM,
        mechanism_subtype    = "spx_vrp",
        predicted_direction  = HypothesisDirection.POSITIVE,
        predicted_magnitude  = "Sharpe 0.7+",
        required_data        = ("SPX options",),
        test_methodology     = "monthly roll",
        extraction_method    = ExtractionMethod.LLM_SYNTHESIS,
        review_state         = HypothesisReviewState.PROPOSED,
        created_ts           = "2026-06-06T11:00:00Z",
        updated_ts           = "2026-06-06T11:00:00Z",
        created_by           = "test",
        tags                 = (),
        synthesizes_paper_ids = tuple(synthesizes_papers),
    )


def _human_hyp(hyp_id: str, *, paper_id: str = ""):
    from engine.research_store.hypothesis import Hypothesis, VerbatimQuote
    from engine.research_store.hypothesis.schema import (
        ExtractionMethod, HypothesisDirection, HypothesisReviewState,
    )
    quotes = ()
    if paper_id:
        quotes = (
            VerbatimQuote(chunk_id="c1", quote_text="enough chars for quote ok"),
            VerbatimQuote(chunk_id="c1", quote_text="another quote sufficient yes"),
        )
    return Hypothesis(
        hypothesis_id        = hyp_id,
        source_paper_id      = paper_id,
        version              = 1,
        parent_hypothesis_id = None,
        source_chunk_ids     = ("c1",) if paper_id else (),
        verbatim_quotes      = quotes,
        claim                = f"human-authored claim {hyp_id}",
        mechanism_family     = MechanismFamily.MOMENTUM,
        mechanism_subtype    = "12_1",
        predicted_direction  = HypothesisDirection.POSITIVE,
        predicted_magnitude  = "Sharpe 0.4+",
        required_data        = ("equity returns",),
        test_methodology     = "decile sort",
        extraction_method    = ExtractionMethod.HUMAN_AUTHORED,
        review_state         = HypothesisReviewState.PROPOSED,
        created_ts           = "2026-06-06T11:00:00Z",
        updated_ts           = "2026-06-06T11:00:00Z",
        created_by           = "user",
        tags                 = (),
    )


def _fake_paper(paper_id: str, *, shelves=("doctrine_method",), title=None):
    """Build a minimal stand-in for the paper registry entry — only the
    attributes the generator reads."""
    class _Shelf:
        def __init__(self, v): self.value = v
    class _Paper:
        def __init__(self):
            self.paper_id = paper_id
            self.title    = title or f"paper {paper_id}"
            self.shelves  = tuple(_Shelf(s) for s in shelves)
    return _Paper()


def _patch_generator_deps(monkeypatch, *, hyps, papers=None,
                            tested=None, fv_created=None, b_verdicts=None):
    """Patch every store-read the generator does so tests don't touch
    real data files. The generator does `from <pkg> import X`, so the
    patch site is the PACKAGE namespace, not the submodule."""
    import engine.research_store.hypothesis as hyp_pkg
    import engine.research_store.papers     as papers_pkg
    import engine.research_store.red_lessons.retrieval as ret_mod
    from engine.research_store.forward_vectors import generator as gen_mod

    monkeypatch.setattr(hyp_pkg, "load_hypotheses",
                          lambda path=None: list(hyps))
    monkeypatch.setattr(papers_pkg, "load_registry",
                          lambda: list(papers or []))
    monkeypatch.setattr(papers_pkg, "latest_per_doi",
                          lambda reg: {p.paper_id: p for p in reg})
    monkeypatch.setattr(ret_mod, "tested_hypothesis_ids",
                          lambda: set(tested or ()))
    monkeypatch.setattr(gen_mod, "_load_fv_created_set",
                          lambda: set(fv_created or ()))
    monkeypatch.setattr(gen_mod, "_load_b_verdicts_by_hid",
                          lambda: dict(b_verdicts or {}))


# ── Track 2 (paper_stated) regression ──
def test_generator_includes_llm_extract_when_paper_exists(monkeypatch):
    _patch_generator_deps(monkeypatch,
        hyps   = [_llm_extract_hyp("h1", "p1")],
        papers = [_fake_paper("p1", shelves=("doctrine_method",))],
    )
    out = generate_forward_vectors()
    assert len(out) == 1
    assert out[0].source_hypothesis_id == "h1"
    assert out[0].source_paper_id == "p1"
    assert out[0].priority_signals["track"] == "paper_stated"
    assert out[0].priority == Priority.HIGH


def test_generator_skips_llm_extract_with_missing_paper(monkeypatch):
    """Old behavior preserved — LLM_EXTRACT with unresolved paper is
    dropped (data integrity guard)."""
    _patch_generator_deps(monkeypatch,
        hyps   = [_llm_extract_hyp("h1", "missing_paper")],
        papers = [],
    )
    assert generate_forward_vectors() == []


# ── Track 1 (brainstorm) ──
def test_generator_skips_llm_synthesis_without_fv_created(monkeypatch):
    """LLM_SYNTHESIS without B+user approval (no fv_created event) is
    suppressed from the queue."""
    _patch_generator_deps(monkeypatch,
        hyps       = [_llm_synthesis_hyp("syn1")],
        fv_created = set(),
    )
    assert generate_forward_vectors() == []


def test_generator_includes_llm_synthesis_when_fv_created(monkeypatch):
    """fv_created event present + B verdict resolvable → row appears."""
    _patch_generator_deps(monkeypatch,
        hyps       = [_llm_synthesis_hyp("syn1")],
        fv_created = {"syn1"},
        b_verdicts = {"syn1": {"confidence": 0.78,
                                 "verdict_type": "APPROVE_FOR_PIPELINE"}},
    )
    out = generate_forward_vectors()
    assert len(out) == 1
    assert out[0].source_hypothesis_id == "syn1"
    assert out[0].source_paper_id == ""               # no single source
    assert out[0].priority_signals["track"] == "brainstorm"
    assert out[0].priority == Priority.HIGH           # conf 0.78
    assert "Synthesis" in out[0].paper_title


def test_generator_skips_llm_synthesis_with_missing_verdict(monkeypatch):
    """fv_created event present but B verdict unresolvable → defensive
    skip (shouldn't happen in normal flow but log + skip not crash)."""
    _patch_generator_deps(monkeypatch,
        hyps       = [_llm_synthesis_hyp("syn1")],
        fv_created = {"syn1"},
        b_verdicts = {},
    )
    assert generate_forward_vectors() == []


def test_generator_llm_synthesis_priority_from_decay_address(monkeypatch):
    """addresses_decay_in non-empty → HIGH even at low confidence."""
    _patch_generator_deps(monkeypatch,
        hyps       = [_llm_synthesis_hyp("syn1")],
        fv_created = {"syn1"},
        b_verdicts = {"syn1": {"confidence": 0.4,
                                 "addresses_decay_in": "carry_g10",
                                 "verdict_type": "APPROVE_FOR_PIPELINE"}},
    )
    out = generate_forward_vectors()
    assert out[0].priority == Priority.HIGH


# ── Track 3 (human_authored) ──
def test_generator_includes_human_authored_unconditionally(monkeypatch):
    """HUMAN_AUTHORED has no gate — accepted as-is."""
    _patch_generator_deps(monkeypatch,
        hyps = [_human_hyp("manual1")],
    )
    out = generate_forward_vectors()
    assert len(out) == 1
    assert out[0].priority_signals["track"] == "human_authored"
    assert out[0].priority == Priority.MEDIUM
    assert out[0].source_paper_id == ""


def test_generator_includes_human_authored_with_paper_link(monkeypatch):
    """HUMAN_AUTHORED with source_paper_id surfaces the paper title."""
    _patch_generator_deps(monkeypatch,
        hyps   = [_human_hyp("manual1", paper_id="p1")],
        papers = [_fake_paper("p1", title="Replication Target Paper")],
    )
    out = generate_forward_vectors()
    assert out[0].paper_title == "Replication Target Paper"
    assert out[0].source_paper_id == "p1"


# ── Mixed-batch / tested-filter / ordering ──
def test_generator_three_tracks_coexist_in_one_batch(monkeypatch):
    """All 3 extraction methods land in the same queue, each tagged
    with its own track."""
    _patch_generator_deps(monkeypatch,
        hyps   = [
            _llm_extract_hyp("h1", "p1"),
            _llm_synthesis_hyp("syn1"),
            _human_hyp("manual1"),
        ],
        papers = [_fake_paper("p1", shelves=("doctrine_method",))],
        fv_created = {"syn1"},
        b_verdicts = {"syn1": {"confidence": 0.6,
                                 "verdict_type": "APPROVE_FOR_PIPELINE"}},
    )
    out = generate_forward_vectors()
    assert len(out) == 3
    tracks = {fv.priority_signals["track"] for fv in out}
    assert tracks == {"paper_stated", "brainstorm", "human_authored"}


def test_generator_filters_tested(monkeypatch):
    """tested_hypothesis_ids filter applies to ALL tracks."""
    _patch_generator_deps(monkeypatch,
        hyps   = [
            _llm_extract_hyp("h1", "p1"),
            _llm_synthesis_hyp("syn1"),
        ],
        papers = [_fake_paper("p1")],
        tested = {"h1", "syn1"},
        fv_created = {"syn1"},
        b_verdicts = {"syn1": {"confidence": 0.8}},
    )
    assert generate_forward_vectors() == []


def test_generator_skips_unknown_extraction_method(monkeypatch):
    """Defensive: a future ExtractionMethod we don't yet know how to
    route should be skipped + logged, not crash."""
    class _Custom:
        class _Em: value = "future_method"
        def __init__(self, hyp_id):
            self.hypothesis_id = hyp_id
            self.source_paper_id = ""
            self.version = 1
            self.extraction_method = _Custom._Em()
            self.mechanism_family = MechanismFamily.CARRY
            class _Dir: value = "positive"
            self.predicted_direction = _Dir()
            self.mechanism_subtype = "x"
            self.claim = "x"
            self.predicted_magnitude = "x"
            self.required_data = ("x",)
            self.test_methodology = "x"
            self.synthesizes_paper_ids = ()
    _patch_generator_deps(monkeypatch,
        hyps = [_Custom("future_hyp")],
    )
    # Should not raise; should not return the unknown row
    assert generate_forward_vectors() == []


# ── extraction_method tag wiring ──
def test_generator_tags_include_track_label(monkeypatch):
    """The track label MUST also land in the FV's tags so query
    consumers can filter without parsing priority_signals."""
    _patch_generator_deps(monkeypatch,
        hyps   = [_llm_extract_hyp("h1", "p1")],
        papers = [_fake_paper("p1")],
    )
    out = generate_forward_vectors()
    assert "track:paper_stated" in out[0].tags
