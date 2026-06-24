"""tests/test_synthesis_anchor_wiring.py — Stage C Phase E.

Tests:
  - SynthesisInput backward compat (anchor_library default empty)
  - _format_input includes ANCHOR LIBRARY section when populated
  - _format_input skips ANCHOR LIBRARY section when empty (no noise)
  - candidate parser extracts orthogonal_to_anchors
  - tool schema requires orthogonal_to_anchors
  - _load_anchor_library produces correct AnchorRef shape
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _mk_anchor_ref(paper_id="abc12345", tier="T2_ANCHOR",
                    first_author="Test", year=2020,
                    anchor_summary="Test anchor summary."):
    from engine.agents.papers_curator.synthesis import AnchorRef
    return AnchorRef(
        paper_id=paper_id, tier=tier, first_author=first_author,
        year=year, anchor_summary=anchor_summary,
    )


def _mk_synth_input(*, anchor_library=()):
    from engine.agents.papers_curator.synthesis import SynthesisInput
    return SynthesisInput(
        recent_summaries=(), deployed_sleeves=(), recent_events=(),
        doctrine_snippets=(), snapshot_ts="2026-06-07T00:00:00Z",
        anchor_library=anchor_library,
    )


# ────────────────────────────────────────────────────────────────────
# Schema backward compat
# ────────────────────────────────────────────────────────────────────
def test_synthesis_input_default_anchor_library_empty():
    """Pre-Phase-E gatherer code that doesn't pass anchor_library
    must still work."""
    from engine.agents.papers_curator.synthesis import SynthesisInput
    inp = SynthesisInput(
        recent_summaries=(), deployed_sleeves=(), recent_events=(),
        doctrine_snippets=(), snapshot_ts="x",
    )
    assert inp.anchor_library == ()


def test_synthesized_candidate_default_orthogonal_to_anchors_empty():
    from engine.agents.papers_curator.synthesis import (
        SynthesizedCandidate,
    )
    c = SynthesizedCandidate(
        claim="x", mechanism_family="CARRY", mechanism_subtype="",
        predicted_direction="positive", predicted_magnitude="moderate",
        required_data=(), test_methodology="",
        synthesizes_event_ids=(), synthesizes_paper_ids=(),
        addresses_decay_in=None, cochrane_frame="risk",
        novelty_vs_known="x", estimated_n_trials_in_family=10,
        graveyard_conflicts=(), doctrine_conflicts=(),
        expected_outcome_prior="x",
        generation_ts="t", model="m",
    )
    assert c.orthogonal_to_anchors == ()


# ────────────────────────────────────────────────────────────────────
# _format_input — anchor library injection
# ────────────────────────────────────────────────────────────────────
def test_format_input_includes_anchor_section_when_populated():
    from engine.agents.papers_curator.synthesis import _format_input
    inp = _mk_synth_input(anchor_library=(
        _mk_anchor_ref(paper_id="kmpv2018", tier="T2_ANCHOR",
                        first_author="Koijen", year=2018,
                        anchor_summary="Carry factor anchor "
                          "(KMPV 2018): cross-asset carry premium."),
        _mk_anchor_ref(paper_id="hlz2016", tier="T1_DOCTRINE",
                        first_author="Harvey", year=2016,
                        anchor_summary="t|>=3 hurdle for new factors."),
    ))
    out = _format_input(inp)
    assert "ANCHOR LIBRARY" in out
    assert "T1 DOCTRINE" in out
    assert "T2 ANCHOR" in out
    assert "kmpv2018" in out
    assert "hlz2016" in out
    assert "Koijen 2018" in out
    assert "Harvey 2016" in out
    assert "MUST CITE" in out.upper()
    # Each anchor summary present
    assert "Carry factor anchor" in out


def test_format_input_skips_anchor_section_when_empty():
    """No anchors loaded → don't pollute prompt with empty section."""
    from engine.agents.papers_curator.synthesis import _format_input
    inp = _mk_synth_input(anchor_library=())
    out = _format_input(inp)
    assert "ANCHOR LIBRARY" not in out


def test_format_input_groups_anchors_by_tier():
    """T1 listed before T2 for prompt readability."""
    from engine.agents.papers_curator.synthesis import _format_input
    inp = _mk_synth_input(anchor_library=(
        _mk_anchor_ref(paper_id="t2A", tier="T2_ANCHOR"),
        _mk_anchor_ref(paper_id="t1A", tier="T1_DOCTRINE"),
        _mk_anchor_ref(paper_id="t2B", tier="T2_ANCHOR"),
    ))
    out = _format_input(inp)
    pos_t1 = out.find("T1 DOCTRINE")
    pos_t2 = out.find("T2 ANCHOR (mechanism")
    assert 0 < pos_t1 < pos_t2


# ────────────────────────────────────────────────────────────────────
# Candidate parser — orthogonal_to_anchors extraction
# ────────────────────────────────────────────────────────────────────
def _mock_synthesis_payload(*, orthogonal_to_anchors):
    """Build a complete valid candidate dict with caller-supplied
    orthogonal_to_anchors."""
    return {
        "candidates": [{
            "claim": "test claim",
            "mechanism_family": "CARRY",
            "mechanism_subtype": "test",
            "predicted_direction": "positive",
            "predicted_magnitude": "moderate",
            "required_data": ["x"],
            "test_methodology": "engine.x",
            "synthesizes_paper_ids": ["p1"],
            "synthesizes_event_ids": ["e1"],
            "cochrane_frame": "risk",
            "novelty_vs_known": "x",
            "estimated_n_trials_in_family": 5,
            "graveyard_conflicts": [],
            "doctrine_conflicts": [],
            "expected_outcome_prior": "likely_RED",
            "orthogonal_to_anchors": orthogonal_to_anchors,
        }],
    }


def test_parser_extracts_orthogonal_to_anchors(monkeypatch):
    from engine.agents.papers_curator import synthesis as syn
    payload = _mock_synthesis_payload(orthogonal_to_anchors=[
        {"anchor_paper_id": "kmpv2018",
          "why_orthogonal": "uses commodity-only universe, not cross-asset"},
        {"anchor_paper_id": "jt1993",
          "why_orthogonal": "monthly horizon vs JT's 12-1 quarterly"},
    ])
    monkeypatch.setattr(syn, "llm_call", lambda **kw: SimpleNamespace(
        text="", tool_calls=(SimpleNamespace(
            name="emit_synthesis", input=payload),),
        model="m"))
    out = syn.run_synthesis(_mk_synth_input())
    assert len(out) == 1
    c = out[0]
    assert len(c.orthogonal_to_anchors) == 2
    assert c.orthogonal_to_anchors[0]["anchor_paper_id"] == "kmpv2018"
    assert "commodity-only" in c.orthogonal_to_anchors[0]["why_orthogonal"]


def test_parser_handles_missing_orthogonal_field(monkeypatch):
    """LLM accidentally omits field → tuple stays empty (schema's
    minItems=1 should have caught it server-side, but defense in depth)."""
    from engine.agents.papers_curator import synthesis as syn
    payload = _mock_synthesis_payload(orthogonal_to_anchors=[])
    monkeypatch.setattr(syn, "llm_call", lambda **kw: SimpleNamespace(
        text="", tool_calls=(SimpleNamespace(
            name="emit_synthesis", input=payload),),
        model="m"))
    out = syn.run_synthesis(_mk_synth_input())
    assert out[0].orthogonal_to_anchors == ()


# ────────────────────────────────────────────────────────────────────
# Tool schema includes new field with constraints
# ────────────────────────────────────────────────────────────────────
def test_tool_schema_requires_orthogonal_to_anchors():
    from engine.agents.papers_curator.synthesis import _TOOL_DEFINITION
    item_schema = _TOOL_DEFINITION["input_schema"]["properties"][
        "candidates"]["items"]
    assert "orthogonal_to_anchors" in item_schema["required"]
    assert "orthogonal_to_anchors" in item_schema["properties"]
    field = item_schema["properties"]["orthogonal_to_anchors"]
    assert field["minItems"] == 1   # forces ≥1 anchor referenced


def test_tool_schema_orthogonal_items_have_required_fields():
    from engine.agents.papers_curator.synthesis import _TOOL_DEFINITION
    item_schema = (_TOOL_DEFINITION["input_schema"]["properties"]
                    ["candidates"]["items"]["properties"]
                    ["orthogonal_to_anchors"]["items"])
    assert "anchor_paper_id" in item_schema["required"]
    assert "why_orthogonal" in item_schema["required"]


# ────────────────────────────────────────────────────────────────────
# _load_anchor_library — registry → AnchorRef
# ────────────────────────────────────────────────────────────────────
def test_load_anchor_library_filters_to_t1_t2(monkeypatch):
    """Only T1_DOCTRINE + T2_ANCHOR papers become anchors; T3 / UNCL
    skipped."""
    from engine.agents.papers_curator import synthesis_context as sc
    from engine.research_store.papers.schema import PaperTier

    def _mk_paper(paper_id, tier, summary=""):
        p = SimpleNamespace(
            paper_id=paper_id, version=1, tier=tier,
            tier_anchor_summary=summary, doi=f"10.x/{paper_id}",
            authors=("X.",), year=2020,
        )
        return p

    fake_reg = [
        _mk_paper("t1a", PaperTier.T1_DOCTRINE),
        _mk_paper("t2a", PaperTier.T2_ANCHOR, summary="Anchor for X."),
        _mk_paper("t2b_empty_summary", PaperTier.T2_ANCHOR, summary=""),
        _mk_paper("t3", PaperTier.T3_RECENT, summary="Recent paper."),
        _mk_paper("unc", PaperTier.UNCLASSIFIED),
    ]
    monkeypatch.setattr(
        "engine.research_store.papers.store.load_registry",
        lambda: fake_reg,
    )
    out = sc._load_anchor_library()
    pids = {a.paper_id for a in out}
    # T1 always included (even without summary); T2 only when summary
    # is populated; T3/UNCL skipped
    assert "t1a" in pids
    assert "t2a" in pids
    assert "t2b_empty" not in pids   # T2 without summary skipped
    assert "t3" not in pids
    assert "unc" not in pids


def test_load_anchor_library_degrades_on_load_failure(monkeypatch):
    """If load_registry raises, return () (don't break synthesis)."""
    from engine.agents.papers_curator import synthesis_context as sc
    def _broken():
        raise RuntimeError("registry corrupted")
    monkeypatch.setattr(
        "engine.research_store.papers.store.load_registry", _broken)
    assert sc._load_anchor_library() == ()
