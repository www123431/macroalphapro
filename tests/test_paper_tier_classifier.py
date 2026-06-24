"""tests/test_paper_tier_classifier.py — Stage C Phase A.

Tests:
  - schema backward compat (pre-Phase-A entries load as UNCLASSIFIED)
  - tier field serialization roundtrip
  - amend_entry carries tier forward + supports set_tier override
  - classifier (LLM mocked): batched output, malformed handling,
    confidence floor coerces to UNCLASSIFIED
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _mk_paper(*, paper_id="p1", title="Test Paper", year=2020,
               authors=("Author", "Coauthor"), venue="JF",
               abstract="abstract text", tier_value="UNCLASSIFIED"):
    from engine.research_store.papers.schema import (
        PaperRegistryEntry, FulltextStatus, PaperTier,
    )
    from engine.research_store.papers.shelves import Shelf
    return PaperRegistryEntry(
        paper_id=paper_id, version=1, parent_paper_id=None,
        doi="10.1/x", title=title, year=year, authors=authors,
        venue=venue, abstract=abstract,
        fulltext_status=FulltextStatus.METADATA_ONLY,
        pdf_source_kind="", pdf_source_url="", n_chunks=0,
        ingested_ts="",
        referenced_by_lessons=(), referenced_by_factors=(),
        referenced_by_sleeves=(), referenced_by_doctrines=(),
        shelves=(Shelf.DOCTRINE_METHOD,), shelf_notes={},
        created_ts="2026-01-01T00:00:00Z",
        updated_ts="2026-01-01T00:00:00Z",
        created_by="test", tags=(), note="",
        tier=PaperTier(tier_value),
    )


# ────────────────────────────────────────────────────────────────────
# Schema backward compat
# ────────────────────────────────────────────────────────────────────
def test_pre_phase_a_entry_loads_as_unclassified():
    """Existing registry rows (no `tier` field) deserialize cleanly
    with tier=UNCLASSIFIED."""
    from engine.research_store.papers.schema import (
        PaperRegistryEntry, PaperTier,
    )
    legacy_dict = {
        "paper_id": "p1", "version": 1, "parent_paper_id": None,
        "doi": "10.1/x", "title": "Legacy", "year": 2015,
        "authors": ["A"], "venue": "JF", "abstract": "x",
        "fulltext_status": "ingested",
        "pdf_source_kind": "", "pdf_source_url": "",
        "n_chunks": 10, "ingested_ts": "",
        "referenced_by_lessons": [], "referenced_by_factors": [],
        "referenced_by_sleeves": [], "referenced_by_doctrines": [],
        "shelves": ["doctrine_method"], "shelf_notes": {},
        "created_ts": "2026-01-01T00:00:00Z",
        "updated_ts": "2026-01-01T00:00:00Z",
        "created_by": "test", "tags": [], "note": "",
        # no tier / tier_classified_ts / tier_rationale fields
    }
    p = PaperRegistryEntry.from_dict(legacy_dict)
    assert p.tier == PaperTier.UNCLASSIFIED
    assert p.tier_classified_ts == ""
    assert p.tier_rationale == ""


def test_tier_serialization_roundtrip():
    """Setting tier + serializing + deserializing preserves value."""
    from engine.research_store.papers.schema import (
        PaperRegistryEntry, PaperTier,
    )
    p = _mk_paper(tier_value="T1_DOCTRINE")
    # Need to also set rationale + ts for roundtrip
    import dataclasses
    p2 = dataclasses.replace(
        p, tier=PaperTier.T1_DOCTRINE,
        tier_classified_ts="2026-06-07T00:00:00Z",
        tier_rationale="defines DSR threshold",
    )
    d = p2.to_dict()
    assert d["tier"] == "T1_DOCTRINE"
    assert d["tier_classified_ts"] == "2026-06-07T00:00:00Z"
    p3 = PaperRegistryEntry.from_dict(d)
    assert p3.tier == PaperTier.T1_DOCTRINE
    assert p3.tier_rationale == "defines DSR threshold"


# ────────────────────────────────────────────────────────────────────
# amend_entry tier behavior
# ────────────────────────────────────────────────────────────────────
def test_amend_carries_tier_forward_by_default():
    """Calling amend_entry without set_tier → new version inherits
    tier from prior."""
    from engine.research_store.papers.amend import amend_entry
    from engine.research_store.papers.schema import PaperTier
    import dataclasses
    prior = _mk_paper()
    prior_with_tier = dataclasses.replace(
        prior, tier=PaperTier.T1_DOCTRINE,
        tier_classified_ts="2026-06-07T00:00:00Z",
        tier_rationale="DSR",
    )
    new = amend_entry(prior=prior_with_tier, add_tags=("x",))
    assert new.tier == PaperTier.T1_DOCTRINE
    assert new.tier_classified_ts == "2026-06-07T00:00:00Z"
    assert new.tier_rationale == "DSR"


def test_amend_set_tier_overrides():
    from engine.research_store.papers.amend import amend_entry
    from engine.research_store.papers.schema import PaperTier
    prior = _mk_paper()
    new = amend_entry(
        prior              = prior,
        set_tier           = PaperTier.T2_ANCHOR,
        set_tier_rationale = "canonical mechanism",
        set_tier_classified_ts = "2026-06-07T01:00:00Z",
    )
    assert new.tier == PaperTier.T2_ANCHOR
    assert new.tier_rationale == "canonical mechanism"
    assert new.tier_classified_ts == "2026-06-07T01:00:00Z"


# ────────────────────────────────────────────────────────────────────
# Classifier (LLM mocked)
# ────────────────────────────────────────────────────────────────────
def _mock_llm_result(*, proposals):
    return SimpleNamespace(
        text="",
        tool_calls=(SimpleNamespace(
            name="emit_tier_classifications",
            input={"proposals": proposals},
        ),),
        model="claude-sonnet-4-6",
    )


def test_classifier_happy_path(monkeypatch):
    from engine.research_store.papers import tier_classifier as tc
    from engine.research_store.papers.schema import PaperTier
    monkeypatch.setattr(tc, "llm_call", lambda **kw: _mock_llm_result(
        proposals=[
            {"paper_id": "p1", "tier": "T1_DOCTRINE",
              "rationale": "DSR defines our gate", "confidence": 0.95},
            {"paper_id": "p2", "tier": "T2_ANCHOR",
              "rationale": "canonical momentum paper", "confidence": 0.88},
            {"paper_id": "p3", "tier": "T3_RECENT",
              "rationale": "narrow application", "confidence": 0.75},
        ],
    ))
    papers = [_mk_paper(paper_id=f"p{i}") for i in (1, 2, 3)]
    out = tc.classify_papers_batch(papers)
    assert [p.tier for p in out] == [
        PaperTier.T1_DOCTRINE, PaperTier.T2_ANCHOR, PaperTier.T3_RECENT,
    ]


def test_classifier_confidence_floor_coerces_to_unclassified(monkeypatch):
    """Proposal with confidence < 0.7 gets coerced to UNCLASSIFIED
    with the original tier noted in rationale."""
    from engine.research_store.papers import tier_classifier as tc
    from engine.research_store.papers.schema import PaperTier
    monkeypatch.setattr(tc, "llm_call", lambda **kw: _mock_llm_result(
        proposals=[
            {"paper_id": "p1", "tier": "T1_DOCTRINE",
              "rationale": "looks like methodology", "confidence": 0.5},
        ],
    ))
    out = tc.classify_papers_batch([_mk_paper(paper_id="p1")])
    assert out[0].tier == PaperTier.UNCLASSIFIED
    assert "T1_DOCTRINE" in out[0].rationale
    assert "below 0.7" in out[0].rationale


def test_classifier_missing_paper_id_in_response(monkeypatch):
    """Model returns proposals for SOME papers but not others →
    missing ones default to UNCLASSIFIED."""
    from engine.research_store.papers import tier_classifier as tc
    from engine.research_store.papers.schema import PaperTier
    monkeypatch.setattr(tc, "llm_call", lambda **kw: _mock_llm_result(
        proposals=[
            {"paper_id": "p1", "tier": "T2_ANCHOR",
              "rationale": "x", "confidence": 0.85},
            # p2 missing
        ],
    ))
    papers = [_mk_paper(paper_id="p1"), _mk_paper(paper_id="p2")]
    out = tc.classify_papers_batch(papers)
    assert len(out) == 2
    assert out[0].tier == PaperTier.T2_ANCHOR
    assert out[1].tier == PaperTier.UNCLASSIFIED
    assert "did not return" in out[1].rationale


def test_classifier_invalid_tier_falls_back(monkeypatch):
    """Model emits a string not in the PaperTier enum → UNCLASSIFIED."""
    from engine.research_store.papers import tier_classifier as tc
    from engine.research_store.papers.schema import PaperTier
    monkeypatch.setattr(tc, "llm_call", lambda **kw: _mock_llm_result(
        proposals=[
            {"paper_id": "p1", "tier": "INVENTED_TIER",
              "rationale": "x", "confidence": 0.9},
        ],
    ))
    out = tc.classify_papers_batch([_mk_paper(paper_id="p1")])
    assert out[0].tier == PaperTier.UNCLASSIFIED


def test_classifier_llm_failure_returns_empty(monkeypatch):
    from engine.research_store.papers import tier_classifier as tc
    def _broken(**kw): raise RuntimeError("anthropic down")
    monkeypatch.setattr(tc, "llm_call", _broken)
    assert tc.classify_papers_batch([_mk_paper()]) == []


def test_classifier_empty_input():
    from engine.research_store.papers import tier_classifier as tc
    assert tc.classify_papers_batch([]) == []


# ────────────────────────────────────────────────────────────────────
# Workload registration
# ────────────────────────────────────────────────────────────────────
def test_workload_registered():
    from engine.llm.call import _resolve_workload
    provider, model = _resolve_workload("papers_tier_classifier")
    assert provider == "anthropic"
    assert "claude" in model
