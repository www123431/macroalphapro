"""tests/test_citation_verifier.py — Phase 2.2a.

Citation verifier tests. Mocks registry + chroma + LLM so tests are
fast and deterministic. Covers:
  - paper not in registry  → unresolved CitationCheck (likely hallucinated)
  - paper resolved but chroma empty → resolved+0 chunks degraded check
  - chroma returns chunks + LLM verifies → normal CitationCheck
  - LLM fails → neutral 0.5 fallback
  - LLM returns malformed payload → degraded check still returns
  - confidence clamped to [0, 1]
  - supporting_chunks capped at 5
  - aggregate roll-up: empty / mixed / all-unresolved / low-confidence flag
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _mock_registry(monkeypatch, *, papers: list[dict] | None = None):
    """Patch papers registry lookup. Each dict needs paper_id + doi."""
    class _Paper:
        def __init__(self, d):
            self.paper_id = d["paper_id"]
            self.doi      = d.get("doi", "")
    from engine.research_store import papers as papers_mod
    fake = [_Paper(d) for d in (papers or [])]
    monkeypatch.setattr(papers_mod, "load_registry", lambda: fake)
    monkeypatch.setattr(papers_mod, "latest_per_doi",
                          lambda reg: {p.doi: p for p in reg if p.doi})


def _mock_chroma(monkeypatch, *, ids=None, docs=None, raises=False):
    """Patch papers_chroma.get_collection so coll.query returns the
    given (ids, docs) tuple."""
    from engine.research_store.red_lessons import papers_chroma as pc
    class _Coll:
        def query(self, **kw):
            if raises:
                raise RuntimeError("chroma down")
            return {
                "ids":       [ids or []],
                "documents": [docs or []],
            }
    monkeypatch.setattr(pc, "get_collection", lambda: _Coll())


def _mock_llm(monkeypatch, *, tool_input=None, text="", raise_exc=None,
                tool_name="emit_verification"):
    from engine.agents.papers_curator import citation_verifier as mod
    from engine.llm.call import LLMCallResult, ToolCall
    def _fake_call(**kw):
        if raise_exc is not None:
            raise raise_exc
        tcs = ()
        if tool_input is not None:
            tcs = (ToolCall(id="tc", name=tool_name, input=tool_input),)
        return LLMCallResult(
            text=text, tool_calls=tcs, stop_reason="tool_use",
            model="deepseek-v4-pro", provider="deepseek",
            cost_usd=0.001, latency_ms=900,
            cache_read_tokens=0, raw_usage={},
        )
    monkeypatch.setattr(mod, "llm_call", _fake_call)


# ─────────────────────────────────────────────────────────────────────
# verify_one_citation — paper not in registry
# ─────────────────────────────────────────────────────────────────────
def test_paper_not_in_registry_returns_unresolved(monkeypatch):
    from engine.agents.papers_curator.citation_verifier import verify_one_citation
    _mock_registry(monkeypatch, papers=[])
    c = verify_one_citation(claim="x", paper_id="ghost-paper-id")
    assert c.paper_resolved is False
    assert c.confidence == 0.0
    assert c.chunks_queried == 0
    assert "not in registry" in c.verifier_notes.lower()


# ─────────────────────────────────────────────────────────────────────
# verify_one_citation — paper resolved, chroma empty
# ─────────────────────────────────────────────────────────────────────
def test_paper_resolved_but_chroma_empty_returns_degraded(monkeypatch):
    """Paper exists in registry but fulltext not ingested → cannot
    verify. Conf 0 but flagged as 'fulltext not ingested' (different
    failure mode than hallucination)."""
    from engine.agents.papers_curator.citation_verifier import verify_one_citation
    _mock_registry(monkeypatch, papers=[{"paper_id": "p1", "doi": "10.x/y"}])
    _mock_chroma(monkeypatch, ids=[], docs=[])
    c = verify_one_citation(claim="x", paper_id="p1")
    assert c.paper_resolved is True
    assert c.chunks_queried == 0
    assert c.confidence == 0.0
    assert "not ingested" in c.verifier_notes.lower()


# ─────────────────────────────────────────────────────────────────────
# verify_one_citation — happy path
# ─────────────────────────────────────────────────────────────────────
def test_happy_path_returns_llm_confidence(monkeypatch):
    from engine.agents.papers_curator.citation_verifier import verify_one_citation
    _mock_registry(monkeypatch, papers=[{"paper_id": "p1", "doi": "10.x/y"}])
    _mock_chroma(monkeypatch,
                   ids  = ["10.x/y::p0001", "10.x/y::p0002"],
                   docs = ["VRP delivers Sharpe 0.7 post-cost",
                            "Variance risk premium on SPX..."])
    _mock_llm(monkeypatch, tool_input={
        "confidence":        0.78,
        "supporting_chunks": ["10.x/y::p0001"],
        "verifier_notes":    "Chunks directly cite Sharpe 0.7+ for VRP",
    })
    c = verify_one_citation(claim="VRP post-2010 stability",
                              paper_id="p1")
    assert c.paper_resolved is True
    assert c.chunks_queried == 2
    assert c.confidence == 0.78
    assert c.supporting_chunks == ("10.x/y::p0001",)
    assert "VRP" in c.verifier_notes


# ─────────────────────────────────────────────────────────────────────
# verify_one_citation — LLM failures
# ─────────────────────────────────────────────────────────────────────
def test_llm_failure_returns_neutral_fallback(monkeypatch):
    """LLM down → 0.5 neutral (not 0 — paper exists + chunks fetched
    so we can't rule out support; just can't verify)."""
    from engine.agents.papers_curator.citation_verifier import verify_one_citation
    _mock_registry(monkeypatch, papers=[{"paper_id": "p1", "doi": "10.x/y"}])
    _mock_chroma(monkeypatch, ids=["x"], docs=["chunk text"])
    _mock_llm(monkeypatch, raise_exc=RuntimeError("api down"))
    c = verify_one_citation(claim="x", paper_id="p1")
    assert c.paper_resolved is True
    assert c.chunks_queried == 1
    assert c.confidence == 0.5
    assert "unavailable" in c.verifier_notes.lower()


def test_llm_no_tool_call_returns_neutral_fallback(monkeypatch):
    from engine.agents.papers_curator.citation_verifier import verify_one_citation
    _mock_registry(monkeypatch, papers=[{"paper_id": "p1", "doi": "10.x/y"}])
    _mock_chroma(monkeypatch, ids=["x"], docs=["chunk text"])
    _mock_llm(monkeypatch, tool_input=None, text="prose response")
    c = verify_one_citation(claim="x", paper_id="p1")
    assert c.confidence == 0.5


# ─────────────────────────────────────────────────────────────────────
# Payload sanitization
# ─────────────────────────────────────────────────────────────────────
def test_confidence_clamped_to_unit_interval(monkeypatch):
    from engine.agents.papers_curator.citation_verifier import verify_one_citation
    _mock_registry(monkeypatch, papers=[{"paper_id": "p1", "doi": "10.x/y"}])
    _mock_chroma(monkeypatch, ids=["x"], docs=["chunk text"])
    _mock_llm(monkeypatch, tool_input={
        "confidence": 1.5,  # out of range
        "supporting_chunks": [],
        "verifier_notes": "n/a",
    })
    c = verify_one_citation(claim="x", paper_id="p1")
    assert c.confidence == 1.0


def test_supporting_chunks_capped_at_5(monkeypatch):
    from engine.agents.papers_curator.citation_verifier import verify_one_citation
    _mock_registry(monkeypatch, papers=[{"paper_id": "p1", "doi": "10.x/y"}])
    _mock_chroma(monkeypatch, ids=["x"], docs=["chunk text"])
    _mock_llm(monkeypatch, tool_input={
        "confidence": 0.8,
        "supporting_chunks": [f"chunk_{i}" for i in range(20)],
        "verifier_notes": "n/a",
    })
    c = verify_one_citation(claim="x", paper_id="p1")
    assert len(c.supporting_chunks) == 5


def test_verifier_notes_truncated_at_240(monkeypatch):
    from engine.agents.papers_curator.citation_verifier import verify_one_citation
    _mock_registry(monkeypatch, papers=[{"paper_id": "p1", "doi": "10.x/y"}])
    _mock_chroma(monkeypatch, ids=["x"], docs=["chunk text"])
    _mock_llm(monkeypatch, tool_input={
        "confidence": 0.5,
        "supporting_chunks": [],
        "verifier_notes": "Q" * 500,
    })
    c = verify_one_citation(claim="x", paper_id="p1")
    assert len(c.verifier_notes) == 240


# ─────────────────────────────────────────────────────────────────────
# verify_citations — batch
# ─────────────────────────────────────────────────────────────────────
def test_verify_citations_empty_returns_empty():
    from engine.agents.papers_curator.citation_verifier import verify_citations
    assert verify_citations(claim="x", paper_ids=()) == ()


def test_verify_citations_mixed_paths(monkeypatch):
    """One paper exists + verified, one paper hallucinated. Both
    returned in order."""
    from engine.agents.papers_curator.citation_verifier import verify_citations
    _mock_registry(monkeypatch,
                     papers=[{"paper_id": "real-paper", "doi": "10.x/y"}])
    _mock_chroma(monkeypatch, ids=["x::p0001"], docs=["chunk text"])
    _mock_llm(monkeypatch, tool_input={
        "confidence": 0.8,
        "supporting_chunks": ["x::p0001"],
        "verifier_notes": "supported",
    })
    out = verify_citations(claim="x", paper_ids=("real-paper", "ghost-paper"))
    assert len(out) == 2
    assert out[0].paper_id == "real-paper"
    assert out[0].paper_resolved is True
    assert out[0].confidence == 0.8
    assert out[1].paper_id == "ghost-paper"
    assert out[1].paper_resolved is False
    assert out[1].confidence == 0.0


# ─────────────────────────────────────────────────────────────────────
# aggregate_citation_quality — roll-up consumed by B + audit event
# ─────────────────────────────────────────────────────────────────────
def test_aggregate_empty_returns_vacuous_ok():
    from engine.agents.papers_curator.citation_verifier import aggregate_citation_quality
    out = aggregate_citation_quality(())
    assert out["n_papers_cited"] == 0
    assert out["mean_confidence"] == 1.0   # vacuous
    assert out["low_confidence_flag"] is False


def test_aggregate_all_resolved_high_conf_flags_clean():
    from engine.agents.papers_curator.citation_verifier import (
        CitationCheck, aggregate_citation_quality,
    )
    checks = (
        CitationCheck(paper_id="p1", paper_resolved=True, chunks_queried=3,
                        confidence=0.85, supporting_chunks=("a",),
                        verifier_notes="ok"),
        CitationCheck(paper_id="p2", paper_resolved=True, chunks_queried=3,
                        confidence=0.75, supporting_chunks=("b",),
                        verifier_notes="ok"),
    )
    out = aggregate_citation_quality(checks)
    assert out["n_resolved"] == 2
    assert out["n_unresolved"] == 0
    assert out["any_unresolved"] is False
    assert out["mean_confidence"] == 0.8
    assert out["low_confidence_flag"] is False


def test_aggregate_any_unresolved_triggers_flag():
    """If even ONE cited paper is unresolved (likely hallucinated),
    flag the whole candidate as low_confidence — that's the strongest
    signal we have."""
    from engine.agents.papers_curator.citation_verifier import (
        CitationCheck, aggregate_citation_quality,
    )
    checks = (
        CitationCheck(paper_id="p1", paper_resolved=True, chunks_queried=3,
                        confidence=0.9, supporting_chunks=("a",),
                        verifier_notes="ok"),
        CitationCheck(paper_id="p2", paper_resolved=False, chunks_queried=0,
                        confidence=0.0, supporting_chunks=(),
                        verifier_notes="not in registry"),
    )
    out = aggregate_citation_quality(checks)
    assert out["any_unresolved"] is True
    assert out["low_confidence_flag"] is True


def test_aggregate_low_mean_triggers_flag_even_when_all_resolved():
    """All papers resolved but mean conf < 0.5 → flag low-confidence.
    LLM judged the chunks don't really support the claim."""
    from engine.agents.papers_curator.citation_verifier import (
        CitationCheck, aggregate_citation_quality,
    )
    checks = (
        CitationCheck(paper_id="p1", paper_resolved=True, chunks_queried=3,
                        confidence=0.3, supporting_chunks=(),
                        verifier_notes="weak support"),
        CitationCheck(paper_id="p2", paper_resolved=True, chunks_queried=3,
                        confidence=0.4, supporting_chunks=(),
                        verifier_notes="partial"),
    )
    out = aggregate_citation_quality(checks)
    assert out["any_unresolved"] is False
    assert out["mean_confidence"] == 0.35
    assert out["low_confidence_flag"] is True   # mean < 0.5


# ─────────────────────────────────────────────────────────────────────
# Prompt content invariants
# ─────────────────────────────────────────────────────────────────────
def test_system_prompt_load_bearing_rules():
    """System prompt must include the calibration scale + tool call
    requirement so the LLM uses the full 0-1 range honestly."""
    from engine.agents.papers_curator.citation_verifier import _SYSTEM_PROMPT
    assert "0.0 - 0.3" in _SYSTEM_PROMPT
    assert "verbatim" in _SYSTEM_PROMPT.lower()
    assert "hallucinated" in _SYSTEM_PROMPT.lower()
    assert "ALWAYS call it" in _SYSTEM_PROMPT
