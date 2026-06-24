"""tests/test_papers_ingest_integration.py — F11 ingest chain regression guard.

Why this exists. The /papers/ingest endpoint is a 3-stage pipeline:

  upload-preview (PDF text → cache)
        |
        v
  ingest (PaperIngestRequest)
        |  1. dedupe + chunk
        |  2. save_entry (papers_registry.jsonl)
        |  3. extract_hypotheses_from_chunks (LLM)  ─┐
        |  4. save_hypothesis (per cand)             | F11 wiring
        |  5. extract_spec (per hyp, LLM)            | (the bit that
        |  6. append_spec                            | broke 3x)
        v                                            ┘
  response: {paper_id, n_chunks, n_hypotheses, n_specs}

The F11 wiring (steps 3-6) broke 3 times in pre-commit code review:
  F11.1 — chunk_paper required a doi; arxiv preprints have none
  F11.2 — wrong call signature for extract_hypotheses_from_chunks +
          treated ExtractorResult as list[Hypothesis]
  F11.3 — extract_spec call signature drift

This test exercises the full ingest() function (NOT the HTTP layer)
with mocked LLM stubs + path-redirected stores, so each of the above
classes of bug surfaces deterministically the next time someone
edits the wiring.

Cost: $0 (LLMs mocked). Wall: ~1 s.
"""
from __future__ import annotations

from pathlib import Path
import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures: redirect store paths + mock LLM-driven extractors
# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def isolate_store_paths(tmp_path, monkeypatch):
    """Repoint the 3 jsonl stores the ingest writes to into tmp_path.
    Returns the tmp dir for downstream assertions."""
    import engine.research_store.papers.store as papers_store
    import engine.research_store.hypothesis.store as hyp_store
    import engine.hypothesis_spec.store as spec_store
    monkeypatch.setattr(
        papers_store, "REGISTRY_PATH",
        tmp_path / "papers_registry.jsonl",
    )
    monkeypatch.setattr(
        hyp_store, "HYPOTHESES_PATH",
        tmp_path / "hypotheses.jsonl",
    )
    monkeypatch.setattr(
        spec_store, "_STORE_PATH",
        tmp_path / "hypothesis_specs.jsonl",
    )
    return tmp_path


@pytest.fixture
def mock_hypothesis_extractor(monkeypatch):
    """Replace LLM-backed extract_hypotheses_from_chunks with a stub
    that returns 2 well-formed HypothesisCandidates. Mirrors the shape
    the route layer expects (ExtractorResult.candidates)."""
    from engine.agents.hypothesis_extractor import extractor as he
    from engine.agents.hypothesis_extractor.extractor import (
        HypothesisCandidate, ExtractorResult,
    )

    def _stub(*, paper_metadata, chunks, **kw):
        # Use the first chunk's id so the post-validation cross-ref
        # against the chunk batch survives (verbatim_quote.chunk_id
        # must be in chunks).
        first_id = chunks[0]["chunk_id"] if chunks else "synthetic_chunk"
        first_text = chunks[0]["text"][:30] if chunks else "synthetic body"
        cand_template = dict(
            mechanism_family    = "VALUE",
            mechanism_subtype   = "book_to_market",
            predicted_direction = "positive",
            predicted_magnitude = "moderate",
            required_data       = ("CRSP",),
            test_methodology    = "decile sort",
            source_chunk_ids    = (first_id,),
            verbatim_quotes     = ({
                "chunk_id":       first_id,
                "quote_text":     first_text,
                "section_ref":    "intro",
                "relevance_note": "",
            },),
        )
        cands = (
            HypothesisCandidate(claim="High book-to-market predicts returns.",
                                 **cand_template),
            HypothesisCandidate(claim="Quality firms outperform junk firms.",
                                 **cand_template),
        )
        return ExtractorResult(
            candidates                = cands,
            notes                     = "stubbed for integration test",
            n_dropped_post_validation = 0,
            drop_reasons              = (),
            raw_response              = {},
        )

    monkeypatch.setattr(he, "extract_hypotheses_from_chunks", _stub)


@pytest.fixture
def spy_spec_extractor(monkeypatch):
    """Replace extract_spec with a stub that records calls + returns
    None (the route layer handles None gracefully — n_specs stays 0).
    Returning a real HypothesisSpec is heavy; this proves the WIRING
    is intact (extractor is invoked per-hypothesis) without needing
    to construct a valid spec.
    """
    from engine.hypothesis_spec import extractor as se
    calls: list[dict] = []

    def _stub(**kw):
        calls.append(kw)
        return None

    monkeypatch.setattr(se, "extract_spec", _stub)
    return calls


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────
def test_papers_ingest_full_chain(isolate_store_paths, mock_hypothesis_extractor,
                                    spy_spec_extractor):
    """Hit papers_ingest() with a pre-populated PDF cache + mocked
    extractors. Validates the entire F11 wiring: chunks → registry →
    hypotheses → spec-extractor invocation."""
    from api.routes_paper_chain import (
        papers_ingest, _cache_put, PaperIngestRequest,
    )

    # Seed the in-memory PDF cache with a synthetic paper body. The
    # body must be long enough that chunk_paper produces ≥ 1 chunk.
    body = ("This is a synthetic paper body for integration testing. " * 50)
    preview_id = _cache_put({
        "text":     body,
        "src_kind": "test_upload",
        "src_url":  "https://example.com/test.pdf",
        "n_pages":  1,
    })

    req = PaperIngestRequest(
        title           = "Integration Test Paper",
        year            = 2024,
        authors         = ["Test Author"],
        venue           = "Test Venue",
        doi             = "",   # no doi → triggers synth path (F11.1)
        abstract        = "Synthetic abstract.",
        shelves         = ["other"],
        # "other" shelf requires an explicit rationale per schema —
        # surface enforces it.
        shelf_notes     = {"other": "integration test synthetic paper"},
        pdf_source_url  = "https://example.com/test.pdf",
        note            = "",
        preview_id      = preview_id,
    )

    resp = papers_ingest(req)

    # 1. Top-level response shape
    assert resp.paper_id, "paper_id must be populated"
    assert resp.title == req.title
    assert resp.n_chunks >= 1, "chunker must produce at least one chunk"
    assert resp.n_hypotheses == 2, ("2 mocked hypotheses must be saved; got "
                                      f"{resp.n_hypotheses}")
    assert resp.n_specs == 0, ("mock returns None spec; should NOT crash but "
                                "leave n_specs=0 — proves the route handles "
                                "extractor failure gracefully")
    assert resp.next_url.startswith("/research/papers/"), "next_url shape"

    # 2. Stores written
    reg_path = isolate_store_paths / "papers_registry.jsonl"
    hyp_path = isolate_store_paths / "hypotheses.jsonl"
    assert reg_path.is_file(), "papers_registry.jsonl must be written"
    assert hyp_path.is_file(), "hypotheses.jsonl must be written"
    n_reg_lines = sum(1 for _ in reg_path.open(encoding="utf-8"))
    n_hyp_lines = sum(1 for _ in hyp_path.open(encoding="utf-8"))
    assert n_reg_lines == 1, "exactly 1 paper entry"
    assert n_hyp_lines == 2, "exactly 2 hypothesis entries"

    # 3. Spec extractor INVOKED per hypothesis (the F11 wiring under test)
    assert len(spy_spec_extractor) == 2, (
        "extract_spec must be called once per saved hypothesis (the F11 "
        "wiring fix); got " + str(len(spy_spec_extractor)) + " calls. "
        "If this is 0, the hypothesis→spec wiring is broken."
    )
    # Each call must carry the right kwargs the extractor signature requires.
    for call_kw in spy_spec_extractor:
        assert "source_hypothesis_id" in call_kw
        assert "claim_text" in call_kw
        assert call_kw["claim_text"], "claim_text must be non-empty"


def test_papers_ingest_persists_ingestion_reason(isolate_store_paths,
                                                    mock_hypothesis_extractor,
                                                    spy_spec_extractor):
    """Phase 1.7 step 3: when the request carries ingestion_reason, it
    must land on the persisted PaperRegistryEntry with the correct
    source enum, free_text truncation, and an intent_category=None
    (LLM extraction is deferred to a later step)."""
    from api.routes_paper_chain import (
        papers_ingest, _cache_put, PaperIngestRequest, PaperIngestReasonIn,
    )
    from engine.research_store.papers import (
        load_registry, IngestionReasonSource,
    )

    body = "Long synthetic paper body. " * 50
    preview_id = _cache_put({
        "text": body, "src_kind": "test_upload",
        "src_url": "https://example.com/r.pdf", "n_pages": 1,
    })
    req = PaperIngestRequest(
        title="Reason Test", year=2024, authors=["A"], venue="V",
        doi="", abstract="abstract",
        shelves=["other"], shelf_notes={"other": "test"},
        pdf_source_url="https://example.com/r.pdf", note="",
        preview_id=preview_id,
        ingestion_reason=PaperIngestReasonIn(
            free_text="want to expand cross-asset breadth into commodity carry",
            source="user",
        ),
    )
    resp = papers_ingest(req)
    assert resp.paper_id

    # Read back from registry; assert ingestion_reason persisted
    entries = load_registry()
    saved = next(e for e in entries if e.paper_id == resp.paper_id)
    assert saved.ingestion_reason is not None
    assert saved.ingestion_reason.source == IngestionReasonSource.USER
    assert "cross-asset breadth" in saved.ingestion_reason.free_text
    assert saved.ingestion_reason.intent_category is None, (
        "intent_category should be None at ingest — LLM extracts later"
    )
    # Schema version bumped per Phase 1.7
    assert saved.schema_version == 2


def test_papers_ingest_agent_source_persists(isolate_store_paths,
                                               mock_hypothesis_extractor,
                                               spy_spec_extractor):
    """source='agent' (clicked through from /incoming) must round-trip
    as IngestionReasonSource.AGENT."""
    from api.routes_paper_chain import (
        papers_ingest, _cache_put, PaperIngestRequest, PaperIngestReasonIn,
    )
    from engine.research_store.papers import (
        load_registry, IngestionReasonSource,
    )

    preview_id = _cache_put({
        "text": "body " * 100, "src_kind": "test_upload",
        "src_url": "x", "n_pages": 1,
    })
    req = PaperIngestRequest(
        title="Agent Source Test", year=2024, authors=["A"], venue="V",
        doi="", abstract="abstract", shelves=["other"],
        shelf_notes={"other": "test"},
        pdf_source_url="x", note="", preview_id=preview_id,
        ingestion_reason=PaperIngestReasonIn(
            free_text="Agent: adjacent to deployed carry, partial breadth expansion",
            source="agent",
        ),
    )
    resp = papers_ingest(req)
    saved = next(e for e in load_registry() if e.paper_id == resp.paper_id)
    assert saved.ingestion_reason is not None
    assert saved.ingestion_reason.source == IngestionReasonSource.AGENT


def test_papers_ingest_empty_reason_persists_as_none(isolate_store_paths,
                                                       mock_hypothesis_extractor,
                                                       spy_spec_extractor):
    """When the user submits with blank reason textarea, the field must
    persist as None (not as IngestionReason with empty free_text)."""
    from api.routes_paper_chain import (
        papers_ingest, _cache_put, PaperIngestRequest, PaperIngestReasonIn,
    )
    from engine.research_store.papers import load_registry

    preview_id = _cache_put({
        "text": "body " * 100, "src_kind": "test_upload",
        "src_url": "x", "n_pages": 1,
    })
    # Empty / whitespace free_text triggers null persistence. Abstract
    # IS required (2026-06-06 change) so this test provides one.
    req = PaperIngestRequest(
        title="Empty Reason", year=2024, authors=["A"], venue="V",
        doi="", abstract="non-empty abstract per 2026-06-06 requirement",
        shelves=["other"],
        shelf_notes={"other": "x"},
        pdf_source_url="x", note="", preview_id=preview_id,
        ingestion_reason=PaperIngestReasonIn(
            free_text="   ",       # whitespace only
            source="user",
        ),
    )
    resp = papers_ingest(req)
    saved = next(e for e in load_registry() if e.paper_id == resp.paper_id)
    assert saved.ingestion_reason is None


def test_papers_ingest_rejects_empty_abstract(isolate_store_paths,
                                                 mock_hypothesis_extractor):
    """2026-06-06: abstract is now load-bearing — the hypothesis
    extractor reads it. Empty abstract MUST fail at ingest."""
    from api.routes_paper_chain import (
        papers_ingest, _cache_put, PaperIngestRequest,
    )
    from fastapi import HTTPException

    preview_id = _cache_put({
        "text": "body " * 100, "src_kind": "test_upload",
        "src_url": "x", "n_pages": 1,
    })
    req = PaperIngestRequest(
        title="Empty Abstract Test", year=2024, authors=["A"], venue="V",
        doi="", abstract="   ",            # whitespace only
        shelves=["other"], shelf_notes={"other": "x"},
        pdf_source_url="x", note="", preview_id=preview_id,
    )
    with pytest.raises(HTTPException) as exc_info:
        papers_ingest(req)
    assert exc_info.value.status_code == 400
    assert "abstract" in str(exc_info.value.detail).lower()


def test_papers_ingest_intent_category_persists(isolate_store_paths,
                                                  mock_hypothesis_extractor,
                                                  spy_spec_extractor):
    """2026-06-06: user-picked intent_category at ingest (from the
    dropdown) must round-trip onto IngestionReason — not wait for
    LLM extraction."""
    from api.routes_paper_chain import (
        papers_ingest, _cache_put, PaperIngestRequest, PaperIngestReasonIn,
    )
    from engine.research_store.papers import load_registry, IntentCategory

    preview_id = _cache_put({
        "text": "body " * 100, "src_kind": "test_upload",
        "src_url": "x", "n_pages": 1,
    })
    req = PaperIngestRequest(
        title="Intent Test", year=2024, authors=["A"], venue="V",
        doi="", abstract="non-empty abstract",
        shelves=["other"], shelf_notes={"other": "x"},
        pdf_source_url="x", note="", preview_id=preview_id,
        # User picked intent but left free_text blank — still persists.
        ingestion_reason=PaperIngestReasonIn(
            free_text="",
            source="user",
            intent_category="expand_breadth",
        ),
    )
    resp = papers_ingest(req)
    saved = next(e for e in load_registry() if e.paper_id == resp.paper_id)
    assert saved.ingestion_reason is not None
    assert saved.ingestion_reason.intent_category == IntentCategory.EXPAND_BREADTH
    assert saved.ingestion_reason.free_text == ""


def test_papers_ingest_shelf_derived_from_intent(isolate_store_paths,
                                                    mock_hypothesis_extractor,
                                                    spy_spec_extractor):
    """2026-06-06 manual-ingest shelf auto-classify: when caller sends
    shelves=["other"] (the silent default) AND a non-OTHER intent_category,
    the backend overrides shelves with the derived classification."""
    from api.routes_paper_chain import (
        papers_ingest, _cache_put, PaperIngestRequest, PaperIngestReasonIn,
    )
    from engine.research_store.papers import load_registry, Shelf

    preview_id = _cache_put({
        "text": "body " * 100, "src_kind": "test_upload",
        "src_url": "x", "n_pages": 1,
    })
    req = PaperIngestRequest(
        title="Methodology Test", year=2024, authors=["A"], venue="V",
        doi="", abstract="non-empty abstract",
        shelves=["other"], shelf_notes={"other": "test"},
        pdf_source_url="x", note="", preview_id=preview_id,
        ingestion_reason=PaperIngestReasonIn(
            free_text="adopt their decay metric",
            source="user",
            intent_category="methodology_borrow",
        ),
    )
    resp = papers_ingest(req)
    saved = next(e for e in load_registry() if e.paper_id == resp.paper_id)
    # methodology_borrow → doctrine_method
    assert Shelf.DOCTRINE_METHOD in saved.shelves
    assert Shelf.OTHER not in saved.shelves


def test_papers_ingest_preclassified_shelf_not_overridden(
        isolate_store_paths, mock_hypothesis_extractor, spy_spec_extractor):
    """If caller already supplied a non-OTHER shelf (the /incoming
    "Ingest now" path passes ?shelf= classified client-side), the
    backend MUST NOT silently re-classify on top of it."""
    from api.routes_paper_chain import (
        papers_ingest, _cache_put, PaperIngestRequest, PaperIngestReasonIn,
    )
    from engine.research_store.papers import load_registry, Shelf

    preview_id = _cache_put({
        "text": "body " * 100, "src_kind": "test_upload",
        "src_url": "x", "n_pages": 1,
    })
    req = PaperIngestRequest(
        title="Preclassified Test", year=2024, authors=["A"], venue="V",
        doi="", abstract="non-empty abstract",
        # caller pre-classified via /incoming JS classifier:
        shelves=["green_critique"], shelf_notes={"green_critique": "from /incoming"},
        pdf_source_url="x", note="", preview_id=preview_id,
        ingestion_reason=PaperIngestReasonIn(
            free_text="x",
            source="agent",
            intent_category="methodology_borrow",   # would map to
                                                     # doctrine_method
                                                     # if backend overrode
        ),
    )
    resp = papers_ingest(req)
    saved = next(e for e in load_registry() if e.paper_id == resp.paper_id)
    # Backend MUST preserve the caller's classification.
    assert Shelf.GREEN_CRITIQUE in saved.shelves
    assert Shelf.DOCTRINE_METHOD not in saved.shelves


def test_papers_ingest_rejects_missing_preview(isolate_store_paths):
    """Stale / missing preview_id must surface as 404 (TTL guard)."""
    from api.routes_paper_chain import papers_ingest, PaperIngestRequest
    from fastapi import HTTPException

    req = PaperIngestRequest(
        title       = "Test",
        year        = 2024,
        authors     = ["A"],
        venue       = "V",
        doi         = "10.0000/test",
        abstract    = "",
        shelves     = ["other"],
        preview_id  = "nonexistent-preview-id-12345",
    )
    with pytest.raises(HTTPException) as exc_info:
        papers_ingest(req)
    assert exc_info.value.status_code == 404
    assert "preview_id" in str(exc_info.value.detail)


def test_papers_ingest_rejects_empty_shelves(isolate_store_paths,
                                               mock_hypothesis_extractor):
    """Shelves are required — schema layer should reject the request."""
    # pydantic may reject at construct-time OR the endpoint may reject
    # at runtime; both are acceptable. The contract is "must fail loudly".
    from api.routes_paper_chain import (
        papers_ingest, _cache_put, PaperIngestRequest,
    )
    from fastapi import HTTPException
    from pydantic import ValidationError

    preview_id = _cache_put({"text": "body" * 50, "src_kind": "test_upload"})
    try:
        req = PaperIngestRequest(
            title=" stub", year=2024, authors=["A"], venue="V",
            doi="10.0000/empty", abstract="", shelves=[],
            preview_id=preview_id,
        )
    except ValidationError:
        # Schema rejected at construct → contract satisfied
        return
    # If schema accepted, endpoint must reject
    with pytest.raises(HTTPException) as exc_info:
        papers_ingest(req)
    assert exc_info.value.status_code == 400
