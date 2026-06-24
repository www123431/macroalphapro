"""Tests for engine.research.discovery.review_ui — local UI for paper nominate.

Senior D shipped 2026-05-30. Tests cover identifier extraction
(arxiv / DOI / OpenAlex / SSRN), metadata fetch routing, nominate flow,
queue write, and the HTTP handler endpoints.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest import mock

import pytest


# ── Identifier extraction ────────────────────────────────────────────────

def test_extract_arxiv_id_bare():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier("2401.12345") == {"type": "arxiv", "id": "2401.12345"}


def test_extract_arxiv_id_url():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier("https://arxiv.org/abs/2401.12345") == \
        {"type": "arxiv", "id": "2401.12345"}


def test_extract_arxiv_id_with_version():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier("https://arxiv.org/abs/2401.12345v3") == \
        {"type": "arxiv", "id": "2401.12345"}


def test_extract_doi_with_url():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier("https://doi.org/10.1287/mnsc.2024.1234") == \
        {"type": "doi", "id": "10.1287/mnsc.2024.1234"}


def test_extract_doi_bare():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier("10.1111/jofi.12345") == \
        {"type": "doi", "id": "10.1111/jofi.12345"}


def test_extract_openalex_work_id():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier("https://openalex.org/W4234567890") == \
        {"type": "openalex", "id": "W4234567890"}


def test_extract_openalex_bare():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier("W4234567890") == \
        {"type": "openalex", "id": "W4234567890"}


def test_extract_ssrn_abstract_url():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier("https://ssrn.com/abstract=4567890") == \
        {"type": "ssrn", "id": "4567890"}


def test_extract_ssrn_papers_cfm_url():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier(
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4567890"
    ) == {"type": "ssrn", "id": "4567890"}


def test_extract_returns_none_for_garbage():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier("") is None
    assert extract_identifier("just some text") is None
    assert extract_identifier(None) is None


def test_extract_strips_trailing_punctuation_from_doi():
    from engine.research.discovery.review_ui import extract_identifier
    assert extract_identifier("(see 10.1234/abc.def.)") == \
        {"type": "doi", "id": "10.1234/abc.def"}


# ── Metadata fetch (mocked HTTP) ─────────────────────────────────────────

def test_fetch_openalex_metadata(monkeypatch):
    from engine.research.discovery import review_ui

    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "https://openalex.org/W123",
        "title": "Test Paper",
        "abstract_inverted_index": {"hello": [0], "world": [1]},
        "authorships": [
            {"author": {"display_name": "A. Smith"},
              "institutions": [{"display_name": "MIT"}]}
        ],
        "primary_location": {"source": {"display_name": "Test Journal"}},
        "publication_date": "2024-01-15",
        "doi": "https://doi.org/10.1234/x",
        "cited_by_count": 10,
    }
    monkeypatch.setattr("requests.get", lambda *a, **kw: mock_response)
    result = review_ui.fetch_metadata({"type": "openalex", "id": "W123"})
    assert result is not None
    assert result["title"] == "Test Paper"
    assert result["abstract"] == "hello world"


def test_fetch_crossref_metadata(monkeypatch):
    from engine.research.discovery import review_ui

    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message": {
        "DOI": "10.1287/mnsc.2024.1234",
        "title": ["Supply Chain Paper"],
        "author": [{"given": "Alice", "family": "Smith"}],
        "issued": {"date-parts": [[2024, 6, 1]]},
        "container-title": ["Management Science"],
    }}
    monkeypatch.setattr("requests.get", lambda *a, **kw: mock_response)
    result = review_ui.fetch_metadata({"type": "doi",
                                            "id": "10.1287/mnsc.2024.1234"})
    assert result is not None
    assert result["title"] == "Supply Chain Paper"
    assert result["doi"] == "10.1287/mnsc.2024.1234"


def test_fetch_returns_none_on_404(monkeypatch):
    from engine.research.discovery import review_ui
    mock_response = mock.MagicMock()
    mock_response.status_code = 404
    monkeypatch.setattr("requests.get", lambda *a, **kw: mock_response)
    assert review_ui.fetch_metadata({"type": "doi", "id": "10.bad/doi"}) is None


def test_fetch_unknown_type_returns_none():
    from engine.research.discovery import review_ui
    assert review_ui.fetch_metadata({"type": "unknown", "id": "x"}) is None


# ── Nominate (full flow) ─────────────────────────────────────────────────

def test_nominate_with_bad_input_returns_error():
    from engine.research.discovery.review_ui import nominate
    result = nominate("not a valid identifier")
    assert "error" in result


def test_nominate_writes_to_queue(monkeypatch, tmp_path):
    """Full nominate happy path writes entry to DISCOVERY_QUEUE."""
    from engine.research.discovery import review_ui

    # Redirect queue file to tmp
    fake_queue = tmp_path / "discovery_queue.jsonl"
    monkeypatch.setattr(review_ui, "DISCOVERY_QUEUE", fake_queue)

    # Mock metadata fetch to return a real-looking record
    monkeypatch.setattr(review_ui, "fetch_metadata", lambda ident: {
        "source":          "openalex",
        "source_id":       ident["id"],
        "title":           "Carry Across Asset Classes",
        "abstract":        "Long-short carry portfolio with Sharpe 1.5 over 1990-2012 monthly CRSP and Bloomberg data.",
        "authors":         "Smith, A.",
        "venue":           "Journal of Finance",
        "doi":             "10.1111/jofi.12345",
        "submitted_date":  "2018-06-01",
    })

    result = review_ui.nominate("https://openalex.org/W123")
    assert result["ok"] is True
    assert "Carry" in result["title"]
    assert result["scoring_method"] == "text"   # abstract present
    assert fake_queue.exists()
    # Verify entry written
    lines = fake_queue.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["source"] == "manual_nominate"
    assert entry["title"] == "Carry Across Asset Classes"
    assert entry["nominated_via"] == "ui"


def test_nominate_falls_back_to_venue_tier_when_no_abstract(monkeypatch, tmp_path):
    """When no abstract is available (e.g. FF 2015 from publisher-locked
    venues), routing should fall back to venue-tier scoring rather than
    using stale text features against an empty abstract."""
    from engine.research.discovery import review_ui

    fake_queue = tmp_path / "discovery_queue.jsonl"
    monkeypatch.setattr(review_ui, "DISCOVERY_QUEUE", fake_queue)

    # Mock metadata with empty abstract — simulates FF 2015 case
    monkeypatch.setattr(review_ui, "fetch_metadata", lambda ident: {
        "source":          "crossref",
        "source_id":       ident["id"],
        "title":           "A five-factor asset pricing model",
        "abstract":        "",     # no abstract available
        "authors":         "Fama; French",
        "venue":           "Journal of Financial Economics",
        "doi":             "10.1016/j.jfineco.2014.10.010",
        "submitted_date":  "2014-09-01",
    })

    result = review_ui.nominate("10.1016/j.jfineco.2014.10.010")
    assert result["ok"] is True
    assert result["scoring_method"] == "venue_tier_fallback"
    # JFE has venue_tier 0.95 → conf 0.95 → routes to review
    assert result["confidence"] >= 0.85
    assert result["routing"] == "review"


def test_nominate_short_abstract_also_triggers_fallback(monkeypatch, tmp_path):
    """Edge case: 30-char abstract is below the 50-char threshold for
    text-scoring → use venue-tier."""
    from engine.research.discovery import review_ui

    fake_queue = tmp_path / "discovery_queue.jsonl"
    monkeypatch.setattr(review_ui, "DISCOVERY_QUEUE", fake_queue)

    monkeypatch.setattr(review_ui, "fetch_metadata", lambda ident: {
        "source":          "crossref",
        "source_id":       ident["id"],
        "title":           "A momentum paper",
        "abstract":        "Short blurb only.",
        "authors":         "X",
        "venue":           "Journal of Finance",
        "doi":             "10.1111/x",
        "submitted_date":  "2020-01-01",
    })
    result = review_ui.nominate("10.1111/x")
    assert result["scoring_method"] == "venue_tier_fallback"


# ── HTTP handler smoke ───────────────────────────────────────────────────

@pytest.fixture
def handler_class(monkeypatch, tmp_path):
    """Provide a handler with paths redirected to tmp."""
    from engine.research.discovery import review_ui
    monkeypatch.setattr(review_ui, "DISCOVERY_QUEUE",
                          tmp_path / "discovery_queue.jsonl")
    monkeypatch.setattr(review_ui, "DISCOVERY_BORDERLINE",
                          tmp_path / "discovery_borderline.jsonl")
    return review_ui.ReviewHandler


def test_render_home_returns_html_with_bookmarklet():
    from engine.research.discovery.review_ui import render_home
    html = render_home()
    assert "<!DOCTYPE html>" in html
    assert "Nominate Paper" in html
    assert "Bookmarklet" in html
    assert "Add to Research Queue" in html
    assert "Primary Review Queue" in html
    assert "Borderline Queue" in html
    # CORS-safe bookmarklet uses fetch + JSON
    assert "fetch(" in html
    assert "javascript:(function" in html


def test_render_home_empty_state(monkeypatch, tmp_path):
    from engine.research.discovery import review_ui
    # Point queues to non-existent files
    monkeypatch.setattr(review_ui, "DISCOVERY_QUEUE", tmp_path / "no.jsonl")
    monkeypatch.setattr(review_ui, "DISCOVERY_BORDERLINE", tmp_path / "no2.jsonl")
    html = review_ui.render_home()
    assert "No reviews queued yet" in html
    assert "No borderline papers yet" in html


def test_read_queue_handles_missing_file(tmp_path):
    from engine.research.discovery.review_ui import read_queue
    assert read_queue(tmp_path / "nonexistent.jsonl") == []


def test_read_queue_skips_invalid_json(tmp_path):
    f = tmp_path / "queue.jsonl"
    f.write_text('{"valid": 1}\nnot json\n{"valid": 2}\n', encoding="utf-8")
    from engine.research.discovery.review_ui import read_queue
    items = read_queue(f)
    assert len(items) == 2     # bad line silently skipped


def test_read_queue_limit():
    from engine.research.discovery.review_ui import read_queue
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                          encoding="utf-8") as f:
        for i in range(50):
            f.write(json.dumps({"i": i}) + "\n")
        fname = Path(f.name)
    try:
        items = read_queue(fname, limit=5)
        assert len(items) == 5
        # Most recent first → highest i
        assert items[0]["i"] == 49
    finally:
        os.unlink(fname)
