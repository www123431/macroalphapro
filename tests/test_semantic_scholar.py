"""tests/test_semantic_scholar.py — Stage A piece 1 client tests.

HTTP layer mocked — tests verify URL composition, fail-OPEN semantics,
and response parsing. No real network calls in CI.

A separate manual smoke test (commented at bottom) hits the live
Semantic Scholar API for end-to-end confirmation; gate it behind
an env flag so CI doesn't depend on network availability.
"""
from __future__ import annotations

import json
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# HTTP mock helper
# ─────────────────────────────────────────────────────────────────────
def _patch_http(monkeypatch, payload_or_factory):
    """Patch _http_get to return either a fixed payload or a callable
    that takes (path, params, timeout) and returns dict / None."""
    from engine.agents.papers_curator import semantic_scholar as ss

    captured: list[dict] = []

    def _fake_get(path, *, params=None, timeout=15.0):
        captured.append({"path": path, "params": params, "timeout": timeout})
        if callable(payload_or_factory):
            return payload_or_factory(path=path, params=params,
                                         timeout=timeout)
        return payload_or_factory

    monkeypatch.setattr(ss, "_http_get", _fake_get)
    # Disable the polite_sleep so tests aren't slow
    monkeypatch.setattr(ss, "_polite_sleep", lambda: None)
    return captured


# ─────────────────────────────────────────────────────────────────────
# search_author_by_name
# ─────────────────────────────────────────────────────────────────────
def test_search_author_by_name_parses_payload(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        search_author_by_name,
    )
    _patch_http(monkeypatch, {
        "data": [
            {"authorId": "33344", "name": "Tim Bollerslev",
             "hIndex": 95, "paperCount": 320,
             "affiliations": [{"name": "Duke University"}]},
        ],
    })
    out = search_author_by_name("Bollerslev")
    assert len(out) == 1
    a = out[0]
    assert a.author_id == "33344"
    assert a.name == "Tim Bollerslev"
    assert a.h_index == 95
    assert a.paper_count == 320
    assert "Duke University" in a.affiliations


def test_search_author_empty_name_returns_empty(monkeypatch):
    """Cost discipline: blank query → don't hit API."""
    from engine.agents.papers_curator.semantic_scholar import (
        search_author_by_name,
    )
    captured = _patch_http(monkeypatch, {"data": []})
    assert search_author_by_name("") == ()
    assert search_author_by_name("   ") == ()
    assert captured == []   # API not even called


def test_search_author_api_failure_returns_empty(monkeypatch):
    """API down → () (caller degrades gracefully)."""
    from engine.agents.papers_curator.semantic_scholar import (
        search_author_by_name,
    )
    _patch_http(monkeypatch, None)
    assert search_author_by_name("Bollerslev") == ()


def test_search_author_respects_limit(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        search_author_by_name,
    )
    captured = _patch_http(monkeypatch, {
        "data": [{"authorId": str(i), "name": f"Author {i}",
                   "hIndex": 0, "paperCount": 0, "affiliations": []}
                  for i in range(20)],
    })
    out = search_author_by_name("test", limit=3)
    assert len(out) == 3
    assert captured[0]["params"]["limit"] == 3


# ─────────────────────────────────────────────────────────────────────
# lookup_author_by_id
# ─────────────────────────────────────────────────────────────────────
def test_lookup_author_by_id_happy_path(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        lookup_author_by_id,
    )
    _patch_http(monkeypatch, {
        "authorId": "33344", "name": "Tim Bollerslev",
        "hIndex": 95, "paperCount": 320,
        "affiliations": [{"name": "Duke University"}],
    })
    a = lookup_author_by_id("33344")
    assert a is not None
    assert a.h_index == 95


def test_lookup_author_404_returns_none(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        lookup_author_by_id,
    )
    _patch_http(monkeypatch, None)
    assert lookup_author_by_id("ghost-id") is None


# ─────────────────────────────────────────────────────────────────────
# author_papers
# ─────────────────────────────────────────────────────────────────────
def _paper_record(*, paper_id, title, year=2024, venue="JF",
                    citation_count=10, doi="10.x/y", arxiv_id=""):
    return {
        "paperId":       paper_id,
        "title":         title,
        "abstract":      "Abstract here",
        "year":          year,
        "venue":         venue,
        "citationCount": citation_count,
        "externalIds":   {"DOI": doi, "ArXiv": arxiv_id},
        "authors":       [{"authorId": "33344", "name": "Tim Bollerslev"}],
        "url":           f"https://semanticscholar.org/paper/{paper_id}",
    }


def test_author_papers_returns_paper_summaries(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        author_papers,
    )
    _patch_http(monkeypatch, {
        "data": [
            _paper_record(paper_id="p1", title="VRP paper 1", year=2024),
            _paper_record(paper_id="p2", title="VRP paper 2", year=2023),
        ],
    })
    out = author_papers("33344", limit=10)
    assert len(out) == 2
    assert out[0].title == "VRP paper 1"
    assert out[0].year == 2024
    assert out[0].venue == "JF"
    assert out[0].citation_count == 10
    assert out[0].doi == "10.x/y"
    assert "Tim Bollerslev" in out[0].author_names


def test_author_papers_min_year_filter(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        author_papers,
    )
    _patch_http(monkeypatch, {
        "data": [
            _paper_record(paper_id="p1", title="recent", year=2024),
            _paper_record(paper_id="p2", title="old",    year=2018),
        ],
    })
    out = author_papers("33344", min_year=2022)
    assert len(out) == 1
    assert out[0].title == "recent"


def test_author_papers_empty_author_id_returns_empty(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        author_papers,
    )
    captured = _patch_http(monkeypatch, {})
    assert author_papers("") == ()
    assert captured == []


# ─────────────────────────────────────────────────────────────────────
# forward_citations — the key institutional research tool
# ─────────────────────────────────────────────────────────────────────
def test_forward_citations_parses_citingPaper_wrapper(monkeypatch):
    """SS wraps each forward citation as {'citingPaper': {...}} — the
    parser must unwrap it. If we silently treat the wrapper as the
    paper, we get all empty PaperSummary."""
    from engine.agents.papers_curator.semantic_scholar import (
        forward_citations,
    )
    _patch_http(monkeypatch, {
        "data": [
            {"citingPaper": _paper_record(paper_id="c1",
                                            title="Citing paper 1",
                                            citation_count=42)},
            {"citingPaper": _paper_record(paper_id="c2",
                                            title="Citing paper 2",
                                            citation_count=15)},
        ],
    })
    out = forward_citations("baseline-paper-id", limit=10)
    assert len(out) == 2
    assert out[0].title == "Citing paper 1"
    assert out[0].citation_count == 42


def test_forward_citations_min_year_filter(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        forward_citations,
    )
    _patch_http(monkeypatch, {
        "data": [
            {"citingPaper": _paper_record(paper_id="c1", title="recent",
                                            year=2024)},
            {"citingPaper": _paper_record(paper_id="c2", title="old",
                                            year=2010)},
        ],
    })
    out = forward_citations("baseline", min_year=2020)
    assert len(out) == 1
    assert out[0].title == "recent"


def test_forward_citations_api_failure_returns_empty(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        forward_citations,
    )
    _patch_http(monkeypatch, None)
    assert forward_citations("x") == ()


# ─────────────────────────────────────────────────────────────────────
# Paper lookup by external id
# ─────────────────────────────────────────────────────────────────────
def test_lookup_paper_by_doi(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        lookup_paper_by_doi,
    )
    captured = _patch_http(monkeypatch,
                              _paper_record(paper_id="ss-paper-1",
                                              title="HXZ replication"))
    p = lookup_paper_by_doi("10.1093/rfs/hhv059")
    assert p is not None
    assert p.title == "HXZ replication"
    # URL has DOI: prefix
    assert "DOI:10.1093" in captured[0]["path"]


def test_lookup_paper_by_arxiv(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        lookup_paper_by_arxiv,
    )
    captured = _patch_http(monkeypatch,
                              _paper_record(paper_id="ss-paper-2",
                                              title="ArXiv paper",
                                              arxiv_id="2606.12345"))
    p = lookup_paper_by_arxiv("2606.12345")
    assert p is not None
    assert p.arxiv_id == "2606.12345"
    assert "arXiv:2606" in captured[0]["path"]


def test_lookup_paper_404_returns_none(monkeypatch):
    from engine.agents.papers_curator.semantic_scholar import (
        lookup_paper_by_doi,
    )
    _patch_http(monkeypatch, None)
    assert lookup_paper_by_doi("ghost-doi") is None


# ─────────────────────────────────────────────────────────────────────
# Parser robustness
# ─────────────────────────────────────────────────────────────────────
def test_paper_summary_handles_missing_fields(monkeypatch):
    """SS returns None for abstract/year/venue/citationCount on older
    or thinly-indexed records. Parser must handle that without crash."""
    from engine.agents.papers_curator.semantic_scholar import (
        lookup_paper_by_doi,
    )
    _patch_http(monkeypatch, {
        "paperId": "p-thin",
        "title":   "Thin record",
        # Everything else missing
    })
    p = lookup_paper_by_doi("10.x/y")
    assert p is not None
    assert p.title == "Thin record"
    assert p.abstract == ""
    assert p.year is None
    assert p.venue == ""
    assert p.citation_count is None
    assert p.author_ids == ()
    assert p.doi == ""
