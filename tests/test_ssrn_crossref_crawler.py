"""tests/test_ssrn_crossref_crawler.py — Stage A piece 5 follow-up.

Tests CrossRef → PaperCandidate adapter + crawl orchestration. Network
is stubbed via monkeypatching _crossref_get so tests are offline +
deterministic.
"""
from __future__ import annotations

import json

import pytest


def _fake_item(*, doi="10.2139/ssrn.1234567", title="A Paper",
                 abstract="An abstract.",
                 authors=(("Jane", "Doe"), ("John", "Smith")),
                 deposited=(2026, 6, 7)):
    return {
        "DOI":       doi,
        "title":     [title] if title else [],
        "abstract":  abstract,
        "author":    [{"given": g, "family": f} for g, f in authors],
        "deposited": {"date-parts": [list(deposited)]},
    }


def _wrap_message(items):
    return {"message": {"items": list(items)}}


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    from engine.agents.papers_curator import store
    cache = tmp_path / "cache.jsonl"
    monkeypatch.setattr(store, "CACHE_PATH", cache)
    return cache


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Don't sleep in tests."""
    monkeypatch.setattr("time.sleep", lambda *a: None)


# ────────────────────────────────────────────────────────────────────
# Markup stripping
# ────────────────────────────────────────────────────────────────────
def test_strip_markup_removes_html_tags():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    assert sc._strip_markup("<span>hello</span> <b>world</b>") == "hello world"
    assert sc._strip_markup("<jats:p>x</jats:p>") == "x"


def test_strip_markup_decodes_entities():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    assert sc._strip_markup("&lt;b&gt;TITLE&lt;/b&gt;") == "<b>TITLE</b>"
    # Combined with tag stripping (entities decoded first leave bare text)
    assert sc._strip_markup("AT&amp;T &lt;Inc&gt;") == "AT&T <Inc>"


def test_strip_markup_collapses_whitespace():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    assert sc._strip_markup("a   b\n\tc") == "a b c"


def test_strip_markup_empty_input():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    assert sc._strip_markup("") == ""
    assert sc._strip_markup(None) == ""


# ────────────────────────────────────────────────────────────────────
# Author extraction
# ────────────────────────────────────────────────────────────────────
def test_extract_authors_standard():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    item = _fake_item(authors=(("Jane", "Doe"), ("John", "Smith")))
    assert sc._extract_authors(item) == ("Jane Doe", "John Smith")


def test_extract_authors_handles_family_only():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    assert sc._extract_authors(
        {"author": [{"family": "Solo"}]}) == ("Solo",)


def test_extract_authors_skips_empty():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    assert sc._extract_authors({"author": [{}, {"family": "X"}]}) == ("X",)


def test_extract_authors_empty_list():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    assert sc._extract_authors({}) == ()
    assert sc._extract_authors({"author": []}) == ()


# ────────────────────────────────────────────────────────────────────
# Item → PaperCandidate adapter
# ────────────────────────────────────────────────────────────────────
def test_to_paper_candidate_basic():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    pc = sc._to_paper_candidate(_fake_item(),
                                  fetched_ts="2026-06-07T00:00:00Z")
    assert pc is not None
    assert pc.source == "ssrn"
    assert pc.source_id == "10.2139/ssrn.1234567"
    assert pc.title == "A Paper"
    assert pc.abstract == "An abstract."
    assert "ssrn_via_crossref" in pc.categories
    assert pc.published_ts == "2026-06-07T00:00:00Z"
    assert "abstract_id=1234567" in pc.abs_url


def test_to_paper_candidate_drops_missing_doi():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    item = _fake_item(doi="")
    assert sc._to_paper_candidate(item, fetched_ts="t") is None


def test_to_paper_candidate_drops_missing_title():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    item = _fake_item(title="")
    assert sc._to_paper_candidate(item, fetched_ts="t") is None


def test_to_paper_candidate_strips_markup_in_title():
    """SSRN CrossRef titles sometimes ship with &lt;b&gt; wrappers."""
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    item = _fake_item(title="&lt;b&gt;TITLE&lt;/b&gt; tail")
    pc = sc._to_paper_candidate(item, fetched_ts="t")
    assert pc.title == "<b>TITLE</b> tail"


def test_to_paper_candidate_strips_markup_in_abstract():
    """Abstracts often ship with <jats:p>…</jats:p>."""
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    item = _fake_item(abstract="<jats:p>This is the abstract.</jats:p>")
    pc = sc._to_paper_candidate(item, fetched_ts="t")
    assert pc.abstract == "This is the abstract."


def test_to_paper_candidate_uses_deposited_date():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    item = _fake_item(deposited=(2026, 6, 7))
    pc = sc._to_paper_candidate(item, fetched_ts="ZZZ")
    assert pc.published_ts == "2026-06-07T00:00:00Z"


def test_to_paper_candidate_falls_back_when_deposited_malformed():
    """Missing date-parts → published_ts = fetched_ts."""
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    item = _fake_item()
    item["deposited"] = {"date-parts": [[]]}
    pc = sc._to_paper_candidate(item, fetched_ts="2026-06-07T05:00:00Z")
    assert pc.published_ts == "2026-06-07T05:00:00Z"


def test_to_paper_candidate_non_ssrn_doi_leaves_url_empty():
    """A DOI without 'ssrn.' shouldn't get an SSRN landing URL guess."""
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    item = _fake_item(doi="10.2139/unexpected.x")
    pc = sc._to_paper_candidate(item, fetched_ts="t")
    assert pc is not None   # still a valid candidate, just no URL
    assert pc.abs_url == ""


# ────────────────────────────────────────────────────────────────────
# crawl_ssrn_via_crossref — orchestration
# ────────────────────────────────────────────────────────────────────
def test_crawl_returns_candidates(monkeypatch):
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    items = [
        _fake_item(doi="10.2139/ssrn.1", title="t1"),
        _fake_item(doi="10.2139/ssrn.2", title="t2"),
        _fake_item(doi="10.2139/ssrn.3", title="t3"),
    ]
    monkeypatch.setattr(sc, "_crossref_get",
        lambda url, **kw: _wrap_message(items))
    out = sc.crawl_ssrn_via_crossref(max_results=10)
    assert [c.source_id for c in out] == [
        "10.2139/ssrn.1", "10.2139/ssrn.2", "10.2139/ssrn.3"]


def test_crawl_respects_max_results(monkeypatch):
    """Cap on items before adapting."""
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    items = [_fake_item(doi=f"10.2139/ssrn.{i}", title=f"t{i}")
              for i in range(10)]
    monkeypatch.setattr(sc, "_crossref_get",
        lambda url, **kw: _wrap_message(items))
    out = sc.crawl_ssrn_via_crossref(max_results=3)
    assert len(out) == 3


def test_crawl_returns_empty_on_api_failure(monkeypatch):
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    monkeypatch.setattr(sc, "_crossref_get", lambda url, **kw: None)
    assert sc.crawl_ssrn_via_crossref() == []


def test_crawl_skips_malformed_items(monkeypatch):
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    items = [
        _fake_item(doi="10.2139/ssrn.ok"),
        {"DOI": "", "title": ["bad-no-doi"]},
        _fake_item(doi="10.2139/ssrn.also_ok", title="also_ok"),
    ]
    monkeypatch.setattr(sc, "_crossref_get",
        lambda url, **kw: _wrap_message(items))
    out = sc.crawl_ssrn_via_crossref()
    assert [c.source_id for c in out] == [
        "10.2139/ssrn.ok", "10.2139/ssrn.also_ok"]


# ────────────────────────────────────────────────────────────────────
# crawl_and_persist_ssrn — full wrapper
# ────────────────────────────────────────────────────────────────────
def test_persist_writes_new_rows(monkeypatch, tmp_cache):
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    items = [_fake_item(doi=f"10.2139/ssrn.{i}", title=f"t{i}")
              for i in (1, 2)]
    monkeypatch.setattr(sc, "_crossref_get",
        lambda url, **kw: _wrap_message(items))

    result = sc.crawl_and_persist_ssrn()
    assert result["n_fetched"] == 2
    assert result["n_new"] == 2
    rows = [json.loads(l) for l in
             tmp_cache.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    assert {r["source_id"] for r in rows} == {
        "10.2139/ssrn.1", "10.2139/ssrn.2"}


def test_persist_dedups_against_existing(monkeypatch, tmp_cache):
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    tmp_cache.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache.write_text(json.dumps({
        "source": "ssrn", "source_id": "10.2139/ssrn.old",
        "title": "old", "authors": [], "abstract": "", "abs_url": "",
        "pdf_url": "", "published_ts": "", "categories": [],
        "fetched_ts": "",
    }) + "\n", encoding="utf-8")
    items = [
        _fake_item(doi="10.2139/ssrn.old", title="dup"),
        _fake_item(doi="10.2139/ssrn.fresh", title="new"),
    ]
    monkeypatch.setattr(sc, "_crossref_get",
        lambda url, **kw: _wrap_message(items))

    result = sc.crawl_and_persist_ssrn()
    assert result["n_fetched"] == 2
    assert result["n_new"] == 1


def test_persist_handles_api_failure_gracefully(monkeypatch, tmp_cache):
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    monkeypatch.setattr(sc, "_crossref_get", lambda url, **kw: None)
    result = sc.crawl_and_persist_ssrn()
    assert result["n_fetched"] == 0
    assert result["n_new"] == 0
    assert result["errors"] == []


# ────────────────────────────────────────────────────────────────────
# Query URL composition
# ────────────────────────────────────────────────────────────────────
def test_build_query_url_includes_required_params():
    from engine.agents.papers_curator import ssrn_crossref_crawler as sc
    u = sc._build_query_url(from_deposit_date="2026-05-01", rows=50)
    assert "from-deposit-date" in u
    assert "rows=50" in u
    assert "sort=deposited" in u
    assert "order=desc" in u
    assert u.startswith(sc._CROSSREF_API)
