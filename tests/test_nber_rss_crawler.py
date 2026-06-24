"""tests/test_nber_rss_crawler.py — Stage A piece 5.

Tests RSS adapter + dedup + error isolation. Network is stubbed via
monkeypatching feedparser.parse so tests are deterministic + offline.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _fake_entry(*, title="A Working Paper",
                 link="https://www.nber.org/papers/w35254#fromrss",
                 summary="abstract text", published="",
                 authors=()):
    return {
        "title":     title,
        "link":      link,
        "summary":   summary,
        "published": published,
        "authors":   list(authors),
    }


def _fake_feed(entries, *, status=200, bozo=False, bozo_exception=None):
    return SimpleNamespace(
        entries        = entries,
        status         = status,
        bozo           = bozo,
        bozo_exception = bozo_exception,
        get            = lambda k, d=None: {
            "status": status, "bozo": bozo,
            "bozo_exception": bozo_exception,
        }.get(k, d),
    )


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Redirect store CACHE_PATH so save_new_candidates is sandboxed."""
    from engine.agents.papers_curator import store
    cache = tmp_path / "cache.jsonl"
    monkeypatch.setattr(store, "CACHE_PATH", cache)
    return cache


# ────────────────────────────────────────────────────────────────────
# WP-number extraction
# ────────────────────────────────────────────────────────────────────
def test_extract_wp_number_standard():
    from engine.agents.papers_curator import nber_rss_crawler as n
    assert n._extract_wp_number(
        "https://www.nber.org/papers/w35254#fromrss") == "w35254"
    assert n._extract_wp_number(
        "https://www.nber.org/papers/w99/foo") == "w99"
    # No paper marker → empty
    assert n._extract_wp_number(
        "https://www.nber.org/some/other") == ""
    assert n._extract_wp_number("") == ""


def test_extract_wp_number_case_insensitive():
    """NBER URLs always lowercase but be defensive."""
    from engine.agents.papers_curator import nber_rss_crawler as n
    assert n._extract_wp_number(
        "https://www.nber.org/Papers/W12345") == "w12345"


# ────────────────────────────────────────────────────────────────────
# Entry → PaperCandidate adapter
# ────────────────────────────────────────────────────────────────────
def test_to_paper_candidate_basic():
    from engine.agents.papers_curator import nber_rss_crawler as n
    e = _fake_entry()
    pc = n._to_paper_candidate(e, fetched_ts="2026-06-07T00:00:00Z")
    assert pc is not None
    assert pc.source == "nber"
    assert pc.source_id == "w35254"
    assert pc.title == "A Working Paper"
    assert pc.abstract == "abstract text"
    assert pc.abs_url == "https://www.nber.org/papers/w35254"
    assert pc.pdf_url == "https://www.nber.org/papers/w35254/w35254.pdf"
    assert "nber_working_paper" in pc.categories


def test_to_paper_candidate_uses_fetched_when_no_published():
    """NBER doesn't carry a date in RSS; published_ts should default
    to fetched_ts so downstream filters don't see empty dates."""
    from engine.agents.papers_curator import nber_rss_crawler as n
    pc = n._to_paper_candidate(
        _fake_entry(published=""),
        fetched_ts="2026-06-07T01:23:45Z",
    )
    assert pc.published_ts == "2026-06-07T01:23:45Z"


def test_to_paper_candidate_authors_empty_by_design():
    """NBER feed doesn't expose authors; we file empty and the LLM
    summarizer recovers from the PDF."""
    from engine.agents.papers_curator import nber_rss_crawler as n
    pc = n._to_paper_candidate(
        _fake_entry(authors=("ignored",)),  # even when present
        fetched_ts="t",
    )
    assert pc.authors == ()


def test_to_paper_candidate_drops_no_wp_number():
    """Entry whose link has no /papers/wNNN → None (defensive)."""
    from engine.agents.papers_curator import nber_rss_crawler as n
    e = _fake_entry(link="https://www.nber.org/topic/labor")
    assert n._to_paper_candidate(e, fetched_ts="t") is None


def test_to_paper_candidate_drops_empty_title():
    from engine.agents.papers_curator import nber_rss_crawler as n
    e = _fake_entry(title="")
    assert n._to_paper_candidate(e, fetched_ts="t") is None


# ────────────────────────────────────────────────────────────────────
# crawl_nber_rss — full fetch path
# ────────────────────────────────────────────────────────────────────
def test_crawl_basic_3_entries(monkeypatch):
    from engine.agents.papers_curator import nber_rss_crawler as n
    feed = _fake_feed([
        _fake_entry(title="T1", link="https://www.nber.org/papers/w1#x"),
        _fake_entry(title="T2", link="https://www.nber.org/papers/w2#x"),
        _fake_entry(title="T3", link="https://www.nber.org/papers/w3#x"),
    ])
    monkeypatch.setattr(
        "feedparser.parse", lambda *a, **kw: feed,
    )
    out = n.crawl_nber_rss()
    assert [c.source_id for c in out] == ["w1", "w2", "w3"]
    assert all(c.source == "nber" for c in out)


def test_crawl_skips_malformed_entries(monkeypatch):
    """Mix of good + bad entries → only good ones returned."""
    from engine.agents.papers_curator import nber_rss_crawler as n
    feed = _fake_feed([
        _fake_entry(title="ok", link="https://www.nber.org/papers/w11#x"),
        _fake_entry(title="", link="https://www.nber.org/papers/w12#x"),
        _fake_entry(title="ok2", link="https://www.nber.org/topic/labor"),
    ])
    monkeypatch.setattr("feedparser.parse", lambda *a, **kw: feed)
    out = n.crawl_nber_rss()
    assert [c.source_id for c in out] == ["w11"]


def test_crawl_403_returns_empty_no_retry(monkeypatch):
    """HTTP 403/404 = deterministic; no retry, return []."""
    from engine.agents.papers_curator import nber_rss_crawler as n
    feed = _fake_feed([], status=403)
    call_count = [0]
    def fake_parse(*a, **kw):
        call_count[0] += 1
        return feed
    monkeypatch.setattr("feedparser.parse", fake_parse)
    assert n.crawl_nber_rss(max_retries=3) == []
    assert call_count[0] == 1   # no retries for deterministic failure


def test_crawl_bozo_retries_then_gives_up(monkeypatch):
    """Bozo (transport issue) + no entries → retry up to max_retries,
    then return [] gracefully."""
    from engine.agents.papers_curator import nber_rss_crawler as n
    feed = _fake_feed([], status=200, bozo=True,
                       bozo_exception=Exception("dns"))
    call_count = [0]
    def fake_parse(*a, **kw):
        call_count[0] += 1
        return feed
    # Patch sleep so the test is fast
    import engine.agents.papers_curator.nber_rss_crawler as nm
    monkeypatch.setattr("feedparser.parse", fake_parse)
    monkeypatch.setattr("time.sleep", lambda *a: None)
    assert n.crawl_nber_rss(max_retries=2) == []
    assert call_count[0] == 3   # 1 initial + 2 retries


def test_crawl_bozo_with_entries_proceeds(monkeypatch):
    """Some RSS feeds set bozo=1 due to namespacing oddities but still
    deliver valid entries. We should USE the entries."""
    from engine.agents.papers_curator import nber_rss_crawler as n
    feed = _fake_feed(
        [_fake_entry(title="x", link="https://www.nber.org/papers/w55#x")],
        status=200, bozo=True, bozo_exception=Exception("ns"),
    )
    monkeypatch.setattr("feedparser.parse", lambda *a, **kw: feed)
    out = n.crawl_nber_rss()
    assert len(out) == 1


# ────────────────────────────────────────────────────────────────────
# crawl_and_persist_nber — full wrapper
# ────────────────────────────────────────────────────────────────────
def test_persist_writes_new_rows(monkeypatch, tmp_cache):
    from engine.agents.papers_curator import nber_rss_crawler as n
    feed = _fake_feed([
        _fake_entry(title="t1", link="https://www.nber.org/papers/w1#x"),
        _fake_entry(title="t2", link="https://www.nber.org/papers/w2#x"),
    ])
    monkeypatch.setattr("feedparser.parse", lambda *a, **kw: feed)

    result = n.crawl_and_persist_nber()
    assert result["n_fetched"] == 2
    assert result["n_new"] == 2
    assert result["errors"] == []

    # cache.jsonl actually has both rows
    rows = [json.loads(l) for l in
             tmp_cache.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    assert {r["source_id"] for r in rows} == {"w1", "w2"}


def test_persist_dedups_against_existing(monkeypatch, tmp_cache):
    """Pre-seed cache with one (nber, w99) row. Crawl finds w99 + w100;
    only w100 gets written."""
    from engine.agents.papers_curator import nber_rss_crawler as n
    tmp_cache.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache.write_text(json.dumps({
        "source": "nber", "source_id": "w99",
        "title": "old", "authors": [], "abstract": "",
        "abs_url": "", "pdf_url": "", "published_ts": "",
        "categories": [], "fetched_ts": "",
    }) + "\n", encoding="utf-8")

    feed = _fake_feed([
        _fake_entry(title="dup", link="https://www.nber.org/papers/w99#x"),
        _fake_entry(title="new", link="https://www.nber.org/papers/w100#x"),
    ])
    monkeypatch.setattr("feedparser.parse", lambda *a, **kw: feed)

    result = n.crawl_and_persist_nber()
    assert result["n_fetched"] == 2
    assert result["n_new"] == 1


def test_persist_handles_fetch_failure_gracefully(monkeypatch, tmp_cache):
    """403 returns empty crawl → result reports n_fetched=0 and
    no exception bubbles up."""
    from engine.agents.papers_curator import nber_rss_crawler as n
    monkeypatch.setattr(
        "feedparser.parse",
        lambda *a, **kw: _fake_feed([], status=403),
    )
    result = n.crawl_and_persist_nber()
    assert result["n_fetched"] == 0
    assert result["n_new"] == 0
    assert result["errors"] == []
