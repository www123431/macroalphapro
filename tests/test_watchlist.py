"""tests/test_watchlist.py — Stage A piece 2 tests.

Covers watchlist storage (load/save/add), author resolution, and the
crawler's integration with semantic_scholar + store.save_new_candidates.

Network is fully mocked — no real SS calls; no real cache.jsonl write
(uses tmp paths).
"""
from __future__ import annotations

from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Watchlist load / save / add
# ─────────────────────────────────────────────────────────────────────
def test_load_missing_file_returns_empty(tmp_path):
    from engine.agents.papers_curator.watchlist import load_watchlist
    assert load_watchlist(path=tmp_path / "does_not_exist.yaml") == []


def test_save_and_load_round_trip(tmp_path):
    from engine.agents.papers_curator.watchlist import (
        load_watchlist, save_watchlist, WatchlistAuthor,
    )
    p = tmp_path / "watchlist.yaml"
    save_watchlist([
        WatchlistAuthor(
            author_id="33344", name="Tim Bollerslev",
            rationale="VRP perspective",
            added_ts="2026-06-07T00:00:00Z", last_crawled_ts="",
        ),
        WatchlistAuthor(
            author_id="", name="Peter Carr",
            rationale="Options pricing",
            added_ts="2026-06-07T00:00:00Z", last_crawled_ts="",
        ),
    ], path=p)
    loaded = load_watchlist(path=p)
    assert len(loaded) == 2
    # Sorted by name (case-insensitive)
    assert loaded[0].name == "Peter Carr"
    assert loaded[1].name == "Tim Bollerslev"
    assert loaded[1].author_id == "33344"
    assert loaded[0].rationale == "Options pricing"


def test_add_author_idempotent_on_name(tmp_path):
    """Adding the same author twice (case-insensitive) is a no-op."""
    from engine.agents.papers_curator.watchlist import (
        add_author, load_watchlist,
    )
    p = tmp_path / "watchlist.yaml"
    add_author("Marcos Lopez de Prado",
                 rationale="multiple testing", path=p)
    add_author("marcos lopez de prado",
                 rationale="repeat", path=p)
    out = load_watchlist(path=p)
    assert len(out) == 1
    assert out[0].rationale == "multiple testing"   # first wins


def test_add_author_sets_added_ts(tmp_path):
    from engine.agents.papers_curator.watchlist import add_author
    p = tmp_path / "watchlist.yaml"
    a = add_author("Test Author", rationale="x", path=p)
    assert a.added_ts.endswith("Z")


# ─────────────────────────────────────────────────────────────────────
# Author resolution via SS
# ─────────────────────────────────────────────────────────────────────
def test_resolve_author_id_persists_back(monkeypatch, tmp_path):
    """resolve_author_id must persist the discovered SS id back to the
    watchlist so subsequent runs don't re-resolve."""
    from engine.agents.papers_curator import semantic_scholar as ss
    from engine.agents.papers_curator.watchlist import (
        add_author, resolve_author_id, load_watchlist,
    )

    class _A:
        author_id   = "33344"
        name        = "Tim Bollerslev"
        h_index     = 95
        paper_count = 320
        affiliations = ()
    monkeypatch.setattr(ss, "search_author_by_name",
                          lambda name, **kw: (_A(),))

    p = tmp_path / "watchlist.yaml"
    add_author("Tim Bollerslev", rationale="VRP", path=p)
    resolved = resolve_author_id("Tim Bollerslev", path=p)
    assert resolved == "33344"
    # Persisted
    out = load_watchlist(path=p)
    assert out[0].author_id == "33344"


def test_resolve_author_id_returns_none_on_no_match(monkeypatch, tmp_path):
    from engine.agents.papers_curator import semantic_scholar as ss
    from engine.agents.papers_curator.watchlist import (
        add_author, resolve_author_id,
    )
    monkeypatch.setattr(ss, "search_author_by_name",
                          lambda name, **kw: ())

    p = tmp_path / "watchlist.yaml"
    add_author("Ghost Author", rationale="x", path=p)
    assert resolve_author_id("Ghost Author", path=p) is None


# ─────────────────────────────────────────────────────────────────────
# Crawler — mocked SS + isolated store path
# ─────────────────────────────────────────────────────────────────────
def _fake_paper(*, paper_id, title, year=2024, venue="JF",
                  abstract="abstract", authors=("Test Author",)):
    """Build a mocked semantic_scholar.PaperSummary."""
    from engine.agents.papers_curator.semantic_scholar import PaperSummary
    return PaperSummary(
        paper_id=paper_id, title=title, abstract=abstract, year=year,
        venue=venue, citation_count=10,
        author_ids=("aid-1",), author_names=tuple(authors),
        doi="", arxiv_id="", url=f"https://x/{paper_id}",
    )


def _patch_crawler_deps(monkeypatch, *, ss_papers_by_author=None,
                          ss_search_results=None,
                          watchlist_path: Path = None):
    """Mock semantic_scholar + redirect store.CACHE_PATH into tmp."""
    from engine.agents.papers_curator import semantic_scholar as ss
    from engine.agents.papers_curator import store as store_mod

    def _fake_author_papers(author_id, *, limit=10, min_year=None):
        return ss_papers_by_author.get(author_id, ()) if ss_papers_by_author else ()

    def _fake_search(name, **kw):
        return ss_search_results.get(name, ()) if ss_search_results else ()

    monkeypatch.setattr(ss, "author_papers", _fake_author_papers)
    monkeypatch.setattr(ss, "search_author_by_name", _fake_search)
    # Redirect cache.jsonl to a fresh tmp file
    if watchlist_path is None:
        watchlist_path = Path("/tmp/test_watchlist_cache.jsonl")
    monkeypatch.setattr(store_mod, "CACHE_PATH", watchlist_path)


def test_crawler_empty_watchlist_returns_clean_zero(tmp_path, monkeypatch):
    from engine.agents.papers_curator.watchlist_crawler import crawl_watchlist
    _patch_crawler_deps(monkeypatch, watchlist_path=tmp_path / "cache.jsonl")
    result = crawl_watchlist(watchlist_path=tmp_path / "missing.yaml")
    assert result["n_authors_total"] == 0
    assert result["n_papers_new"] == 0
    assert result["errors"] == []


def test_crawler_resolves_then_fetches(tmp_path, monkeypatch):
    """Unresolved author → SS search → SS author_papers → cache.jsonl
    write. Verifies the full integration chain."""
    from engine.agents.papers_curator.watchlist import (
        add_author, load_watchlist,
    )
    from engine.agents.papers_curator.watchlist_crawler import crawl_watchlist
    from engine.agents.papers_curator.semantic_scholar import AuthorSummary

    cache = tmp_path / "cache.jsonl"
    wp    = tmp_path / "watchlist.yaml"
    add_author("Tim Bollerslev", rationale="VRP", path=wp)

    _patch_crawler_deps(monkeypatch,
        ss_search_results={"Tim Bollerslev": (
            AuthorSummary(author_id="33344", name="Tim Bollerslev",
                            h_index=95, paper_count=320,
                            affiliations=()),
        )},
        ss_papers_by_author={"33344": (
            _fake_paper(paper_id="p1", title="VRP carry 2024"),
            _fake_paper(paper_id="p2", title="Vol forecasting 2023"),
        )},
        watchlist_path=cache,
    )

    result = crawl_watchlist(watchlist_path=wp)
    assert result["n_authors_total"] == 1
    assert result["n_authors_crawled"] == 1
    assert result["n_papers_fetched"] == 2
    assert result["n_papers_new"] == 2

    # cache.jsonl now has 2 PaperCandidates with source=semantic_scholar
    import json
    rows = [json.loads(ln)
            for ln in cache.read_text(encoding="utf-8").strip().split("\n")]
    assert len(rows) == 2
    assert all(r["source"] == "semantic_scholar" for r in rows)
    assert "VRP carry 2024" in {r["title"] for r in rows}

    # Watchlist updated with resolved author_id
    out = load_watchlist(path=wp)
    assert out[0].author_id == "33344"
    assert out[0].last_crawled_ts != ""


def test_crawler_skips_recently_crawled(tmp_path, monkeypatch):
    """Author crawled within skip_recent_hours → skipped to save API
    quota."""
    from engine.agents.papers_curator.watchlist import (
        save_watchlist, WatchlistAuthor,
    )
    from engine.agents.papers_curator.watchlist_crawler import crawl_watchlist
    import datetime as _dt

    wp = tmp_path / "watchlist.yaml"
    save_watchlist([
        WatchlistAuthor(
            author_id="33344", name="Tim Bollerslev",
            rationale="VRP",
            added_ts="2026-06-01T00:00:00Z",
            # Crawled 1 hour ago
            last_crawled_ts=(_dt.datetime.utcnow()
                              - _dt.timedelta(hours=1)
                              ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    ], path=wp)

    _patch_crawler_deps(monkeypatch,
        ss_papers_by_author={"33344": (
            _fake_paper(paper_id="should-not-fetch", title="x"),
        )},
        watchlist_path=tmp_path / "cache.jsonl",
    )

    result = crawl_watchlist(watchlist_path=wp,
                                skip_recent_hours=24)
    assert result["n_authors_skipped"] == 1
    assert result["n_authors_crawled"] == 0
    assert result["n_papers_fetched"] == 0


def test_crawler_unresolved_authors_tracked(tmp_path, monkeypatch):
    """Authors SS can't find → tracked in result.unresolved_names but
    don't kill the run."""
    from engine.agents.papers_curator.watchlist import add_author
    from engine.agents.papers_curator.watchlist_crawler import crawl_watchlist

    wp = tmp_path / "watchlist.yaml"
    add_author("Ghost Author", rationale="x", path=wp)
    add_author("Another Ghost", rationale="x", path=wp)

    _patch_crawler_deps(monkeypatch,
        ss_search_results={},   # nothing resolves
        watchlist_path=tmp_path / "cache.jsonl",
    )
    result = crawl_watchlist(watchlist_path=wp)
    assert set(result["unresolved_names"]) == {"Ghost Author", "Another Ghost"}
    assert result["n_papers_new"] == 0


def test_crawler_papercandidate_has_correct_source_tag(tmp_path, monkeypatch):
    """All persisted candidates from watchlist must have
    source='semantic_scholar' (NOT 'arxiv') so downstream queries
    can distinguish arxiv RSS vs watchlist origin."""
    from engine.agents.papers_curator.watchlist_crawler import (
        _to_paper_candidate,
    )
    pc = _to_paper_candidate(
        _fake_paper(paper_id="p1", title="t"),
        fetched_ts="2026-06-07T00:00:00Z",
    )
    assert pc.source == "semantic_scholar"
    assert pc.source_id == "p1"
    assert pc.fetched_ts == "2026-06-07T00:00:00Z"


# ────────────────────────────────────────────────────────────────────
# Outage detection — added 2026-06-07 after failure-surface walk
# found silent failure on total network outage / bad SS key.
# ────────────────────────────────────────────────────────────────────
def test_crawler_emits_outage_signal_when_all_authors_unresolved(
    tmp_path, monkeypatch,
):
    """If 0/N authors crawl successfully AND all are unresolved, the
    result must carry an 'outage_suspected:' error so chief_of_staff
    sees the upstream failure (vs interpreting it as 'nothing to
    crawl')."""
    from engine.agents.papers_curator.watchlist import add_author
    from engine.agents.papers_curator.watchlist_crawler import (
        crawl_watchlist,
    )

    wp = tmp_path / "watchlist.yaml"
    add_author("Author A", rationale="x", path=wp)
    add_author("Author B", rationale="y", path=wp)

    _patch_crawler_deps(monkeypatch,
        ss_search_results={},      # SS returns nothing for anyone
        watchlist_path=tmp_path / "cache.jsonl",
    )
    result = crawl_watchlist(watchlist_path=wp)
    assert result["n_authors_crawled"] == 0
    assert len(result["unresolved_names"]) == 2
    assert any(e.startswith("outage_suspected:")
                 for e in result["errors"])


def test_crawler_no_outage_signal_when_partial_success(
    tmp_path, monkeypatch,
):
    """Mixed (some resolved, some not) → no outage signal — it's a
    REAL unresolved-author situation, not an upstream outage."""
    from engine.agents.papers_curator.watchlist import add_author
    from engine.agents.papers_curator.watchlist_crawler import (
        crawl_watchlist,
    )
    from engine.agents.papers_curator.semantic_scholar import (
        AuthorSummary,
    )

    wp = tmp_path / "watchlist.yaml"
    add_author("Resolves", rationale="x", path=wp)
    add_author("Ghost",    rationale="y", path=wp)

    _patch_crawler_deps(monkeypatch,
        ss_search_results={"Resolves": (
            AuthorSummary(author_id="A1", name="Resolves",
                            h_index=10, paper_count=5,
                            affiliations=()),
        )},
        ss_papers_by_author={"A1": (
            _fake_paper(paper_id="p1", title="t1"),
        )},
        watchlist_path=tmp_path / "cache.jsonl",
    )
    result = crawl_watchlist(watchlist_path=wp)
    # One crawled successfully → no outage signal even though
    # the other is unresolved
    assert result["n_authors_crawled"] == 1
    assert "Ghost" in result["unresolved_names"]
    assert not any(e.startswith("outage_suspected:")
                     for e in result["errors"])


def test_crawler_no_outage_signal_when_empty_watchlist(
    tmp_path, monkeypatch,
):
    """Empty watchlist → no outage signal (n_total=0, vacuous)."""
    from engine.agents.papers_curator.watchlist_crawler import (
        crawl_watchlist,
    )
    _patch_crawler_deps(monkeypatch,
        watchlist_path=tmp_path / "cache.jsonl")
    result = crawl_watchlist(watchlist_path=tmp_path / "missing.yaml")
    assert result["n_authors_total"] == 0
    assert not any(e.startswith("outage_suspected:")
                     for e in result["errors"])
