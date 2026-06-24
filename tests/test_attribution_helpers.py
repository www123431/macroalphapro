"""tests/test_attribution_helpers.py — Layer 4 piece 3a.

Tests the join-helper primitives. Uses tmp paths + lru_cache clears
to keep tests isolated.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────
def _seed_cache(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture(autouse=True)
def _clear_caches():
    """Every test runs with cleared lru_cache so prior tests don't
    leak source maps into this one."""
    from engine.agents.attribution import helpers
    helpers.clear_caches()
    yield
    helpers.clear_caches()


# ─────────────────────────────────────────────────────────────────────
# get_paper_source — handles composite and bare ids
# ─────────────────────────────────────────────────────────────────────
def test_get_paper_source_composite_key(tmp_path, monkeypatch):
    from engine.agents.attribution import helpers
    cache = tmp_path / "cache.jsonl"
    _seed_cache(cache, [
        {"source": "arxiv", "source_id": "2606.12345",
         "title": "x", "authors": ["A"]},
        {"source": "semantic_scholar", "source_id": "ss-pid-1",
         "title": "y", "authors": ["B"]},
    ])
    monkeypatch.setattr(helpers, "_CACHE_PATH", cache)
    helpers.clear_caches()

    assert helpers.get_paper_source("arxiv/2606.12345") == "arxiv"
    assert helpers.get_paper_source("semantic_scholar/ss-pid-1") == "semantic_scholar"


def test_get_paper_source_bare_id(tmp_path, monkeypatch):
    """A's synthesizes_paper_ids sometimes carry bare ids (no source
    prefix). The lookup must handle that too."""
    from engine.agents.attribution import helpers
    cache = tmp_path / "cache.jsonl"
    _seed_cache(cache, [
        {"source": "arxiv", "source_id": "2606.12345",
         "title": "x", "authors": []},
    ])
    monkeypatch.setattr(helpers, "_CACHE_PATH", cache)
    helpers.clear_caches()
    assert helpers.get_paper_source("2606.12345") == "arxiv"


def test_get_paper_source_unknown_returns_none(tmp_path, monkeypatch):
    """Cache miss → None (caller treats as 'don't know which source')."""
    from engine.agents.attribution import helpers
    cache = tmp_path / "cache.jsonl"
    _seed_cache(cache, [])
    monkeypatch.setattr(helpers, "_CACHE_PATH", cache)
    helpers.clear_caches()
    assert helpers.get_paper_source("arxiv/never_seen") is None


def test_get_paper_source_empty_returns_none():
    from engine.agents.attribution import helpers
    assert helpers.get_paper_source("") is None


def test_get_paper_source_missing_cache_returns_none(tmp_path, monkeypatch):
    """Missing cache.jsonl → None gracefully."""
    from engine.agents.attribution import helpers
    monkeypatch.setattr(helpers, "_CACHE_PATH",
                          tmp_path / "does_not_exist.jsonl")
    helpers.clear_caches()
    assert helpers.get_paper_source("arxiv/x") is None


# ─────────────────────────────────────────────────────────────────────
# get_paper_authors
# ─────────────────────────────────────────────────────────────────────
def test_get_paper_authors_returns_tuple(tmp_path, monkeypatch):
    from engine.agents.attribution import helpers
    cache = tmp_path / "cache.jsonl"
    _seed_cache(cache, [
        {"source": "arxiv", "source_id": "2606.x",
         "title": "t", "authors": ["McLean", "Pontiff"]},
    ])
    monkeypatch.setattr(helpers, "_CACHE_PATH", cache)
    helpers.clear_caches()
    auths = helpers.get_paper_authors("arxiv/2606.x")
    assert auths == ("McLean", "Pontiff")


def test_get_paper_authors_missing_returns_empty(tmp_path, monkeypatch):
    from engine.agents.attribution import helpers
    cache = tmp_path / "cache.jsonl"
    _seed_cache(cache, [])
    monkeypatch.setattr(helpers, "_CACHE_PATH", cache)
    helpers.clear_caches()
    assert helpers.get_paper_authors("arxiv/x") == ()


# ─────────────────────────────────────────────────────────────────────
# paper_from_watchlist_authors — joining cache to watchlist
# ─────────────────────────────────────────────────────────────────────
def test_paper_from_watchlist_authors_returns_intersection(tmp_path, monkeypatch):
    """Paper with 3 authors, 1 on watchlist → returns 1-tuple."""
    from engine.agents.attribution import helpers
    from engine.agents.papers_curator.watchlist import (
        add_author, load_watchlist,
    )

    cache = tmp_path / "cache.jsonl"
    _seed_cache(cache, [
        {"source": "semantic_scholar", "source_id": "p1",
         "title": "t", "authors": ["Tim Bollerslev",
                                     "Some Coauthor",
                                     "Another Coauthor"]},
    ])
    wl = tmp_path / "watchlist.yaml"
    add_author("Tim Bollerslev", rationale="VRP", path=wl)

    monkeypatch.setattr(helpers, "_CACHE_PATH", cache)
    # Redirect watchlist load to tmp file
    from engine.agents.papers_curator import watchlist as wl_mod
    monkeypatch.setattr(wl_mod, "WATCHLIST_PATH", wl)
    helpers.clear_caches()

    hits = helpers.paper_from_watchlist_authors("semantic_scholar/p1")
    assert hits == ("Tim Bollerslev",)


def test_paper_from_watchlist_no_overlap_returns_empty(tmp_path, monkeypatch):
    from engine.agents.attribution import helpers
    from engine.agents.papers_curator.watchlist import add_author

    cache = tmp_path / "cache.jsonl"
    _seed_cache(cache, [
        {"source": "arxiv", "source_id": "p1",
         "title": "t", "authors": ["Random Researcher"]},
    ])
    wl = tmp_path / "watchlist.yaml"
    add_author("Tim Bollerslev", rationale="VRP", path=wl)

    monkeypatch.setattr(helpers, "_CACHE_PATH", cache)
    from engine.agents.papers_curator import watchlist as wl_mod
    monkeypatch.setattr(wl_mod, "WATCHLIST_PATH", wl)
    helpers.clear_caches()

    assert helpers.paper_from_watchlist_authors("arxiv/p1") == ()


def test_paper_from_watchlist_case_insensitive(tmp_path, monkeypatch):
    """Author name match must be case-insensitive so 'Tim Bollerslev'
    matches 'tim bollerslev' (some cache writes lowercase)."""
    from engine.agents.attribution import helpers
    from engine.agents.papers_curator.watchlist import add_author

    cache = tmp_path / "cache.jsonl"
    _seed_cache(cache, [
        {"source": "ss", "source_id": "p1",
         "title": "t", "authors": ["tim bollerslev"]},
    ])
    wl = tmp_path / "watchlist.yaml"
    add_author("Tim Bollerslev", rationale="VRP", path=wl)

    monkeypatch.setattr(helpers, "_CACHE_PATH", cache)
    from engine.agents.papers_curator import watchlist as wl_mod
    monkeypatch.setattr(wl_mod, "WATCHLIST_PATH", wl)
    helpers.clear_caches()

    assert helpers.paper_from_watchlist_authors("ss/p1") == ("tim bollerslev",)


# ─────────────────────────────────────────────────────────────────────
# Cache invalidation
# ─────────────────────────────────────────────────────────────────────
def test_clear_caches_picks_up_new_cache_rows(tmp_path, monkeypatch):
    """Mid-process cache mutation followed by clear_caches() must
    surface the new rows."""
    from engine.agents.attribution import helpers
    cache = tmp_path / "cache.jsonl"
    _seed_cache(cache, [
        {"source": "arxiv", "source_id": "old",
         "title": "x", "authors": []},
    ])
    monkeypatch.setattr(helpers, "_CACHE_PATH", cache)
    helpers.clear_caches()

    # First read populates cache
    assert helpers.get_paper_source("arxiv/old") == "arxiv"
    assert helpers.get_paper_source("arxiv/new") is None

    # Append new row + clear caches
    with cache.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"source": "arxiv", "source_id": "new",
                              "title": "y", "authors": []}) + "\n")
    helpers.clear_caches()

    assert helpers.get_paper_source("arxiv/new") == "arxiv"
