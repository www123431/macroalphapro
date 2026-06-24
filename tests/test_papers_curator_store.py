"""tests/test_papers_curator_store.py — cache.jsonl persistence.

Tests for load_cache + save_new_candidates including defensive
handling of malformed / partial-field rows (added 2026-06-07 after
failure-surface walk).
"""
from __future__ import annotations

import json

import pytest


def _seed_cache(p, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    from engine.agents.papers_curator import store
    cache = tmp_path / "cache.jsonl"
    monkeypatch.setattr(store, "CACHE_PATH", cache)
    return cache


# ────────────────────────────────────────────────────────────────────
# Defensive load — partial-field rows must be dropped, not loaded
# ────────────────────────────────────────────────────────────────────
def test_load_cache_drops_row_missing_source(tmp_cache):
    """A row that parses as valid JSON but lacks `source` would dedup
    against ('', source_id) — corrupting future writes. Drop it."""
    from engine.agents.papers_curator.store import load_cache
    _seed_cache(tmp_cache, [
        json.dumps({"source": "arxiv", "source_id": "good1",
                     "title": "t1"}),
        json.dumps({"source_id": "missing_source", "title": "t2"}),
    ])
    cache = load_cache()
    assert [c.source_id for c in cache] == ["good1"]


def test_load_cache_drops_row_missing_source_id(tmp_cache):
    """Same logic for source_id."""
    from engine.agents.papers_curator.store import load_cache
    _seed_cache(tmp_cache, [
        json.dumps({"source": "arxiv", "source_id": "good1",
                     "title": "t"}),
        json.dumps({"source": "arxiv", "title": "no source_id"}),
    ])
    cache = load_cache()
    assert [c.source_id for c in cache] == ["good1"]


def test_load_cache_drops_empty_string_keys(tmp_cache):
    """An explicit empty string for source/source_id is still missing
    by our definition."""
    from engine.agents.papers_curator.store import load_cache
    _seed_cache(tmp_cache, [
        json.dumps({"source": "arxiv", "source_id": "good1",
                     "title": "t"}),
        json.dumps({"source": "", "source_id": "x", "title": "t"}),
        json.dumps({"source": "arxiv", "source_id": "", "title": "t"}),
    ])
    cache = load_cache()
    assert [c.source_id for c in cache] == ["good1"]


def test_load_cache_skips_malformed_json(tmp_cache):
    """Pre-existing behavior — bad JSON lines are warned + skipped,
    valid lines still load."""
    from engine.agents.papers_curator.store import load_cache
    _seed_cache(tmp_cache, [
        json.dumps({"source": "arxiv", "source_id": "a",
                     "title": "t1"}),
        "this is not json",
        '{"source": "arxiv", "source_id": "b",',   # truncated
        json.dumps({"source": "arxiv", "source_id": "c",
                     "title": "t3"}),
    ])
    cache = load_cache()
    assert [c.source_id for c in cache] == ["a", "c"]


def test_load_cache_empty_file_returns_empty(tmp_cache):
    tmp_cache.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache.write_text("", encoding="utf-8")
    from engine.agents.papers_curator.store import load_cache
    assert load_cache() == []


def test_load_cache_missing_file_returns_empty(tmp_path, monkeypatch):
    """Fresh-install path — cache.jsonl doesn't exist yet."""
    from engine.agents.papers_curator import store
    monkeypatch.setattr(store, "CACHE_PATH",
                          tmp_path / "never_existed.jsonl")
    assert store.load_cache() == []
