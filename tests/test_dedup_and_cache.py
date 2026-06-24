"""Tests for cross-cron dedup (漏洞 5) + nominate metadata cache (漏洞 6)."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest import mock

import pytest


# ── _is_dup_against_recent_discovery (漏洞 5) ──────────────────────────

@pytest.fixture
def isolated_log(monkeypatch, tmp_path):
    from engine.research.discovery import discovery_pipeline as dp
    log = tmp_path / "discovery_log.jsonl"
    monkeypatch.setattr(dp, "DISCOVERY_LOG", log)
    # Reset cache for each test
    dp._RECENT_DISCOVERY_CACHE = (0.0, [])
    return {"log": log, "dp": dp}


def test_dup_recent_returns_false_when_log_missing(isolated_log):
    assert isolated_log["dp"]._is_dup_against_recent_discovery("Some Title") is False


def test_dup_recent_returns_true_for_recent_match(isolated_log):
    """Same title processed yesterday → dup hit."""
    log = isolated_log["log"]
    yesterday = (datetime.datetime.utcnow()
                  - datetime.timedelta(days=1)).isoformat()
    log.write_text(
        json.dumps({"title": "Cross-Asset Carry Strategies", "ts": yesterday}) + "\n",
        encoding="utf-8",
    )
    assert isolated_log["dp"]._is_dup_against_recent_discovery(
        "Cross-Asset Carry Strategies",
    ) is True


def test_dup_recent_ignores_old_entries(isolated_log):
    """Same title processed 100 days ago → NOT a dup at default 90d."""
    log = isolated_log["log"]
    old = (datetime.datetime.utcnow()
            - datetime.timedelta(days=100)).isoformat()
    log.write_text(
        json.dumps({"title": "Old Title", "ts": old}) + "\n",
        encoding="utf-8",
    )
    # Default 90d window
    assert isolated_log["dp"]._is_dup_against_recent_discovery(
        "Old Title",
    ) is False


def test_dup_recent_handles_token_overlap(isolated_log):
    """Title with ≥70% token overlap is dup."""
    log = isolated_log["log"]
    now = datetime.datetime.utcnow().isoformat()
    log.write_text(
        json.dumps({"title": "Cross-Asset Carry Strategies", "ts": now}) + "\n",
        encoding="utf-8",
    )
    # 4 of 5 tokens match (cross-asset, carry, strategies, in, futures
    # minus stopwords) - actually let me compute: "carry strategies"
    # token-overlap is 100% with "cross-asset carry strategies"
    assert isolated_log["dp"]._is_dup_against_recent_discovery(
        "Cross-Asset Carry Strategies (extended)",
    ) is True


def test_dup_recent_no_match_for_unrelated_title(isolated_log):
    log = isolated_log["log"]
    now = datetime.datetime.utcnow().isoformat()
    log.write_text(
        json.dumps({"title": "Momentum Returns", "ts": now}) + "\n",
        encoding="utf-8",
    )
    assert isolated_log["dp"]._is_dup_against_recent_discovery(
        "Quality Junk Premium",
    ) is False


def test_dup_recent_returns_false_for_empty_title(isolated_log):
    assert isolated_log["dp"]._is_dup_against_recent_discovery("") is False
    assert isolated_log["dp"]._is_dup_against_recent_discovery("  ") is False


# ── Metadata cache (漏洞 6) ───────────────────────────────────────────

@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    from engine.research.discovery import review_ui as rui
    monkeypatch.setattr(rui, "METADATA_CACHE_DIR", tmp_path)
    return {"dir": tmp_path, "rui": rui}


def test_metadata_cache_path_stable(isolated_cache):
    rui = isolated_cache["rui"]
    p1 = rui._metadata_cache_path({"type": "doi", "id": "10.1/x"})
    p2 = rui._metadata_cache_path({"type": "doi", "id": "10.1/x"})
    assert p1 == p2


def test_metadata_cache_path_differs_for_different_ids(isolated_cache):
    rui = isolated_cache["rui"]
    p1 = rui._metadata_cache_path({"type": "doi", "id": "10.1/x"})
    p2 = rui._metadata_cache_path({"type": "doi", "id": "10.1/y"})
    assert p1 != p2


def test_metadata_cache_save_load_roundtrip(isolated_cache):
    rui = isolated_cache["rui"]
    ident = {"type": "doi", "id": "10.1/x"}
    meta = {"title": "Carry Paper", "venue": "JFE", "abstract": "X"}
    rui._save_metadata_cached(ident, meta)
    loaded = rui._load_metadata_cached(ident)
    assert loaded is not None
    assert loaded["title"] == "Carry Paper"
    # Cache metadata field stripped on load
    assert "_cached_at" not in loaded


def test_metadata_cache_returns_none_for_missing(isolated_cache):
    rui = isolated_cache["rui"]
    assert rui._load_metadata_cached({"type": "doi", "id": "10.1/missing"}) is None


def test_metadata_cache_returns_none_for_stale(isolated_cache):
    """Entry older than TTL → cache miss."""
    rui = isolated_cache["rui"]
    ident = {"type": "doi", "id": "10.1/stale"}
    cache_path = rui._metadata_cache_path(ident)
    # Write entry with timestamp 48h ago (beyond default 24h TTL)
    stale_ts = (datetime.datetime.utcnow()
                 - datetime.timedelta(hours=48)).isoformat() + "Z"
    cache_path.write_text(
        json.dumps({"title": "X", "_cached_at": stale_ts}),
        encoding="utf-8",
    )
    assert rui._load_metadata_cached(ident) is None


def test_fetch_metadata_uses_cache(monkeypatch, isolated_cache):
    """When cache hit, _fetch_crossref etc must NOT be called."""
    rui = isolated_cache["rui"]
    ident = {"type": "doi", "id": "10.1/x"}
    rui._save_metadata_cached(ident, {"title": "Cached"})

    def _should_not_call(*a, **kw):
        pytest.fail("fetcher should not be called when cache hit")
    monkeypatch.setattr(rui, "_fetch_crossref", _should_not_call)

    result = rui.fetch_metadata(ident)
    assert result is not None
    assert result["title"] == "Cached"


def test_fetch_metadata_saves_to_cache_on_success(monkeypatch, isolated_cache):
    rui = isolated_cache["rui"]
    ident = {"type": "doi", "id": "10.1/newpaper"}
    monkeypatch.setattr(
        rui, "_fetch_crossref",
        lambda doi: {"title": "Fetched", "venue": "X"},
    )
    rui.fetch_metadata(ident)
    # Subsequent call should hit cache
    cached = rui._load_metadata_cached(ident)
    assert cached is not None
    assert cached["title"] == "Fetched"


def test_fetch_metadata_returns_none_when_fetcher_fails(
    monkeypatch, isolated_cache,
):
    rui = isolated_cache["rui"]
    monkeypatch.setattr(rui, "_fetch_crossref", lambda doi: None)
    result = rui.fetch_metadata({"type": "doi", "id": "10.1/fail"})
    assert result is None
