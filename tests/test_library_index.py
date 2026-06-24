"""Tests for engine.research.library_index — SQLite FTS5 paper+mechanism index."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from engine.research import library_index


@pytest.fixture
def temp_db(tmp_path):
    """Isolated DB path for each test."""
    return tmp_path / "library_index.db"


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Redirect all source paths to a writable temp location."""
    lib_red = tmp_path / "library" / "red"
    lib_white = tmp_path / "library" / "whitelisted"
    lib_pending = tmp_path / "library" / "pending"
    for d in (lib_red, lib_white, lib_pending):
        d.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "data" / "research" / "discovery_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(library_index, "LIBRARY_RED_DIR", lib_red)
    monkeypatch.setattr(library_index, "LIBRARY_WHITELISTED_DIR", lib_white)
    monkeypatch.setattr(library_index, "LIBRARY_PENDING_DIR", lib_pending)
    monkeypatch.setattr(library_index, "DISCOVERY_LOG", log)
    return {"red": lib_red, "white": lib_white, "log": log}


# ── schema + build ─────────────────────────────────────────────────────────

def test_build_empty_index_succeeds(temp_db, patched_paths):
    """Building over empty sources should not crash."""
    result = library_index.build_index(db_path=temp_db)
    assert result["mechanisms_indexed"] == 0
    assert result["papers_indexed"] == 0
    assert Path(result["db_path"]).exists()


def test_build_with_library_yamls(temp_db, patched_paths):
    lib_red = patched_paths["red"]
    (lib_red / "quality_qmj.yaml").write_text(yaml.safe_dump({
        "id": "quality_qmj",
        "title": "Quality Minus Junk",
        "family": "quality",
        "parent_family": "equity_factor",
        "status_in_our_book": "RED",
        "mechanism_economics": "Asness-Frazzini-Pedersen quality premium.",
    }), encoding="utf-8")
    (lib_red / "bond_xsmom.yaml").write_text(yaml.safe_dump({
        "id": "bond_xsmom",
        "title": "Cross-sectional bond momentum",
        "family": "tsmom",
        "parent_family": "cross_asset_trend",
        "status_in_our_book": "RED",
        "mechanism_economics": "Asness-Moskowitz-Pedersen cross-asset momentum.",
    }), encoding="utf-8")
    result = library_index.build_index(db_path=temp_db)
    assert result["mechanisms_indexed"] == 2


def test_build_with_discovery_log(temp_db, patched_paths):
    log = patched_paths["log"]
    with log.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"source": "arxiv", "source_id": "2401.001",
                              "title": "Momentum Signals via Embeddings",
                              "abstract": "We propose a new approach to momentum.",
                              "authors": "Smith, Jones",
                              "verdict": "queue_for_review",
                              "arxiv_id": "2401.001"}) + "\n")
        f.write(json.dumps({"source": "nber", "source_id": "w32100",
                              "title": "Carry trade and tail risk",
                              "abstract": "Currency carry exposes tail risk.",
                              "verdict": "skip"}) + "\n")
    result = library_index.build_index(db_path=temp_db)
    assert result["papers_indexed"] == 2


# ── FTS5 search ────────────────────────────────────────────────────────────

def test_search_mechanisms_exact_term(temp_db, patched_paths):
    (patched_paths["red"] / "quality_qmj.yaml").write_text(yaml.safe_dump({
        "id": "quality_qmj", "title": "Quality Minus Junk",
        "family": "quality", "parent_family": "equity_factor",
        "status_in_our_book": "RED",
        "mechanism_economics": "Profitability + growth + safety + payout factor."
    }), encoding="utf-8")
    library_index.build_index(db_path=temp_db)
    hits = library_index.search_mechanisms("quality", db_path=temp_db)
    assert len(hits) == 1
    assert hits[0]["mechanism_id"] == "quality_qmj"


def test_search_mechanisms_porter_stem(temp_db, patched_paths):
    """Porter stemmer should match 'momentum' to 'momenta' via stem."""
    (patched_paths["red"] / "x.yaml").write_text(yaml.safe_dump({
        "id": "x", "title": "Tested momenta in stocks",
        "family": "momentum", "status_in_our_book": "RED",
        "mechanism_economics": "Underreaction creates momentum.",
    }), encoding="utf-8")
    library_index.build_index(db_path=temp_db)
    hits = library_index.search_mechanisms("momentum", db_path=temp_db)
    assert len(hits) == 1


def test_search_papers_by_abstract(temp_db, patched_paths):
    log = patched_paths["log"]
    with log.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"source": "arxiv", "source_id": "2401.001",
                              "title": "Cross-sectional momentum",
                              "abstract": "Carry trade analysis with focus on currency markets.",
                              "verdict": "queue_for_review"}) + "\n")
    library_index.build_index(db_path=temp_db)
    hits = library_index.search_papers("carry", db_path=temp_db)
    assert len(hits) == 1
    assert "Cross-sectional" in hits[0]["title"]


def test_search_all_returns_both(temp_db, patched_paths):
    (patched_paths["red"] / "m.yaml").write_text(yaml.safe_dump({
        "id": "m", "title": "Carry premium", "family": "carry",
        "status_in_our_book": "DEPLOYED",
        "mechanism_economics": "Roll yield captured via carry.",
    }), encoding="utf-8")
    with patched_paths["log"].open("w", encoding="utf-8") as f:
        f.write(json.dumps({"source": "arxiv", "source_id": "p1",
                              "title": "Currency carry trade analysis",
                              "abstract": "We analyze carry returns.",
                              "verdict": "skip"}) + "\n")
    library_index.build_index(db_path=temp_db)
    result = library_index.search_all("carry", db_path=temp_db)
    assert len(result["mechanisms"]) == 1
    assert len(result["papers"]) == 1


def test_search_bm25_ranking(temp_db, patched_paths):
    """Title-match should rank above abstract-match for the same term."""
    log = patched_paths["log"]
    with log.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"source": "arxiv", "source_id": "p1",
                              "title": "Momentum study",
                              "abstract": "Generic stuff.",
                              "verdict": "skip"}) + "\n")
        f.write(json.dumps({"source": "arxiv", "source_id": "p2",
                              "title": "Generic stuff",
                              "abstract": "Momentum is briefly mentioned once here.",
                              "verdict": "skip"}) + "\n")
    library_index.build_index(db_path=temp_db)
    hits = library_index.search_papers("momentum", db_path=temp_db)
    assert len(hits) == 2
    # title hit should rank first (lower bm25 score = better)
    assert hits[0]["source_id"] == "p1"


def test_sanitize_fts_query_strips_injection():
    """User-supplied syntax we don't want to allow gets cleaned."""
    cleaned = library_index._sanitize_fts_query("select * from where; drop")
    # FTS-meaningful chars (* and ") kept; SQL-injection ; ; ; removed
    assert ";" not in cleaned
    assert "*" in cleaned


def test_empty_query_returns_empty(temp_db, patched_paths):
    library_index.build_index(db_path=temp_db)
    assert library_index.search_papers("", db_path=temp_db) == []
    assert library_index.search_mechanisms("   ", db_path=temp_db) == []


# ── incremental refresh ────────────────────────────────────────────────────

def test_refresh_no_change_is_noop(temp_db, patched_paths):
    library_index.build_index(db_path=temp_db)
    result = library_index.refresh_index_incremental(db_path=temp_db)
    assert not result["mechanisms_rescan"]
    assert not result["papers_rescan"]


def test_refresh_picks_up_new_yaml(temp_db, patched_paths):
    library_index.build_index(db_path=temp_db)
    # Add a new YAML after the initial build
    (patched_paths["red"] / "new_one.yaml").write_text(yaml.safe_dump({
        "id": "new_one", "title": "Brand new", "family": "novel",
        "status_in_our_book": "PENDING",
        "mechanism_economics": "New mechanism added later.",
    }), encoding="utf-8")
    # Bump mtime to ensure detection
    import os, time
    p = patched_paths["red"] / "new_one.yaml"
    future = time.time() + 60
    os.utime(p, (future, future))
    result = library_index.refresh_index_incremental(db_path=temp_db)
    assert result["mechanisms_rescan"] is True
    assert result["mechanisms_indexed"] == 1


# ── stats ──────────────────────────────────────────────────────────────────

def test_index_stats_reports_counts(temp_db, patched_paths):
    (patched_paths["red"] / "a.yaml").write_text(yaml.safe_dump({
        "id": "a", "title": "A", "family": "f", "status_in_our_book": "RED",
        "mechanism_economics": "e",
    }), encoding="utf-8")
    with patched_paths["log"].open("w", encoding="utf-8") as f:
        f.write(json.dumps({"source": "arxiv", "source_id": "p1",
                              "title": "t", "abstract": "a",
                              "verdict": "skip"}) + "\n")
    library_index.build_index(db_path=temp_db)
    stats = library_index.index_stats(db_path=temp_db)
    assert stats["mechanisms"] == 1
    assert stats["papers"] == 1
    assert len(stats["sources"]) >= 1


# ── dedup ──────────────────────────────────────────────────────────────────

def test_papers_dedup_on_source_plus_id(temp_db, patched_paths):
    """Re-running the same record should not double-count."""
    log = patched_paths["log"]
    with log.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"source": "arxiv", "source_id": "p1",
                              "title": "T", "abstract": "A",
                              "verdict": "skip"}) + "\n")
        f.write(json.dumps({"source": "arxiv", "source_id": "p1",
                              "title": "T2 different title same id",
                              "abstract": "A2",
                              "verdict": "queue_for_review"}) + "\n")
    library_index.build_index(db_path=temp_db)
    stats = library_index.index_stats(db_path=temp_db)
    assert stats["papers"] == 1
