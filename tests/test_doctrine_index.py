"""tests/test_doctrine_index.py — Tier-2 query_doctrine module tests.

ChromaDB + SentenceTransformer is heavy to install in CI but available
locally. Tests that need the real vector index gate with skipif so the
suite passes on machines without the deps.

Covers:
  - Frontmatter parsing (name / description / metadata.type extraction)
  - Files without frontmatter → silently skipped (not a parse error)
  - MEMORY.md index file always skipped
  - iter_memory_entries over a tmp memory dir
  - Empty topic_hint → () (cost discipline, no chroma fire)
  - Missing memory dir → () (graceful, no crash)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_HAS_CHROMA = importlib.util.find_spec("chromadb") is not None


# ─────────────────────────────────────────────────────────────────────
# Frontmatter parsing — independent of chroma
# ─────────────────────────────────────────────────────────────────────
def _write_memory_file(p: Path, *, name: str, description: str,
                        entry_type: str = "feedback", body: str = "test body"):
    p.write_text(f"""---
name: {name}
description: {description}
metadata:
  type: {entry_type}
---

{body}
""", encoding="utf-8")


def test_parse_memory_file_extracts_frontmatter_fields(tmp_path):
    from engine.agents.papers_curator.doctrine_index import _parse_memory_file
    p = tmp_path / "feedback_test_2026-06-07.md"
    _write_memory_file(p,
        name="feedback-test-2026-06-07",
        description="Test feedback for parse verification",
        entry_type="feedback",
        body="The full body of the memory entry goes here. "
              "Multiple sentences supported.")
    e = _parse_memory_file(p)
    assert e is not None
    assert e.name == "feedback-test-2026-06-07"
    assert e.description == "Test feedback for parse verification"
    assert e.entry_type == "feedback"
    assert "full body" in e.body


def test_parse_skips_memory_index_file(tmp_path):
    from engine.agents.papers_curator.doctrine_index import _parse_memory_file
    p = tmp_path / "MEMORY.md"
    p.write_text("# Memory Index\n\n- some entries\n", encoding="utf-8")
    assert _parse_memory_file(p) is None


def test_parse_skips_file_without_frontmatter(tmp_path):
    from engine.agents.papers_curator.doctrine_index import _parse_memory_file
    p = tmp_path / "notes.md"
    p.write_text("# Just notes\nNo frontmatter here.\n", encoding="utf-8")
    assert _parse_memory_file(p) is None


def test_parse_handles_quoted_values(tmp_path):
    """YAML supports double + single-quoted strings — frontmatter often
    uses quotes for description fields with colons inside."""
    from engine.agents.papers_curator.doctrine_index import _parse_memory_file
    p = tmp_path / "feedback_quoted.md"
    p.write_text("""---
name: "feedback-quoted-test"
description: "Has colons: like this"
metadata:
  type: 'feedback'
---

body
""", encoding="utf-8")
    e = _parse_memory_file(p)
    assert e is not None
    assert e.name == "feedback-quoted-test"
    assert e.description == "Has colons: like this"
    assert e.entry_type == "feedback"


def test_parse_falls_back_to_filename_stem_for_missing_name(tmp_path):
    """If frontmatter has no `name:` (defensive), use filename stem."""
    from engine.agents.papers_curator.doctrine_index import _parse_memory_file
    p = tmp_path / "fallback_test.md"
    p.write_text("""---
description: only description
metadata:
  type: project
---

body
""", encoding="utf-8")
    e = _parse_memory_file(p)
    assert e is not None
    assert e.name == "fallback_test"


# ─────────────────────────────────────────────────────────────────────
# iter_memory_entries
# ─────────────────────────────────────────────────────────────────────
def test_iter_memory_entries_yields_all_parseable(tmp_path):
    from engine.agents.papers_curator.doctrine_index import iter_memory_entries
    for i in range(3):
        _write_memory_file(tmp_path / f"entry_{i}.md",
            name=f"entry-{i}", description=f"desc {i}", body=f"body {i}")
    # Plus a MEMORY.md (skipped) and a no-frontmatter file (skipped)
    (tmp_path / "MEMORY.md").write_text("idx", encoding="utf-8")
    (tmp_path / "raw.md").write_text("no frontmatter", encoding="utf-8")
    entries = list(iter_memory_entries(memory_dir=tmp_path))
    assert len(entries) == 3
    assert {e.name for e in entries} == {"entry-0", "entry-1", "entry-2"}


def test_iter_memory_entries_missing_dir_returns_empty(tmp_path):
    from engine.agents.papers_curator.doctrine_index import iter_memory_entries
    entries = list(iter_memory_entries(memory_dir=tmp_path / "does_not_exist"))
    assert entries == []


# ─────────────────────────────────────────────────────────────────────
# query_doctrine — cost-discipline / fail-open paths (no chroma needed)
# ─────────────────────────────────────────────────────────────────────
def test_query_empty_topic_hint_returns_empty():
    """Empty topic → don't fire chroma, save cost."""
    from engine.agents.papers_curator.doctrine_index import query_doctrine
    assert query_doctrine("") == ()
    assert query_doctrine("   ") == ()


def test_query_with_no_chroma_collection_returns_empty(monkeypatch):
    """If chroma client/collection unavailable, query returns ()
    so caller (A's gatherer / B's runner) falls back gracefully."""
    from engine.agents.papers_curator import doctrine_index as di
    monkeypatch.setattr(di, "get_doctrine_collection", lambda: None)
    monkeypatch.setattr(di, "ingest_doctrine",
                          lambda **kw: {"error": "stubbed"})
    out = di.query_doctrine("test topic")
    assert out == ()


# ─────────────────────────────────────────────────────────────────────
# Full pipeline test — needs ChromaDB
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not _HAS_CHROMA, reason="chromadb not installed")
def test_ingest_and_query_end_to_end(tmp_path, monkeypatch):
    """Full ingest + retrieve flow against synthetic memory dir."""
    from engine.agents.papers_curator import doctrine_index as di

    # Redirect chroma persistent dir into tmp
    monkeypatch.setattr(di, "_DOCTRINE_DIR", tmp_path / "doctrine_chroma")
    # Reset module-level singletons so this test uses fresh chroma
    monkeypatch.setattr(di, "_chroma_client", None)
    monkeypatch.setattr(di, "_chroma_collection", None)

    mem = tmp_path / "memory"
    mem.mkdir()
    _write_memory_file(mem / "feedback_carry.md",
        name="feedback-carry-doctrine-2026-05",
        description="Carry trade institutional sizing rules",
        body="Cross-asset carry sleeve sized at vol-target 6% per "
              "Politis-Romano paired bootstrap calibration.")
    _write_memory_file(mem / "project_vrp.md",
        name="project-vrp-research-2026-06",
        description="Variance risk premium research plan",
        body="VRP on SPX delivers Sharpe 0.7 post-cost. Bekaert-Hoerova "
              "shows cross-asset diversification lifts portfolio Sharpe.")
    _write_memory_file(mem / "feedback_graveyard.md",
        name="feedback-graveyard-2026-04",
        description="PEAD family exhausted graveyard rule",
        body="Do not propose more PEAD-family candidates. We have 12 "
              "RED verdicts in last 90 days; family is over-mined.")

    ing = di.ingest_doctrine(memory_dir=mem, force=True)
    assert ing["n_added"] == 3
    assert ing["n_unparseable"] == 0

    # Query for VRP — should retrieve project_vrp first
    hits = di.query_doctrine("variance risk premium options",
                                top_k=3, memory_dir=mem,
                                auto_ingest=False)
    assert len(hits) >= 1
    assert hits[0].name == "project-vrp-research-2026-06"

    # Query for PEAD-related text — graveyard rule should appear in
    # top-K (exact ranking is embedding-dependent and brittle; check
    # presence rather than ranking).
    hits2 = di.query_doctrine("PEAD post-earnings family over-mined",
                                 top_k=3, memory_dir=mem,
                                 auto_ingest=False)
    names = {h.name for h in hits2}
    assert "feedback-graveyard-2026-04" in names
