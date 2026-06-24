"""tests/test_ingestion_reason_schema.py — Phase 1.7 step 2 schema guards.

Verify the new IngestionReason + IngestionReasonSource + IntentCategory
types round-trip cleanly, that PaperRegistryEntry's added field is
forward/backward compatible with pre-1.7 jsonl rows, and that the
controlled vocab (8 intent categories + 2 sources) holds.
"""
from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────
# Controlled vocab sanity
# ──────────────────────────────────────────────────────────────────────
def test_intent_category_set():
    """8 categories per spec
    [[spec-papers-curator-full-architecture-2026-06-05]]."""
    from engine.research_store.papers import IntentCategory
    expected = {
        "expand_breadth", "improve_existing_sleeve", "address_decay",
        "methodology_borrow", "challenge_doctrine", "curiosity",
        "fact_check", "author_trust", "other",
    }
    actual = {c.value for c in IntentCategory}
    assert actual == expected


def test_ingestion_source_set():
    """2 sources only — USER and AGENT. No third "edited" state per
    design lock 2026-06-06."""
    from engine.research_store.papers import IngestionReasonSource
    actual = {s.value for s in IngestionReasonSource}
    assert actual == {"user", "agent"}


def test_schema_version_bumped():
    from engine.research_store.papers import REGISTRY_SCHEMA_VERSION
    assert REGISTRY_SCHEMA_VERSION == 2, (
        "schema_version must be 2 after Phase 1.7 step 2 (ingestion_reason)"
    )


# ──────────────────────────────────────────────────────────────────────
# IngestionReason round-trip
# ──────────────────────────────────────────────────────────────────────
def test_ingestion_reason_user_round_trip():
    from engine.research_store.papers import (
        IngestionReason, IngestionReasonSource, IntentCategory,
    )
    r = IngestionReason(
        free_text       = "want to expand our cross-asset breadth",
        intent_category = IntentCategory.EXPAND_BREADTH,
        source          = IngestionReasonSource.USER,
        user_ts         = "2026-06-06T13:00:00Z",
    )
    d = r.to_dict()
    assert d["source"] == "user"
    assert d["intent_category"] == "expand_breadth"
    r2 = IngestionReason.from_dict(d)
    assert r == r2


def test_ingestion_reason_agent_round_trip():
    from engine.research_store.papers import (
        IngestionReason, IngestionReasonSource, IntentCategory,
    )
    r = IngestionReason(
        free_text       = "ESG fragility adjacency to risk monitoring",
        intent_category = IntentCategory.METHODOLOGY_BORROW,
        source          = IngestionReasonSource.AGENT,
        user_ts         = "2026-06-06T13:00:00Z",
    )
    d = r.to_dict()
    assert d["source"] == "agent"
    r2 = IngestionReason.from_dict(d)
    assert r2.source == IngestionReasonSource.AGENT


def test_ingestion_reason_no_category_yet():
    """intent_category=None is valid (LLM hasn't normalized yet)."""
    from engine.research_store.papers import (
        IngestionReason, IngestionReasonSource,
    )
    r = IngestionReason(
        free_text       = "raw user text",
        intent_category = None,
        source          = IngestionReasonSource.USER,
        user_ts         = "2026-06-06T13:00:00Z",
    )
    d = r.to_dict()
    assert d["intent_category"] is None
    r2 = IngestionReason.from_dict(d)
    assert r2.intent_category is None


def test_ingestion_reason_free_text_truncated_to_200():
    """Long free_text gets trimmed in from_dict to keep stored size
    bounded."""
    from engine.research_store.papers import (
        IngestionReason, IngestionReasonSource,
    )
    long = "x" * 500
    r = IngestionReason.from_dict({
        "free_text":       long,
        "intent_category": None,
        "source":          "user",
        "user_ts":         "2026-06-06T13:00:00Z",
    })
    assert len(r.free_text) == 200


def test_unknown_intent_category_falls_to_other():
    """Forward-compat: if a future enum value appears in legacy data,
    we map it to OTHER rather than failing."""
    from engine.research_store.papers import IngestionReason, IntentCategory
    r = IngestionReason.from_dict({
        "free_text":       "x",
        "intent_category": "some_future_label",
        "source":          "user",
        "user_ts":         "",
    })
    assert r.intent_category == IntentCategory.OTHER


def test_unknown_source_falls_to_user():
    """Forward-compat / corrupt-row: unknown source defaults to USER
    (the less-confident attribution)."""
    from engine.research_store.papers import IngestionReason, IngestionReasonSource
    r = IngestionReason.from_dict({
        "free_text":       "x",
        "intent_category": None,
        "source":          "system",
        "user_ts":         "",
    })
    assert r.source == IngestionReasonSource.USER


# ──────────────────────────────────────────────────────────────────────
# PaperRegistryEntry — ingestion_reason field round-trip + backward compat
# ──────────────────────────────────────────────────────────────────────
def _minimal_entry_dict(**overrides) -> dict:
    """A minimal-valid PaperRegistryEntry dict shape; tests override
    specific fields."""
    base = {
        "paper_id": "test-paper-001",
        "version": 1,
        "parent_paper_id": None,
        "doi": "10.0000/test.001",
        "title": "Test Paper",
        "year": 2024,
        "authors": ["Test Author"],
        "venue": "Test Venue",
        "abstract": "Test abstract",
        "fulltext_status": "metadata_only",
        "pdf_source_kind": "",
        "pdf_source_url": "",
        "n_chunks": 0,
        "ingested_ts": "",
        "referenced_by_lessons": [],
        "referenced_by_factors": [],
        "referenced_by_sleeves": [],
        "referenced_by_doctrines": [],
        "shelves": ["other"],
        "shelf_notes": {"other": "test"},
        "created_ts": "2024-01-01T00:00:00Z",
        "updated_ts": "2024-01-01T00:00:00Z",
        "created_by": "test",
        "tags": [],
        "note": "",
    }
    base.update(overrides)
    return base


def test_paper_entry_round_trip_with_ingestion_reason():
    from engine.research_store.papers import (
        PaperRegistryEntry, IngestionReason, IngestionReasonSource,
        IntentCategory,
    )
    reason = IngestionReason(
        free_text="testing breadth expansion",
        intent_category=IntentCategory.EXPAND_BREADTH,
        source=IngestionReasonSource.USER,
        user_ts="2026-06-06T13:00:00Z",
    )
    d = _minimal_entry_dict()
    e = PaperRegistryEntry.from_dict(d)
    # construct a new entry WITH ingestion_reason via dataclasses.replace
    import dataclasses
    e_with = dataclasses.replace(e, ingestion_reason=reason)
    d2 = e_with.to_dict()
    assert d2["ingestion_reason"] is not None
    assert d2["ingestion_reason"]["source"] == "user"
    e3 = PaperRegistryEntry.from_dict(d2)
    assert e3.ingestion_reason is not None
    assert e3.ingestion_reason.source == IngestionReasonSource.USER


def test_paper_entry_backward_compat_pre_phase_17():
    """Pre-1.7 entries (no ingestion_reason key in jsonl row) load
    cleanly with ingestion_reason=None."""
    from engine.research_store.papers import PaperRegistryEntry
    d = _minimal_entry_dict()
    assert "ingestion_reason" not in d
    e = PaperRegistryEntry.from_dict(d)
    assert e.ingestion_reason is None


def test_paper_entry_to_dict_emits_null_when_none():
    """to_dict must emit ingestion_reason=None on disk (not omit it)
    so the column exists in jsonl for downstream consumers expecting
    the v2 shape."""
    from engine.research_store.papers import PaperRegistryEntry
    e = PaperRegistryEntry.from_dict(_minimal_entry_dict())
    d = e.to_dict()
    assert "ingestion_reason" in d
    assert d["ingestion_reason"] is None
