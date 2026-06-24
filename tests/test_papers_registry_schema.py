"""tests/test_papers_registry_schema.py — Q-A schema integrity tests.

Mirror the test depth from test_red_lessons_schema.py: controlled vocab
sanity + round-trip + validation + store I/O.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research_store.papers import (
    FulltextStatus,
    PaperRegistryEntry,
    REGISTRY_SCHEMA_VERSION,
    SHELF_DOCS,
    Shelf,
    find_by_doi,
    latest_per_doi,
    load_registry,
    save_entry,
)


# ───────────────── controlled vocab sanity ────────────────────────────


def test_every_shelf_has_docs():
    required = {"definition", "queryability_hint"}
    for s in Shelf:
        assert s in SHELF_DOCS, f"missing docs for {s}"
        assert required.issubset(set(SHELF_DOCS[s].keys()))


def test_shelves_total_count():
    """8 shelves: 7 categorical + 1 escape hatch (OTHER)."""
    assert len(list(Shelf)) == 8


def test_fulltext_status_values():
    expected = {"ingested", "metadata_only", "paywalled", "unattempted"}
    assert {s.value for s in FulltextStatus} == expected


# ───────────────── round-trip ─────────────────────────────────────────


def _minimal_entry() -> PaperRegistryEntry:
    return PaperRegistryEntry(
        paper_id              = PaperRegistryEntry.new_id(),
        version               = 1,
        parent_paper_id       = None,
        doi                   = "10.1111/j.1540-6261.2014.12148.x",
        title                 = "Betting Against Beta",
        year                  = 2014,
        authors               = ("Frazzini", "Pedersen"),
        venue                 = "Journal of Financial Economics",
        abstract              = "We present a model where investors face leverage constraints...",
        fulltext_status       = FulltextStatus.METADATA_ONLY,
        pdf_source_kind       = "",
        pdf_source_url        = "",
        n_chunks              = 0,
        ingested_ts           = "",
        referenced_by_lessons    = ("L_k1bab_v1",),
        referenced_by_factors    = ("K1_BAB",),
        referenced_by_sleeves    = ("k1_bab_sleeve",),
        referenced_by_doctrines  = (),
        shelves                = (Shelf.GREEN_MOTIVATION, Shelf.DOCTRINE_METHOD),
        shelf_notes            = {
            Shelf.DOCTRINE_METHOD.value:
                "Canonical low-vol-anomaly framework; referenced across "
                "BAB/MIN-VOL/idiosyncratic-vol candidates."
        },
        created_ts = "2026-06-03T12:00:00Z",
        updated_ts = "2026-06-03T12:00:00Z",
        created_by = "tests",
        tags       = ("seed", "doctrine"),
        note       = "Q-A test entry",
    )


def test_round_trip_dict():
    e = _minimal_entry()
    e2 = PaperRegistryEntry.from_dict(e.to_dict())
    assert e == e2


def test_round_trip_json():
    e = _minimal_entry()
    s = json.dumps(e.to_dict(), ensure_ascii=False)
    e2 = PaperRegistryEntry.from_dict(json.loads(s))
    assert e == e2


def test_multi_shelf_serialization():
    e = _minimal_entry()
    # Ensure all 8 shelves serialize cleanly
    e2 = PaperRegistryEntry(**{**e.__dict__, "shelves": tuple(Shelf)})
    e3 = PaperRegistryEntry.from_dict(e2.to_dict())
    assert e2 == e3
    assert set(e3.shelves) == set(Shelf)


# ───────────────── validation ─────────────────────────────────────────


def test_validate_passes_for_minimal_well_formed_entry():
    assert _minimal_entry().validate() == []


def test_validate_rejects_empty_title():
    e = _minimal_entry()
    bad = PaperRegistryEntry(**{**e.__dict__, "title": "  "})
    assert any("title" in err for err in bad.validate())


def test_validate_rejects_no_shelves():
    e = _minimal_entry()
    bad = PaperRegistryEntry(**{**e.__dict__, "shelves": ()})
    assert any("shelf" in err for err in bad.validate())


def test_validate_rejects_implausible_year():
    e = _minimal_entry()
    bad = PaperRegistryEntry(**{**e.__dict__, "year": 1800})
    assert any("year" in err for err in bad.validate())


def test_validate_requires_note_for_OTHER_shelf():
    e = _minimal_entry()
    bad = PaperRegistryEntry(**{**e.__dict__,
                                "shelves": (Shelf.OTHER,),
                                "shelf_notes": {}})
    assert any("OTHER" in err for err in bad.validate())


def test_validate_OTHER_with_note_passes():
    e = _minimal_entry()
    ok = PaperRegistryEntry(**{**e.__dict__,
                               "shelves": (Shelf.OTHER,),
                               "shelf_notes": {Shelf.OTHER.value: "novel mechanism, no fit"}})
    assert ok.validate() == []


def test_validate_ingested_requires_chunks_and_ts():
    e = _minimal_entry()
    bad = PaperRegistryEntry(**{**e.__dict__,
                                "fulltext_status": FulltextStatus.INGESTED,
                                "n_chunks": 0,
                                "ingested_ts": ""})
    errs = bad.validate()
    assert any("n_chunks" in err for err in errs)
    assert any("ingested_ts" in err for err in errs)


def test_validate_ingested_with_metadata_passes():
    e = _minimal_entry()
    ok = PaperRegistryEntry(**{**e.__dict__,
                               "fulltext_status": FulltextStatus.INGESTED,
                               "n_chunks": 12,
                               "ingested_ts": "2026-06-03T12:00:00Z",
                               "pdf_source_kind": "nber"})
    assert ok.validate() == []


# ───────────────── store I/O ──────────────────────────────────────────


def test_store_round_trip(tmp_path: Path):
    p = tmp_path / "registry.jsonl"
    e = _minimal_entry()
    save_entry(e, path=p)
    loaded = load_registry(p)
    assert len(loaded) == 1
    assert loaded[0] == e


def test_store_validation_strict_blocks(tmp_path: Path):
    p = tmp_path / "registry.jsonl"
    e = _minimal_entry()
    bad = PaperRegistryEntry(**{**e.__dict__, "title": ""})
    with pytest.raises(ValueError):
        save_entry(bad, path=p, validate_strict=True)


def test_find_by_doi(tmp_path: Path):
    p = tmp_path / "registry.jsonl"
    e1 = _minimal_entry()
    e2 = PaperRegistryEntry(**{**e1.__dict__,
                               "paper_id": PaperRegistryEntry.new_id(),
                               "version": 2,
                               "parent_paper_id": e1.paper_id,
                               "n_chunks": 10,
                               "fulltext_status": FulltextStatus.INGESTED,
                               "ingested_ts": "2026-06-03T13:00:00Z"})
    save_entry(e1, path=p)
    save_entry(e2, path=p)
    found = find_by_doi(e1.doi, load_registry(p))
    assert found is not None
    assert found.version == 2
    assert found.fulltext_status == FulltextStatus.INGESTED


def test_find_by_doi_returns_none_for_unknown(tmp_path: Path):
    p = tmp_path / "registry.jsonl"
    save_entry(_minimal_entry(), path=p)
    assert find_by_doi("10.9999/nonexistent.0000", load_registry(p)) is None


def test_latest_per_doi_handles_no_doi(tmp_path: Path):
    """Entries with empty DOI are grouped by paper_id (each unique)."""
    p = tmp_path / "registry.jsonl"
    e1 = _minimal_entry()
    e2 = PaperRegistryEntry(**{**e1.__dict__,
                               "paper_id": PaperRegistryEntry.new_id(),
                               "doi": ""})
    e3 = PaperRegistryEntry(**{**e1.__dict__,
                               "paper_id": PaperRegistryEntry.new_id(),
                               "doi": ""})
    save_entry(e1, path=p)
    save_entry(e2, path=p)
    save_entry(e3, path=p)
    latest = latest_per_doi(load_registry(p))
    assert len(latest) == 3  # e1 by DOI, e2 + e3 each by paper_id


def test_load_registry_missing_file_returns_empty(tmp_path: Path):
    p = tmp_path / "nope.jsonl"
    assert load_registry(p) == []


def test_load_registry_skips_corrupt_lines(tmp_path: Path):
    p = tmp_path / "mixed.jsonl"
    e = _minimal_entry()
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(e.to_dict()) + "\n")
        f.write("{broken json\n")
        f.write("\n")
        f.write(json.dumps(e.to_dict()) + "\n")
    assert len(load_registry(p)) == 2
