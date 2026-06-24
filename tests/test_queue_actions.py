"""Tests for engine.research.discovery.queue_actions — promote + skip."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect queue + library + rejected log to tmp."""
    from engine.research.discovery import queue_actions as qa
    queue = tmp_path / "discovery_queue.jsonl"
    border = tmp_path / "discovery_borderline.jsonl"
    rejected = tmp_path / "discovery_rejected.jsonl"
    lib = tmp_path / "mechanism_library"
    lib.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(qa, "DISCOVERY_QUEUE", queue)
    monkeypatch.setattr(qa, "DISCOVERY_BORDERLINE", border)
    monkeypatch.setattr(qa, "DISCOVERY_REJECTED", rejected)
    monkeypatch.setattr(qa, "LIBRARY_DIR", lib)
    monkeypatch.setattr(qa, "REPO_ROOT", tmp_path)
    return {"queue": queue, "border": border, "rejected": rejected,
              "lib": lib, "tmp": tmp_path}


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


@pytest.fixture
def sample_entry():
    return {
        "source":      "manual_nominate",
        "source_id":   "10.1111/jofi.12345",
        "title":       "A Five-Factor Asset Pricing Model",
        "venue":       "Journal of Finance",
        "authors":     "Fama; French",
        "submitted_date": "2014-09-01",
        "doi":         "10.1111/jofi.12345",
        "abs_url":     "https://doi.org/10.1111/jofi.12345",
        "routing":     {"family": "factor_model", "adjusted_confidence": 0.95},
        "extraction":  {
            "family_guess": "factor_model",
            "parent_family_guess": "equity_factor",
            "economic_intuition": "Profitability + investment factor model.",
            "mechanism_proposal": "Five-factor extension of FF3.",
            "required_data_tokens": ["crsp_dsf"],
        },
        "credibility": {"score": 0.7},
        "ts":          "2024-09-01T10:00:00Z",
    }


# ── find_entry / remove_entry ────────────────────────────────────────────

def test_find_entry_in_review_queue(isolated_paths, sample_entry):
    from engine.research.discovery import queue_actions as qa
    _write_jsonl(isolated_paths["queue"], [sample_entry])
    entry, qname = qa.find_entry("10.1111/jofi.12345")
    assert entry is not None
    assert qname == "review"
    assert entry["title"] == sample_entry["title"]


def test_find_entry_in_borderline_queue(isolated_paths, sample_entry):
    from engine.research.discovery import queue_actions as qa
    _write_jsonl(isolated_paths["border"], [sample_entry])
    entry, qname = qa.find_entry("10.1111/jofi.12345")
    assert qname == "borderline"


def test_find_entry_missing_returns_none(isolated_paths):
    from engine.research.discovery import queue_actions as qa
    entry, qname = qa.find_entry("10.nope/missing")
    assert entry is None
    assert qname is None


def test_find_entry_empty_id_returns_none(isolated_paths):
    from engine.research.discovery import queue_actions as qa
    assert qa.find_entry("") == (None, None)


def test_remove_entry_strips_from_queue(isolated_paths, sample_entry):
    from engine.research.discovery import queue_actions as qa
    other = {"source_id": "other", "title": "Other"}
    _write_jsonl(isolated_paths["queue"], [sample_entry, other])
    removed, qname = qa.remove_entry("10.1111/jofi.12345")
    assert removed is not None
    # The other entry should remain
    remaining = [json.loads(l) for l in
                  isolated_paths["queue"].read_text(encoding="utf-8").splitlines()
                  if l.strip()]
    assert len(remaining) == 1
    assert remaining[0]["source_id"] == "other"


# ── promote ──────────────────────────────────────────────────────────────

def test_promote_writes_library_yaml(isolated_paths, sample_entry):
    from engine.research.discovery import queue_actions as qa
    _write_jsonl(isolated_paths["queue"], [sample_entry])
    result = qa.promote("10.1111/jofi.12345")
    assert result["ok"] is True
    assert result["original_queue"] == "review"
    # YAML created
    yaml_files = list(isolated_paths["lib"].glob("*.yaml"))
    assert len(yaml_files) == 1
    with yaml_files[0].open(encoding="utf-8") as f:
        stub = yaml.safe_load(f)
    assert stub["title"] == sample_entry["title"]
    assert stub["family"] == "factor_model"
    assert stub["status_in_our_book"] == "PENDING"
    assert "promotion_metadata" in stub


def test_promote_removes_from_queue(isolated_paths, sample_entry):
    from engine.research.discovery import queue_actions as qa
    _write_jsonl(isolated_paths["queue"], [sample_entry])
    qa.promote("10.1111/jofi.12345")
    # Queue file now empty
    remaining = isolated_paths["queue"].read_text(encoding="utf-8").strip()
    assert remaining == ""


def test_promote_unique_mechanism_id_on_collision(isolated_paths, sample_entry):
    from engine.research.discovery import queue_actions as qa
    # Existing YAML with the slug
    slug_path = isolated_paths["lib"] / "a_five_factor_asset_pricing_model.yaml"
    slug_path.write_text("id: existing\n", encoding="utf-8")
    _write_jsonl(isolated_paths["queue"], [sample_entry])
    result = qa.promote("10.1111/jofi.12345")
    assert result["mechanism_id"].endswith("_2")
    # both files coexist
    assert slug_path.exists()
    assert (isolated_paths["lib"] /
              f"{result['mechanism_id']}.yaml").exists()


def test_promote_missing_id_raises(isolated_paths):
    from engine.research.discovery import queue_actions as qa
    with pytest.raises(ValueError):
        qa.promote("10.nope/notfound")


def test_promote_from_borderline(isolated_paths, sample_entry):
    from engine.research.discovery import queue_actions as qa
    _write_jsonl(isolated_paths["border"], [sample_entry])
    result = qa.promote("10.1111/jofi.12345")
    assert result["original_queue"] == "borderline"


# ── skip ──────────────────────────────────────────────────────────────────

def test_skip_appends_to_rejected_log(isolated_paths, sample_entry):
    from engine.research.discovery import queue_actions as qa
    _write_jsonl(isolated_paths["queue"], [sample_entry])
    result = qa.skip("10.1111/jofi.12345", reason="not_relevant")
    assert result["ok"] is True
    lines = isolated_paths["rejected"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["skip_reason"] == "not_relevant"
    assert record["from_queue"] == "review"
    assert "skipped_at" in record


def test_skip_removes_from_queue(isolated_paths, sample_entry):
    from engine.research.discovery import queue_actions as qa
    _write_jsonl(isolated_paths["queue"], [sample_entry])
    qa.skip("10.1111/jofi.12345")
    remaining = isolated_paths["queue"].read_text(encoding="utf-8").strip()
    assert remaining == ""


def test_skip_missing_raises(isolated_paths):
    from engine.research.discovery import queue_actions as qa
    with pytest.raises(ValueError):
        qa.skip("10.nope/missing")


def test_skip_preserves_original_fields(isolated_paths, sample_entry):
    from engine.research.discovery import queue_actions as qa
    _write_jsonl(isolated_paths["queue"], [sample_entry])
    qa.skip("10.1111/jofi.12345")
    record = json.loads(
        isolated_paths["rejected"].read_text(encoding="utf-8").splitlines()[0]
    )
    # Original metadata still present
    assert record["title"] == sample_entry["title"]
    assert record["venue"] == sample_entry["venue"]


# ── build_mechanism_stub ─────────────────────────────────────────────────

def test_build_stub_handles_missing_extraction(isolated_paths):
    from engine.research.discovery import queue_actions as qa
    minimal = {"title": "X", "source_id": "10.1/x"}
    stub = qa.build_mechanism_stub(minimal)
    assert stub["title"] == "X"
    assert stub["family"] == "unknown"
    assert stub["required_data"] == []


def test_build_stub_default_status_is_pending(isolated_paths):
    from engine.research.discovery import queue_actions as qa
    stub = qa.build_mechanism_stub({"title": "X"})
    assert stub["status_in_our_book"] == "PENDING"


def test_build_stub_custom_status(isolated_paths):
    from engine.research.discovery import queue_actions as qa
    stub = qa.build_mechanism_stub({"title": "X"}, target_status="WHITELISTED")
    assert stub["status_in_our_book"] == "WHITELISTED"


# ── slug generation ─────────────────────────────────────────────────────

def test_slug_from_title_strips_punctuation(isolated_paths):
    from engine.research.discovery.queue_actions import _slug_from_title
    assert _slug_from_title("A Five-Factor Asset Pricing Model") == \
        "a_five_factor_asset_pricing_model"


def test_slug_from_title_handles_empty(isolated_paths):
    from engine.research.discovery.queue_actions import _slug_from_title
    assert _slug_from_title("") == "untitled"
    assert _slug_from_title("", fallback="custom") == "custom"
