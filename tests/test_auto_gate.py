"""Tests for engine.research.discovery.auto_gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from engine.research.discovery import auto_gate as ag


@pytest.fixture
def stub_yaml(tmp_path):
    p = tmp_path / "carry_stub.yaml"
    p.write_text(yaml.safe_dump({
        "id": "carry_stub",
        "title": "Test Carry Stub",
        "family": "carry",
        "status_in_our_book": "PENDING",
        "mechanism_economics": "Cross-asset carry returns.",
    }), encoding="utf-8")
    return p


# ── _can_infer_template ──────────────────────────────────────────────────

def test_can_infer_template_for_known_families():
    assert ag._can_infer_template("carry") == "factor_quartile"
    assert ag._can_infer_template("CARRY") == "factor_quartile"  # case-insens
    assert ag._can_infer_template("tsmom") == "factor_quartile"
    assert ag._can_infer_template("momentum") == "factor_quartile"


def test_can_infer_template_returns_none_for_unknown():
    assert ag._can_infer_template("totally_made_up") is None
    assert ag._can_infer_template("") is None
    assert ag._can_infer_template(None) is None


# ── _synthetic_factor_panel ──────────────────────────────────────────────

def test_synthetic_panel_shape():
    factor, price, returns = ag._synthetic_factor_panels(n_periods=120, n_tickers=20)
    assert factor.shape == (120, 20)
    assert price.shape == (120, 20)
    assert returns.shape == (120, 20)
    # No NaN in core data
    assert not factor.isna().all().any()
    # Prices should be positive (above $50 start)
    assert (price > 0).all().all()


def test_synthetic_panel_reproducible():
    f1, p1, r1 = ag._synthetic_factor_panels(seed=123)
    f2, p2, r2 = ag._synthetic_factor_panels(seed=123)
    assert (f1 == f2).all().all()
    assert (p1 == p2).all().all()
    assert (r1 == r2).all().all()


# ── auto_gate result for unsupported family ──────────────────────────────

def test_auto_gate_skips_unknown_family(tmp_path):
    p = tmp_path / "weird_family.yaml"
    p.write_text(yaml.safe_dump({
        "id": "weird", "title": "Weird", "family": "totally_made_up",
        "status_in_our_book": "PENDING",
        "mechanism_economics": "test",
    }), encoding="utf-8")
    result = ag.auto_gate(p, write_ledger=False)
    assert result.ok is True
    assert result.skipped is True
    assert "no auto-gate template" in (result.skip_reason or "")


def test_auto_gate_yaml_parse_failure(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("not: valid: yaml: at: all: ::", encoding="utf-8")
    result = ag.auto_gate(p, write_ledger=False)
    assert result.ok is False
    assert "yaml parse" in (result.error or "").lower()


# ── auto_gate happy path (mocked run_gate) ───────────────────────────────

def test_auto_gate_provisional_synthetic_flag_set(stub_yaml, monkeypatch):
    """Result must carry provisional_synthetic=True so downstream
    knows the verdict is from fake data."""
    # Mock run_gate to return a fake verdict quickly (avoids the
    # FF factor file dependency)
    from engine.research import pipeline as _p
    monkeypatch.setattr(_p, "run_gate", lambda *a, **kw: {
        "name": kw.get("name", "x"),
        "available": True,
        "verdict": "RED",
        "standalone_sharpe": 0.0,
        "alpha_t_ff5umd": 0.1,
        "deflated_sr": 0.0,
    })
    result = ag.auto_gate(stub_yaml, write_ledger=False)
    assert result.ok is True
    assert result.provisional_synthetic is True
    assert "synthetic" in result.provisional_note.lower()
    assert result.verdict == "RED"


def test_auto_gate_to_dict_includes_synthetic_field(stub_yaml, monkeypatch):
    from engine.research import pipeline as _p
    monkeypatch.setattr(_p, "run_gate", lambda *a, **kw: {
        "name": "x", "available": True, "verdict": "YELLOW",
        "standalone_sharpe": 0.3, "alpha_t_ff5umd": 1.5,
        "deflated_sr": 0.4,
    })
    result = ag.auto_gate(stub_yaml, write_ledger=False)
    d = result.to_dict()
    assert "provisional_synthetic" in d
    assert d["provisional_synthetic"] is True
    assert "provisional_note" in d


def test_auto_gate_handles_template_missing(stub_yaml, monkeypatch):
    """If TEMPLATES dict doesn't have the inferred template, error gracefully."""
    monkeypatch.setattr(
        "engine.research.templates.TEMPLATES", {},
    )
    # Wait — _can_infer says factor_quartile, but TEMPLATES is empty
    result = ag.auto_gate(stub_yaml, write_ledger=False)
    # Either ok=False or some failure mode — should NOT crash
    assert isinstance(result, ag.AutoGateResult)


# ── _annotate_ledger_synthetic ──────────────────────────────────────────

def test_annotate_ledger_synthetic_adds_flag(tmp_path, monkeypatch):
    fake_ledger = tmp_path / "gate_runs.jsonl"
    fake_ledger.write_text(
        json.dumps({"name": "auto_gate__x", "verdict": "RED"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ag, "GATE_RUNS", fake_ledger)
    ag._annotate_ledger_synthetic("auto_gate__x")
    lines = fake_ledger.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    assert rec.get("provisional_synthetic") is True


def test_annotate_ledger_handles_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "GATE_RUNS", tmp_path / "missing.jsonl")
    # Should not raise
    ag._annotate_ledger_synthetic("any_name")


# ── Symmetric author-track on skip ───────────────────────────────────────

def test_skip_updates_author_track_with_fail(tmp_path, monkeypatch):
    """Skip must symmetrically +1 fail for the first author (Tier 1 ②)."""
    from engine.research.discovery import queue_actions as qa
    from engine.research.discovery import credibility_scorer as cs

    queue = tmp_path / "discovery_queue.jsonl"
    rejected = tmp_path / "discovery_rejected.jsonl"
    track = tmp_path / "author_track.jsonl"

    monkeypatch.setattr(qa, "DISCOVERY_QUEUE", queue)
    monkeypatch.setattr(qa, "DISCOVERY_BORDERLINE",
                          tmp_path / "discovery_borderline.jsonl")
    monkeypatch.setattr(qa, "DISCOVERY_REJECTED", rejected)
    monkeypatch.setattr(qa, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cs, "AUTHOR_TRACK_PATH", track)

    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(
        json.dumps({
            "source_id": "10.1/test", "title": "X",
            "authors": "Doe, Jane; Smith, John",
        }) + "\n",
        encoding="utf-8",
    )
    qa.skip("10.1/test", reason="off_topic")
    # Author track ledger should now have +1 fail for "doe, jane"
    assert track.exists()
    records = [json.loads(l) for l in track.read_text(encoding="utf-8").splitlines()
                  if l.strip()]
    assert any(r.get("outcome") == "fail" and "doe, jane" in r.get("author", "")
                  for r in records)
