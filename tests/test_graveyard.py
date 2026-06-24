"""Tests for engine.research.graveyard v2 senior-design."""
from __future__ import annotations

import datetime
import json

import pytest
import yaml

from engine.research import graveyard


@pytest.fixture(autouse=True)
def isolated_disk(tmp_path, monkeypatch):
    """Stand up isolated library / gate_runs / discovery_log paths."""
    lib_dir = tmp_path / "mechanism_library"
    lib_dir.mkdir()
    gate_runs = tmp_path / "gate_runs.jsonl"
    discovery_log = tmp_path / "discovery_log.jsonl"

    monkeypatch.setattr(graveyard, "LIBRARY_DIR", lib_dir)
    monkeypatch.setattr(graveyard, "GATE_RUNS", gate_runs)
    monkeypatch.setattr(graveyard, "DISCOVERY_LOG", discovery_log)
    graveyard.clear_cache()
    yield {
        "lib_dir": lib_dir,
        "gate_runs": gate_runs,
        "discovery_log": discovery_log,
    }
    graveyard.clear_cache()


def _write_library_yaml(lib_dir, name, **kw):
    fields = {
        "id": name, "family": kw.get("family", "test_family"),
        "parent_family": kw.get("parent_family", "test_parent"),
        "status_in_our_book": kw.get("status", "UNTESTED"),
        "purpose": kw.get("purpose", "candidate"),
        "required_data": kw.get("required_data", []),
        "mechanism_economics": kw.get("economics", "test economics"),
        "our_test_record": kw.get("test_record"),
        "last_audited": kw.get("audited", "2026-05-30"),
    }
    (lib_dir / f"{name}.yaml").write_text(
        yaml.safe_dump(fields, sort_keys=False), encoding="utf-8"
    )


def _write_gate_run(gate_runs_path, **kw):
    row = {
        "name": kw.get("name", "test_run"),
        "verdict": kw.get("verdict", "RED"),
        "ts": kw.get("ts", "2024-01-15T10:00:00Z"),
        "standalone_sharpe": kw.get("sharpe", -0.3),
        "alpha_t_ff5umd": kw.get("alpha_t", -1.5),
        "deflated_sr": kw.get("dsr", 0.0),
        "mechanism": kw.get("mechanism", "test mechanism"),
        "n_months": kw.get("n_months", 84),
    }
    with gate_runs_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _write_discovery_entry(disc_path, **kw):
    row = {
        "arxiv_id": kw.get("arxiv_id", "2401.test"),
        "title": kw.get("title", "Test Discovery Paper"),
        "verdict": kw.get("verdict", "skip"),
        "stage": kw.get("stage", "dedup"),
        "reason": kw.get("reason", "title overlap"),
        "ts": kw.get("ts", "2024-01-15T10:00:00Z"),
        "extraction": kw.get("extraction", {}),
    }
    with disc_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ── build_graveyard from 4 sources ──────────────────────────────────────

def test_build_graveyard_empty_returns_empty(isolated_disk):
    g = graveyard.build_graveyard(use_cache=False)
    assert g == []


def test_build_graveyard_picks_up_library_red(isolated_disk):
    _write_library_yaml(isolated_disk["lib_dir"], "dead_mech",
                          status="RED", family="momentum")
    g = graveyard.build_graveyard(use_cache=False)
    assert len(g) == 1
    assert g[0].source == "library_red"
    assert g[0].family == "momentum"


def test_build_graveyard_picks_up_negative_evidence(isolated_disk):
    _write_library_yaml(isolated_disk["lib_dir"], "dead_anchor",
                          status="UNTESTED",
                          purpose="negative_evidence", family="value")
    g = graveyard.build_graveyard(use_cache=False)
    assert len(g) == 1
    assert g[0].source == "library_negative_evidence"


def test_build_graveyard_picks_up_gate_runs_red(isolated_disk):
    _write_gate_run(isolated_disk["gate_runs"], verdict="RED",
                      name="test_v1")
    g = graveyard.build_graveyard(use_cache=False)
    assert len(g) == 1
    assert g[0].source == "gate_runs_red"


def test_build_graveyard_skips_non_red_gate_runs(isolated_disk):
    _write_gate_run(isolated_disk["gate_runs"], verdict="GREEN", name="ok")
    _write_gate_run(isolated_disk["gate_runs"], verdict="YELLOW", name="meh")
    g = graveyard.build_graveyard(use_cache=False)
    assert g == []


def test_build_graveyard_picks_up_discovery_dedup_rejects(isolated_disk):
    _write_discovery_entry(isolated_disk["discovery_log"],
                              verdict="skip", stage="dedup")
    g = graveyard.build_graveyard(use_cache=False)
    assert len(g) == 1
    assert g[0].source == "discovery_rejected"


# ── Failure mode classification ─────────────────────────────────────────

def test_failure_mode_classification_decay():
    assert graveyard._classify_failure_mode(
        "post-publication decay observed"
    ) == graveyard.FailureMode.DECAY_POSTPUB.value


def test_failure_mode_classification_regime():
    assert graveyard._classify_failure_mode(
        "junk premium regime hostile to factor"
    ) == graveyard.FailureMode.REGIME_HOSTILE.value


def test_failure_mode_classification_decomposition():
    assert graveyard._classify_failure_mode(
        "absorbed by FF5 + UMD decomposition"
    ) == graveyard.FailureMode.DECOMPOSITION_CONTAM.value


# ── Detectors ───────────────────────────────────────────────────────────

def test_paper_id_match_exact():
    cand = graveyard.CandidateInfo(arxiv_id="2401.123")
    entry = graveyard.GraveyardEntry(
        source="x", source_id="y", name="n", family=None, parent_family=None,
        required_data=[], economics_text="", title="",
        failure_reason="", failure_mode="other",
        death_date=None, source_weight=0.5, revival_potential=0.5,
        extra={"arxiv_id": "2401.123"},
    )
    assert graveyard._paper_id_match(cand, entry) == 1.0


def test_title_overlap_high():
    cand = graveyard.CandidateInfo(title="Cross Sectional Momentum")
    entry = graveyard.GraveyardEntry(
        source="x", source_id="y", name="n", family=None, parent_family=None,
        required_data=[], economics_text="", title="Cross Sectional Momentum Strategy",
        failure_reason="", failure_mode="other",
        death_date=None, source_weight=0.5, revival_potential=0.5, extra={},
    )
    score = graveyard._title_overlap(cand, entry)
    assert score >= 0.6


def test_family_match_same():
    cand = graveyard.CandidateInfo(family="momentum")
    entry = graveyard.GraveyardEntry(
        source="x", source_id="y", name="n", family="momentum",
        parent_family="equity_factor",
        required_data=[], economics_text="", title="",
        failure_reason="", failure_mode="other",
        death_date=None, source_weight=0.5, revival_potential=0.5, extra={},
    )
    assert graveyard._family_match(cand, entry) == 1.0


def test_data_signature_overlap():
    cand = graveyard.CandidateInfo(required_data=["crsp_dsf", "fred_macro"])
    entry = graveyard.GraveyardEntry(
        source="x", source_id="y", name="n", family=None, parent_family=None,
        required_data=["crsp_dsf", "fred_macro"], economics_text="",
        title="", failure_reason="", failure_mode="other",
        death_date=None, source_weight=0.5, revival_potential=0.5, extra={},
    )
    assert graveyard._data_overlap(cand, entry) == 1.0


# ── Match API end-to-end ─────────────────────────────────────────────────

def test_check_against_graveyard_exact_paper_blocks(isolated_disk):
    _write_discovery_entry(
        isolated_disk["discovery_log"],
        arxiv_id="2401.exact",
        title="Title",
        verdict="skip", stage="dedup",
        extraction={"arxiv_id": "2401.exact"},
    )
    cand = graveyard.CandidateInfo(arxiv_id="2401.exact", title="Title")
    result = graveyard.check_against_graveyard(cand, use_cache=False)
    assert result.matched
    assert "paper_id_match" in result.signals_matched


def test_check_against_graveyard_family_with_red_blocks(isolated_disk):
    _write_library_yaml(isolated_disk["lib_dir"], "dead", status="RED",
                          family="my_family")
    cand = graveyard.CandidateInfo(
        title="New Paper",
        family="my_family",
        parent_family="x",
        required_data=[], economics_text="",
    )
    result = graveyard.check_against_graveyard(cand, use_cache=False)
    assert result.matched
    assert "family_match" in result.signals_matched
    assert result.recommendation in ("block", "warn", "review")


def test_check_against_graveyard_no_match_returns_allow(isolated_disk):
    _write_library_yaml(isolated_disk["lib_dir"], "dead",
                          status="RED", family="dead_family")
    cand = graveyard.CandidateInfo(
        title="Totally Unrelated Mechanism",
        family="my_distinct_family",
        parent_family="other_parent",
        required_data=["unique_data"],
        economics_text="unique economic concept",
    )
    result = graveyard.check_against_graveyard(cand, use_cache=False)
    assert not result.matched
    assert result.recommendation == "allow"


# ── Temporal decay ──────────────────────────────────────────────────────

def test_apply_temporal_decay_old_revivable_downgrades():
    """10-year-old entry with revival_potential >= 0.5 → block downgraded to warn."""
    rec = graveyard._apply_temporal_decay("block", entry_age_years=10.0,
                                            revival_potential=0.7)
    assert rec == "warn"


def test_apply_temporal_decay_recent_keeps_block():
    rec = graveyard._apply_temporal_decay("block", entry_age_years=2.0,
                                            revival_potential=0.7)
    assert rec == "block"


def test_apply_temporal_decay_old_low_revival_keeps_block():
    rec = graveyard._apply_temporal_decay("block", entry_age_years=10.0,
                                            revival_potential=0.1)
    assert rec == "block"


# ── Cousin-count escalation ─────────────────────────────────────────────

def test_cousin_count_escalates(isolated_disk):
    """Multiple dead in same family → escalate."""
    _write_library_yaml(isolated_disk["lib_dir"], "dead1",
                          status="RED", family="busy_family",
                          required_data=["x"], economics="unique_a")
    _write_library_yaml(isolated_disk["lib_dir"], "dead2",
                          status="RED", family="busy_family",
                          required_data=["y"], economics="unique_b")
    # Candidate has only a weak family match — but 2 cousins makes it warn
    cand = graveyard.CandidateInfo(
        title="A new candidate",
        family="busy_family",
        parent_family="z",
        required_data=["z"],
        economics_text="z",
    )
    result = graveyard.check_against_graveyard(cand, use_cache=False)
    assert result.cousin_count_in_family == 2


# ── Convenience helpers ─────────────────────────────────────────────────

def test_dead_in_family_returns_filtered(isolated_disk):
    _write_library_yaml(isolated_disk["lib_dir"], "d1", status="RED",
                          family="alpha")
    _write_library_yaml(isolated_disk["lib_dir"], "d2", status="RED",
                          family="beta")
    graveyard.clear_cache()
    result = graveyard.dead_in_family("alpha")
    assert len(result) == 1
    assert result[0].name == "d1"


def test_summarize_graveyard_aggregates(isolated_disk):
    _write_library_yaml(isolated_disk["lib_dir"], "d1", status="RED",
                          family="momentum")
    _write_gate_run(isolated_disk["gate_runs"], verdict="RED", name="g1")
    graveyard.clear_cache()
    summary = graveyard.summarize_graveyard()
    assert summary["total"] == 2
    assert summary["by_source"]["library_red"] == 1
    assert summary["by_source"]["gate_runs_red"] == 1


def test_cache_invalidates_on_disk_change(isolated_disk):
    """Cache rebuilds when disk mtimes change."""
    g1 = graveyard.build_graveyard()
    assert g1 == []
    # Now write a new entry
    _write_library_yaml(isolated_disk["lib_dir"], "new", status="RED")
    # mtime check should detect and rebuild
    g2 = graveyard.build_graveyard()
    assert len(g2) == 1


def test_to_dataframe_round_trip(isolated_disk):
    _write_library_yaml(isolated_disk["lib_dir"], "d1", status="RED",
                          family="momentum")
    graveyard.clear_cache()
    df = graveyard.graveyard_to_dataframe()
    assert not df.empty
    assert "source" in df.columns
    assert "failure_mode" in df.columns
