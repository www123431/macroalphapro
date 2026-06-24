"""Unit tests for engine.research.library_writer (NEW1 gate→library loop).

Critical properties:
1. Mapped candidate writes to the right YAML
2. Unmapped candidate becomes an orphan (no YAML write)
3. our_observed.gate_run_ids accumulates set-style (no dup on re-run)
4. summary_sharpe_observed averages across runs
5. UNTESTED+GREEN promotes to YELLOW (NEVER auto-DEPLOY)
6. audit_signature is NEVER auto-flipped
7. published-literature decay fields are NEVER overwritten
8. dry_run=True does not write to disk
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml as _pyyaml

from engine.research import library_writer as LW


@pytest.fixture
def tmp_library(tmp_path, monkeypatch):
    """Stand up a temp library + map + ledgers for isolated tests."""
    lib_dir = tmp_path / "mechanism_library"
    lib_dir.mkdir()
    map_path = tmp_path / "candidate_to_mechanism_map.yaml"
    orphan_log = tmp_path / "orphan_candidates.jsonl"
    update_log = tmp_path / "library_updates.jsonl"

    monkeypatch.setattr(LW, "LIBRARY_DIR", lib_dir)
    monkeypatch.setattr(LW, "MAP_PATH", map_path)
    monkeypatch.setattr(LW, "ORPHAN_LOG", orphan_log)
    monkeypatch.setattr(LW, "UPDATE_LOG", update_log)
    return {
        "lib_dir": lib_dir,
        "map_path": map_path,
        "orphan_log": orphan_log,
        "update_log": update_log,
    }


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(_pyyaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_yaml(path: Path) -> dict:
    return _pyyaml.safe_load(path.read_text(encoding="utf-8"))


def _seed_mechanism(lib_dir: Path, mid: str, status: str = "UNTESTED") -> Path:
    yaml_path = lib_dir / f"{mid}.yaml"
    _write_yaml(yaml_path, {
        "_schema_version": 2,
        "id": mid,
        "post_pub_decay": {
            "mclean_pontiff_2016": {
                "delta_range_estimate": [-0.5, -0.2],
                "verified": False,
            },
            "our_observed": {
                "gate_run_ids": [],
                "summary_sharpe_observed": None,
                "delta_vs_published_lit": None,
                "last_updated": None,
            },
        },
        "status_in_our_book": status,
        "currently_unexplored_in_our_book": status == "UNTESTED",
        "our_test_record": None,
        "audit_checklist_passed": {
            "paper_exists_in_master_index": False,
        },
        "audit_signature": "pending",
    })
    return yaml_path


def _seed_map(map_path: Path, mappings: dict) -> None:
    _write_yaml(map_path, {"_schema_version": 1, "mappings": mappings})


# ── Test 1: mapped candidate writes correctly ───────────────────────────

def test_mapped_candidate_updates_yaml(tmp_library):
    lib = tmp_library
    _seed_mechanism(lib["lib_dir"], "quality_qmj", status="RED")
    _seed_map(lib["map_path"], {"qty_v1": "quality_qmj"})

    gate_run = {
        "name": "qty_v1", "ts": "2026-05-29T10:00:00Z",
        "standalone_sharpe": -0.67, "verdict": "RED",
    }
    result = LW.update_library_from_gate_run(gate_run, all_gate_runs=[gate_run])

    assert result["orphan"] is False
    assert result["mechanism_id"] == "quality_qmj"
    yaml_data = _read_yaml(lib["lib_dir"] / "quality_qmj.yaml")
    obs = yaml_data["post_pub_decay"]["our_observed"]
    assert obs["gate_run_ids"] == ["2026-05-29T10:00:00Z"]
    assert obs["summary_sharpe_observed"] == -0.67


# ── Test 2: unmapped candidate goes to orphan log ───────────────────────

def test_unmapped_candidate_is_orphan(tmp_library):
    lib = tmp_library
    _seed_map(lib["map_path"], {"mapped_v1": "post_earnings_drift"})
    _seed_mechanism(lib["lib_dir"], "post_earnings_drift", status="DEPLOYED")

    gate_run = {"name": "unmapped_v99", "ts": "2026-05-30T00:00:00Z", "verdict": "RED"}
    result = LW.update_library_from_gate_run(gate_run, all_gate_runs=[gate_run])

    assert result["orphan"] is True
    assert lib["orphan_log"].exists()
    rows = [json.loads(l) for l in lib["orphan_log"].read_text(encoding="utf-8").splitlines()]
    assert any(r["candidate_name"] == "unmapped_v99" for r in rows)


# ── Test 3: mapping points to null (tracked, no YAML) → orphan ──────────

def test_null_mapping_is_orphan(tmp_library):
    lib = tmp_library
    _seed_map(lib["map_path"], {"tracked_v1": None})

    gate_run = {"name": "tracked_v1", "ts": "2026-05-30T00:00:00Z", "verdict": "RED"}
    result = LW.update_library_from_gate_run(gate_run, all_gate_runs=[gate_run])
    assert result["orphan"] is True


# ── Test 4: gate_run_ids set-style append (no dup) ──────────────────────

def test_gate_run_ids_no_dup_on_rerun(tmp_library):
    lib = tmp_library
    _seed_mechanism(lib["lib_dir"], "post_earnings_drift", status="DEPLOYED")
    _seed_map(lib["map_path"], {"pead_v1": "post_earnings_drift"})

    gate_run = {"name": "pead_v1", "ts": "2026-05-29T10:00:00Z",
                "standalone_sharpe": 0.5, "verdict": "GREEN"}
    LW.update_library_from_gate_run(gate_run, all_gate_runs=[gate_run])
    LW.update_library_from_gate_run(gate_run, all_gate_runs=[gate_run])

    obs = _read_yaml(lib["lib_dir"] / "post_earnings_drift.yaml")["post_pub_decay"]["our_observed"]
    assert obs["gate_run_ids"] == ["2026-05-29T10:00:00Z"]    # not duplicated


# ── Test 5: summary_sharpe averages across runs ──────────────────────────

def test_summary_sharpe_averages(tmp_library):
    lib = tmp_library
    _seed_mechanism(lib["lib_dir"], "post_earnings_drift", status="DEPLOYED")
    _seed_map(lib["map_path"], {"pead_v1": "post_earnings_drift",
                                 "pead_v2": "post_earnings_drift"})

    r1 = {"name": "pead_v1", "ts": "2026-05-29T10:00:00Z",
          "standalone_sharpe": 0.4, "verdict": "GREEN"}
    r2 = {"name": "pead_v2", "ts": "2026-05-30T10:00:00Z",
          "standalone_sharpe": 0.6, "verdict": "GREEN"}
    LW.update_library_from_gate_run(r1, all_gate_runs=[r1, r2])
    LW.update_library_from_gate_run(r2, all_gate_runs=[r1, r2])

    obs = _read_yaml(lib["lib_dir"] / "post_earnings_drift.yaml")["post_pub_decay"]["our_observed"]
    assert obs["summary_sharpe_observed"] == 0.5    # (0.4 + 0.6) / 2


# ── Test 6: UNTESTED + GREEN promotes to YELLOW (not DEPLOYED) ──────────

def test_untested_green_promotes_to_yellow(tmp_library):
    lib = tmp_library
    _seed_mechanism(lib["lib_dir"], "equity_xsmom_jt", status="UNTESTED")
    _seed_map(lib["map_path"], {"jt_v1": "equity_xsmom_jt"})

    gate_run = {"name": "jt_v1", "ts": "2026-05-29T10:00:00Z",
                "standalone_sharpe": 0.8, "verdict": "GREEN"}
    result = LW.update_library_from_gate_run(gate_run, all_gate_runs=[gate_run])

    assert result["promoted_to_candidate"] is True
    yaml_data = _read_yaml(lib["lib_dir"] / "equity_xsmom_jt.yaml")
    assert yaml_data["status_in_our_book"] == "YELLOW"
    assert yaml_data["currently_unexplored_in_our_book"] is False


# ── Test 7: UNTESTED + RED does NOT promote ─────────────────────────────

def test_untested_red_does_not_promote(tmp_library):
    lib = tmp_library
    _seed_mechanism(lib["lib_dir"], "equity_xsmom_jt", status="UNTESTED")
    _seed_map(lib["map_path"], {"jt_v1": "equity_xsmom_jt"})

    gate_run = {"name": "jt_v1", "ts": "2026-05-29T10:00:00Z",
                "standalone_sharpe": -0.3, "verdict": "RED"}
    result = LW.update_library_from_gate_run(gate_run, all_gate_runs=[gate_run])

    assert result["promoted_to_candidate"] is False
    yaml_data = _read_yaml(lib["lib_dir"] / "equity_xsmom_jt.yaml")
    assert yaml_data["status_in_our_book"] == "UNTESTED"


# ── Test 8: audit_signature NEVER auto-flipped ──────────────────────────

def test_audit_signature_never_auto_flipped(tmp_library):
    lib = tmp_library
    _seed_mechanism(lib["lib_dir"], "post_earnings_drift", status="DEPLOYED")
    _seed_map(lib["map_path"], {"pead_v1": "post_earnings_drift"})

    gate_run = {"name": "pead_v1", "ts": "2026-05-29T10:00:00Z",
                "standalone_sharpe": 0.5, "verdict": "GREEN"}
    LW.update_library_from_gate_run(gate_run, all_gate_runs=[gate_run])

    yaml_data = _read_yaml(lib["lib_dir"] / "post_earnings_drift.yaml")
    assert yaml_data["audit_signature"] == "pending"
    assert yaml_data["audit_checklist_passed"]["paper_exists_in_master_index"] is False


# ── Test 9: published-lit decay fields NEVER overwritten ────────────────

def test_published_lit_decay_preserved(tmp_library):
    lib = tmp_library
    _seed_mechanism(lib["lib_dir"], "post_earnings_drift", status="DEPLOYED")
    _seed_map(lib["map_path"], {"pead_v1": "post_earnings_drift"})

    original = _read_yaml(lib["lib_dir"] / "post_earnings_drift.yaml")
    original_decay = original["post_pub_decay"]["mclean_pontiff_2016"]["delta_range_estimate"]

    gate_run = {"name": "pead_v1", "ts": "2026-05-29T10:00:00Z",
                "standalone_sharpe": 0.5, "verdict": "GREEN"}
    LW.update_library_from_gate_run(gate_run, all_gate_runs=[gate_run])

    updated = _read_yaml(lib["lib_dir"] / "post_earnings_drift.yaml")
    assert updated["post_pub_decay"]["mclean_pontiff_2016"]["delta_range_estimate"] == original_decay


# ── Test 10: dry_run does not write ─────────────────────────────────────

def test_dry_run_does_not_write(tmp_library):
    lib = tmp_library
    _seed_mechanism(lib["lib_dir"], "post_earnings_drift", status="DEPLOYED")
    _seed_map(lib["map_path"], {"pead_v1": "post_earnings_drift"})

    original_text = (lib["lib_dir"] / "post_earnings_drift.yaml").read_text(encoding="utf-8")

    gate_run = {"name": "pead_v1", "ts": "2026-05-29T10:00:00Z",
                "standalone_sharpe": 0.5, "verdict": "GREEN"}
    result = LW.update_library_from_gate_run(
        gate_run, all_gate_runs=[gate_run], dry_run=True)

    assert result["orphan"] is False
    new_text = (lib["lib_dir"] / "post_earnings_drift.yaml").read_text(encoding="utf-8")
    assert new_text == original_text
    assert not lib["update_log"].exists()


# ── Test 11: missing name skips gracefully ──────────────────────────────

def test_missing_name_skips(tmp_library):
    result = LW.update_library_from_gate_run({"ts": "2026-05-29T10:00:00Z"},
                                               all_gate_runs=[])
    assert result["orphan"] is True
    assert result["mechanism_id"] is None


# ── Test 12: mapping to nonexistent YAML logs orphan ─────────────────────

def test_mapping_to_missing_yaml_is_orphan(tmp_library):
    lib = tmp_library
    _seed_map(lib["map_path"], {"v1": "ghost_mechanism"})

    gate_run = {"name": "v1", "ts": "2026-05-29T10:00:00Z", "verdict": "RED"}
    result = LW.update_library_from_gate_run(gate_run, all_gate_runs=[gate_run])
    assert result["orphan"] is True
    assert lib["orphan_log"].exists()
