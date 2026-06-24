"""Tests for engine.research.discovery.forward_oos_runner."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from engine.research.discovery import forward_oos_runner as fr
    from engine.research.discovery import forward_oos_observer as fo

    runs_dir = tmp_path / "forward_oos_runs"
    watchlist = tmp_path / "watchlist.jsonl"
    lib = tmp_path / "library"
    lib.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(fr, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(fr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fo, "WATCHLIST_PATH", watchlist)
    monkeypatch.setattr(fo, "LIBRARY_DIR", lib)

    return {"tmp": tmp_path, "fr": fr, "fo": fo,
              "runs_dir": runs_dir, "watchlist": watchlist, "lib": lib}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


# ── _runs_file_for ───────────────────────────────────────────────────────

def test_runs_file_path_uses_safe_id(isolated):
    fr = isolated["fr"]
    path = fr._runs_file_for("a/b\\c")
    assert "/" not in path.name
    assert "\\" not in path.name


# ── read_runs / _append_run ──────────────────────────────────────────────

def test_read_runs_empty_when_no_file(isolated):
    assert isolated["fr"].read_runs("nonexistent") == []


def test_append_then_read_roundtrip(isolated):
    fr = isolated["fr"]
    run = fr.SimulationRun(
        mechanism_id="test_a",
        ts="2024-01-01T00:00:00Z",
        data_mode="synthetic",
        state_before="tracking",
        state_after="tracking",
        sharpe=0.5,
    )
    fr._append_run(run)
    out = fr.read_runs("test_a")
    assert len(out) == 1
    assert out[0]["sharpe"] == 0.5


# ── run_watchlist_pass — main loop ──────────────────────────────────────

def test_pass_skips_not_ready_mechanisms(isolated):
    fr, fo = isolated["fr"], isolated["fo"]
    # Register a mechanism but DON'T write a YAML for it
    fo.register_for_forward_oos("no_yaml_mech")
    summary = fr.run_watchlist_pass()
    assert summary["scanned"] == 1
    assert summary["skipped_not_ready"] == 1
    assert summary["simulated"] == 0


def test_pass_transitions_registered_to_awaiting_data_when_yaml_exists(isolated):
    """If YAML exists but no bindings → transition to awaiting_data."""
    fr, fo = isolated["fr"], isolated["fo"]
    fo.register_for_forward_oos("partial_mech")
    _write_yaml(isolated["lib"] / "partial_mech.yaml", {
        "id": "partial_mech", "title": "X", "family": "carry",
        "required_data": ["crsp_dsf"],
        # NO bindings
    })
    fr.run_watchlist_pass()
    entries = fo._read_watchlist()
    assert entries[0]["state"] == "awaiting_data"


def test_pass_simulates_ready_mechanism(isolated, monkeypatch):
    """Ready mechanism gets simulated + transitions to tracking."""
    fr, fo = isolated["fr"], isolated["fo"]
    fo.register_for_forward_oos("ready_mech")
    _write_yaml(isolated["lib"] / "ready_mech.yaml", {
        "id": "ready_mech", "title": "X", "family": "carry",
        "bindings": {"top_frac": 0.2},
        "tunable_bindings": ["top_frac"],
        "required_data": ["crsp_dsf"],
    })
    # Mock _simulate_mechanism so we don't have to run the whole gate
    monkeypatch.setattr(
        fr, "_simulate_mechanism",
        lambda *a, **kw: (0.3, 1.2, 0.2, "synthetic", None),
    )

    summary = fr.run_watchlist_pass()
    assert summary["simulated"] == 1
    assert summary["transitioned"] == 1
    assert summary["errors"] == 0
    # Run file should exist
    runs = fr.read_runs("ready_mech")
    assert len(runs) == 1
    assert runs[0]["sharpe"] == 0.3
    # State should be tracking
    entries = fo._read_watchlist()
    assert entries[0]["state"] == "tracking"


def test_pass_is_idempotent_within_a_day(isolated, monkeypatch):
    """Running pass twice same day → second run skipped."""
    fr, fo = isolated["fr"], isolated["fo"]
    fo.register_for_forward_oos("dup_mech")
    _write_yaml(isolated["lib"] / "dup_mech.yaml", {
        "id": "dup_mech", "title": "X", "family": "carry",
        "bindings": {"top_frac": 0.2},
        "tunable_bindings": ["top_frac"],
        "required_data": ["crsp_dsf"],
    })
    monkeypatch.setattr(
        fr, "_simulate_mechanism",
        lambda *a, **kw: (0.3, 1.2, 0.2, "synthetic", None),
    )
    # First pass: should simulate
    summary1 = fr.run_watchlist_pass()
    assert summary1["simulated"] == 1
    # Second pass same day: should skip
    summary2 = fr.run_watchlist_pass()
    assert summary2["simulated"] == 0
    assert summary2["skipped_already_today"] == 1


def test_pass_graduates_when_track_until_passed(isolated, monkeypatch):
    """track_until in the past → state transitions to graduated."""
    fr, fo = isolated["fr"], isolated["fo"]
    fo.register_for_forward_oos("late_mech", track_days=1)
    # Backdate track_until
    entries = fo._read_watchlist()
    entries[0]["track_until"] = "2020-01-01"   # long past
    fo._write_watchlist(entries)
    _write_yaml(isolated["lib"] / "late_mech.yaml", {
        "id": "late_mech", "title": "X", "family": "carry",
        "bindings": {"top_frac": 0.2},
        "tunable_bindings": ["top_frac"],
        "required_data": ["crsp_dsf"],
    })
    monkeypatch.setattr(
        fr, "_simulate_mechanism",
        lambda *a, **kw: (0.5, 1.0, 0.4, "synthetic", None),
    )
    summary = fr.run_watchlist_pass()
    assert summary["graduated"] == 1
    final = fo._read_watchlist()
    assert final[0]["state"] == "graduated"


def test_pass_skips_already_graduated(isolated, monkeypatch):
    """Graduated mechanisms are NOT re-simulated."""
    fr, fo = isolated["fr"], isolated["fo"]
    fo.register_for_forward_oos("grad_mech")
    fo.update_state("grad_mech", "graduated")
    monkeypatch.setattr(
        fr, "_simulate_mechanism",
        lambda *a, **kw: pytest.fail("should not be called"),
    )
    summary = fr.run_watchlist_pass()
    assert summary["simulated"] == 0
    assert summary["scanned"] == 1
    assert summary["skipped_not_ready"] == 0   # because we exit early


def test_pass_error_handling(isolated, monkeypatch):
    """Simulator error → records error, doesn't crash whole pass."""
    fr, fo = isolated["fr"], isolated["fo"]
    fo.register_for_forward_oos("error_mech")
    _write_yaml(isolated["lib"] / "error_mech.yaml", {
        "id": "error_mech", "title": "X", "family": "carry",
        "bindings": {"top_frac": 0.2},
        "tunable_bindings": ["top_frac"],
        "required_data": ["crsp_dsf"],
    })
    monkeypatch.setattr(
        fr, "_simulate_mechanism",
        lambda *a, **kw: (None, None, None, "synthetic", "simulated_error"),
    )
    summary = fr.run_watchlist_pass()
    assert summary["errors"] == 1
    assert summary["simulated"] == 0
    runs = fr.read_runs("error_mech")
    assert runs[0]["error"] == "simulated_error"
