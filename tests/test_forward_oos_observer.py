"""Tests for engine.research.discovery.forward_oos_observer."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from engine.research.discovery import forward_oos_observer as fo
    monkeypatch.setattr(fo, "WATCHLIST_PATH",
                          tmp_path / "watchlist.jsonl")
    monkeypatch.setattr(fo, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(fo, "GATE_RUNS", tmp_path / "gate_runs.jsonl")
    return {"tmp": tmp_path, "fo": fo}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# ── register_for_forward_oos ─────────────────────────────────────────────

def test_register_creates_watchlist_entry(isolated):
    fo = isolated["fo"]
    entry = fo.register_for_forward_oos(
        "test_carry",
        auto_gate_result={"verdict": "RED", "sharpe": 0.1, "deflated_sr": 0.05},
    )
    assert entry.mechanism_id == "test_carry"
    assert entry.state == "registered"
    assert entry.auto_gate_verdict == "RED"
    assert entry.auto_gate_sharpe == 0.1
    # File should exist
    assert isolated["fo"].WATCHLIST_PATH.exists()


def test_register_idempotent_on_duplicate_id(isolated):
    """Re-registering same mechanism_id doesn't create duplicate or
    overwrite original timestamp."""
    fo = isolated["fo"]
    fo.register_for_forward_oos("test_x",
                                    auto_gate_result={"verdict": "GREEN"})
    # Simulate same id re-promoted later
    import time
    time.sleep(0.01)
    fo.register_for_forward_oos("test_x",
                                    auto_gate_result={"verdict": "RED"})
    entries = fo._read_watchlist()
    assert len(entries) == 1
    # Original GREEN preserved (not overwritten with RED)
    assert entries[0]["auto_gate_verdict"] == "GREEN"


def test_register_track_until_is_90d_default(isolated):
    fo = isolated["fo"]
    entry = fo.register_for_forward_oos("test_y")
    until = datetime.date.fromisoformat(entry.track_until)
    today = datetime.date.today()
    delta = (until - today).days
    assert 89 <= delta <= 91     # 90 ± 1 for timezone slop


def test_register_custom_track_days(isolated):
    fo = isolated["fo"]
    entry = fo.register_for_forward_oos("test_z", track_days=30)
    until = datetime.date.fromisoformat(entry.track_until)
    today = datetime.date.today()
    delta = (until - today).days
    assert 29 <= delta <= 31


# ── get_watchlist ────────────────────────────────────────────────────────

def test_get_watchlist_returns_reversed(isolated):
    """Most recent first."""
    fo = isolated["fo"]
    fo.register_for_forward_oos("first")
    import time; time.sleep(0.01)
    fo.register_for_forward_oos("second")
    entries = fo.get_watchlist()
    assert entries[0]["mechanism_id"] == "second"
    assert entries[1]["mechanism_id"] == "first"


def test_get_watchlist_empty_no_file(isolated):
    assert isolated["fo"].get_watchlist() == []


# ── check_implementation_status ──────────────────────────────────────────

def test_check_implementation_no_yaml(isolated):
    result = isolated["fo"].check_implementation_status("nonexistent")
    assert result["yaml_exists"] is False
    assert result["ready_for_paper_trade"] is False


def test_check_implementation_yaml_no_bindings(isolated):
    fo = isolated["fo"]
    _write_yaml(fo.LIBRARY_DIR / "stub.yaml", {
        "id": "stub", "title": "X", "family": "carry",
        "tunable_bindings": ["top_frac"],
        "required_data": ["crsp_dsf"],
        # NO bindings: dict
    })
    result = fo.check_implementation_status("stub")
    assert result["yaml_exists"] is True
    assert result["has_bindings"] is False
    assert result["ready_for_paper_trade"] is False


def test_check_implementation_ready(isolated):
    fo = isolated["fo"]
    _write_yaml(fo.LIBRARY_DIR / "ready.yaml", {
        "id": "ready", "title": "X", "family": "carry",
        "bindings": {"top_frac": 0.2, "vol_target": 0.10},
        "tunable_bindings": ["top_frac", "vol_target"],
        "required_data": ["crsp_dsf"],
    })
    result = fo.check_implementation_status("ready")
    assert result["ready_for_paper_trade"] is True


# ── update_state ─────────────────────────────────────────────────────────

def test_update_state_valid(isolated):
    fo = isolated["fo"]
    fo.register_for_forward_oos("test_a")
    updated = fo.update_state("test_a", "tracking")
    assert updated is True
    entries = fo._read_watchlist()
    assert entries[0]["state"] == "tracking"


def test_update_state_invalid_raises(isolated):
    fo = isolated["fo"]
    fo.register_for_forward_oos("test_b")
    with pytest.raises(ValueError):
        fo.update_state("test_b", "not_a_real_state")


def test_update_state_unknown_mechanism_returns_false(isolated):
    assert isolated["fo"].update_state("nope", "tracking") is False


# ── compute_calibration_delta ────────────────────────────────────────────

def test_calibration_no_real_runs(isolated):
    fo = isolated["fo"]
    fo.register_for_forward_oos("c1",
                                    auto_gate_result={"sharpe": 0.3, "verdict": "RED"})
    delta = fo.compute_calibration_delta("c1")
    assert delta["has_real_runs"] is False
    assert delta["auto_gate_sharpe"] == 0.3


def test_calibration_with_real_runs(isolated):
    fo = isolated["fo"]
    fo.register_for_forward_oos("c2",
                                    auto_gate_result={"sharpe": 0.2, "verdict": "RED"})
    # Write real gate runs that are NOT provisional_synthetic
    now = datetime.datetime.utcnow().isoformat() + "Z"
    later = (datetime.datetime.utcnow()
              + datetime.timedelta(seconds=1)).isoformat() + "Z"
    _write_jsonl(fo.GATE_RUNS, [
        {"name": "auto_gate__c2", "verdict": "RED", "standalone_sharpe": 0.1,
         "ts": now, "provisional_synthetic": True},   # skip (synthetic)
        {"name": "c2_real_run", "mechanism": "c2", "verdict": "GREEN",
         "standalone_sharpe": 0.85, "ts": later},
    ])
    delta = fo.compute_calibration_delta("c2")
    assert delta["has_real_runs"] is True
    assert delta["n_real_runs"] == 1
    assert delta["real_sharpe_mean"] == 0.85
    assert delta["auto_gate_sharpe"] == 0.2
    # delta = real - auto-gate = 0.85 - 0.2 = 0.65 (auto-gate underestimated)
    assert delta["calibration_delta"] == pytest.approx(0.65, abs=0.01)
    assert delta["verdict_mismatch"] is True   # RED vs GREEN


def test_calibration_unknown_mechanism(isolated):
    delta = isolated["fo"].compute_calibration_delta("nope")
    assert "error" in delta


# ── watchlist_summary ───────────────────────────────────────────────────

def test_watchlist_summary_empty(isolated):
    s = isolated["fo"].watchlist_summary()
    assert s["total"] == 0
    assert s["by_state"] == {}


def test_watchlist_summary_counts_by_state(isolated):
    fo = isolated["fo"]
    fo.register_for_forward_oos("a")
    fo.register_for_forward_oos("b")
    fo.update_state("b", "tracking")
    s = fo.watchlist_summary()
    assert s["total"] == 2
    assert s["by_state"]["registered"] == 1
    assert s["by_state"]["tracking"] == 1


def test_watchlist_summary_overdue_count(isolated):
    """track_until in the past + state not graduated/retired → overdue."""
    fo = isolated["fo"]
    fo.register_for_forward_oos("late", track_days=1)
    # Manually backdate the entry to simulate elapsed time
    entries = fo._read_watchlist()
    entries[0]["track_until"] = "2020-01-01"   # long past
    fo._write_watchlist(entries)
    s = fo.watchlist_summary()
    assert s["overdue_for_review"] == 1


# ── Integration: promote auto-registers ─────────────────────────────────

def test_promote_registers_to_watchlist(monkeypatch, tmp_path):
    """End-to-end: queue_actions.promote should auto-register the
    new library entry into the watchlist."""
    from engine.research.discovery import queue_actions as qa
    from engine.research.discovery import forward_oos_observer as fo

    queue = tmp_path / "queue.jsonl"
    lib = tmp_path / "library"
    watchlist = tmp_path / "watchlist.jsonl"

    monkeypatch.setattr(qa, "DISCOVERY_QUEUE", queue)
    monkeypatch.setattr(qa, "DISCOVERY_BORDERLINE", tmp_path / "border.jsonl")
    monkeypatch.setattr(qa, "DISCOVERY_REJECTED", tmp_path / "rej.jsonl")
    monkeypatch.setattr(qa, "LIBRARY_DIR", lib)
    monkeypatch.setattr(qa, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fo, "WATCHLIST_PATH", watchlist)
    monkeypatch.setattr(fo, "LIBRARY_DIR", lib)

    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(
        json.dumps({
            "source_id": "10.1/test", "title": "Test Carry",
            "authors": "Doe, Jane",
            "extraction": {"family_guess": "carry"},
        }) + "\n",
        encoding="utf-8",
    )

    result = qa.promote("10.1/test", auto_gate=False)
    assert result["ok"] is True
    assert "forward_oos_watchlist" in result
    assert result["forward_oos_watchlist"]["registered"] is True
    # Watchlist file should now contain the entry
    assert watchlist.exists()
