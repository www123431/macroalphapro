"""tests/test_chief_of_staff_runner_strengthen_scan_step.py — Stage B P3c.

Verifies wiring of step 1.7 (sleeve_strengthen_scan) into
run_weekly_session. Step 1.7 runs after sleeve_fix_proposer (1.5) and
before A's synthesis (2).
"""
from __future__ import annotations

import pytest


def _stub_all_substeps(monkeypatch):
    """Stub D/A/B/substrate/sleeve_fix/memo/emit so strengthen_scan
    tests stay focused. Returns sequence list mutated by step stubs."""
    from engine.agents.book_monitor import runner as d_mod
    from engine.agents.papers_curator import synthesis_runner as a_mod
    from engine.agents.strengthener import runner as b_mod
    from engine.agents.chief_of_staff import runner as cos_mod
    from engine.agents.chief_of_staff import memo as mm
    from engine.agents.chief_of_staff import substrate as sub_mod
    from engine.agents.strengthener import sleeve_fix_proposer as sfp
    from engine.research_store import emit

    monkeypatch.setattr(mm, "generate_memo", lambda **kw: None)
    monkeypatch.setattr(cos_mod, "_count_pending_b_approvals",
                          lambda: 0)
    monkeypatch.setattr(emit, "chief_of_staff_session_run",
                          lambda **kw: "ev_x")
    monkeypatch.setattr(sub_mod, "run_weekly_substrate",
        lambda **kw: sub_mod.SubstrateRunResult(
            run_ts="t", run_date="2026-06-07", dry_run=False,
            enabled_sources=(), arxiv_result={}, nber_result={},
            ssrn_result={}, watchlist_result={},
            forward_citation_result={},
            total_fetched=0, total_new=0, errors=[]))

    sequence: list[str] = []

    def _fake_d(**kw):
        sequence.append("D")
        return {"n_events_scanned": 0, "n_hits_total": 0,
                 "n_hits_fresh": 0, "n_emitted": 0,
                 "event_ids": [], "errors": []}
    def _fake_a(**kw):
        sequence.append("A")
        return {"run_ts": "x", "dry_run": False, "snapshot": {},
                 "candidates": [], "n_candidates": 0,
                 "written_hypothesis_ids": [], "n_written": 0,
                 "errors": [], "event_id": None}
    def _fake_b(**kw):
        sequence.append("B")
        return {"run_ts": "x", "dry_run": False, "n_candidates": 0,
                 "n_reviewed": 0, "n_persisted": 0, "verdicts": [],
                 "errors": []}
    def _fake_sf(**kw):
        sequence.append("SLEEVE_FIX")
        return {"run_ts": "t", "dry_run": False,
                 "n_signals_seen": 0, "n_already_done": 0,
                 "n_proposed": 0, "n_persisted": 0,
                 "proposed_ids": [], "errors": []}

    monkeypatch.setattr(d_mod, "run_book_monitor", _fake_d)
    monkeypatch.setattr(a_mod, "run_synthesis_pipeline", _fake_a)
    monkeypatch.setattr(b_mod, "run_strengthener_pipeline", _fake_b)
    monkeypatch.setattr(sfp, "propose_sleeve_fixes", _fake_sf)
    return sequence


# ────────────────────────────────────────────────────────────────────
# Step 1.7 runs AFTER sleeve_fix (1.5) + BEFORE A (2)
# ────────────────────────────────────────────────────────────────────
def test_strengthen_scan_runs_between_sleeve_fix_and_a(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_strengthen_scan as ssscan

    sequence = _stub_all_substeps(monkeypatch)

    def _fake_scan(**kw):
        sequence.append("STRENGTHEN_SCAN")
        return {"run_ts": "t", "iso_week": "2026-W23", "dry_run": False,
                 "n_sleeves_eligible": 9, "n_sleeves_scanned": 3,
                 "n_sleeves_skipped": 0,
                 "n_proposals_total": 5, "n_proposals_persisted": 5,
                 "proposed_ids": ["h1", "h2", "h3", "h4", "h5"],
                 "per_sleeve": [], "errors": []}
    monkeypatch.setattr(ssscan, "run_sleeve_strengthen_scan", _fake_scan)

    result = run_weekly_session(session_id="cos-test")
    # Ordering: D → SLEEVE_FIX → STRENGTHEN_SCAN → A → B
    assert sequence == ["D", "SLEEVE_FIX", "STRENGTHEN_SCAN", "A", "B"]
    assert result.strengthen_scan_result is not None
    assert result.strengthen_scan_result["n_proposals_total"] == 5


def test_strengthen_scan_disabled_skips_step(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_strengthen_scan as ssscan

    sequence = _stub_all_substeps(monkeypatch)
    def _broken(**kw):
        pytest.fail("run_sleeve_strengthen_scan should NOT be called "
                     "when run_strengthen_scan=False")
    monkeypatch.setattr(ssscan, "run_sleeve_strengthen_scan", _broken)

    result = run_weekly_session(session_id="cos-test",
                                  run_strengthen_scan=False)
    assert "STRENGTHEN_SCAN" not in sequence
    assert result.strengthen_scan_result is None


# ────────────────────────────────────────────────────────────────────
# Failure isolation
# ────────────────────────────────────────────────────────────────────
def test_scan_exception_does_not_block_ab(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_strengthen_scan as ssscan

    sequence = _stub_all_substeps(monkeypatch)
    def _broken(**kw):
        raise RuntimeError("Sonnet 503")
    monkeypatch.setattr(ssscan, "run_sleeve_strengthen_scan", _broken)

    result = run_weekly_session(session_id="cos-test")
    assert "A" in sequence
    assert "B" in sequence
    assert any("strengthen_scan_step" in e for e in result.errors)
    assert result.strengthen_scan_result is None


def test_scan_internal_errors_propagated(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_strengthen_scan as ssscan

    _stub_all_substeps(monkeypatch)
    monkeypatch.setattr(ssscan, "run_sleeve_strengthen_scan",
        lambda **kw: {"run_ts": "t", "iso_week": "2026-W23",
                       "dry_run": False, "n_sleeves_eligible": 9,
                       "n_sleeves_scanned": 2, "n_sleeves_skipped": 0,
                       "n_proposals_total": 0, "n_proposals_persisted": 0,
                       "proposed_ids": [], "per_sleeve": [],
                       "errors": ["broken_sleeve: proposer: 503"]})

    result = run_weekly_session(session_id="cos-test")
    assert any("strengthen_scan: broken_sleeve" in e
                 for e in result.errors)


# ────────────────────────────────────────────────────────────────────
# kwarg propagation
# ────────────────────────────────────────────────────────────────────
def test_scan_kwargs_propagate(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_strengthen_scan as ssscan

    _stub_all_substeps(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(ssscan, "run_sleeve_strengthen_scan",
        lambda **kw: (captured.update(kw),
                       {"run_ts": "t", "iso_week": "2026-W23",
                        "dry_run": kw.get("dry_run", False),
                        "n_sleeves_eligible": 0, "n_sleeves_scanned": 0,
                        "n_sleeves_skipped": 0,
                        "n_proposals_total": 0,
                        "n_proposals_persisted": 0,
                        "proposed_ids": [], "per_sleeve": [],
                        "errors": []})[1])

    run_weekly_session(session_id="cos-test",
                        strengthen_max_sleeves=2,
                        strengthen_force=True)
    assert captured.get("max_sleeves") == 2
    assert captured.get("force") is True


# ────────────────────────────────────────────────────────────────────
# dry_run propagation
# ────────────────────────────────────────────────────────────────────
def test_dry_run_propagates_to_scan(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_strengthen_scan as ssscan

    _stub_all_substeps(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(ssscan, "run_sleeve_strengthen_scan",
        lambda **kw: (captured.update(kw),
                       {"run_ts": "t", "iso_week": "2026-W23",
                        "dry_run": True,
                        "n_sleeves_eligible": 0, "n_sleeves_scanned": 0,
                        "n_sleeves_skipped": 0,
                        "n_proposals_total": 0,
                        "n_proposals_persisted": 0,
                        "proposed_ids": [], "per_sleeve": [],
                        "errors": []})[1])

    run_weekly_session(session_id="cos-test", dry_run=True)
    assert captured.get("dry_run") is True
