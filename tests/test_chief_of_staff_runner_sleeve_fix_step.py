"""tests/test_chief_of_staff_runner_sleeve_fix_step.py — Stage B P2 piece 2.

Verifies wiring of step 1.5 (sleeve_fix_proposer) into run_weekly_session.
Step 1.5 runs after D (so it sees fresh signals) and before A/B.
"""
from __future__ import annotations

import pytest


def _stub_dab_substrate_emit_memo(monkeypatch):
    """Common stubs for D/A/B/substrate/memo/emit so each test focuses
    on the sleeve_fix step. Returns the (sequence list, captured-calls
    dict) so each test can verify ordering / kwargs."""
    from engine.agents.book_monitor import runner as d_mod
    from engine.agents.papers_curator import synthesis_runner as a_mod
    from engine.agents.strengthener import runner as b_mod
    from engine.agents.chief_of_staff import runner as cos_mod
    from engine.agents.chief_of_staff import memo as mm
    from engine.agents.chief_of_staff import substrate as sub_mod
    from engine.research_store import emit
    # Stub P3c (strengthen_scan) so step 1.5 tests don't iterate library
    # YAML / hit Sonnet at step 1.7
    from engine.agents.strengthener import sleeve_strengthen_scan as ssscan
    monkeypatch.setattr(ssscan, "run_sleeve_strengthen_scan",
        lambda **kw: {"run_ts": "t", "iso_week": "2026-W23",
                       "dry_run": kw.get("dry_run", False),
                       "n_sleeves_eligible": 0, "n_sleeves_scanned": 0,
                       "n_sleeves_skipped": 0,
                       "n_proposals_total": 0,
                       "n_proposals_persisted": 0,
                       "proposed_ids": [], "per_sleeve": [],
                       "errors": []})

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

    monkeypatch.setattr(d_mod, "run_book_monitor", _fake_d)
    monkeypatch.setattr(a_mod, "run_synthesis_pipeline", _fake_a)
    monkeypatch.setattr(b_mod, "run_strengthener_pipeline", _fake_b)
    return sequence


# ────────────────────────────────────────────────────────────────────
# Step 1.5 runs by default + ordering: D → SLEEVE_FIX → A → B
# ────────────────────────────────────────────────────────────────────
def test_sleeve_fix_runs_after_d_before_a(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_fix_proposer as sfp

    sequence = _stub_dab_substrate_emit_memo(monkeypatch)
    captured: dict = {}
    def _fake_sf(**kw):
        sequence.append("SLEEVE_FIX")
        captured.update(kw)
        return {"run_ts": "t", "dry_run": False,
                 "n_signals_seen": 2, "n_already_done": 0,
                 "n_proposed": 2, "n_persisted": 2,
                 "proposed_ids": ["h1", "h2"], "errors": []}
    monkeypatch.setattr(sfp, "propose_sleeve_fixes", _fake_sf)

    result = run_weekly_session(session_id="cos-test")
    assert sequence == ["D", "SLEEVE_FIX", "A", "B"]
    assert result.sleeve_fix_result is not None
    assert result.sleeve_fix_result["n_proposed"] == 2
    assert result.sleeve_fix_result["proposed_ids"] == ["h1", "h2"]


def test_sleeve_fix_disabled_skips_step(monkeypatch):
    """propose_sleeve_fixes=False → step skipped; sleeve_fix_result=None."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_fix_proposer as sfp

    sequence = _stub_dab_substrate_emit_memo(monkeypatch)
    def _broken(**kw):
        pytest.fail("propose_sleeve_fixes should NOT be called when "
                     "propose_sleeve_fixes=False")
    monkeypatch.setattr(sfp, "propose_sleeve_fixes", _broken)

    result = run_weekly_session(session_id="cos-test",
                                  propose_sleeve_fixes=False)
    assert sequence == ["D", "A", "B"]
    assert result.sleeve_fix_result is None


# ────────────────────────────────────────────────────────────────────
# Failure isolation
# ────────────────────────────────────────────────────────────────────
def test_sleeve_fix_exception_does_not_block_ab(monkeypatch):
    """sleeve_fix raising → A/B still run; error captured with
    'sleeve_fix_step:' prefix."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_fix_proposer as sfp

    sequence = _stub_dab_substrate_emit_memo(monkeypatch)
    def _broken(**kw):
        raise RuntimeError("proposer crashed")
    monkeypatch.setattr(sfp, "propose_sleeve_fixes", _broken)

    result = run_weekly_session(session_id="cos-test")
    assert "A" in sequence
    assert "B" in sequence
    assert any("sleeve_fix_step" in e for e in result.errors)
    assert result.sleeve_fix_result is None


def test_sleeve_fix_internal_errors_propagated(monkeypatch):
    """Errors[] in result dict propagated with 'sleeve_fix:' prefix."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_fix_proposer as sfp

    _stub_dab_substrate_emit_memo(monkeypatch)
    monkeypatch.setattr(sfp, "propose_sleeve_fixes", lambda **kw: {
        "run_ts": "t", "dry_run": False, "n_signals_seen": 2,
        "n_already_done": 0, "n_proposed": 2, "n_persisted": 1,
        "proposed_ids": ["h1", "h2"],
        "errors": ["persist:h2: disk full"],
    })

    result = run_weekly_session(session_id="cos-test")
    assert any("sleeve_fix: persist:h2: disk full" in e
                 for e in result.errors)


# ────────────────────────────────────────────────────────────────────
# kwarg propagation
# ────────────────────────────────────────────────────────────────────
def test_sleeve_fix_kwargs_propagate(monkeypatch):
    """Caller's sleeve_fix_* kwargs flow through to proposer."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_fix_proposer as sfp

    _stub_dab_substrate_emit_memo(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(sfp, "propose_sleeve_fixes", lambda **kw: (
        captured.update(kw),
        {"run_ts": "t", "dry_run": kw.get("dry_run", False),
          "n_signals_seen": 0, "n_already_done": 0,
          "n_proposed": 0, "n_persisted": 0,
          "proposed_ids": [], "errors": []},
    )[1])

    run_weekly_session(session_id="cos-test",
                        sleeve_fix_days=7,
                        sleeve_fix_max_signals=3)
    assert captured.get("days") == 7
    assert captured.get("max_signals") == 3


# ────────────────────────────────────────────────────────────────────
# dry_run propagation
# ────────────────────────────────────────────────────────────────────
def test_dry_run_propagates_to_sleeve_fix(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.strengthener import sleeve_fix_proposer as sfp

    _stub_dab_substrate_emit_memo(monkeypatch)
    captured: dict = {}
    monkeypatch.setattr(sfp, "propose_sleeve_fixes", lambda **kw: (
        captured.update(kw),
        {"run_ts": "t", "dry_run": True,
          "n_signals_seen": 0, "n_already_done": 0,
          "n_proposed": 0, "n_persisted": 0,
          "proposed_ids": [], "errors": []},
    )[1])

    run_weekly_session(session_id="cos-test", dry_run=True)
    assert captured.get("dry_run") is True
