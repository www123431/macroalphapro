"""tests/test_chief_of_staff_runner_substrate_step.py — Stage A piece 7b.

Tests the wiring of step 0 (substrate refresh) into run_weekly_session.

This complements test_chief_of_staff_runner.py (which covers D/A/B
sequencing) with tests specifically for: substrate runs before D, can
be disabled, errors are isolated, dry_run propagates, sources kwarg
honoured.
"""
from __future__ import annotations

import pytest


# ────────────────────────────────────────────────────────────────────
# Helper — stub D/A/B/memo/emit so this layer's tests are deterministic
# ────────────────────────────────────────────────────────────────────
def _stub_dab_and_memo(monkeypatch):
    """Stub D/A/B substeps + memo + emit so substrate-step tests stay
    focused. Returns the captured-calls dict + substrate-call list."""
    from engine.agents.book_monitor import runner as d_mod
    from engine.agents.papers_curator import synthesis_runner as a_mod
    from engine.agents.strengthener import runner as b_mod
    from engine.agents.chief_of_staff import runner as cos_mod
    from engine.agents.chief_of_staff import memo as mm
    from engine.research_store import emit
    # Stub P2 + P3c LLM-burning steps so substrate-step tests stay free
    from engine.agents.strengthener import sleeve_fix_proposer as sfp
    from engine.agents.strengthener import sleeve_strengthen_scan as ssscan

    monkeypatch.setattr(mm, "generate_memo", lambda **kw: None)
    monkeypatch.setattr(cos_mod, "_count_pending_b_approvals",
                          lambda: 0)
    monkeypatch.setattr(emit, "chief_of_staff_session_run",
                          lambda **kw: "ev_x")
    monkeypatch.setattr(sfp, "propose_sleeve_fixes",
        lambda **kw: {"run_ts": "t", "dry_run": False,
                       "n_signals_seen": 0, "n_already_done": 0,
                       "n_proposed": 0, "n_persisted": 0,
                       "proposed_ids": [], "errors": []})
    monkeypatch.setattr(ssscan, "run_sleeve_strengthen_scan",
        lambda **kw: {"run_ts": "t", "iso_week": "2026-W23",
                       "dry_run": False, "n_sleeves_eligible": 0,
                       "n_sleeves_scanned": 0, "n_sleeves_skipped": 0,
                       "n_proposals_total": 0,
                       "n_proposals_persisted": 0,
                       "proposed_ids": [], "per_sleeve": [],
                       "errors": []})

    sequence: list[str] = []

    def _fake_d(**kw):
        sequence.append("D")
        return {"n_events_scanned": 0, "n_hits_total": 0,
                 "n_hits_fresh": 0, "n_emitted": 0,
                 "event_ids": [], "errors": []}

    def _fake_a(**kw):
        sequence.append("A")
        return {"run_ts": "x", "dry_run": False,
                 "snapshot": {}, "candidates": [], "n_candidates": 0,
                 "written_hypothesis_ids": [], "n_written": 0,
                 "errors": [], "event_id": None}

    def _fake_b(**kw):
        sequence.append("B")
        return {"run_ts": "x", "dry_run": False,
                 "n_candidates": 0, "n_reviewed": 0, "n_persisted": 0,
                 "verdicts": [], "errors": []}

    monkeypatch.setattr(d_mod, "run_book_monitor", _fake_d)
    monkeypatch.setattr(a_mod, "run_synthesis_pipeline", _fake_a)
    monkeypatch.setattr(b_mod, "run_strengthener_pipeline", _fake_b)
    return sequence


# ────────────────────────────────────────────────────────────────────
# Step 0 runs by default + BEFORE D
# ────────────────────────────────────────────────────────────────────
def test_substrate_runs_before_d_by_default(monkeypatch):
    """Default refresh_substrate=True → step 0 runs; ordering matters."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.chief_of_staff import substrate as sub_mod

    sequence = _stub_dab_and_memo(monkeypatch)
    captured = {}
    def _fake_sub(**kw):
        sequence.append("SUBSTRATE")
        captured.update(kw)
        return sub_mod.SubstrateRunResult(
            run_ts="t", run_date="2026-06-07", dry_run=kw.get("dry_run", False),
            enabled_sources=kw.get("enabled_sources", ()),
            arxiv_result={"n_fetched": 5, "n_new": 3, "errors": []},
            nber_result={}, ssrn_result={}, watchlist_result={},
            forward_citation_result={},
            total_fetched=5, total_new=3, errors=[],
        )
    monkeypatch.setattr(sub_mod, "run_weekly_substrate", _fake_sub)

    result = run_weekly_session(session_id="cos-test")
    # Order: SUBSTRATE before D
    assert sequence == ["SUBSTRATE", "D", "A", "B"]
    # substrate_result is on the SessionResult dataclass
    assert result.substrate_result is not None
    assert result.substrate_result["total_fetched"] == 5
    assert result.substrate_result["total_new"] == 3


def test_substrate_disabled_skips_step(monkeypatch):
    """refresh_substrate=False → run_weekly_substrate not called;
    substrate_result is None."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.chief_of_staff import substrate as sub_mod

    sequence = _stub_dab_and_memo(monkeypatch)

    def _fake_sub(**kw):
        pytest.fail("run_weekly_substrate should NOT be called when "
                      "refresh_substrate=False")
    monkeypatch.setattr(sub_mod, "run_weekly_substrate", _fake_sub)

    result = run_weekly_session(session_id="cos-test",
                                  refresh_substrate=False)
    assert sequence == ["D", "A", "B"]
    assert result.substrate_result is None


# ────────────────────────────────────────────────────────────────────
# Error isolation
# ────────────────────────────────────────────────────────────────────
def test_substrate_exception_does_not_block_dab(monkeypatch):
    """run_weekly_substrate raising → D/A/B still run; error in
    SessionResult.errors with 'substrate_step:' prefix."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.chief_of_staff import substrate as sub_mod

    sequence = _stub_dab_and_memo(monkeypatch)

    def _broken(**kw):
        sequence.append("SUBSTRATE_RAISED")
        raise RuntimeError("substrate down")
    monkeypatch.setattr(sub_mod, "run_weekly_substrate", _broken)

    result = run_weekly_session(session_id="cos-test")
    # D/A/B still ran
    assert "D" in sequence
    assert "A" in sequence
    assert "B" in sequence
    assert any("substrate_step" in e for e in result.errors)
    assert result.substrate_result is None


def test_substrate_internal_errors_propagated(monkeypatch):
    """When run_weekly_substrate returns with errors[] populated,
    those errors get the 'substrate:' prefix on SessionResult.errors."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.chief_of_staff import substrate as sub_mod

    _stub_dab_and_memo(monkeypatch)

    def _fake(**kw):
        return sub_mod.SubstrateRunResult(
            run_ts="t", run_date="2026-06-07", dry_run=False,
            enabled_sources=(), arxiv_result={}, nber_result={},
            ssrn_result={}, watchlist_result={},
            forward_citation_result={},
            total_fetched=0, total_new=0,
            errors=["arxiv: timeout", "nber: feed bozo"],
        )
    monkeypatch.setattr(sub_mod, "run_weekly_substrate", _fake)

    result = run_weekly_session(session_id="cos-test")
    assert any("substrate: arxiv: timeout" in e for e in result.errors)
    assert any("substrate: nber: feed bozo" in e for e in result.errors)


# ────────────────────────────────────────────────────────────────────
# dry_run propagation
# ────────────────────────────────────────────────────────────────────
def test_dry_run_propagates_to_substrate(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.chief_of_staff import substrate as sub_mod

    _stub_dab_and_memo(monkeypatch)
    captured = {}
    def _fake(**kw):
        captured.update(kw)
        return sub_mod.SubstrateRunResult(
            run_ts="t", run_date="2026-06-07",
            dry_run=kw.get("dry_run", False),
            enabled_sources=kw.get("enabled_sources", ()),
            arxiv_result={}, nber_result={}, ssrn_result={},
            watchlist_result={}, forward_citation_result={},
            total_fetched=0, total_new=0, errors=[],
        )
    monkeypatch.setattr(sub_mod, "run_weekly_substrate", _fake)

    run_weekly_session(session_id="cos-test", dry_run=True)
    assert captured.get("dry_run") is True


# ────────────────────────────────────────────────────────────────────
# substrate_sources kwarg
# ────────────────────────────────────────────────────────────────────
def test_substrate_sources_kwarg_passed_through(monkeypatch):
    """Caller passes substrate_sources=('arxiv', 'nber') → only those
    sources are activated."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.chief_of_staff import substrate as sub_mod

    _stub_dab_and_memo(monkeypatch)
    captured = {}
    def _fake(**kw):
        captured.update(kw)
        return sub_mod.SubstrateRunResult(
            run_ts="t", run_date="2026-06-07", dry_run=False,
            enabled_sources=kw.get("enabled_sources", ()),
            arxiv_result={}, nber_result={}, ssrn_result={},
            watchlist_result={}, forward_citation_result={},
            total_fetched=0, total_new=0, errors=[],
        )
    monkeypatch.setattr(sub_mod, "run_weekly_substrate", _fake)

    run_weekly_session(session_id="cos-test",
                        substrate_sources=("arxiv", "nber"))
    assert captured.get("enabled_sources") == ("arxiv", "nber")


def test_substrate_default_sources_is_all_sources(monkeypatch):
    """No substrate_sources → ALL_SOURCES are enabled."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    from engine.agents.chief_of_staff import substrate as sub_mod

    _stub_dab_and_memo(monkeypatch)
    captured = {}
    def _fake(**kw):
        captured.update(kw)
        return sub_mod.SubstrateRunResult(
            run_ts="t", run_date="2026-06-07", dry_run=False,
            enabled_sources=kw.get("enabled_sources", ()),
            arxiv_result={}, nber_result={}, ssrn_result={},
            watchlist_result={}, forward_citation_result={},
            total_fetched=0, total_new=0, errors=[],
        )
    monkeypatch.setattr(sub_mod, "run_weekly_substrate", _fake)

    run_weekly_session(session_id="cos-test")
    assert captured.get("enabled_sources") == sub_mod.ALL_SOURCES
