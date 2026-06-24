"""tests/test_chief_of_staff_runner.py — Phase 2.0 step 14a.

Orchestrator tests. Each substep is mocked at its module-level entry
point so this layer's contract (sequencing, error aggregation, emit
wiring, dry_run propagation) is verified independently of the substep
internals.
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _patch_substeps(monkeypatch, *,
                      d_result=None, a_result=None, b_result=None,
                      d_raises=False, a_raises=False, b_raises=False,
                      pending_b=0):
    """Patch each substep's runner. Returns a dict that captures the
    kwargs they were called with.

    ALSO patches generate_memo (step 14b) to a no-op so 14a tests
    don't accidentally fire real Sonnet calls (each would burn ~$0.03
    and add ~10s latency).

    ALSO patches run_weekly_substrate (piece 7b step 0) to a no-op
    SubstrateRunResult so 14a tests don't accidentally hit the
    network for arxiv/NBER/SSRN/SS. The captured calls dict gets a
    'SUBSTRATE' entry mirroring D/A/B."""
    from engine.agents.book_monitor import runner as d_mod
    from engine.agents.papers_curator import synthesis_runner as a_mod
    from engine.agents.strengthener import runner as b_mod
    from engine.agents.chief_of_staff import runner as cos_mod
    from engine.agents.chief_of_staff import memo as mm
    from engine.agents.chief_of_staff import substrate as sub_mod
    monkeypatch.setattr(mm, "generate_memo", lambda **kw: None)

    calls = {"SUBSTRATE": None, "D": None, "A": None, "B": None}

    # Stub substrate to a zero-counts SubstrateRunResult so production
    # runner.run_weekly_session() doesn't burn network on this layer's
    # tests (which only check D/A/B sequencing).
    def _fake_substrate(**kw):
        calls["SUBSTRATE"] = dict(kw)
        return sub_mod.SubstrateRunResult(
            run_ts="2026-06-07T00:00:00Z", run_date="2026-06-07",
            dry_run=kw.get("dry_run", False),
            enabled_sources=kw.get("enabled_sources", ()),
            arxiv_result={}, nber_result={}, ssrn_result={},
            watchlist_result={}, forward_citation_result={},
            total_fetched=0, total_new=0, errors=[],
        )
    monkeypatch.setattr(sub_mod, "run_weekly_substrate", _fake_substrate)

    # Stub sleeve_fix_proposer (Stage B P2 piece 2) so 14a tests don't
    # hit the doctrine_signal_detected store. Returns a zero-counts
    # shell matching the real result shape.
    from engine.agents.strengthener import sleeve_fix_proposer as sfp_mod
    def _fake_sleeve_fix(**kw):
        calls.setdefault("SLEEVE_FIX", None)
        calls["SLEEVE_FIX"] = dict(kw)
        return {
            "run_ts": "2026-06-07T00:00:00Z",
            "dry_run": kw.get("dry_run", False),
            "n_signals_seen": 0, "n_already_done": 0,
            "n_proposed": 0, "n_persisted": 0,
            "proposed_ids": [], "errors": [],
        }
    monkeypatch.setattr(sfp_mod, "propose_sleeve_fixes",
                          _fake_sleeve_fix)

    # Stub strengthen_scan (Stage B P3c step 1.7) so 14a tests don't
    # iterate library YAML / hit Sonnet 13 times.
    from engine.agents.strengthener import (
        sleeve_strengthen_scan as ssscan_mod,
    )
    def _fake_strengthen_scan(**kw):
        calls.setdefault("STRENGTHEN_SCAN", None)
        calls["STRENGTHEN_SCAN"] = dict(kw)
        return {
            "run_ts":                "2026-06-07T00:00:00Z",
            "iso_week":              "2026-W23",
            "dry_run":               kw.get("dry_run", False),
            "n_sleeves_eligible":    0,
            "n_sleeves_scanned":     0,
            "n_sleeves_skipped":     0,
            "n_proposals_total":     0,
            "n_proposals_persisted": 0,
            "proposed_ids":          [],
            "per_sleeve":            [],
            "errors":                [],
        }
    monkeypatch.setattr(ssscan_mod, "run_sleeve_strengthen_scan",
                          _fake_strengthen_scan)

    def _fake_d(**kw):
        calls["D"] = dict(kw)
        if d_raises: raise RuntimeError("D blew up")
        return d_result if d_result is not None else {
            "n_events_scanned": 100, "n_hits_total": 0, "n_hits_fresh": 0,
            "n_emitted": 0, "event_ids": [], "errors": [],
        }
    def _fake_a(**kw):
        calls["A"] = dict(kw)
        if a_raises: raise RuntimeError("A blew up")
        return a_result if a_result is not None else {
            "run_ts": "2026-06-06T13:00:00Z", "dry_run": False,
            "snapshot": {"recent_summaries": 0, "deployed_sleeves": 0,
                          "recent_events": 0, "doctrine_snippets": 0},
            "candidates": [], "n_candidates": 0,
            "written_hypothesis_ids": [], "n_written": 0,
            "errors": [], "event_id": None,
        }
    def _fake_b(**kw):
        calls["B"] = dict(kw)
        if b_raises: raise RuntimeError("B blew up")
        return b_result if b_result is not None else {
            "run_ts": "2026-06-06T13:00:00Z", "dry_run": False,
            "n_candidates": 0, "n_reviewed": 0, "n_persisted": 0,
            "verdicts": [], "errors": [],
        }

    monkeypatch.setattr(d_mod, "run_book_monitor", _fake_d)
    monkeypatch.setattr(a_mod, "run_synthesis_pipeline", _fake_a)
    monkeypatch.setattr(b_mod, "run_strengthener_pipeline", _fake_b)

    # Stub the pending-approval lookup
    monkeypatch.setattr(cos_mod, "_count_pending_b_approvals",
                          lambda: pending_b)
    return calls


def _patch_emit(monkeypatch, *, raises=False):
    """Patch the session-summary emit so tests don't write events."""
    from engine.research_store import emit
    captured: list = []
    def _fake(**kw):
        if raises: raise RuntimeError("emit broken")
        captured.append(kw)
        return f"ev_session_{len(captured)}"
    monkeypatch.setattr(emit, "chief_of_staff_session_run", _fake)
    return captured


# ─────────────────────────────────────────────────────────────────────
# Happy path — sequence + emit
# ─────────────────────────────────────────────────────────────────────
def test_substeps_called_in_order(monkeypatch):
    """D → A → B; each substep gets the right kwargs."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    calls = _patch_substeps(monkeypatch)
    _patch_emit(monkeypatch)
    result = run_weekly_session(session_id="cos-test")
    assert calls["D"] is not None
    assert calls["A"] is not None
    assert calls["B"] is not None
    assert result.session_id == "cos-test"


def test_default_session_id_includes_today(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    _patch_substeps(monkeypatch)
    _patch_emit(monkeypatch)
    result = run_weekly_session()
    assert result.session_id.startswith("cos-")
    assert len(result.session_id) == len("cos-2026-06-06")


def test_session_tag_propagated_to_A(monkeypatch):
    """A's persisted hypotheses must carry the session id as a tag so
    the audit trail correlates ('which hypotheses came from this
    session?')."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    calls = _patch_substeps(monkeypatch)
    _patch_emit(monkeypatch)
    run_weekly_session(session_id="cos-2026-06-06")
    a_tags = calls["A"]["extra_tags"]
    assert "session:cos-2026-06-06" in a_tags


def test_a_extra_tags_combine_with_session_tag(monkeypatch):
    """Caller can pass additional tags; both they and the session tag
    end up on A's writes."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    calls = _patch_substeps(monkeypatch)
    _patch_emit(monkeypatch)
    run_weekly_session(session_id="cos-2026-06-06",
                         a_extra_tags=("audit_mode",))
    a_tags = calls["A"]["extra_tags"]
    assert "audit_mode" in a_tags
    assert "session:cos-2026-06-06" in a_tags


def test_emit_aggregates_substep_metrics(monkeypatch):
    """The session emit must reflect what the substeps reported."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    _patch_substeps(monkeypatch,
        d_result={"n_events_scanned": 100, "n_hits_total": 3,
                   "n_hits_fresh": 3, "n_emitted": 3,
                   "event_ids": ["ev_d_1", "ev_d_2", "ev_d_3"], "errors": []},
        a_result={"run_ts": "x", "dry_run": False,
                   "snapshot": {}, "candidates": [], "n_candidates": 2,
                   "written_hypothesis_ids": ["h1", "h2"], "n_written": 2,
                   "errors": [], "event_id": "ev_a"},
        b_result={"run_ts": "x", "dry_run": False, "n_candidates": 2,
                   "n_reviewed": 2, "n_persisted": 2,
                   "verdicts": [], "errors": []},
        pending_b=5,
    )
    captured = _patch_emit(monkeypatch)
    result = run_weekly_session(session_id="cos-test")
    assert len(captured) == 1
    ev = captured[0]
    assert ev["d_emitted"] == 3
    assert ev["a_n_candidates"] == 2
    assert ev["a_n_written"] == 2
    assert ev["b_n_reviewed"] == 2
    assert ev["b_n_pending_approval"] == 5
    assert set(ev["parent_event_ids"]) == {"ev_d_1", "ev_d_2", "ev_d_3", "ev_a"}
    # Roll-up fields on SessionResult too
    assert result.d_emitted == 3
    assert result.a_n_written == 2
    assert result.b_n_pending_approval == 5


# ─────────────────────────────────────────────────────────────────────
# dry_run propagation
# ─────────────────────────────────────────────────────────────────────
def test_dry_run_propagates_to_each_substep(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    calls = _patch_substeps(monkeypatch)
    captured = _patch_emit(monkeypatch)
    result = run_weekly_session(dry_run=True)
    assert calls["D"]["dry_run"] is True
    assert calls["A"]["dry_run"] is True
    assert calls["B"]["dry_run"] is True
    # dry_run also SKIPS the session emit (the audit trail is for
    # real runs; dry-run is for human preview)
    assert captured == []
    assert result.session_event_id is None


# ─────────────────────────────────────────────────────────────────────
# Per-substep fail-safe
# ─────────────────────────────────────────────────────────────────────
def test_d_exception_doesnt_kill_a_or_b(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    calls = _patch_substeps(monkeypatch, d_raises=True)
    _patch_emit(monkeypatch)
    result = run_weekly_session()
    # A and B still ran
    assert calls["A"] is not None
    assert calls["B"] is not None
    assert any("D_step" in e for e in result.errors)


def test_a_exception_doesnt_kill_b(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    calls = _patch_substeps(monkeypatch, a_raises=True)
    _patch_emit(monkeypatch)
    result = run_weekly_session()
    assert calls["B"] is not None
    assert any("A_step" in e for e in result.errors)


def test_b_exception_doesnt_kill_emit(monkeypatch):
    from engine.agents.chief_of_staff.runner import run_weekly_session
    _patch_substeps(monkeypatch, b_raises=True)
    captured = _patch_emit(monkeypatch)
    result = run_weekly_session()
    # The session emit still happened so the audit trail has an entry
    assert len(captured) == 1
    assert any("B_step" in e for e in result.errors)


def test_substep_internal_errors_aggregated(monkeypatch):
    """When a substep returns errors in its result dict (not raised),
    chief_of_staff still records them in its own errors list with
    a prefix so the source is obvious."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    _patch_substeps(monkeypatch,
        d_result={"n_events_scanned": 0, "n_hits_total": 0,
                   "n_hits_fresh": 0, "n_emitted": 0,
                   "event_ids": [], "errors": ["disk read failed"]},
        a_result={"snapshot": {}, "candidates": [], "n_candidates": 0,
                   "written_hypothesis_ids": [], "n_written": 0,
                   "errors": ["LLM rate limit"], "event_id": None},
    )
    _patch_emit(monkeypatch)
    result = run_weekly_session()
    assert any("D: disk read failed" in e for e in result.errors)
    assert any("A: LLM rate limit" in e for e in result.errors)


def test_emit_exception_recorded_not_raised(monkeypatch):
    """A broken emit at the end must NOT raise — the substeps already
    ran, their results are in the SessionResult; the audit-trail
    failure is logged but doesn't kill the caller."""
    from engine.agents.chief_of_staff.runner import run_weekly_session
    _patch_substeps(monkeypatch)
    _patch_emit(monkeypatch, raises=True)
    result = run_weekly_session()
    assert result.session_event_id is None
    assert any("session_emit" in e for e in result.errors)
