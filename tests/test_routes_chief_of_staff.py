"""tests/test_routes_chief_of_staff.py — Phase 2.0 step 15 backend.

POST /api/chief_of_staff/run endpoint tests. Mocks the runner so
tests are fast + deterministic + no LLM cost.
"""
from __future__ import annotations

import dataclasses as _dc

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def _mock_result(**overrides):
    """Build a SessionResult dict for the mocked runner to return."""
    from engine.agents.chief_of_staff.runner import SessionResult
    base = dict(
        session_id           = "cos-test",
        run_ts               = "2026-06-07T13:00:00Z",
        dry_run              = True,
        d_result             = {"n_events_scanned": 100, "n_hits_total": 0,
                                  "n_hits_fresh": 0, "n_emitted": 0,
                                  "event_ids": [], "errors": []},
        a_result             = {"snapshot": {"recent_summaries": 6,
                                               "deployed_sleeves": 5,
                                               "recent_events": 40,
                                               "doctrine_snippets": 0},
                                  "n_candidates": 0, "n_written": 0,
                                  "written_hypothesis_ids": [],
                                  "candidates": [],
                                  "errors": [], "event_id": None},
        b_result             = {"n_candidates": 0, "n_reviewed": 0,
                                  "n_persisted": 0, "verdicts": [], "errors": []},
        session_event_id     = None,
        errors               = [],
        d_emitted            = 0,
        a_n_candidates       = 0,
        a_n_written          = 0,
        b_n_reviewed         = 0,
        b_n_pending_approval = 0,
        memo                 = None,
    )
    base.update(overrides)
    return SessionResult(**base)


def _patch_runner(monkeypatch, factory):
    """Patch the runner. Factory is callable(**kw) → SessionResult."""
    from engine.agents.chief_of_staff import runner as cos_mod
    monkeypatch.setattr(cos_mod, "run_weekly_session", factory)


# ─────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────
def test_default_is_dry_run(monkeypatch):
    captured = {}
    def _fake(**kw):
        captured.update(kw)
        return _mock_result()
    _patch_runner(monkeypatch, _fake)

    r = client.post("/api/chief_of_staff/run", json={})
    assert r.status_code == 200
    assert captured["dry_run"] is True
    body = r.json()
    assert body["dry_run"] is True
    assert body["session_id"] == "cos-test"


def test_explicit_persist_mode_passes_through(monkeypatch):
    captured = {}
    def _fake(**kw):
        captured.update(kw)
        return _mock_result(dry_run=False, session_event_id="ev-cos-1")
    _patch_runner(monkeypatch, _fake)

    r = client.post("/api/chief_of_staff/run",
                      json={"dry_run": False,
                            "session_id": "cos-custom"})
    assert r.status_code == 200
    body = r.json()
    assert body["dry_run"] is False
    assert body["session_event_id"] == "ev-cos-1"
    assert captured["dry_run"] is False
    assert captured["session_id"] == "cos-custom"


def test_session_result_dataclass_fully_serialized(monkeypatch):
    """All SessionResult fields must round-trip through the route's
    pydantic model — missing one = a contract regression."""
    rich = _mock_result(
        dry_run=False,
        a_n_candidates=2, a_n_written=2,
        b_n_reviewed=2, b_n_pending_approval=5,
        d_emitted=3,
        session_event_id="ev-session",
        memo={"session_id": "cos-test", "headline": "Test memo headline",
               "bullets": ["b1", "b2", "b3"], "whats_next": "x",
               "generated_ts": "2026-06-07T13:00:00Z",
               "model": "claude-sonnet-4-6"},
    )
    _patch_runner(monkeypatch, lambda **kw: rich)
    r = client.post("/api/chief_of_staff/run", json={"dry_run": False})
    assert r.status_code == 200
    body = r.json()
    assert body["a_n_candidates"] == 2
    assert body["b_n_pending_approval"] == 5
    assert body["d_emitted"] == 3
    assert body["memo"]["headline"] == "Test memo headline"


def test_errors_returned_as_200_with_array(monkeypatch):
    """Runner-recorded errors come back as HTTP 200 + errors[] populated.
    The runner's fail-safe contract is preserved through the route."""
    _patch_runner(monkeypatch, lambda **kw: _mock_result(
        errors=["D: disk read failed", "A: rate limited"]))
    r = client.post("/api/chief_of_staff/run", json={})
    assert r.status_code == 200
    assert r.json()["errors"] == ["D: disk read failed", "A: rate limited"]


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────
def test_request_validation_rejects_wrong_type(monkeypatch):
    _patch_runner(monkeypatch, lambda **kw: _mock_result())
    r = client.post("/api/chief_of_staff/run",
                      json={"dry_run": "not_a_bool"})
    assert r.status_code == 422


def test_500_on_import_failure(monkeypatch):
    """If chief_of_staff runner can't be imported, 500. Production
    code-deployment bug — should surface loudly."""
    import sys
    monkeypatch.setitem(sys.modules,
                          "engine.agents.chief_of_staff.runner", None)
    r = client.post("/api/chief_of_staff/run", json={})
    assert r.status_code == 500


# ─────────────────────────────────────────────────────────────────────
# Route registration
# ─────────────────────────────────────────────────────────────────────
def test_route_registered():
    paths = [r.path for r in app.routes]
    assert "/api/chief_of_staff/run" in paths
