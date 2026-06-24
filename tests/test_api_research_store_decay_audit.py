"""tests/test_api_research_store_decay_audit.py — G.4.

Tests the /api/research_store/decay_audit/{subject_id} endpoint that
backs the /lab/decay/detail canonical Tier C section.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def _client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def _fake_event(
    *, event_id="evt_1", event_type="decay_alert",
    subject_id="x", verdict="MARGINAL", ts="2026-06-09T10:00:00Z",
    summary="s", metrics=None, tags=("decay_watch",),
):
    class _V:
        def __init__(s, v): s.value = v
    return SimpleNamespace(
        event_id=event_id, event_type=event_type, subject_id=subject_id,
        verdict=_V(verdict), ts=ts, summary=summary,
        metrics=metrics or {}, tags=tags, family=None, actor="test",
        parent_event_ids=(),
        to_dict=lambda: {
            "event_id": event_id, "event_type": event_type,
            "subject_id": subject_id, "verdict": verdict,
            "ts": ts, "summary": summary, "metrics": metrics or {},
            "tags": list(tags), "family": None, "actor": "test",
            "parent_event_ids": [],
        },
    )


def test_endpoint_returns_canonical_decay_alerts_only():
    """Endpoint filters by `decay_watch` tag — legacy SLM alerts
    without that tag must NOT appear in the response."""
    canonical = _fake_event(
        event_id="evt_canonical",
        tags=("decay_watch", "review_recommended"),
        metrics={"triggers_hit": ["A", "B"],
                  "worst_best_sharpe_ratio": 0.10,
                  "severity": "MARGINAL"},
    )
    legacy = _fake_event(event_id="evt_legacy", tags=("slm_legacy",))

    def _fake_filter(event_type=None, subject_id=None, limit=None, **kw):
        if event_type == "decay_alert":
            return [canonical, legacy]
        return []
    with patch("engine.research_store.store.filter_events", _fake_filter):
        r = _client().get("/api/research_store/decay_audit/cross_asset_carry")
    assert r.status_code == 200
    d = r.json()
    assert d["subject_id"] == "cross_asset_carry"
    assert d["n_decay_alerts"] == 1
    ids = [e["event_id"] for e in d["decay_alerts"]]
    assert "evt_canonical" in ids
    assert "evt_legacy" not in ids


def test_endpoint_includes_factor_verdicts_for_subject():
    """Factor verdicts for the subject are surfaced so the detail
    page can show the audit chain that led to a decay alert."""
    verdict = _fake_event(
        event_id="evt_verdict", event_type="factor_verdict_filed",
        subject_id="cross_asset_carry", verdict="GREEN",
        metrics={"sharpe": 0.5, "subsample_stability": {}},
    )
    def _fake_filter(event_type=None, subject_id=None, limit=None, **kw):
        if event_type == "decay_alert":
            return []
        if event_type == "factor_verdict_filed":
            return [verdict]
        return []
    with patch("engine.research_store.store.filter_events", _fake_filter):
        r = _client().get("/api/research_store/decay_audit/cross_asset_carry")
    d = r.json()
    assert d["n_factor_verdicts"] == 1
    assert d["factor_verdicts"][0]["event_id"] == "evt_verdict"


def test_endpoint_returns_empty_for_unknown_subject():
    with patch("engine.research_store.store.filter_events",
                  return_value=[]):
        r = _client().get("/api/research_store/decay_audit/no_such_sleeve")
    assert r.status_code == 200
    d = r.json()
    assert d["n_decay_alerts"] == 0
    assert d["n_factor_verdicts"] == 0


def test_endpoint_clamps_limit_to_safe_range():
    """Out-of-range limit gets clamped (1..500) — no DOS via large limit."""
    calls = {}
    def _fake_filter(event_type=None, subject_id=None, limit=None, **kw):
        calls.setdefault("limit_seen", []).append(limit)
        return []
    with patch("engine.research_store.store.filter_events", _fake_filter):
        # too small
        _client().get("/api/research_store/decay_audit/X?limit=0")
        # too large
        _client().get("/api/research_store/decay_audit/X?limit=99999")
    assert all(1 <= L <= 500 for L in calls["limit_seen"])


# ────────────────────────────────────────────────────────────────────
# G.5 — Acknowledge endpoint
# ────────────────────────────────────────────────────────────────────
def _emit_fake_decay_alert():
    """Helper: emit a fresh decay_alert so we have a real event_id to
    acknowledge. Uses the real research_store (events.jsonl) — this
    is integration, not pure unit. Each test run creates its own
    target event."""
    import datetime as _dt
    from engine.research_store import emit, registry
    from engine.research_store.schema import SubjectType
    if registry.resolve("ack_test_sleeve") is None:
        registry.register_subject(
            "ack_test_sleeve",
            subject_type=SubjectType.sleeve,
            family="deployed_book",
            description="Test sleeve subject for ack tests.",
        )
    return emit.decay_alert(
        subject_id="ack_test_sleeve", verdict="MARGINAL",
        metrics={"triggers_hit": ["A", "B"], "severity": "MARGINAL",
                  "worst_best_sharpe_ratio": 0.15},
        artifacts={}, summary="Test alert for ack",
        parent_event_ids=(),
        tags=("decay_watch", "ack_test"),
        actor="test",
    )


def test_ack_endpoint_validates_action_enum():
    """Bad action → 422."""
    eid = _emit_fake_decay_alert()
    r = _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "totally_wrong", "reason": "A reasonable rationale here."},
    )
    assert r.status_code == 422
    assert "action" in r.json()["detail"]


def test_ack_endpoint_validates_reason_length():
    """Reason shorter than 10 chars → 422 (institutional standard)."""
    eid = _emit_fake_decay_alert()
    r = _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "reviewed_no_action", "reason": "ok"},
    )
    assert r.status_code == 422
    assert "10" in r.json()["detail"]


def test_ack_endpoint_404_on_unknown_event():
    r = _client().post(
        "/api/research_store/decay_alert/no_such_event/acknowledge",
        json={"action": "reviewed_no_action",
              "reason": "A reasonable rationale here."},
    )
    assert r.status_code == 404


def test_ack_endpoint_emits_followup_event_with_parent_chain():
    """Successful ack returns new event_id; new event has parent_event_ids
    pointing to original + `acknowledged` tag."""
    from engine.research_store import store
    eid = _emit_fake_decay_alert()
    r = _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "reviewed_no_action",
              "reason": "Reviewed 2026-06-09 audit; no allocation change."},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["ok"]
    assert j["original_event_id"] == eid
    ack_eid = j["ack_event_id"]
    # Verify the ack event exists in store with correct shape
    ack_event = store.by_event_id(ack_eid)
    assert ack_event is not None
    assert "acknowledged" in (ack_event.tags or ())
    assert eid in (ack_event.parent_event_ids or ())
    assert ack_event.subject_id == "ack_test_sleeve"


def test_ack_marks_original_as_acknowledged_in_audit_endpoint():
    """After ack, the audit endpoint's response shows the original
    event with `is_ack_event=False` AND `ack_info` populated."""
    eid = _emit_fake_decay_alert()
    _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "scheduled_review",
              "reason": "Will revisit at next portfolio review."},
    )
    r = _client().get("/api/research_store/decay_audit/ack_test_sleeve")
    d = r.json()
    # Find the original we just acked
    original = next((e for e in d["decay_alerts"]
                       if e["event_id"] == eid), None)
    assert original is not None
    assert original["is_ack_event"] is False
    assert original["ack_info"] is not None
    # H refactor: ack_info shape is now {is_acknowledged, latest_*, history}
    assert original["ack_info"]["is_acknowledged"] is True
    assert original["ack_info"]["latest_action"] == "scheduled_review"
    assert "Will revisit" in original["ack_info"]["latest_reason"]


def test_ack_endpoint_rejects_non_decay_alert_event():
    """Trying to ack a factor_verdict event (wrong type) → 422."""
    from engine.research_store import emit, registry
    from engine.research_store.schema import SubjectType
    # Register subject + emit a non-decay event
    if registry.resolve("ack_test_factor") is None:
        registry.register_subject(
            "ack_test_factor", subject_type=SubjectType.factor,
            family="test", description="non-decay subject")
    fid = emit.factor_verdict(
        subject_id="ack_test_factor", verdict="GREEN",
        metrics={"sharpe": 0.5}, artifacts={},
        summary="test", actor="test",
    )
    r = _client().post(
        f"/api/research_store/decay_alert/{fid}/acknowledge",
        json={"action": "reviewed_no_action",
              "reason": "A reasonable rationale here."},
    )
    assert r.status_code == 422
    assert "decay_alert" in r.json()["detail"]


# ────────────────────────────────────────────────────────────────────
# H — Unacknowledge endpoint + chain walk
# ────────────────────────────────────────────────────────────────────
def test_unack_validates_reason_length():
    eid = _emit_fake_decay_alert()
    _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "reviewed_no_action",
              "reason": "Initial ack with adequate rationale here."},
    )
    r = _client().post(
        f"/api/research_store/decay_alert/{eid}/unacknowledge",
        json={"reason": "no"},
    )
    assert r.status_code == 422
    assert "10" in r.json()["detail"]


def test_unack_404_on_unknown_event():
    r = _client().post(
        "/api/research_store/decay_alert/no_such_event/unacknowledge",
        json={"reason": "Reasonable rationale for re-opening this alert."},
    )
    assert r.status_code == 404


def test_unack_422_when_not_currently_acked():
    """Cannot unack an alert that was never acked."""
    eid = _emit_fake_decay_alert()
    r = _client().post(
        f"/api/research_store/decay_alert/{eid}/unacknowledge",
        json={"reason": "Reasonable rationale for re-opening this alert."},
    )
    assert r.status_code == 422
    assert "not currently acknowledged" in r.json()["detail"]


def test_unack_emits_event_chained_to_latest_ack():
    """Successful unack writes an event with unacknowledged tag
    and parent_event_ids pointing to the latest ack."""
    from engine.research_store import store
    eid = _emit_fake_decay_alert()
    ack = _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "reviewed_no_action",
              "reason": "Initial ack with adequate rationale here."},
    ).json()
    ack_eid = ack["ack_event_id"]
    r = _client().post(
        f"/api/research_store/decay_alert/{eid}/unacknowledge",
        json={"reason": "Found new data; re-opening to review."},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["ok"]
    assert j["original_event_id"] == eid
    assert j["reverted_ack_event_id"] == ack_eid
    unack_event = store.by_event_id(j["unack_event_id"])
    assert "unacknowledged" in (unack_event.tags or ())
    assert ack_eid in (unack_event.parent_event_ids or ())


def test_unack_then_audit_shows_open_state_with_history():
    """After ack → unack, audit endpoint shows is_acknowledged=False
    AND the history list contains both events newest-first."""
    eid = _emit_fake_decay_alert()
    _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "reviewed_no_action",
              "reason": "First ack with adequate rationale here."},
    )
    _client().post(
        f"/api/research_store/decay_alert/{eid}/unacknowledge",
        json={"reason": "Changed my mind; reopening for review now."},
    )
    r = _client().get("/api/research_store/decay_audit/ack_test_sleeve")
    d = r.json()
    original = next((e for e in d["decay_alerts"]
                       if e["event_id"] == eid), None)
    assert original is not None
    assert original["ack_info"] is not None
    assert original["ack_info"]["is_acknowledged"] is False
    history = original["ack_info"]["history"]
    assert len(history) >= 2
    # Newest first — unack is newest
    assert history[0]["kind"] == "unacknowledged"
    assert history[1]["kind"] == "acknowledged"


def test_reack_after_unack_supersedes_to_acked():
    """ack → unack → ack again. Final state = acked."""
    eid = _emit_fake_decay_alert()
    _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "reviewed_no_action",
              "reason": "First ack with adequate rationale here."},
    )
    _client().post(
        f"/api/research_store/decay_alert/{eid}/unacknowledge",
        json={"reason": "Changed my mind initially; reopening now."},
    )
    _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "scheduled_review",
              "reason": "Decided to review at next cycle as planned."},
    )
    r = _client().get("/api/research_store/decay_audit/ack_test_sleeve")
    d = r.json()
    original = next((e for e in d["decay_alerts"]
                       if e["event_id"] == eid), None)
    assert original["ack_info"]["is_acknowledged"] is True
    assert original["ack_info"]["latest_action"] == "scheduled_review"
    history = original["ack_info"]["history"]
    assert len(history) >= 3


# ────────────────────────────────────────────────────────────────────
# I — Inbox composer hides acked alerts by default
# ────────────────────────────────────────────────────────────────────
def test_inbox_hides_acked_decay_alerts_by_default():
    """source_decay_alerts_canonical skips events whose latest state
    is `acknowledged`. Inbox is for action items, not for history."""
    from engine.inbox.composer import source_decay_alerts_canonical
    eid = _emit_fake_decay_alert()
    # Before ack: appears in inbox
    items_before = source_decay_alerts_canonical()
    assert any(it["metadata"].get("event_id") == eid for it in items_before)
    # Ack it
    _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "reviewed_no_action",
              "reason": "Acked for inbox-hide test verification."},
    )
    items_after = source_decay_alerts_canonical()
    assert not any(it["metadata"].get("event_id") == eid for it in items_after)


def test_inbox_show_acked_param_surfaces_acked():
    """Opt-in via show_acked=True for audit views."""
    from engine.inbox.composer import source_decay_alerts_canonical
    eid = _emit_fake_decay_alert()
    _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "reviewed_no_action",
              "reason": "Acked for show_acked param test verification."},
    )
    items = source_decay_alerts_canonical(show_acked=True)
    assert any(it["metadata"].get("event_id") == eid for it in items)


def test_inbox_unack_re_surfaces_in_inbox():
    """After unack, the alert is open again and re-appears in inbox."""
    from engine.inbox.composer import source_decay_alerts_canonical
    eid = _emit_fake_decay_alert()
    _client().post(
        f"/api/research_store/decay_alert/{eid}/acknowledge",
        json={"action": "reviewed_no_action",
              "reason": "Ack for unack-resurface test verification."},
    )
    # Hidden after ack
    assert not any(it["metadata"].get("event_id") == eid
                       for it in source_decay_alerts_canonical())
    # Unack
    _client().post(
        f"/api/research_store/decay_alert/{eid}/unacknowledge",
        json={"reason": "Re-opening for resurface test verification."},
    )
    # Re-surfaces
    assert any(it["metadata"].get("event_id") == eid
                  for it in source_decay_alerts_canonical())


# ────────────────────────────────────────────────────────────────────
# L+M — Verdict detail endpoint
# ────────────────────────────────────────────────────────────────────
def test_verdict_detail_404_on_unknown():
    r = _client().get("/api/research_store/verdict/no_such_event_id")
    assert r.status_code == 404


def test_verdict_detail_422_on_wrong_event_type():
    """A decay_alert event is not a factor_verdict event — should 422."""
    eid = _emit_fake_decay_alert()
    r = _client().get(f"/api/research_store/verdict/{eid}")
    assert r.status_code == 422
    assert "factor_verdict_filed" in r.json()["detail"]


def test_verdict_detail_returns_full_event_with_metrics():
    """Successful fetch returns the event dict with all metric blocks
    intact (anchor_orthogonality, subsample_stability, etc. if they
    were present at emit time)."""
    from engine.research_store import emit, registry
    from engine.research_store.schema import SubjectType
    if registry.resolve("test_verdict_subject") is None:
        registry.register_subject(
            "test_verdict_subject", subject_type=SubjectType.factor,
            family="test", description="Test verdict subject",
        )
    fid = emit.factor_verdict(
        subject_id="test_verdict_subject",
        verdict="MARGINAL",
        metrics={
            "sharpe": 0.75, "nw_t_stat": 1.85, "n_months": 240,
            "specification_robustness": {
                "verdict": "MARGINAL_OVERFIT",
                "stability_score": 0.45,
                "base_sharpe": 0.75, "sharpe_median": 0.34,
                "neighborhood_size": 8, "successful_cells": 9,
                "cells_tested": [],
            },
            "anchor_orthogonality": {
                "anchor_library": "ken_french_ff5_mom",
                "alpha_nw_t": 0.50,
                "betas": {"RMW": 0.55},
                "beta_nw_t": {"RMW": 5.5},
                "r2": 0.30,
            },
        },
        artifacts={}, summary="L+M verdict detail smoke",
        actor="test",
    )
    r = _client().get(f"/api/research_store/verdict/{fid}")
    assert r.status_code == 200
    d = r.json()
    assert d["event"]["event_id"] == fid
    assert d["event"]["verdict"] == "MARGINAL"
    assert d["event"]["metrics"]["specification_robustness"]["verdict"] == "MARGINAL_OVERFIT"
    assert d["event"]["metrics"]["anchor_orthogonality"]["alpha_nw_t"] == 0.50


def test_endpoint_live_real_smoke_event():
    """End-to-end against the REAL events.jsonl — the C smoke event
    we emitted earlier must surface (event_id d95aa546-...)."""
    r = _client().get("/api/research_store/decay_audit/cross_asset_carry")
    assert r.status_code == 200
    d = r.json()
    # At least one decay_alert with decay_watch tag should be visible
    # (added during C dev). If the project's events.jsonl has been
    # wiped this test won't find it — skip in that case.
    if d["n_decay_alerts"] == 0:
        import pytest
        pytest.skip("no real decay_alert events in store (clean env)")
    assert any(
        "decay_watch" in e["tags"] for e in d["decay_alerts"]
    )
