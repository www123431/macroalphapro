"""tests/test_strengthener_approvals.py — Phase 2.0 step 12.

Covers both layers:
  - approval_view: filtering, resolution merge, FIFO ordering
  - HTTP route: list + resolve endpoints

No real I/O — uses tmp_path for both verdicts.jsonl and resolutions.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────
# Fixture writers
# ─────────────────────────────────────────────────────────────────────
def _approve_verdict_dict(hid: str, *, ts: str = "2026-06-06T12:00:00Z"):
    return {
        "hypothesis_id":               hid,
        "verdict_type":                "APPROVE_FOR_PIPELINE",
        "one_line_summary":            f"Approve {hid}",
        "confidence":                  0.72,
        "reasoning":                   f"strong differentiator for {hid}",
        "similar_to_deployed":         None,
        "replaces_decaying":           None,
        "blocking_doctrine_id":        None,
        "proposed_amendment_summary":  None,
        "recommended_pipeline_action": "run f14b strict gate",
        "risk_flags":                  ["short sample"],
        "review_ts":                   ts,
        "model":                       "claude-sonnet-4-6",
    }


def _amendment_verdict_dict(hid: str, *, ts: str = "2026-06-06T12:30:00Z"):
    return {
        "hypothesis_id":               hid,
        "verdict_type":                "DOCTRINE_AMENDMENT_NEEDED",
        "one_line_summary":            f"Amendment proposal for {hid}",
        "confidence":                  0.65,
        "reasoning":                   "blocked only by stale doctrine",
        "similar_to_deployed":         None,
        "replaces_decaying":           None,
        "blocking_doctrine_id":        "project-cross-asset-breadth-2026-05-28",
        "proposed_amendment_summary":  "carve out EM sov QMJ from the ban",
        "recommended_pipeline_action": None,
        "risk_flags":                  ["doctrine risk"],
        "review_ts":                   ts,
        "model":                       "claude-sonnet-4-6",
    }


def _reject_verdict_dict(hid: str, *, ts: str = "2026-06-06T13:00:00Z"):
    return {
        "hypothesis_id":               hid,
        "verdict_type":                "REJECT",
        "one_line_summary":            f"reject {hid}",
        "confidence":                  0.85,
        "reasoning":                   "too similar",
        "similar_to_deployed":         "cross_asset_carry",
        "replaces_decaying":           None,
        "blocking_doctrine_id":        None,
        "proposed_amendment_summary":  None,
        "recommended_pipeline_action": None,
        "risk_flags":                  [],
        "review_ts":                   ts,
        "model":                       "claude-sonnet-4-6",
    }


def _seed(tmp_path: Path, verdicts: list[dict],
            resolutions: list[dict] | None = None) -> tuple[Path, Path]:
    vp = tmp_path / "verdicts.jsonl"
    rp = tmp_path / "resolutions.jsonl"
    vp.parent.mkdir(parents=True, exist_ok=True)
    with vp.open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v) + "\n")
    if resolutions:
        with rp.open("w", encoding="utf-8") as f:
            for r in resolutions:
                f.write(json.dumps(r) + "\n")
    return vp, rp


def _patch_paths(monkeypatch, vp: Path, rp: Path):
    """Redirect both modules' default paths to tmp."""
    from engine.agents.strengthener import approval_view as av
    monkeypatch.setattr(av, "_DEFAULT_VERDICTS_PATH", vp)
    monkeypatch.setattr(av, "_DEFAULT_RESOLUTIONS_PATH", rp)


# ─────────────────────────────────────────────────────────────────────
# view: list_pending_approvals — direct module tests
# ─────────────────────────────────────────────────────────────────────
def test_empty_returns_zero(tmp_path, monkeypatch):
    from engine.agents.strengthener.approval_view import list_pending_approvals
    vp, rp = _seed(tmp_path, [])
    out = list_pending_approvals(verdicts_path=vp, resolutions_path=rp)
    assert out == {"n_pending": 0, "n_resolved": 0, "rows": []}


def test_reject_verdicts_not_surfaced(tmp_path):
    """REJECT means B decided — no human action; must not appear."""
    from engine.agents.strengthener.approval_view import list_pending_approvals
    vp, rp = _seed(tmp_path, [_reject_verdict_dict("syn1")])
    out = list_pending_approvals(verdicts_path=vp, resolutions_path=rp)
    assert out["n_pending"] == 0
    assert out["rows"] == []


def test_approve_and_amendment_surfaced(tmp_path):
    from engine.agents.strengthener.approval_view import list_pending_approvals
    vp, rp = _seed(tmp_path, [
        _approve_verdict_dict("syn1"),
        _amendment_verdict_dict("syn2"),
        _reject_verdict_dict("syn3"),
    ])
    out = list_pending_approvals(verdicts_path=vp, resolutions_path=rp)
    assert out["n_pending"] == 2
    types = {r["verdict_type"] for r in out["rows"]}
    assert types == {"APPROVE_FOR_PIPELINE", "DOCTRINE_AMENDMENT_NEEDED"}


def test_resolved_verdicts_hidden_by_default(tmp_path):
    from engine.agents.strengthener.approval_view import list_pending_approvals
    vp, rp = _seed(tmp_path, [
        _approve_verdict_dict("syn1"),
        _approve_verdict_dict("syn2"),
    ], resolutions=[
        {"hypothesis_id": "syn1", "decision": "approved",
         "rationale": "ok", "resolved_ts": "2026-06-06T14:00:00Z",
         "resolved_by": "user"},
    ])
    out = list_pending_approvals(verdicts_path=vp, resolutions_path=rp)
    assert out["n_pending"] == 1
    assert out["n_resolved"] == 1
    assert out["rows"][0]["hypothesis_id"] == "syn2"


def test_include_resolved_surfaces_both(tmp_path):
    from engine.agents.strengthener.approval_view import list_pending_approvals
    vp, rp = _seed(tmp_path, [
        _approve_verdict_dict("syn1"),
        _approve_verdict_dict("syn2"),
    ], resolutions=[
        {"hypothesis_id": "syn1", "decision": "approved",
         "rationale": "ok", "resolved_ts": "2026-06-06T14:00:00Z",
         "resolved_by": "user"},
    ])
    out = list_pending_approvals(verdicts_path=vp, resolutions_path=rp,
                                    include_resolved=True)
    assert len(out["rows"]) == 2
    # Pending row first, resolved row after
    assert out["rows"][0]["hypothesis_id"] == "syn2"
    assert out["rows"][0]["resolved"] is False
    assert out["rows"][1]["hypothesis_id"] == "syn1"
    assert out["rows"][1]["resolved"] is True
    assert out["rows"][1]["resolution"]["decision"] == "approved"


def test_pending_ordered_fifo_oldest_first(tmp_path):
    """Principal queue must be FIFO — oldest review first."""
    from engine.agents.strengthener.approval_view import list_pending_approvals
    vp, rp = _seed(tmp_path, [
        _approve_verdict_dict("newest", ts="2026-06-06T15:00:00Z"),
        _approve_verdict_dict("oldest", ts="2026-06-06T08:00:00Z"),
        _approve_verdict_dict("middle", ts="2026-06-06T12:00:00Z"),
    ])
    out = list_pending_approvals(verdicts_path=vp, resolutions_path=rp)
    ids = [r["hypothesis_id"] for r in out["rows"]]
    assert ids == ["oldest", "middle", "newest"]


def test_latest_resolution_wins(tmp_path):
    """If a verdict has multiple resolution rows, latest by ts wins."""
    from engine.agents.strengthener.approval_view import list_pending_approvals
    vp, rp = _seed(tmp_path, [_approve_verdict_dict("syn1")], resolutions=[
        {"hypothesis_id": "syn1", "decision": "deferred",
         "rationale": "later", "resolved_ts": "2026-06-06T14:00:00Z",
         "resolved_by": "user"},
        {"hypothesis_id": "syn1", "decision": "approved",
         "rationale": "ok now", "resolved_ts": "2026-06-06T15:00:00Z",
         "resolved_by": "user"},
    ])
    out = list_pending_approvals(verdicts_path=vp, resolutions_path=rp,
                                    include_resolved=True)
    assert out["n_resolved"] == 1
    assert out["rows"][0]["resolution"]["decision"] == "approved"


# ─────────────────────────────────────────────────────────────────────
# view: append_resolution
# ─────────────────────────────────────────────────────────────────────
def test_append_resolution_writes_row(tmp_path):
    from engine.agents.strengthener.approval_view import append_resolution
    rp = tmp_path / "resolutions.jsonl"
    r = append_resolution(
        hypothesis_id="syn1", decision="approved",
        rationale="looks good", path=rp,
    )
    assert r.decision == "approved"
    assert rp.is_file()
    rows = [json.loads(ln) for ln in rp.read_text(encoding="utf-8").strip().split("\n")]
    assert len(rows) == 1
    assert rows[0]["decision"] == "approved"
    assert rows[0]["resolved_by"] == "user"


def test_append_resolution_rejects_unknown_decision(tmp_path):
    from engine.agents.strengthener.approval_view import append_resolution
    import pytest
    with pytest.raises(ValueError):
        append_resolution(hypothesis_id="syn1", decision="maybe",
                            path=tmp_path / "r.jsonl")


# ─────────────────────────────────────────────────────────────────────
# HTTP route
# ─────────────────────────────────────────────────────────────────────
def test_route_list_returns_pending(tmp_path, monkeypatch):
    vp, rp = _seed(tmp_path, [
        _approve_verdict_dict("syn1"),
        _reject_verdict_dict("syn2"),
    ])
    _patch_paths(monkeypatch, vp, rp)
    r = client.get("/api/strengthener/approvals")
    assert r.status_code == 200
    body = r.json()
    assert body["n_pending"] == 1
    assert body["rows"][0]["verdict_type"] == "APPROVE_FOR_PIPELINE"


def test_route_resolve_writes_resolution(tmp_path, monkeypatch):
    vp, rp = _seed(tmp_path, [_approve_verdict_dict("syn1")])
    _patch_paths(monkeypatch, vp, rp)
    r = client.post("/api/strengthener/approvals/resolve",
                      json={"hypothesis_id": "syn1",
                            "decision":      "approved",
                            "rationale":     "yes ship it"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["decision"] == "approved"
    # And the pending list now shows it resolved
    r2 = client.get("/api/strengthener/approvals")
    assert r2.json()["n_pending"] == 0
    assert r2.json()["n_resolved"] == 1


def test_route_resolve_rejects_bad_decision(tmp_path, monkeypatch):
    vp, rp = _seed(tmp_path, [_approve_verdict_dict("syn1")])
    _patch_paths(monkeypatch, vp, rp)
    r = client.post("/api/strengthener/approvals/resolve",
                      json={"hypothesis_id": "syn1",
                            "decision":      "maybe"})
    assert r.status_code == 400


def test_route_resolve_validates_request_shape(tmp_path, monkeypatch):
    """Missing required fields → 422 from pydantic."""
    vp, rp = _seed(tmp_path, [_approve_verdict_dict("syn1")])
    _patch_paths(monkeypatch, vp, rp)
    r = client.post("/api/strengthener/approvals/resolve",
                      json={"decision": "approved"})
    assert r.status_code == 422


def test_route_include_resolved_query_param(tmp_path, monkeypatch):
    vp, rp = _seed(tmp_path, [_approve_verdict_dict("syn1")], resolutions=[
        {"hypothesis_id": "syn1", "decision": "approved",
         "rationale": "ok", "resolved_ts": "2026-06-06T14:00:00Z",
         "resolved_by": "user"},
    ])
    _patch_paths(monkeypatch, vp, rp)
    r1 = client.get("/api/strengthener/approvals")
    assert r1.json()["rows"] == []
    r2 = client.get("/api/strengthener/approvals?include_resolved=true")
    assert len(r2.json()["rows"]) == 1


# ─────────────────────────────────────────────────────────────────────
# Phase 2.1a: forward_vector_created event on `approved`
# ─────────────────────────────────────────────────────────────────────
def test_resolve_approved_emits_forward_vector_created(tmp_path, monkeypatch):
    """When the principal clicks `approved`, the route MUST emit
    forward_vector_created so generate_forward_vectors (P2.1b) can
    pick the hypothesis up. The decision=rejected or deferred paths
    do NOT emit (those don't qualify for /research/forward queue)."""
    vp, rp = _seed(tmp_path, [_approve_verdict_dict("syn1", ts="2026-06-06T11:00:00Z")])
    _patch_paths(monkeypatch, vp, rp)

    # Patch emit + hypothesis lookup
    captured: list = []
    from engine.research_store import emit as emit_mod
    monkeypatch.setattr(emit_mod, "forward_vector_created",
                          lambda **kw: captured.append(kw) or "ev_fv")
    # Patch hypothesis store so find_by_id returns a synthetic row
    class _FakeHyp:
        class _Fam: value = "VOL_RISK_PREMIUM"
        class _Em:  value = "llm_synthesis"
        mechanism_family = _Fam()
        extraction_method = _Em()
    from engine.research_store.hypothesis import store as hyp_store
    monkeypatch.setattr(hyp_store, "find_by_id", lambda hid: _FakeHyp() if hid == "syn1" else None)

    r = client.post("/api/strengthener/approvals/resolve",
                      json={"hypothesis_id": "syn1",
                            "decision":      "approved",
                            "rationale":     "yes ship"})
    assert r.status_code == 200
    assert len(captured) == 1
    ev = captured[0]
    assert ev["hypothesis_id"] == "syn1"
    assert ev["verdict_type"] == "APPROVE_FOR_PIPELINE"
    assert ev["b_confidence"] == 0.72
    assert ev["extraction_method"] == "llm_synthesis"
    assert ev["mechanism_family"] == "VOL_RISK_PREMIUM"


def test_resolve_rejected_does_not_emit(tmp_path, monkeypatch):
    vp, rp = _seed(tmp_path, [_approve_verdict_dict("syn1")])
    _patch_paths(monkeypatch, vp, rp)
    captured: list = []
    from engine.research_store import emit as emit_mod
    monkeypatch.setattr(emit_mod, "forward_vector_created",
                          lambda **kw: captured.append(kw) or "ev_fv")
    r = client.post("/api/strengthener/approvals/resolve",
                      json={"hypothesis_id": "syn1",
                            "decision":      "rejected"})
    assert r.status_code == 200
    assert captured == []


def test_resolve_deferred_does_not_emit(tmp_path, monkeypatch):
    vp, rp = _seed(tmp_path, [_approve_verdict_dict("syn1")])
    _patch_paths(monkeypatch, vp, rp)
    captured: list = []
    from engine.research_store import emit as emit_mod
    monkeypatch.setattr(emit_mod, "forward_vector_created",
                          lambda **kw: captured.append(kw) or "ev_fv")
    r = client.post("/api/strengthener/approvals/resolve",
                      json={"hypothesis_id": "syn1",
                            "decision":      "deferred"})
    assert r.status_code == 200
    assert captured == []


def test_resolve_emit_failure_does_not_roll_back_resolution(tmp_path, monkeypatch):
    """If forward_vector_created emit blows up, the resolution row
    MUST still persist. Resolution is the authoritative record; the
    event is an audit/downstream signal — emit failures are logged
    + swallowed, not propagated."""
    vp, rp = _seed(tmp_path, [_approve_verdict_dict("syn1")])
    _patch_paths(monkeypatch, vp, rp)
    from engine.research_store import emit as emit_mod
    def _boom(**kw):
        raise RuntimeError("registry down")
    monkeypatch.setattr(emit_mod, "forward_vector_created", _boom)

    r = client.post("/api/strengthener/approvals/resolve",
                      json={"hypothesis_id": "syn1",
                            "decision":      "approved"})
    assert r.status_code == 200   # not 500
    # Resolution still recorded
    r2 = client.get("/api/strengthener/approvals?include_resolved=true")
    assert r2.json()["n_resolved"] == 1


# ─────────────────────────────────────────────────────────────────────
# Phase 2.0 step 13: DOCTRINE_AMENDMENT_NEEDED handler
# ─────────────────────────────────────────────────────────────────────
def _amendment_verdict_dict(hid: str, *, blocking="project-test-doctrine",
                              ts: str = "2026-06-06T12:30:00Z"):
    return {
        "hypothesis_id":               hid,
        "verdict_type":                "DOCTRINE_AMENDMENT_NEEDED",
        "one_line_summary":            f"Amendment proposal for {hid}",
        "confidence":                  0.7,
        "reasoning":                   "candidate strong but blocked by stale doctrine X",
        "similar_to_deployed":         None,
        "replaces_decaying":           None,
        "blocking_doctrine_id":        blocking,
        "proposed_amendment_summary":  "Carve out EM sov QMJ from the cross-asset-breadth ban",
        "recommended_pipeline_action": None,
        "risk_flags":                  ["doctrine risk"],
        "review_ts":                   ts,
        "model":                       "claude-sonnet-4-6",
    }


def test_resolve_amendment_writes_draft_and_emits_proposed(tmp_path, monkeypatch):
    """AMENDMENT approval path: writes draft + emits memory_amendment_proposed,
    does NOT emit forward_vector_created."""
    vp, rp = _seed(tmp_path, [_amendment_verdict_dict("syn-amend")])
    _patch_paths(monkeypatch, vp, rp)
    # Redirect amendment-drafts dir into tmp
    from engine.agents.strengthener import approval_view as av
    monkeypatch.setattr(av, "_DEFAULT_AMENDMENT_DRAFTS_DIR",
                          tmp_path / "drafts")

    # Capture both possible emits
    from engine.research_store import emit as emit_mod
    fv_calls: list = []
    amend_calls: list = []
    monkeypatch.setattr(emit_mod, "forward_vector_created",
                          lambda **kw: fv_calls.append(kw) or "ev_fv")
    monkeypatch.setattr(emit_mod, "memory_amendment_proposed",
                          lambda **kw: amend_calls.append(kw) or "ev_amend")

    r = client.post("/api/strengthener/approvals/resolve",
                      json={"hypothesis_id": "syn-amend",
                            "decision":      "approved",
                            "rationale":     "yes draft it"})
    assert r.status_code == 200

    # forward_vector_created MUST NOT fire for amendment path
    assert fv_calls == []
    # memory_amendment_proposed MUST fire
    assert len(amend_calls) == 1
    ev = amend_calls[0]
    assert ev["hypothesis_id"] == "syn-amend"
    assert ev["blocking_doctrine_id"] == "project-test-doctrine"
    assert "EM sov QMJ" in ev["proposed_amendment_summary"]
    assert ev["b_confidence"] == 0.7

    # Draft file written
    draft_dir = tmp_path / "drafts"
    files = list(draft_dir.glob("amendment_*.md"))
    assert len(files) == 1
    body = files[0].read_text(encoding="utf-8")
    assert "project-test-doctrine" in body
    assert "EM sov QMJ" in body
    assert "stale doctrine" in body
    # The Next-step manual instruction must surface
    assert "memory_doctrine_locked" in body


def test_resolve_amendment_missing_blocking_id_skipped(tmp_path, monkeypatch):
    """Defensive: AMENDMENT verdict without blocking_doctrine_id is
    malformed (B's parser should reject upstream); resolve still
    succeeds + no emit + no draft."""
    bad = _amendment_verdict_dict("syn-bad")
    bad["blocking_doctrine_id"] = ""
    vp, rp = _seed(tmp_path, [bad])
    _patch_paths(monkeypatch, vp, rp)
    from engine.agents.strengthener import approval_view as av
    monkeypatch.setattr(av, "_DEFAULT_AMENDMENT_DRAFTS_DIR",
                          tmp_path / "drafts")
    from engine.research_store import emit as emit_mod
    amend_calls: list = []
    monkeypatch.setattr(emit_mod, "memory_amendment_proposed",
                          lambda **kw: amend_calls.append(kw) or "x")

    r = client.post("/api/strengthener/approvals/resolve",
                      json={"hypothesis_id": "syn-bad",
                            "decision":      "approved"})
    assert r.status_code == 200
    assert amend_calls == []
    # No draft file
    assert not (tmp_path / "drafts").exists() or \
           list((tmp_path / "drafts").glob("*")) == []


def test_resolve_unknown_verdict_type_skipped(tmp_path, monkeypatch):
    """Future-proof: an unknown verdict_type should be logged + skipped,
    not crash + not mis-emit."""
    weird = _approve_verdict_dict("syn-weird")
    weird["verdict_type"] = "FUTURE_VERDICT_TYPE"
    vp, rp = _seed(tmp_path, [weird])
    _patch_paths(monkeypatch, vp, rp)
    from engine.research_store import emit as emit_mod
    fv_calls: list = []
    amend_calls: list = []
    monkeypatch.setattr(emit_mod, "forward_vector_created",
                          lambda **kw: fv_calls.append(kw) or "x")
    monkeypatch.setattr(emit_mod, "memory_amendment_proposed",
                          lambda **kw: amend_calls.append(kw) or "x")
    r = client.post("/api/strengthener/approvals/resolve",
                      json={"hypothesis_id": "syn-weird",
                            "decision":      "approved"})
    assert r.status_code == 200
    assert fv_calls == []
    assert amend_calls == []


# ─────────────────────────────────────────────────────────────────────
# Direct unit tests for write_amendment_draft
# ─────────────────────────────────────────────────────────────────────
def test_write_amendment_draft_file_structure(tmp_path):
    from engine.agents.strengthener.approval_view import write_amendment_draft
    p = write_amendment_draft(
        hypothesis_id              = "hyp-xyz",
        blocking_doctrine_id       = "feedback-test-doctrine-2026",
        proposed_amendment_summary = "Allow X in case Y",
        b_reasoning                = "evidence reasoning A B C",
        b_confidence               = 0.78,
        drafts_dir                 = tmp_path / "d",
    )
    assert p.is_file()
    body = p.read_text(encoding="utf-8")
    # Headers + content surfaces
    assert "feedback-test-doctrine-2026" in body
    assert "hyp-xyz" in body
    assert "0.78" in body
    assert "Allow X in case Y" in body
    assert "evidence reasoning A B C" in body
    # Manual-next-step instruction surfaces
    assert "memory_doctrine_locked" in body
    # NOT autonomous warning
    assert "do not autonomously rewrite" in body.lower()


def test_write_amendment_draft_handles_empty_summary_gracefully(tmp_path):
    """Edge: if B's proposed_amendment_summary is empty, the file still
    writes (with a placeholder) — caller checked blocking_doctrine_id
    upstream, that's the real prereq."""
    from engine.agents.strengthener.approval_view import write_amendment_draft
    p = write_amendment_draft(
        hypothesis_id              = "hyp-empty",
        blocking_doctrine_id       = "doc-x",
        proposed_amendment_summary = "",
        b_reasoning                = "x",
        b_confidence               = 0.5,
        drafts_dir                 = tmp_path / "d",
    )
    body = p.read_text(encoding="utf-8")
    assert "(empty" in body   # placeholder hint


def test_resolve_skips_emit_when_no_verdict_found(tmp_path, monkeypatch):
    """If the user POSTs approved for a hypothesis_id that has no
    verdict in verdicts.jsonl (shouldn't happen via UI but defensive),
    we DO record the resolution but SKIP the emit (nothing to base
    it on)."""
    # No verdicts seeded; only an empty store
    vp = tmp_path / "verdicts.jsonl"
    rp = tmp_path / "resolutions.jsonl"
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_text("", encoding="utf-8")
    _patch_paths(monkeypatch, vp, rp)
    captured: list = []
    from engine.research_store import emit as emit_mod
    monkeypatch.setattr(emit_mod, "forward_vector_created",
                          lambda **kw: captured.append(kw) or "ev_fv")
    r = client.post("/api/strengthener/approvals/resolve",
                      json={"hypothesis_id": "ghost",
                            "decision":      "approved"})
    assert r.status_code == 200
    assert captured == []   # no verdict → no emit
