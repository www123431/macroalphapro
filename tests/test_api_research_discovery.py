"""Tests for the /api/research/discovery/* endpoints integrated into
the existing MacroAlphaPro Terminal FastAPI app.

Per user 2026-05-30 senior architecture call: the paper-discovery UI
should be part of the existing Research page, not a separate localhost
server. Endpoints live under /api/research/discovery/* matching the
existing /api/research/{graveyard, gate-runs, pit-audit} convention.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def client():
    """FastAPI TestClient against the real app."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


# ── POST /api/research/discovery/nominate ────────────────────────────────

def test_nominate_endpoint_exists(client):
    """Endpoint registered + responds (will fail without url field)."""
    r = client.post("/api/research/discovery/nominate", json={})
    assert r.status_code == 400


def test_nominate_endpoint_validation(client):
    """Missing url + id returns 400."""
    r = client.post("/api/research/discovery/nominate",
                       json={"title": "Just a title"})
    assert r.status_code == 400
    body = r.json()
    assert "missing" in body.get("detail", "").lower()


def test_nominate_endpoint_calls_nominate(client, monkeypatch):
    """Endpoint forwards to review_ui.nominate and returns its dict."""
    from engine.research.discovery import review_ui
    fake_result = {
        "ok": True,
        "title": "Carry Paper",
        "confidence": 0.55,
        "routing": "review",
        "queued_to": "discovery_queue.jsonl",
        "ident_type": "doi",
        "ident_id": "10.1/x",
    }
    monkeypatch.setattr(review_ui, "nominate", lambda x: fake_result)
    r = client.post("/api/research/discovery/nominate",
                       json={"url": "10.1/x"})
    assert r.status_code == 200
    assert r.json() == fake_result


def test_nominate_endpoint_handles_exception(client, monkeypatch):
    """Exceptions in nominate() return 500 with detail."""
    from engine.research.discovery import review_ui
    def _raise(_):
        raise RuntimeError("simulated fetch failure")
    monkeypatch.setattr(review_ui, "nominate", _raise)
    r = client.post("/api/research/discovery/nominate",
                       json={"url": "garbage"})
    assert r.status_code == 500


def test_nominate_endpoint_accepts_id_field(client, monkeypatch):
    """Alternate body field: {id: "..."} should also work."""
    from engine.research.discovery import review_ui
    monkeypatch.setattr(review_ui, "nominate",
                          lambda x: {"ok": True, "title": "X",
                                       "confidence": 0.5, "routing": "review",
                                       "queued_to": "q.jsonl",
                                       "ident_type": "arxiv",
                                       "ident_id": "2401.00001"})
    r = client.post("/api/research/discovery/nominate",
                       json={"id": "2401.00001"})
    assert r.status_code == 200


# ── GET /api/research/discovery/queues ───────────────────────────────────

def test_queues_endpoint_returns_dict_shape(client, monkeypatch, tmp_path):
    """Endpoint returns {review: [...], borderline: [...]}."""
    from engine.research.discovery import review_ui
    review_file = tmp_path / "review.jsonl"
    border_file = tmp_path / "border.jsonl"
    review_file.write_text(
        json.dumps({"title": "Review item", "ts": "2024-01-01"}) + "\n",
        encoding="utf-8",
    )
    border_file.write_text(
        json.dumps({"title": "Borderline item", "ts": "2024-01-01"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(review_ui, "DISCOVERY_QUEUE", review_file)
    monkeypatch.setattr(review_ui, "DISCOVERY_BORDERLINE", border_file)

    r = client.get("/api/research/discovery/queues?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert "review" in body
    assert "borderline" in body
    assert len(body["review"]) == 1
    assert body["review"][0]["title"] == "Review item"
    assert body["borderline"][0]["title"] == "Borderline item"


def test_queues_endpoint_empty_files(client, monkeypatch, tmp_path):
    """Missing queue files return empty lists, not 500."""
    from engine.research.discovery import review_ui
    monkeypatch.setattr(review_ui, "DISCOVERY_QUEUE", tmp_path / "no1.jsonl")
    monkeypatch.setattr(review_ui, "DISCOVERY_BORDERLINE", tmp_path / "no2.jsonl")
    r = client.get("/api/research/discovery/queues")
    assert r.status_code == 200
    assert r.json() == {"review": [], "borderline": []}


def test_queues_endpoint_limit_param(client, monkeypatch, tmp_path):
    """Limit param caps each list independently."""
    from engine.research.discovery import review_ui
    review_file = tmp_path / "review.jsonl"
    with review_file.open("w", encoding="utf-8") as f:
        for i in range(50):
            f.write(json.dumps({"i": i, "ts": f"2024-01-{i+1:02d}"}) + "\n")
    monkeypatch.setattr(review_ui, "DISCOVERY_QUEUE", review_file)
    monkeypatch.setattr(review_ui, "DISCOVERY_BORDERLINE",
                          tmp_path / "missing.jsonl")
    r = client.get("/api/research/discovery/queues?limit=5")
    assert r.status_code == 200
    assert len(r.json()["review"]) == 5


# ── GET /api/research/discovery/bookmarklet ──────────────────────────────

def test_promote_endpoint_calls_queue_action(client, monkeypatch):
    """POST /promote forwards to queue_actions.promote()."""
    from engine.research.discovery import queue_actions
    fake = {"ok": True, "mechanism_id": "x", "library_path": "lib/x.yaml",
              "original_queue": "review", "title": "X"}
    monkeypatch.setattr(queue_actions, "promote",
                          lambda *args, **kw: fake)
    r = client.post("/api/research/discovery/promote",
                       json={"source_id": "10.1/x"})
    assert r.status_code == 200
    assert r.json() == fake


def test_promote_endpoint_missing_returns_404(client, monkeypatch):
    from engine.research.discovery import queue_actions
    def _missing(*args, **kw):
        raise ValueError("not found")
    monkeypatch.setattr(queue_actions, "promote", _missing)
    r = client.post("/api/research/discovery/promote",
                       json={"source_id": "missing"})
    assert r.status_code == 404


def test_skip_endpoint_calls_queue_action(client, monkeypatch):
    from engine.research.discovery import queue_actions
    fake = {"ok": True, "rejected_path": "data/research/rejected.jsonl",
              "original_queue": "borderline", "title": "X"}
    monkeypatch.setattr(queue_actions, "skip",
                          lambda *args, **kw: fake)
    r = client.post("/api/research/discovery/skip",
                       json={"source_id": "10.1/x", "reason": "off_topic"})
    assert r.status_code == 200
    assert r.json() == fake


def test_skip_endpoint_default_reason(client, monkeypatch):
    """When reason field omitted, action still proceeds with default."""
    from engine.research.discovery import queue_actions
    captured = {}
    def _capture(*args, **kw):
        if args:
            captured["source_id"] = args[0]
        captured.update(kw)
        return {"ok": True, "rejected_path": "x", "original_queue": "review",
                "title": "Y"}
    monkeypatch.setattr(queue_actions, "skip", _capture)
    r = client.post("/api/research/discovery/skip",
                       json={"source_id": "10.1/y"})
    assert r.status_code == 200
    assert captured.get("reason") == "user_skip"


def test_promote_validates_request_body(client):
    """Missing source_id field → 422 from pydantic."""
    r = client.post("/api/research/discovery/promote",
                       json={"reason": "nothing"})
    assert r.status_code == 422


def test_bookmarklet_endpoint(client):
    r = client.get("/api/research/discovery/bookmarklet")
    assert r.status_code == 200
    body = r.json()
    for key in ("bookmarklet", "endpoint", "instructions"):
        assert key in body
    assert body["endpoint"] == "/api/research/discovery/nominate"
    assert body["bookmarklet"].startswith("javascript:")
    # Should target the real API endpoint
    assert "/api/research/discovery/nominate" in body["bookmarklet"]
    # Should use localhost:8000 (the real app port, not 8770)
    assert "localhost:8000" in body["bookmarklet"]
    # NO EMOJI per [[feedback-no-emoji-2026-05-30]]
    for emoji_char in ("📌", "📚", "🟢", "🟠"):
        assert emoji_char not in body["bookmarklet"]
