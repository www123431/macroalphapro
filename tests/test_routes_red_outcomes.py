"""tests/test_routes_red_outcomes.py — Stage A piece 6a.

GET /api/research/red_outcomes endpoint tests. Stubs the underlying
store + hypothesis registry + papers registry so tests are fast,
deterministic, offline.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def _ev(**kw):
    """Build a ResearchEvent-shaped namespace with sensible defaults."""
    base = dict(
        event_id="ev_default",
        event_type="factor_verdict_filed",
        subject_type="factor",
        subject_id="auto_default",
        family="behavioral_alpha",
        ts="2026-06-05T12:00:00Z",
        verdict="RED",
        summary="default summary",
        metrics={"score": 1},
        parent_event_ids=(),
        artifacts=(),
        tags=(),
        actor="engine.test",
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ────────────────────────────────────────────────────────────────────
# Happy path: 2 RED verdicts with one JOIN-able to hypothesis + paper
# ────────────────────────────────────────────────────────────────────
def test_red_outcomes_basic(monkeypatch, tmp_path):
    """Two RED verdicts; one has full join chain (hypothesis →
    source_paper_id → title); the other has no source_hypothesis_id."""
    from engine.research_store import store as st
    from engine.research_store.hypothesis import store as hyp_st
    from engine.agents.papers_curator import store as cache_st

    red_events = [
        _ev(event_id="ev_red1", subject_id="auto_aaa",
             family="behavioral_alpha", ts="2026-06-05T12:00:00Z",
             summary="rejected for overfit",
             metrics={"score": 1}),
        _ev(event_id="ev_red2", subject_id="auto_bbb",
             family="momentum_overlay", ts="2026-06-04T08:00:00Z",
             summary="DSR failed", metrics={"score": 2}),
    ]
    pipeline_starts = [
        _ev(event_type="candidate_pipeline_started",
             event_id="ev_pls1", subject_id="auto_aaa",
             ts="2026-06-04T00:00:00Z",
             metrics={"source_hypothesis_id": "h_alpha_1"}),
        # auto_bbb has NO matching pipeline_started — orphan
    ]
    def _fake_filter(event_type=None, verdict=None, since=None,
                      limit=None, **kw):
        if event_type == "factor_verdict_filed" and verdict == "RED":
            return list(red_events)
        if event_type == "candidate_pipeline_started":
            return list(pipeline_starts)
        return []
    monkeypatch.setattr(st, "filter_events", _fake_filter)

    # Mock the hypothesis registry
    h = SimpleNamespace(
        hypothesis_id="h_alpha_1", version=1,
        source_paper_id="paper_kmpv_2018",
        synthesizes_paper_ids=(),
    )
    monkeypatch.setattr(hyp_st, "load_hypotheses", lambda: [h])

    # Mock the papers registry (write a real file)
    import json, pathlib
    papers_path = (pathlib.Path(__file__).resolve().parent.parent
                    / "data" / "research_store" / "papers_registry.jsonl")
    # Don't touch the real one — just stub the path constant
    import api.routes_research_tools as rrt
    fake_registry = tmp_path / "papers_registry.jsonl"
    fake_registry.write_text(json.dumps({
        "paper_id": "paper_kmpv_2018", "title": "Carry",
    }) + "\n", encoding="utf-8")
    # The route reads via REPO_ROOT/data/research_store/papers_registry.jsonl;
    # monkeypatch the module's REPO_ROOT to tmp_path's parent layout.
    monkeypatch.setattr(rrt, "REPO_ROOT", tmp_path.parent)
    # Build the expected path
    (tmp_path.parent / "data" / "research_store").mkdir(
        parents=True, exist_ok=True)
    real_target = (tmp_path.parent / "data" / "research_store"
                    / "papers_registry.jsonl")
    real_target.write_text(json.dumps({
        "paper_id": "paper_kmpv_2018", "title": "Carry",
    }) + "\n", encoding="utf-8")

    r = client.get("/api/research/red_outcomes?days=30&limit=50")
    assert r.status_code == 200
    body = r.json()
    assert body["n_total"] == 2
    assert body["n_returned"] == 2
    items = body["items"]
    # Item 1: full JOIN
    by_subject = {it["subject_id"]: it for it in items}
    assert by_subject["auto_aaa"]["source_hypothesis_id"] == "h_alpha_1"
    assert by_subject["auto_aaa"]["source_paper_id"] == "paper_kmpv_2018"
    assert by_subject["auto_aaa"]["source_paper_title"] == "Carry"
    assert by_subject["auto_aaa"]["family"] == "behavioral_alpha"
    assert by_subject["auto_aaa"]["score"] == 1
    # Item 2: no pipeline_started match → nulls
    assert by_subject["auto_bbb"]["source_hypothesis_id"] is None
    assert by_subject["auto_bbb"]["source_paper_id"] is None
    assert by_subject["auto_bbb"]["source_paper_title"] is None
    assert by_subject["auto_bbb"]["score"] == 2


# ────────────────────────────────────────────────────────────────────
# Empty case
# ────────────────────────────────────────────────────────────────────
def test_red_outcomes_empty(monkeypatch):
    """No RED events in window → n_total=0, items=[]."""
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events",
                          lambda **kw: [])

    r = client.get("/api/research/red_outcomes?days=30")
    assert r.status_code == 200
    body = r.json()
    assert body["n_total"] == 0
    assert body["n_returned"] == 0
    assert body["items"] == []


# ────────────────────────────────────────────────────────────────────
# JOIN gracefully falls back when hypothesis store can't load
# ────────────────────────────────────────────────────────────────────
def test_red_outcomes_handles_hypothesis_store_failure(
    monkeypatch, tmp_path,
):
    """If load_hypotheses raises, items still come back with null
    source_* fields — the verdict itself is still useful."""
    from engine.research_store import store as st
    from engine.research_store.hypothesis import store as hyp_st

    monkeypatch.setattr(st, "filter_events", lambda **kw:
        [_ev(event_id="ev_red1", subject_id="auto_aaa")]
        if kw.get("verdict") == "RED" else []
    )
    def _raise():
        raise RuntimeError("hypothesis store broken")
    monkeypatch.setattr(hyp_st, "load_hypotheses", _raise)

    r = client.get("/api/research/red_outcomes")
    assert r.status_code == 200
    body = r.json()
    assert body["n_total"] == 1
    assert body["items"][0]["source_hypothesis_id"] is None
    assert body["items"][0]["source_paper_id"] is None


# ────────────────────────────────────────────────────────────────────
# Synthesizes_paper_ids fallback (used when source_paper_id is empty)
# ────────────────────────────────────────────────────────────────────
def test_red_outcomes_uses_synthesizes_paper_ids_fallback(
    monkeypatch, tmp_path,
):
    """LLM_SYNTHESIS hypotheses have source_paper_id='' but populate
    synthesizes_paper_ids; first paper there should resolve."""
    from engine.research_store import store as st
    from engine.research_store.hypothesis import store as hyp_st
    import api.routes_research_tools as rrt

    monkeypatch.setattr(st, "filter_events", lambda **kw: (
        [_ev(event_id="ev_red1", subject_id="auto_aaa")]
        if kw.get("verdict") == "RED"
        else (
            [_ev(event_type="candidate_pipeline_started",
                  subject_id="auto_aaa",
                  metrics={"source_hypothesis_id": "h_synth_1"})]
            if kw.get("event_type") == "candidate_pipeline_started"
            else []
        )
    ))
    h = SimpleNamespace(
        hypothesis_id="h_synth_1", version=1,
        source_paper_id="",  # empty
        synthesizes_paper_ids=("paper_carry_2018", "paper_other_2017"),
    )
    monkeypatch.setattr(hyp_st, "load_hypotheses", lambda: [h])

    # Papers registry has the first one
    import json
    (tmp_path / "data" / "research_store").mkdir(parents=True)
    (tmp_path / "data" / "research_store"
     / "papers_registry.jsonl").write_text(json.dumps({
        "paper_id": "paper_carry_2018", "title": "Synth-derived Carry",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(rrt, "REPO_ROOT", tmp_path)

    r = client.get("/api/research/red_outcomes")
    assert r.status_code == 200
    body = r.json()
    it = body["items"][0]
    assert it["source_paper_id"] == "paper_carry_2018"
    assert it["source_paper_title"] == "Synth-derived Carry"


# ────────────────────────────────────────────────────────────────────
# Validation — days / limit bounds
# ────────────────────────────────────────────────────────────────────
def test_red_outcomes_rejects_invalid_days():
    r = client.get("/api/research/red_outcomes?days=0")
    assert r.status_code == 422   # ge=1


def test_red_outcomes_rejects_invalid_limit():
    r = client.get("/api/research/red_outcomes?limit=999999")
    assert r.status_code == 422   # le=500


# ────────────────────────────────────────────────────────────────────
# Response schema stability — required keys present
# ────────────────────────────────────────────────────────────────────
def test_red_outcomes_response_schema(monkeypatch):
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: (
        [_ev(event_id="ev_red1", subject_id="auto_aaa")]
        if kw.get("verdict") == "RED" else []
    ))
    r = client.get("/api/research/red_outcomes")
    body = r.json()
    required_top = {"since", "n_total", "n_returned", "items"}
    assert required_top <= set(body.keys())
    if body["items"]:
        item = body["items"][0]
        required_item = {"event_id", "subject_id", "family",
                          "verdict_ts", "score", "summary",
                          "source_hypothesis_id", "source_paper_id",
                          "source_paper_title"}
        assert required_item <= set(item.keys())
