"""tests/test_routes_papers_curator_synthesis.py — Phase 2.0 step 5b.

POST /api/papers_curator/synthesis/run endpoint tests.

We mock the runner (`run_synthesis_pipeline`) so:
  - tests are fast + deterministic
  - no LLM cost
  - the route's contract (request validation + response shape +
    structured error pass-through) is verified independently of
    runner internals (which have their own tests).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def _mock_result(**overrides) -> dict:
    base = {
        "run_ts":                 "2026-06-06T13:00:00Z",
        "dry_run":                True,
        "snapshot":               {"snapshot_ts": "2026-06-06T13:00:00Z",
                                    "recent_summaries": 6,
                                    "deployed_sleeves": 5,
                                    "recent_events": 40,
                                    "doctrine_snippets": 0},
        "candidates":             [],
        "n_candidates":           0,
        "written_hypothesis_ids": [],
        "n_written":              0,
        "errors":                 [],
    }
    base.update(overrides)
    return base


def _patch_runner(monkeypatch, result_factory):
    """Monkeypatch run_synthesis_pipeline AT THE ROUTE'S IMPORT SITE.
    The route imports lazily inside the handler, so we patch the
    source module."""
    from engine.agents.papers_curator import synthesis_runner
    monkeypatch.setattr(synthesis_runner, "run_synthesis_pipeline",
                          result_factory)


# ─────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────
def test_dry_run_default(monkeypatch):
    captured = {}
    def _fake(**kw):
        captured.update(kw)
        return _mock_result()
    _patch_runner(monkeypatch, _fake)

    r = client.post("/api/papers_curator/synthesis/run", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["dry_run"] is True
    assert body["snapshot"]["recent_summaries"] == 6
    assert body["n_candidates"] == 0
    assert body["errors"] == []
    # The default request should have hit the runner with dry_run=True
    assert captured["dry_run"] is True


def test_explicit_persist_mode_passed_through(monkeypatch):
    captured = {}
    def _fake(**kw):
        captured.update(kw)
        return _mock_result(dry_run=False, n_written=2,
                             written_hypothesis_ids=["h1", "h2"])
    _patch_runner(monkeypatch, _fake)

    r = client.post("/api/papers_curator/synthesis/run",
                      json={"dry_run": False,
                            "extra_tags": ["session:cos-2026-06-06"]})
    assert r.status_code == 200
    body = r.json()
    assert body["dry_run"] is False
    assert body["n_written"] == 2
    assert body["written_hypothesis_ids"] == ["h1", "h2"]
    # Tags propagated as tuple to the runner
    assert captured["dry_run"] is False
    assert tuple(captured["extra_tags"]) == ("session:cos-2026-06-06",)


def test_candidates_returned_with_full_payload(monkeypatch):
    """The candidates list must include the rich generation metadata
    (cochrane_frame / conflicts / prior) — the UI renders them."""
    rich_candidate = {
        "claim": "EM carry refresh", "mechanism_family": "carry",
        "mechanism_subtype": "qmj_em_sov", "predicted_direction": "positive",
        "predicted_magnitude": "Sharpe 0.5+",
        "required_data": ["EM bond returns"],
        "test_methodology": "long-short decile",
        "synthesizes_paper_ids": ["arxiv/p1"],
        "synthesizes_event_ids": ["ev1"],
        "addresses_decay_in": None,
        "cochrane_frame": "risk",
        "novelty_vs_known": "extension",
        "estimated_n_trials_in_family": 5,
        "graveyard_conflicts": [],
        "doctrine_conflicts": [],
        "expected_outcome_prior": "marginal_per_HXZ",
        "generation_ts": "2026-06-06T13:00:00Z",
        "model": "claude-sonnet-4-6",
    }
    _patch_runner(monkeypatch, lambda **kw: _mock_result(
        candidates=[rich_candidate], n_candidates=1))

    r = client.post("/api/papers_curator/synthesis/run", json={})
    assert r.status_code == 200
    body = r.json()
    c = body["candidates"][0]
    assert c["cochrane_frame"] == "risk"
    assert c["expected_outcome_prior"] == "marginal_per_HXZ"
    assert c["synthesizes_paper_ids"] == ["arxiv/p1"]


# ─────────────────────────────────────────────────────────────────────
# Structured error pass-through
# ─────────────────────────────────────────────────────────────────────
def test_errors_returned_as_200_with_error_array(monkeypatch):
    """When the runner records an error (e.g. LLM failed), the route
    must return 200 + the error in response.errors, NOT a 500. The UI
    needs to render partial results."""
    _patch_runner(monkeypatch, lambda **kw: _mock_result(
        errors=["synthesize: rate limited"]))

    r = client.post("/api/papers_curator/synthesis/run", json={})
    assert r.status_code == 200
    assert r.json()["errors"] == ["synthesize: rate limited"]


def test_unrecoverable_500_on_import_failure(monkeypatch):
    """If the runner module itself can't be imported (e.g. an install
    issue), the endpoint surfaces 500 — that's a deployment bug."""
    import sys
    # Force ImportError by injecting a broken module entry
    monkeypatch.setitem(
        sys.modules,
        "engine.agents.papers_curator.synthesis_runner",
        None,  # `import None` raises TypeError → handler should 500
    )
    r = client.post("/api/papers_curator/synthesis/run", json={})
    assert r.status_code == 500


# ─────────────────────────────────────────────────────────────────────
# Request validation
# ─────────────────────────────────────────────────────────────────────
def test_request_validation_rejects_wrong_type(monkeypatch):
    """dry_run must be a bool — sending a string is a 422 from
    FastAPI's pydantic validation, BEFORE we hit the runner."""
    _patch_runner(monkeypatch, lambda **kw: _mock_result())
    r = client.post("/api/papers_curator/synthesis/run",
                      json={"dry_run": "not_a_bool"})
    assert r.status_code == 422


def test_defaults_applied_when_fields_omitted(monkeypatch):
    captured = {}
    def _fake(**kw):
        captured.update(kw)
        return _mock_result()
    _patch_runner(monkeypatch, _fake)
    r = client.post("/api/papers_curator/synthesis/run", json={})
    assert r.status_code == 200
    assert captured["summaries_days"] == 14
    assert captured["events_days"] == 30
    assert captured["extra_tags"] == ()
