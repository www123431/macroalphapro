"""tests/test_factor_spec_api.py — Tier C-2d.1 API layer.

HTTP route coverage for /api/strengthener/factor_specs and
/factor_specs/resolve. Backend store paths redirected to tmp;
dispatcher mocked so the API test stays offline + fast.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def _spec_payload(spec_hash: str, *,
                    source_hypothesis_id: str = "hid_A",
                    family_hint:          str = "MOMENTUM",
                    signal_kind:          str = "time_series_momentum"):
    return {
        "spec_hash":            spec_hash,
        "source_hypothesis_id": source_hypothesis_id,
        "family_hint":          family_hint,
        "persisted_ts":         "2026-06-08T03:00:00Z",
        "spec": {
            "hypothesis_id":           source_hypothesis_id,
            "signal_kind":             signal_kind,
            "universe":                "us_equities_sector_etf",
            "date_range":              "2020-01:2024-12",
            "signal_inputs":           ["etf.adj_close.spy"],
            "rebal":                   "weekly",
            "weighting":               "signed_signal_volatility_targeted",
            "expected_holding_period": "weekly",
            "min_obs_months":          24,
            "pit_audits":              ["lookahead"],
            "cost_model":              "engine.execution.cost_model.basic",
            "rationale":               "test rationale",
            "extracted_ts":            "2026-06-08T00:00:00Z",
            "model":                   "claude-sonnet-4-6",
        },
    }


@pytest.fixture
def tmp_store_paths(tmp_path, monkeypatch):
    """Redirect factor_spec_store paths to tmp + mock dispatcher to
    avoid touching live data."""
    from engine.agents.strengthener import factor_spec_store as fss
    from engine.agents.strengthener import factor_dispatcher as fd
    specs = tmp_path / "factor_specs.jsonl"
    resolutions = tmp_path / "factor_spec_resolutions.jsonl"
    monkeypatch.setattr(fss, "_DEFAULT_SPECS_PATH", specs)
    monkeypatch.setattr(fss, "_DEFAULT_RESOLUTIONS_PATH", resolutions)
    # Dispatcher stub: returns a fake successful run
    monkeypatch.setattr(fd, "dispatch_factor_spec",
                          lambda spec, **kw: {
                              "dispatch_event_id": "ev_disp_X",
                              "verdict_event_id":  "ev_verd_X",
                              "template_result":   {
                                  "verdict": "GREEN",
                                  "summary": "stub green for API test",
                              },
                              "refusal":           None,
                          })
    return specs, resolutions


def _seed(specs_path: Path, rows: list[dict]):
    specs_path.parent.mkdir(parents=True, exist_ok=True)
    with specs_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ────────────────────────────────────────────────────────────────────
# GET /factor_specs
# ────────────────────────────────────────────────────────────────────
def test_list_empty(tmp_store_paths):
    r = client.get("/api/strengthener/factor_specs")
    assert r.status_code == 200
    body = r.json()
    assert body["n_pending"] == 0
    assert body["n_resolved"] == 0
    assert body["rows"] == []


def test_list_returns_pending_specs(tmp_store_paths):
    specs_path, _ = tmp_store_paths
    _seed(specs_path, [_spec_payload("hash_aaaa00", source_hypothesis_id="A"),
                         _spec_payload("hash_bbbb11", source_hypothesis_id="B")])
    r = client.get("/api/strengthener/factor_specs")
    assert r.status_code == 200
    body = r.json()
    assert body["n_pending"] == 2
    assert body["rows"][0]["spec_hash"] in {"hash_aaaa00", "hash_bbbb11"}
    # Spec shape present
    spec = body["rows"][0]["spec"]
    assert spec["signal_kind"] == "time_series_momentum"
    assert spec["universe"] == "us_equities_sector_etf"


# ────────────────────────────────────────────────────────────────────
# POST /factor_specs/resolve
# ────────────────────────────────────────────────────────────────────
def test_resolve_approved_returns_dispatch_metadata(tmp_store_paths):
    specs_path, _ = tmp_store_paths
    _seed(specs_path, [_spec_payload("hash_resolve_aa")])
    r = client.post(
        "/api/strengthener/factor_specs/resolve",
        json={"spec_hash": "hash_resolve_aa",
                "decision": "approved",
                "rationale": "looks good"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["spec_hash"] == "hash_resolve_aa"
    assert body["decision"] == "approved"
    assert body["dispatch_event_id"] == "ev_disp_X"
    assert body["verdict_event_id"] == "ev_verd_X"
    assert body["template_verdict"] == "GREEN"
    assert body["template_summary"] == "stub green for API test"
    assert body["refusal_reason"] is None


def test_resolve_rejected_skips_dispatch(tmp_store_paths):
    specs_path, _ = tmp_store_paths
    _seed(specs_path, [_spec_payload("hash_reject_aa")])
    r = client.post(
        "/api/strengthener/factor_specs/resolve",
        json={"spec_hash": "hash_reject_aa",
                "decision": "rejected",
                "rationale": "too similar"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "rejected"
    assert body["dispatch_event_id"] is None
    assert body["template_verdict"] is None


def test_resolve_unknown_spec_hash_404_400(tmp_store_paths):
    r = client.post(
        "/api/strengthener/factor_specs/resolve",
        json={"spec_hash": "definitely_not_real",
                "decision": "approved"},
    )
    assert r.status_code == 400
    assert "not found" in r.json()["detail"]


def test_resolve_bad_decision_400(tmp_store_paths):
    specs_path, _ = tmp_store_paths
    _seed(specs_path, [_spec_payload("hash_x")])
    r = client.post(
        "/api/strengthener/factor_specs/resolve",
        json={"spec_hash": "hash_x", "decision": "yolo"},
    )
    assert r.status_code == 400


# ────────────────────────────────────────────────────────────────────
# After resolve, list_pending omits resolved (default)
# ────────────────────────────────────────────────────────────────────
def test_list_after_resolve_excludes_by_default(tmp_store_paths):
    specs_path, _ = tmp_store_paths
    _seed(specs_path, [_spec_payload("hash_post")])
    # Resolve it
    client.post("/api/strengthener/factor_specs/resolve",
                 json={"spec_hash": "hash_post",
                         "decision": "approved"})
    # Default list omits resolved
    r = client.get("/api/strengthener/factor_specs")
    assert r.json()["n_pending"] == 0
    assert r.json()["n_resolved"] == 1
    assert r.json()["rows"] == []
    # include_resolved=true surfaces it
    r2 = client.get("/api/strengthener/factor_specs"
                     "?include_resolved=true")
    assert r2.json()["n_resolved"] == 1
    assert len(r2.json()["rows"]) == 1
    assert r2.json()["rows"][0]["resolved"] is True
