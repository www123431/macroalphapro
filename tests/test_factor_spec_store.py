"""tests/test_factor_spec_store.py — Tier C-2d.1 backend.

Tests for the factor SPEC approval persistence layer +
API endpoints. Fully offline:
  - llm_call mocked (no Sonnet)
  - dispatch_factor_spec mocked where needed
  - file paths redirected to tmp_path
  - registry / event store redirected to tmp via factor_verdict_emit's
    fixture pattern (reused — defense against the C-2c test pollution
    incident)
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest


def _spec_obj(**kw):
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    base = dict(
        hypothesis_id="hid_test",
        signal_kind="time_series_momentum",
        universe="us_equities_sector_etf",
        date_range="2020-01:2024-12",
        signal_inputs=("etf.adj_close.spy",),
        rebal="weekly",
        weighting="signed_signal_volatility_targeted",
        expected_holding_period="weekly",
        min_obs_months=24,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="test",
        extracted_ts="2026-06-08T00:00:00Z",
        model="claude-sonnet-4-6",
    )
    base.update(kw)
    return FactorSpec(**base)


def _hyp(**kw):
    """Build a Hypothesis-shaped namespace for is_factor_hypothesis."""
    base = dict(
        hypothesis_id="hid_factor_test",
        claim="profitability predicts cross-section",
        mechanism_family=SimpleNamespace(value="PROFITABILITY"),
        mechanism_subtype="gross_profitability_cross_section",
        predicted_direction=SimpleNamespace(value="positive"),
        predicted_magnitude="Sharpe 0.4-0.7",
        required_data=("CRSP", "Compustat"),
        test_methodology="Fama-MacBeth on GP/A",
        extraction_method=SimpleNamespace(value="llm_extract"),
        source_chunk_ids=("chunk_1",),
        synthesizes_paper_ids=(),
        synthesizes_event_ids=(),
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """Redirect store paths to tmp."""
    from engine.agents.strengthener import factor_spec_store as fss
    specs = tmp_path / "factor_specs.jsonl"
    resolutions = tmp_path / "factor_spec_resolutions.jsonl"
    monkeypatch.setattr(fss, "_DEFAULT_SPECS_PATH", specs)
    monkeypatch.setattr(fss, "_DEFAULT_RESOLUTIONS_PATH", resolutions)
    return specs, resolutions


# ────────────────────────────────────────────────────────────────────
# extract_and_persist_pending
# ────────────────────────────────────────────────────────────────────
def test_extract_and_persist_writes_pending_row(tmp_store, monkeypatch):
    specs_path, _ = tmp_store
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener import factor_spec_store as fss
    # Stub the LLM call: return a valid FactorSpec
    s = _spec_obj()
    monkeypatch.setattr(fse, "extract_factor_spec", lambda h: s)
    h = _hyp()
    sh = fss.extract_and_persist_pending(h, family_hint="PROFITABILITY")
    assert sh is not None
    assert specs_path.exists()
    rows = [json.loads(l) for l in
              specs_path.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    assert len(rows) == 1
    r = rows[0]
    assert r["spec_hash"] == sh
    assert r["source_hypothesis_id"] == "hid_factor_test"
    assert r["family_hint"] == "PROFITABILITY"
    assert r["spec"]["signal_kind"] == "time_series_momentum"


def test_extract_and_persist_idempotent_on_same_spec_hash(
    tmp_store, monkeypatch,
):
    specs_path, _ = tmp_store
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener import factor_spec_store as fss
    monkeypatch.setattr(fse, "extract_factor_spec",
                          lambda h: _spec_obj())
    h = _hyp()
    sh1 = fss.extract_and_persist_pending(h, family_hint="X")
    sh2 = fss.extract_and_persist_pending(h, family_hint="X")
    assert sh1 == sh2
    # Only ONE row on disk (no duplicate)
    rows = [json.loads(l) for l in
              specs_path.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    assert len(rows) == 1


def test_extract_and_persist_returns_none_when_extractor_returns_none(
    tmp_store, monkeypatch,
):
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener import factor_spec_store as fss
    monkeypatch.setattr(fse, "extract_factor_spec", lambda h: None)
    sh = fss.extract_and_persist_pending(_hyp(), family_hint="X")
    assert sh is None


# ────────────────────────────────────────────────────────────────────
# list_pending_factor_specs — queue shape
# ────────────────────────────────────────────────────────────────────
def test_list_pending_returns_empty_for_no_store(tmp_store):
    from engine.agents.strengthener.factor_spec_store import (
        list_pending_factor_specs,
    )
    digest = list_pending_factor_specs()
    assert digest["n_pending"] == 0
    assert digest["rows"] == []


def test_list_pending_returns_rows(tmp_store, monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener.factor_spec_store import (
        extract_and_persist_pending, list_pending_factor_specs,
    )
    # 2 different specs (different hypothesis_id → different hash)
    monkeypatch.setattr(fse, "extract_factor_spec",
                          lambda h: _spec_obj(
                              hypothesis_id=h.hypothesis_id))
    extract_and_persist_pending(_hyp(hypothesis_id="A"),
                                   family_hint="MOMENTUM")
    extract_and_persist_pending(_hyp(hypothesis_id="B"),
                                   family_hint="CARRY")
    digest = list_pending_factor_specs()
    assert digest["n_pending"] == 2
    assert digest["n_resolved"] == 0
    # FIFO by persisted_ts
    assert digest["rows"][0]["source_hypothesis_id"] == "A"
    assert digest["rows"][1]["source_hypothesis_id"] == "B"
    # Spec shape on the row
    assert digest["rows"][0]["spec"]["signal_kind"] == \
        "time_series_momentum"


def test_list_pending_excludes_resolved_by_default(tmp_store, monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener.factor_spec_store import (
        extract_and_persist_pending, list_pending_factor_specs,
        resolve_factor_spec,
    )
    from engine.agents.strengthener import factor_dispatcher as fd
    monkeypatch.setattr(fse, "extract_factor_spec",
                          lambda h: _spec_obj(
                              hypothesis_id=h.hypothesis_id))
    # Stub dispatcher so we don't hit real DB
    monkeypatch.setattr(fd, "dispatch_factor_spec",
                          lambda spec, **kw: {
                              "dispatch_event_id": "ev_dummy",
                              "verdict_event_id":  None,
                              "template_result":   {"verdict": "RED"},
                              "refusal":           None,
                          })
    sh = extract_and_persist_pending(_hyp(), family_hint="X")
    resolve_factor_spec(sh, decision="rejected", rationale="test")

    digest = list_pending_factor_specs()
    assert digest["n_pending"] == 0
    assert digest["n_resolved"] == 1
    assert digest["rows"] == []   # default: exclude resolved

    digest_full = list_pending_factor_specs(include_resolved=True)
    assert len(digest_full["rows"]) == 1
    assert digest_full["rows"][0]["resolved"] is True
    assert digest_full["rows"][0]["resolution"]["decision"] == "rejected"


# ────────────────────────────────────────────────────────────────────
# resolve_factor_spec — dispatch trigger
# ────────────────────────────────────────────────────────────────────
def test_resolve_approved_triggers_dispatch(tmp_store, monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_store import (
        extract_and_persist_pending, resolve_factor_spec,
    )
    monkeypatch.setattr(fse, "extract_factor_spec",
                          lambda h: _spec_obj(
                              hypothesis_id=h.hypothesis_id))
    captured = {}
    def _spy_dispatch(spec, **kw):
        captured["spec"]   = spec
        captured["kwargs"] = kw
        return {
            "dispatch_event_id": "ev_disp_1",
            "verdict_event_id":  "ev_verd_1",
            "template_result":   {"verdict": "GREEN",
                                    "summary": "test green"},
            "refusal":           None,
        }
    monkeypatch.setattr(fd, "dispatch_factor_spec", _spy_dispatch)

    sh = extract_and_persist_pending(_hyp(), family_hint="MOMENTUM")
    out = resolve_factor_spec(sh, decision="approved",
                                  rationale="looks good")

    # Dispatcher called with spec_approved=True
    assert captured["kwargs"]["spec_approved"] is True
    assert captured["kwargs"]["family_hint"] == "MOMENTUM"
    # Response carries dispatch ids
    assert out["dispatch_event_id"] == "ev_disp_1"
    assert out["verdict_event_id"] == "ev_verd_1"
    assert out["dispatch_result"]["template_result"]["verdict"] == "GREEN"


def test_resolve_rejected_skips_dispatch(tmp_store, monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_store import (
        extract_and_persist_pending, resolve_factor_spec,
    )
    monkeypatch.setattr(fse, "extract_factor_spec",
                          lambda h: _spec_obj(
                              hypothesis_id=h.hypothesis_id))
    called = {"n": 0}
    def _spy(spec, **kw):
        called["n"] += 1
        return {}
    monkeypatch.setattr(fd, "dispatch_factor_spec", _spy)

    sh = extract_and_persist_pending(_hyp(), family_hint="X")
    out = resolve_factor_spec(sh, decision="rejected",
                                  rationale="too similar")
    assert called["n"] == 0
    assert out["dispatch_event_id"] is None
    assert out["dispatch_result"] is None


def test_resolve_unknown_spec_hash_raises(tmp_store):
    from engine.agents.strengthener.factor_spec_store import (
        resolve_factor_spec,
    )
    with pytest.raises(ValueError, match="not found"):
        resolve_factor_spec("ffffffff00000000", decision="approved")


def test_resolve_unknown_decision_raises(tmp_store):
    from engine.agents.strengthener.factor_spec_store import (
        resolve_factor_spec,
    )
    with pytest.raises(ValueError, match="decision must be"):
        resolve_factor_spec("ffffffff00000000", decision="yolo")


def test_resolve_dispatch_exception_does_not_block_resolution(
    tmp_store, monkeypatch,
):
    """Dispatcher raising must NOT block the resolution row write —
    the principal made a decision, record it regardless."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_store import (
        extract_and_persist_pending, resolve_factor_spec,
        list_pending_factor_specs,
    )
    monkeypatch.setattr(fse, "extract_factor_spec",
                          lambda h: _spec_obj(
                              hypothesis_id=h.hypothesis_id))
    def _boom(spec, **kw):
        raise RuntimeError("data unreachable")
    monkeypatch.setattr(fd, "dispatch_factor_spec", _boom)
    sh = extract_and_persist_pending(_hyp(), family_hint="X")
    out = resolve_factor_spec(sh, decision="approved")
    # Resolution still recorded
    assert out["decision"] == "approved"
    assert out["dispatch_event_id"] is None  # because dispatch raised
    digest = list_pending_factor_specs(include_resolved=True)
    assert digest["n_resolved"] == 1


# ────────────────────────────────────────────────────────────────────
# is_spec_approved
# ────────────────────────────────────────────────────────────────────
def test_is_spec_approved_returns_false_for_pending(tmp_store, monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener.factor_spec_store import (
        extract_and_persist_pending, is_spec_approved,
    )
    monkeypatch.setattr(fse, "extract_factor_spec",
                          lambda h: _spec_obj())
    sh = extract_and_persist_pending(_hyp(), family_hint="X")
    assert is_spec_approved(sh) is False


def test_is_spec_approved_true_after_approve(tmp_store, monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_store import (
        extract_and_persist_pending, resolve_factor_spec,
        is_spec_approved,
    )
    monkeypatch.setattr(fse, "extract_factor_spec",
                          lambda h: _spec_obj(
                              hypothesis_id=h.hypothesis_id))
    monkeypatch.setattr(fd, "dispatch_factor_spec",
                          lambda spec, **kw: {})
    sh = extract_and_persist_pending(_hyp(), family_hint="X")
    resolve_factor_spec(sh, decision="approved")
    assert is_spec_approved(sh) is True


def test_is_spec_approved_latest_wins_after_redecide(
    tmp_store, monkeypatch,
):
    """Re-resolve overrides: most recent decision wins."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_store import (
        extract_and_persist_pending, resolve_factor_spec,
        is_spec_approved,
    )
    monkeypatch.setattr(fse, "extract_factor_spec",
                          lambda h: _spec_obj(
                              hypothesis_id=h.hypothesis_id))
    monkeypatch.setattr(fd, "dispatch_factor_spec",
                          lambda spec, **kw: {})
    sh = extract_and_persist_pending(_hyp(), family_hint="X")
    resolve_factor_spec(sh, decision="approved")
    resolve_factor_spec(sh, decision="rejected",
                           rationale="changed my mind")
    assert is_spec_approved(sh) is False
