"""Phase 2.1 (2026-06-13): claim-shape router tests.

The router is Stage 0 of the extractor pipeline. It tightly classifies
a hypothesis into ONE canonical claim shape before Stage 1 (the existing
extract_factor_spec) chooses signal_kind / universe / signal_inputs.

Coverage:
  - happy path: TESTABLE_DIRECT shape → is_actionable=True, refusal=None
  - TESTABLE_FUTURE shape (DECAY_STUDY / CAPACITY) → NEEDS_NEW_TEMPLATE
  - NOT_TESTABLE shape (FACTOR_STRUCTURE) → WRONG_HYPOTHESIS_TYPE
  - UNCLEAR shape → LOW_CONFIDENCE_CLASSIFY
  - low confidence on testable shape → LOW_CONFIDENCE_CLASSIFY
  - tool not called / bad enum / LLM exception → safe refusal
  - shape_hint_for_extractor produces correct guardrail line
  - ExtractionResult dataclass + refusal_reason property
  - end-to-end: routing-aware extractor wires into burndown_executor
"""
from __future__ import annotations

import dataclasses as _dc
from unittest.mock import MagicMock

import pytest

from engine.agents.strengthener.claim_shape_router import (
    ClaimShape, ClaimShapeVerdict, classify_claim_shape,
    shape_hint_for_extractor, SHAPE_TO_SIGNAL_KIND,
    TESTABLE_DIRECT, TESTABLE_FUTURE, NOT_TESTABLE,
    _MIN_CONFIDENCE,
)


# ── Fake hypothesis builder ──────────────────────────────────────


def _fake_hypothesis(
    claim="X stocks beat Y stocks", test_methodology="",
    hypothesis_id="t-h", mechanism_family_value="value",
    mechanism_subtype="value_book_to_market",
):
    h = MagicMock()
    h.hypothesis_id = hypothesis_id
    h.mechanism_family = MagicMock()
    h.mechanism_family.value = mechanism_family_value
    h.mechanism_subtype = mechanism_subtype
    h.claim = claim
    h.test_methodology = test_methodology
    return h


def _fake_llm_result(shape_value, confidence, rationale="ok"):
    """Build a MagicMock that mimics llm_call result with one tool call."""
    tc = MagicMock()
    tc.name = "classify_shape"
    tc.input = {
        "shape":      shape_value,
        "confidence": confidence,
        "rationale":  rationale,
    }
    res = MagicMock()
    res.tool_calls = [tc]
    return res


def _stub_fn(shape_value, confidence, rationale="ok"):
    """LLM stub that returns the canned shape verdict."""
    return lambda **kwargs: _fake_llm_result(shape_value, confidence, rationale)


# ── Taxonomy sanity ─────────────────────────────────────────────


def test_taxonomy_partitioning_is_disjoint():
    """TESTABLE_DIRECT / TESTABLE_FUTURE / NOT_TESTABLE must not overlap."""
    assert not (TESTABLE_DIRECT & TESTABLE_FUTURE)
    assert not (TESTABLE_DIRECT & NOT_TESTABLE)
    assert not (TESTABLE_FUTURE & NOT_TESTABLE)


def test_shape_to_signal_kind_covers_all_testable_direct():
    """Every TESTABLE_DIRECT shape must map to exactly one signal_kind."""
    for shape in TESTABLE_DIRECT:
        assert shape in SHAPE_TO_SIGNAL_KIND, (
            f"{shape.value} is TESTABLE_DIRECT but missing from "
            f"SHAPE_TO_SIGNAL_KIND map"
        )


def test_shape_to_signal_kind_targets_exist_in_dispatcher():
    """Every signal_kind named in SHAPE_TO_SIGNAL_KIND must exist in
    factor_dispatcher.TEMPLATE_REGISTRY (otherwise Stage 1 has nothing
    to route to)."""
    from engine.agents.strengthener.factor_dispatcher import TEMPLATE_REGISTRY
    for shape, sk in SHAPE_TO_SIGNAL_KIND.items():
        assert sk in TEMPLATE_REGISTRY, (
            f"shape {shape.value} maps to '{sk}' which is not in "
            f"TEMPLATE_REGISTRY (Stage 1 would refuse)"
        )


# ── Happy path: TESTABLE_DIRECT ──────────────────────────────────


def test_classify_spanning_claim_actionable():
    h = _fake_hypothesis(
        claim="MOM is not spanned by FF5 (alpha-t around 3.0)",
        test_methodology="Regress MOM on MKT+SMB+HML+RMW+CMA",
    )
    v = classify_claim_shape(h, llm_call_fn=_stub_fn("SPANNING", 0.92))
    assert v.shape == ClaimShape.SPANNING
    assert v.is_actionable
    assert v.refusal is None
    assert v.confidence >= 0.9


def test_classify_vrp_claim_actionable():
    h = _fake_hypothesis(
        claim="VIX systematically exceeds realized SPX vol; short variance pays",
    )
    v = classify_claim_shape(h, llm_call_fn=_stub_fn("VRP", 0.88))
    assert v.shape == ClaimShape.VRP
    assert v.is_actionable


def test_classify_factor_combination_actionable():
    h = _fake_hypothesis(
        claim="50/50 value + momentum combination beats either alone",
    )
    v = classify_claim_shape(
        h, llm_call_fn=_stub_fn("FACTOR_COMBINATION", 0.85),
    )
    assert v.shape == ClaimShape.FACTOR_COMBINATION
    assert v.is_actionable


# ── TESTABLE_FUTURE refusals ─────────────────────────────────────


def test_classify_decay_study_refuses_needs_new_template():
    h = _fake_hypothesis(
        claim="PEAD weakened substantially after the 1990s",
    )
    v = classify_claim_shape(
        h, llm_call_fn=_stub_fn("DECAY_STUDY", 0.90),
    )
    assert v.shape == ClaimShape.DECAY_STUDY
    assert not v.is_actionable
    assert v.refusal == "NEEDS_NEW_TEMPLATE"


def test_classify_capacity_refuses_needs_new_template():
    h = _fake_hypothesis(
        claim="Small-cap value capacity is limited to about 2 billion USD",
    )
    v = classify_claim_shape(
        h, llm_call_fn=_stub_fn("CAPACITY", 0.78),
    )
    assert v.shape == ClaimShape.CAPACITY
    assert v.refusal == "NEEDS_NEW_TEMPLATE"


# ── NOT_TESTABLE refusal ─────────────────────────────────────────


def test_classify_factor_structure_refuses_wrong_type():
    h = _fake_hypothesis(
        claim="HXZ4 is preferable to FF5 for explaining the cross-section",
    )
    v = classify_claim_shape(
        h, llm_call_fn=_stub_fn("FACTOR_STRUCTURE", 0.90),
    )
    assert v.shape == ClaimShape.FACTOR_STRUCTURE
    assert v.refusal == "WRONG_HYPOTHESIS_TYPE"


# ── UNCLEAR / low confidence refusals ────────────────────────────


def test_classify_unclear_refuses():
    h = _fake_hypothesis()
    v = classify_claim_shape(h, llm_call_fn=_stub_fn("UNCLEAR", 0.30))
    assert v.shape == ClaimShape.UNCLEAR
    assert v.refusal == "LOW_CONFIDENCE_CLASSIFY"


def test_classify_low_confidence_on_testable_refuses():
    """Even a testable shape choice must clear the confidence threshold."""
    h = _fake_hypothesis()
    v = classify_claim_shape(
        h, llm_call_fn=_stub_fn("CROSS_SECTIONAL_ALPHA", 0.40),
    )
    # Below threshold → router refuses
    assert v.refusal == "LOW_CONFIDENCE_CLASSIFY"
    assert not v.is_actionable


def test_classify_at_confidence_threshold_proceeds():
    """Exactly at the threshold → actionable."""
    h = _fake_hypothesis()
    v = classify_claim_shape(
        h, llm_call_fn=_stub_fn("CROSS_SECTIONAL_ALPHA", _MIN_CONFIDENCE),
    )
    assert v.is_actionable


# ── Error paths (LLM exceptions / bad payload) ───────────────────


def test_classify_llm_exception_returns_unclear_with_refusal():
    h = _fake_hypothesis()

    def boom(**kw):
        raise RuntimeError("LLM provider down")

    v = classify_claim_shape(h, llm_call_fn=boom)
    assert v.shape == ClaimShape.UNCLEAR
    assert v.refusal == "CLASSIFIER_UNAVAILABLE"


def test_classify_tool_not_called_refuses():
    h = _fake_hypothesis()

    def no_tool(**kw):
        res = MagicMock()
        res.tool_calls = []
        return res

    v = classify_claim_shape(h, llm_call_fn=no_tool)
    assert v.refusal == "CLASSIFIER_NO_TOOL_CALL"


def test_classify_invalid_enum_refuses():
    h = _fake_hypothesis()

    def bad_enum(**kw):
        return _fake_llm_result("NOT_A_REAL_SHAPE", 0.95)

    v = classify_claim_shape(h, llm_call_fn=bad_enum)
    assert v.refusal == "CLASSIFIER_INVALID_ENUM"


def test_classify_confidence_out_of_range_clamps():
    h = _fake_hypothesis()
    v = classify_claim_shape(
        h, llm_call_fn=_stub_fn("CROSS_SECTIONAL_ALPHA", 1.5),
    )
    assert 0.0 <= v.confidence <= 1.0


# ── shape_hint_for_extractor ─────────────────────────────────────


def test_shape_hint_includes_target_signal_kind():
    line = shape_hint_for_extractor(ClaimShape.SPANNING)
    assert "spanning_test" in line
    assert "ROUTER_HINT" in line


def test_shape_hint_for_non_testable_returns_empty():
    assert shape_hint_for_extractor(ClaimShape.FACTOR_STRUCTURE) == ""
    assert shape_hint_for_extractor(ClaimShape.UNCLEAR) == ""


# ── ExtractionResult + extract_factor_spec_with_routing ──────────


def test_extraction_result_refusal_reason_property():
    from engine.agents.strengthener.factor_spec_extractor import ExtractionResult
    v = ClaimShapeVerdict(
        shape=ClaimShape.DECAY_STUDY, confidence=0.85, rationale="x",
        refusal="NEEDS_NEW_TEMPLATE",
    )
    r = ExtractionResult(spec=None, router_verdict=v)
    assert r.refusal_reason == "NEEDS_NEW_TEMPLATE"


def test_extraction_result_no_router_verdict_no_refusal():
    from engine.agents.strengthener.factor_spec_extractor import ExtractionResult
    r = ExtractionResult(spec=None, router_verdict=None)
    assert r.refusal_reason is None


# ── End-to-end: burndown_executor surfaces router refusal ────────


def test_burndown_executor_surfaces_router_refusal(monkeypatch):
    """When router refuses (e.g. DECAY_STUDY → NEEDS_NEW_TEMPLATE), the
    burndown executor's outcome.extraction_error must contain the specific
    router refusal code (not the generic EXTRACT_RETURNED_NONE)."""
    from engine.research import burndown_executor as bx
    from engine.agents.strengthener.factor_spec_extractor import ExtractionResult

    monkeypatch.setenv("BURNDOWN_EXTERNAL_AUDIT_DISABLED", "1")

    # Stub _log_extraction_failure so we can inspect the err_msg it sees
    captured = {}
    def fake_log(hid, fam, msg):
        captured["err_msg"] = msg
        return "evt-fake"
    monkeypatch.setattr("engine.research.burndown_executor.BurndownExecutor._log_extraction_failure",
                          lambda self, hid, fam, msg: fake_log(hid, fam, msg))

    fake_hyp = MagicMock()
    monkeypatch.setattr(bx, "_load_hypothesis_by_id", lambda hid, **kw: fake_hyp)

    # Stub extractor: returns ExtractionResult with router refusal
    refusing_verdict = ClaimShapeVerdict(
        shape=ClaimShape.DECAY_STUDY, confidence=0.88, rationale="r",
        refusal="NEEDS_NEW_TEMPLATE",
    )
    def fake_extract(h):
        return ExtractionResult(spec=None, router_verdict=refusing_verdict)

    stream = bx.BurndownExecutor(
        cron_run_id="t",
        spec_extractor_fn=fake_extract,
        dispatcher_fn=lambda *a, **kw: {},   # unused — router refuses first
    )
    cand = MagicMock(hypothesis_id="t-h", family="DECAY")
    outcome = stream.execute_one(cand)
    assert outcome.extraction_ok is False
    assert outcome.extraction_error == "ROUTER_REFUSAL_NEEDS_NEW_TEMPLATE"
    assert captured["err_msg"] == "ROUTER_REFUSAL_NEEDS_NEW_TEMPLATE"


def test_burndown_executor_legacy_optional_factor_spec_still_works(monkeypatch):
    """Existing tests inject fake_extract that returns Optional[FactorSpec]
    directly (not ExtractionResult). The unwrap code must handle both."""
    from engine.research import burndown_executor as bx

    monkeypatch.setenv("BURNDOWN_EXTERNAL_AUDIT_DISABLED", "1")
    fake_hyp = MagicMock()
    monkeypatch.setattr(bx, "_load_hypothesis_by_id", lambda hid, **kw: fake_hyp)

    fake_spec = MagicMock(universe="ken_french_ff5_mom", signal_kind="factor_combination")

    def fake_extract_legacy(h):
        return fake_spec   # legacy contract: Optional[FactorSpec]

    def fake_dispatch(spec, **kwargs):
        return {
            "refusal": None,
            "template_result": {"verdict": "MARGINAL", "summary": "", "metrics": {}},
            "spec_hash": "h", "dispatch_event_id": "evt", "prediction_id": None,
        }

    stream = bx.BurndownExecutor(
        cron_run_id="t",
        spec_extractor_fn=fake_extract_legacy,
        dispatcher_fn=fake_dispatch,
    )
    cand = MagicMock(hypothesis_id="t-h", family="VALUE")
    outcome = stream.execute_one(cand)
    assert outcome.extraction_ok is True
    assert outcome.verdict == "MARGINAL"
