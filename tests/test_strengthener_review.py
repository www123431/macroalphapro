"""tests/test_strengthener_review.py — Phase 2.0 step 11a module tests.

LLM mocked so tests are fast + deterministic. Verifies:
  - Happy path for each of the 3 verdict types
  - Cross-field validation (similar/replaces mutually exclusive,
    amendment needs blocking + summary, approve needs pipeline action)
  - Failure modes (no tool call, wrong tool name, raises)
  - Prompt content invariants (HXZ prior, default REJECT, hard rules)
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────
def _minimal_input():
    from engine.agents.strengthener.review import (
        StrengthenerInput, HypothesisRef, SleeveContextRef,
        DoctrineContextRef, FamilyVerdictRef,
    )
    return StrengthenerInput(
        hypothesis = HypothesisRef(
            hypothesis_id       = "hyp-abc",
            claim               = "EM sovereign QMJ delivers Sharpe 0.6 OOS",
            mechanism_family    = "CARRY",
            mechanism_subtype   = "qmj_em_sovereign",
            predicted_direction = "positive",
            predicted_magnitude = "Sharpe 0.5+",
            required_data       = ("EM sovereign bond returns",),
            test_methodology    = "long-short decile sort",
            extraction_method   = "llm_synthesis",
            synthesizes_paper_ids = ("arxiv/p1",),
            synthesizes_event_ids = ("ev1",),
            addresses_decay_in  = None,
            created_ts          = "2026-06-06T13:00:00Z",
        ),
        deployed_sleeves = (
            SleeveContextRef(
                sleeve_id="cross_asset_carry", family="CARRY",
                ann_sharpe_live=0.83, months_since_deploy=7,
                last_decay_alert=None,
            ),
        ),
        doctrine_snippets = (
            DoctrineContextRef(
                memory_file_id="project-cross-asset-breadth-2026-05-28",
                headline="equity single-name exhausted",
                snippet="12+ RED categories blocked",
                relevance_note="family CARRY adjacency",
            ),
        ),
        family_verdicts = (
            FamilyVerdictRef(
                event_id="ev_carry_red_1", subject_id="auto_carry_x",
                verdict="RED", ts="2026-06-04T12:00:00Z",
                summary="CARRY family RED autopilot verdict",
            ),
        ),
        snapshot_ts="2026-06-06T13:00:00Z",
    )


def _approve_payload(**overrides):
    base = {
        "verdict_type":                "APPROVE_FOR_PIPELINE",
        "one_line_summary":            "Differentiated from deployed carry; worth strict-gate budget",
        "confidence":                  0.72,
        "reasoning":                   "Although CARRY family has a RED cluster, this candidate "
                                       "targets EM sovereign which is orthogonal to cross_asset_carry "
                                       "(G10 only). Sharpe target 0.5+ is plausible given paper evidence.",
        "similar_to_deployed":         None,
        "replaces_decaying":           None,
        "blocking_doctrine_id":        None,
        "proposed_amendment_summary":  None,
        "recommended_pipeline_action": "run f14b strict gate with EM sov bond data",
        "risk_flags":                  ["data not free", "short sample window"],
    }
    base.update(overrides)
    return base


def _reject_payload(**overrides):
    base = {
        "verdict_type":                "REJECT",
        "one_line_summary":            "Too similar to deployed cross_asset_carry; same family already RED-clustering",
        "confidence":                  0.85,
        "reasoning":                   "Mechanism overlap with cross_asset_carry (G10 carry); "
                                       "CARRY family has 10 RED verdicts in last 30 days per "
                                       "ev_carry_red_1 cluster signal.",
        "similar_to_deployed":         "cross_asset_carry",
        "replaces_decaying":           None,
        "blocking_doctrine_id":        None,
        "proposed_amendment_summary":  None,
        "recommended_pipeline_action": None,
        "risk_flags":                  ["family in RED cluster"],
    }
    base.update(overrides)
    return base


def _amendment_payload(**overrides):
    base = {
        "verdict_type":                "DOCTRINE_AMENDMENT_NEEDED",
        "one_line_summary":            "Candidate is strong; blocked only by stale doctrine — propose amendment",
        "confidence":                  0.78,
        "reasoning":                   "The candidate provides 3 fresh papers showing sub-mechanism X is "
                                       "still alpha-generative; doctrine project-cross-asset-breadth-2026-05-28 "
                                       "is overly restrictive for this sub-family.",
        "similar_to_deployed":         None,
        "replaces_decaying":           None,
        "blocking_doctrine_id":        "project-cross-asset-breadth-2026-05-28",
        "proposed_amendment_summary":  "Carve out EM sovereign QMJ from the cross-asset-breadth ban: "
                                       "fresh paper evidence shows it survives post-pub decay.",
        "recommended_pipeline_action": None,
        "risk_flags":                  ["doctrine change risk"],
    }
    base.update(overrides)
    return base


def _mock_llm(monkeypatch, tool_input=None, *, text="", raise_exc=None,
                tool_name="emit_review"):
    from engine.agents.strengthener import review as rm
    from engine.llm.call import LLMCallResult, ToolCall

    def _fake_call(**kw):
        if raise_exc is not None:
            raise raise_exc
        tool_calls = ()
        if tool_input is not None:
            tool_calls = (ToolCall(id="tc", name=tool_name, input=tool_input),)
        return LLMCallResult(
            text=text, tool_calls=tool_calls, stop_reason="tool_use",
            model="claude-sonnet-4-6", provider="anthropic",
            cost_usd=0.04, latency_ms=3200,
            cache_read_tokens=0, raw_usage={},
        )
    monkeypatch.setattr(rm, "llm_call", _fake_call)


# ─────────────────────────────────────────────────────────────────────
# Happy paths — three verdict types parse cleanly
# ─────────────────────────────────────────────────────────────────────
def test_approve_verdict_parses(monkeypatch):
    from engine.agents.strengthener.review import (
        run_strengthener_review, VerdictType,
    )
    _mock_llm(monkeypatch, tool_input=_approve_payload())
    v = run_strengthener_review(_minimal_input())
    assert v is not None
    assert v.verdict_type == VerdictType.APPROVE_FOR_PIPELINE
    assert v.recommended_pipeline_action
    assert v.confidence == 0.72
    assert "EM sov" in v.recommended_pipeline_action
    assert v.model == "claude-sonnet-4-6"


def test_reject_verdict_parses(monkeypatch):
    from engine.agents.strengthener.review import (
        run_strengthener_review, VerdictType,
    )
    _mock_llm(monkeypatch, tool_input=_reject_payload())
    v = run_strengthener_review(_minimal_input())
    assert v is not None
    assert v.verdict_type == VerdictType.REJECT
    assert v.similar_to_deployed == "cross_asset_carry"
    assert v.replaces_decaying is None
    assert v.recommended_pipeline_action is None


def test_doctrine_amendment_verdict_parses(monkeypatch):
    from engine.agents.strengthener.review import (
        run_strengthener_review, VerdictType,
    )
    _mock_llm(monkeypatch, tool_input=_amendment_payload())
    v = run_strengthener_review(_minimal_input())
    assert v is not None
    assert v.verdict_type == VerdictType.DOCTRINE_AMENDMENT_NEEDED
    assert v.blocking_doctrine_id == "project-cross-asset-breadth-2026-05-28"
    assert v.proposed_amendment_summary
    assert "EM sovereign" in v.proposed_amendment_summary


# ─────────────────────────────────────────────────────────────────────
# Cross-field validation
# ─────────────────────────────────────────────────────────────────────
def test_similar_and_replaces_both_set_rejects(monkeypatch):
    """Mutually exclusive — can't be both 'too similar' and 'good
    replacement' at the same time."""
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=_reject_payload(
        similar_to_deployed="x", replaces_decaying="y"))
    assert run_strengthener_review(_minimal_input()) is None


def test_amendment_missing_blocking_id_rejects(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=_amendment_payload(
        blocking_doctrine_id=None))
    assert run_strengthener_review(_minimal_input()) is None


def test_amendment_missing_summary_rejects(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=_amendment_payload(
        proposed_amendment_summary=None))
    assert run_strengthener_review(_minimal_input()) is None


def test_approve_missing_pipeline_action_rejects(monkeypatch):
    """APPROVE_FOR_PIPELINE without a recommended_pipeline_action is
    half-baked — the runner needs to know what to do next."""
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=_approve_payload(
        recommended_pipeline_action=None))
    assert run_strengthener_review(_minimal_input()) is None


def test_confidence_clamped_to_unit_interval(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=_approve_payload(confidence=1.5))
    v = run_strengthener_review(_minimal_input())
    assert v is not None
    assert v.confidence == 1.0


def test_invalid_confidence_falls_back_to_default(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=_approve_payload(confidence="not_a_number"))
    v = run_strengthener_review(_minimal_input())
    assert v is not None
    assert v.confidence == 0.5


def test_risk_flags_capped_at_five(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=_approve_payload(
        risk_flags=[f"flag_{i}" for i in range(10)]))
    v = run_strengthener_review(_minimal_input())
    assert v is not None
    assert len(v.risk_flags) == 5


# ─────────────────────────────────────────────────────────────────────
# Failure modes
# ─────────────────────────────────────────────────────────────────────
def test_llm_exception_returns_none(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, raise_exc=RuntimeError("api down"))
    assert run_strengthener_review(_minimal_input()) is None


def test_no_tool_call_returns_none(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=None,
                text="I think this is a reject but I won't use the tool")
    assert run_strengthener_review(_minimal_input()) is None


def test_wrong_tool_name_returns_none(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input={"verdict_type": "REJECT"},
                tool_name="some_other_tool")
    assert run_strengthener_review(_minimal_input()) is None


def test_invalid_verdict_type_returns_none(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=_reject_payload(
        verdict_type="MAYBE_LATER"))
    assert run_strengthener_review(_minimal_input()) is None


def test_missing_one_line_summary_returns_none(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=_reject_payload(one_line_summary=""))
    assert run_strengthener_review(_minimal_input()) is None


def test_oversize_one_line_summary_returns_none(monkeypatch):
    from engine.agents.strengthener.review import run_strengthener_review
    _mock_llm(monkeypatch, tool_input=_reject_payload(
        one_line_summary="x" * 250))
    assert run_strengthener_review(_minimal_input()) is None


# ─────────────────────────────────────────────────────────────────────
# Prompt content invariants — load-bearing rules in the system prompt
# ─────────────────────────────────────────────────────────────────────
def test_system_prompt_carries_hxz_prior_and_default_reject():
    from engine.agents.strengthener.review import _SYSTEM_PROMPT
    assert "Hou-Xue-Zhang" in _SYSTEM_PROMPT
    assert "65%" in _SYSTEM_PROMPT
    assert "REJECT is NOT a failure" in _SYSTEM_PROMPT
    assert "DOCTRINE_AMENDMENT_NEEDED" in _SYSTEM_PROMPT


def test_system_prompt_comfort_bias_guard_present():
    """Anti-mental-rut Stage C (2026-06-07): the comfort-bias guard
    flags 'comfortable' candidates as a risk and biases B toward more
    skepticism on them, not less. If this block is silently dropped,
    B loses its anti-rut discipline against confirmation bias."""
    from engine.agents.strengthener.review import _SYSTEM_PROMPT
    assert "COMFORT-BIAS GUARD" in _SYSTEM_PROMPT
    assert "solo" in _SYSTEM_PROMPT
    assert "selection bias" in _SYSTEM_PROMPT
    assert "comfort_bias_risk" in _SYSTEM_PROMPT
    # Make sure the prompt explicitly states it's NOT softening REJECT
    assert "NOT a softening" in _SYSTEM_PROMPT


def test_user_prompt_surfaces_all_input_sections():
    """All 5 input sections must appear in the user message so the
    LLM can address each. Missing a section = regression."""
    from engine.agents.strengthener.review import _format_input
    msg = _format_input(_minimal_input())
    assert "HYPOTHESIS UNDER REVIEW" in msg
    assert "CITATION VERIFICATION" in msg     # Phase 2.2c — load-bearing
    assert "DEPLOYED SLEEVES" in msg
    assert "ACTIVE DOCTRINE SNIPPETS" in msg
    assert "RECENT FAMILY VERDICTS" in msg
    # The specific candidate ID + sleeve + doctrine id must propagate
    assert "hyp-abc" in msg
    assert "cross_asset_carry" in msg
    assert "project-cross-asset-breadth-2026-05-28" in msg


# ─────────────────────────────────────────────────────────────────────
# Phase 2.2c: CITATION VERIFICATION block
# ─────────────────────────────────────────────────────────────────────
def test_citation_block_none_states_unverified():
    """citation_quality=None → message must clearly tell the model the
    citations are un-checked (B should be more cautious)."""
    from engine.agents.strengthener.review import _format_input
    msg = _format_input(_minimal_input())  # default citation_quality=None
    assert "not verified" in msg
    assert "un-checked" in msg


def test_citation_block_renders_quality_dict():
    """citation_quality dict present → all key fields surface."""
    from engine.agents.strengthener.review import _format_input
    import dataclasses
    si = _minimal_input()
    hyp = dataclasses.replace(si.hypothesis, citation_quality={
        "n_papers_cited":      2,
        "n_resolved":          2,
        "n_unresolved":        0,
        "mean_confidence":     0.82,
        "min_confidence":      0.75,
        "any_unresolved":      False,
        "low_confidence_flag": False,
    })
    si2 = dataclasses.replace(si, hypothesis=hyp)
    msg = _format_input(si2)
    assert "papers cited        : 2" in msg
    assert "mean confidence     : 0.82" in msg
    assert "min confidence      : 0.75" in msg
    assert "OK" in msg


def test_citation_block_flags_low_confidence_loudly():
    """When any_unresolved=True (hallucinated citation suspected), the
    message MUST explicitly tell B to weight heavily toward REJECT."""
    from engine.agents.strengthener.review import _format_input
    import dataclasses
    si = _minimal_input()
    hyp = dataclasses.replace(si.hypothesis, citation_quality={
        "n_papers_cited":      2,
        "n_resolved":          1,
        "n_unresolved":        1,
        "mean_confidence":     0.40,
        "min_confidence":      0.00,
        "any_unresolved":      True,
        "low_confidence_flag": True,
    })
    si2 = dataclasses.replace(si, hypothesis=hyp)
    msg = _format_input(si2)
    assert "LOW CONFIDENCE" in msg
    assert "hallucinated" in msg.lower()
    assert "REJECT" in msg


def test_runner_hypothesis_ref_propagates_citation_quality():
    """B's runner must copy citation_quality from the Hypothesis row
    into the HypothesisRef so the prompt sees it."""
    from engine.agents.strengthener.runner import _hypothesis_ref
    from engine.research_store.hypothesis import Hypothesis
    from engine.research_store.hypothesis.schema import (
        ExtractionMethod, HypothesisDirection, HypothesisReviewState,
    )
    from engine.research_store.red_lessons.mechanism_families import MechanismFamily
    h = Hypothesis(
        hypothesis_id        = "h1", source_paper_id = "",
        version              = 1, parent_hypothesis_id = None,
        source_chunk_ids     = (), verbatim_quotes = (),
        claim                = "x",
        mechanism_family     = MechanismFamily.CARRY,
        mechanism_subtype    = "",
        predicted_direction  = HypothesisDirection.POSITIVE,
        predicted_magnitude  = "x", required_data = ("y",),
        test_methodology     = "z",
        extraction_method    = ExtractionMethod.LLM_SYNTHESIS,
        review_state         = HypothesisReviewState.PROPOSED,
        created_ts           = "2026-06-07T00:00:00Z",
        updated_ts           = "2026-06-07T00:00:00Z",
        created_by           = "test", tags = (),
        synthesizes_paper_ids = ("p1",),
        synthesizes_event_ids = (),
        addresses_decay_in    = None,
        citation_quality      = {"mean_confidence": 0.7,
                                  "low_confidence_flag": False,
                                  "any_unresolved": False,
                                  "n_papers_cited": 1, "n_resolved": 1,
                                  "n_unresolved": 0, "min_confidence": 0.7},
    )
    ref = _hypothesis_ref(h)
    assert ref.citation_quality is not None
    assert ref.citation_quality["mean_confidence"] == 0.7
