"""tests/test_autopilot_pre_compute_da.py — Phase 2.0 step 6 module tests.

Pre-compute DA gate: mocked LLM, verifies the call shape + parse + error
fallthrough match the post-compute DA conventions so the F14b wiring
(step 7) can rely on the same fail-OPEN contract.
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────
# Fake spec — minimal shape autopilot_pre_compute_da._build_user_message reads
# ─────────────────────────────────────────────────────────────────────
class _F:
    def __init__(self, v): self.value = v

class _Leg:
    def __init__(self, signal_type): self.signal_type = _F(signal_type)

class _Universe:
    def __init__(self, asset_class, subset):
        self.asset_class = _F(asset_class)
        self.subset      = _F(subset)

class _Construction:
    def __init__(self, weighting, rebalance):
        self.weighting = _F(weighting)
        self.rebalance = _F(rebalance)

class _Spec:
    def __init__(self, *, family="EARNINGS_DRIFT", signal_type="pead_sue",
                  source_hypothesis_id="hyp-abc", claim_text=None):
        self.family               = _F(family)
        self.legs                 = (_Leg(signal_type),)
        self.universe             = _Universe("equity", "us_large")
        self.construction         = _Construction("equal", "monthly")
        self.source_hypothesis_id = source_hypothesis_id
        self.claim_text           = claim_text


# ─────────────────────────────────────────────────────────────────────
# Mock LLM helper
# ─────────────────────────────────────────────────────────────────────
def _mock_llm(monkeypatch, *, tool_input=None, text="", raise_exc=None,
                tool_name="emit_pre_compute_verdict"):
    from engine.agents import autopilot_pre_compute_da as mod
    from engine.llm.call import LLMCallResult, ToolCall
    def _fake_call(**kw):
        if raise_exc is not None:
            raise raise_exc
        tcs = ()
        if tool_input is not None:
            tcs = (ToolCall(id="tc", name=tool_name, input=tool_input),)
        return LLMCallResult(
            text=text, tool_calls=tcs, stop_reason="tool_use",
            model="deepseek-v4-pro", provider="deepseek",
            cost_usd=0.005, latency_ms=2100,
            cache_read_tokens=0, raw_usage={},
        )
    # Module-level binding (matches synthesis/review pattern)
    monkeypatch.setattr(mod, "llm_call", _fake_call)


# ─────────────────────────────────────────────────────────────────────
# Happy paths — proceed + skip
# ─────────────────────────────────────────────────────────────────────
def test_proceed_verdict_parses(monkeypatch):
    from engine.agents.autopilot_pre_compute_da import (
        run_autopilot_pre_compute_da, PreComputeVerdict,
    )
    _mock_llm(monkeypatch, tool_input={
        "worth_running": True,
        "attack_vector": "Novel data source + addresses carry decay",
        "reasoning":     "Cites EM sov bond returns we have on disk; targets cross_asset_carry which is in DECAY_WATCH.",
        "confidence":    0.78,
    })
    v = run_autopilot_pre_compute_da(
        spec        = _Spec(claim_text="EM sov QMJ test"),
        claim_text  = "test claim",
        graveyard_matches = [],
    )
    assert v is not None
    assert isinstance(v, PreComputeVerdict)
    assert v.worth_running is True
    assert v.confidence == 0.78
    assert "data source" in v.attack_vector


def test_skip_verdict_parses(monkeypatch):
    from engine.agents.autopilot_pre_compute_da import (
        run_autopilot_pre_compute_da, PreComputeVerdict,
    )
    _mock_llm(monkeypatch, tool_input={
        "worth_running": False,
        "attack_vector": "EARNINGS_DRIFT family has 7 recent REDs; same-cell weighting tweak",
        "reasoning":     "Graveyard shows 7 PEAD weighting-variant RED verdicts in last 90d. This spec is the same cell with another weighting tweak.",
        "confidence":    0.85,
    })
    v = run_autopilot_pre_compute_da(
        spec       = _Spec(family="EARNINGS_DRIFT"),
        claim_text = "PEAD weighted by SUE^2",
        graveyard_matches = [{"family": "EARNINGS_DRIFT",
                                "signal_type": "pead_sue",
                                "verdict": "RED", "score": 1}] * 7,
        family_recent_test_count = 15,
    )
    assert v is not None
    assert v.worth_running is False
    assert "EARNINGS_DRIFT" in v.attack_vector
    assert v.confidence == 0.85


# ─────────────────────────────────────────────────────────────────────
# Failure paths — fail-OPEN
# ─────────────────────────────────────────────────────────────────────
def test_llm_exception_returns_none(monkeypatch):
    """LLM failure → None → caller treats as 'no gate decision, proceed'.
    Important: pre-compute DA being down MUST NOT block research."""
    from engine.agents.autopilot_pre_compute_da import run_autopilot_pre_compute_da
    _mock_llm(monkeypatch, raise_exc=RuntimeError("api down"))
    assert run_autopilot_pre_compute_da(
        spec=_Spec(), claim_text="x", graveyard_matches=[],
    ) is None


def test_no_tool_call_returns_none(monkeypatch):
    from engine.agents.autopilot_pre_compute_da import run_autopilot_pre_compute_da
    _mock_llm(monkeypatch, tool_input=None, text="Going to think about this...")
    assert run_autopilot_pre_compute_da(
        spec=_Spec(), claim_text="x", graveyard_matches=[],
    ) is None


def test_wrong_tool_name_returns_none(monkeypatch):
    from engine.agents.autopilot_pre_compute_da import run_autopilot_pre_compute_da
    _mock_llm(monkeypatch, tool_input={"worth_running": True},
                tool_name="some_other_tool")
    assert run_autopilot_pre_compute_da(
        spec=_Spec(), claim_text="x", graveyard_matches=[],
    ) is None


# ─────────────────────────────────────────────────────────────────────
# User-message content invariants
# ─────────────────────────────────────────────────────────────────────
def test_user_message_surfaces_graveyard_hits():
    from engine.agents.autopilot_pre_compute_da import _build_user_message
    msg = _build_user_message(
        spec       = _Spec(family="EARNINGS_DRIFT"),
        claim_text = "x",
        graveyard_matches = [
            {"family": "EARNINGS_DRIFT", "signal_type": "pead_sue",
             "verdict": "RED", "score": 1},
            {"family": "EARNINGS_DRIFT", "signal_type": "pead_revisions",
             "verdict": "RED", "score": 0},
        ],
        family_recent_test_count = 12,
        paper_age_years          = 11.0,
        addresses_decay_in       = None,
    )
    assert "GRAVEYARD HITS (2)" in msg
    assert "pead_sue" in msg
    assert "FAMILY n_TRIALS THIS QUARTER: 12" in msg
    assert "paper_age:    11.0" in msg


def test_user_message_surfaces_addresses_decay_hint():
    """When the candidate addresses a known decay, surface that —
    it's a strong signal toward proceed."""
    from engine.agents.autopilot_pre_compute_da import _build_user_message
    msg = _build_user_message(
        spec       = _Spec(),
        claim_text = "x",
        graveyard_matches = [],
        family_recent_test_count = 0,
        paper_age_years          = None,
        addresses_decay_in       = "cross_asset_carry",
    )
    assert "addresses_decay_in: cross_asset_carry" in msg


def test_user_message_truncates_long_claim():
    from engine.agents.autopilot_pre_compute_da import _build_user_message
    msg = _build_user_message(
        spec       = _Spec(),
        claim_text = "x" * 5000,
        graveyard_matches = [],
        family_recent_test_count = 0,
        paper_age_years          = None,
        addresses_decay_in       = None,
    )
    # Claim block is truncated to 500 chars
    assert msg.count("x") <= 500 + 50


# ─────────────────────────────────────────────────────────────────────
# System prompt content invariants
# ─────────────────────────────────────────────────────────────────────
def test_system_prompt_default_stance_proceed():
    """The default stance MUST be PROCEED — the candidate already
    cleared upstream gates, this is the cheap-veto opportunity."""
    from engine.agents.autopilot_pre_compute_da import _SYSTEM_PROMPT
    assert "Default stance: PROCEED" in _SYSTEM_PROMPT


def test_system_prompt_load_bearing_skip_criteria():
    from engine.agents.autopilot_pre_compute_da import _SYSTEM_PROMPT
    assert "GRAVEYARD REDUNDANCY" in _SYSTEM_PROMPT
    assert "METHODOLOGY DEAD-END" in _SYSTEM_PROMPT
    assert "POST-PUB DECAY" in _SYSTEM_PROMPT
    assert "n_TRIALS BUDGET" in _SYSTEM_PROMPT


def test_system_prompt_acknowledges_upstream_gates():
    """The prompt MUST tell the DA that upstream gates exist —
    otherwise the DA over-rejects."""
    from engine.agents.autopilot_pre_compute_da import _SYSTEM_PROMPT
    assert "D's graveyard" in _SYSTEM_PROMPT
    assert "B's institutional skeptical review" in _SYSTEM_PROMPT
    assert "principal's approval" in _SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────
# decide_pre_compute_gate — wrapper used by autopilot_live
# ─────────────────────────────────────────────────────────────────────
def test_gate_skip_kwarg_bypasses_llm(monkeypatch):
    """skip=True must short-circuit BEFORE the LLM call. Caller uses
    this for force-compute override."""
    from engine.agents.autopilot_pre_compute_da import decide_pre_compute_gate
    # Set up LLM mock that would raise if called — we expect it NOT
    # to be called
    _mock_llm(monkeypatch, raise_exc=RuntimeError("DA should not fire"))
    proceed, v = decide_pre_compute_gate(
        spec=_Spec(), claim_text="x", graveyard_matches=[], skip=True,
    )
    assert proceed is True
    assert v is None


def test_gate_da_failure_fails_open_proceed(monkeypatch):
    """LLM failure → (True, None) — research pipeline must not block
    when DA is down. Bias toward fail-open."""
    from engine.agents.autopilot_pre_compute_da import decide_pre_compute_gate
    _mock_llm(monkeypatch, raise_exc=RuntimeError("rate limit"))
    proceed, v = decide_pre_compute_gate(
        spec=_Spec(), claim_text="x", graveyard_matches=[],
    )
    assert proceed is True
    assert v is None


def test_gate_proceed_passes_verdict_through(monkeypatch):
    """Approved verdict → (True, verdict) — caller surfaces the
    attack_vector in the eventual capability_evidence markdown."""
    from engine.agents.autopilot_pre_compute_da import decide_pre_compute_gate
    _mock_llm(monkeypatch, tool_input={
        "worth_running": True,
        "attack_vector": "novel data source",
        "reasoning":     "fresh OptionMetrics tier we hadn't used",
        "confidence":    0.7,
    })
    proceed, v = decide_pre_compute_gate(
        spec=_Spec(), claim_text="x", graveyard_matches=[],
    )
    assert proceed is True
    assert v is not None
    assert v.attack_vector == "novel data source"


def test_gate_skip_returns_verdict_for_emit(monkeypatch):
    """Rejected verdict → (False, verdict) — caller MUST emit the
    candidate_skipped_pre_compute event using the verdict payload."""
    from engine.agents.autopilot_pre_compute_da import decide_pre_compute_gate
    _mock_llm(monkeypatch, tool_input={
        "worth_running": False,
        "attack_vector": "graveyard hit",
        "reasoning":     "7 same-cell REDs",
        "confidence":    0.9,
    })
    proceed, v = decide_pre_compute_gate(
        spec=_Spec(), claim_text="x", graveyard_matches=[{"x": 1}] * 7,
    )
    assert proceed is False
    assert v is not None
    assert v.worth_running is False
    assert v.attack_vector == "graveyard hit"
    assert v.confidence == 0.9
