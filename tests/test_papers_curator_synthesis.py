"""tests/test_papers_curator_synthesis.py — Phase 2.0 step 3 module tests.

Cross-source synthesis call. LLM call mocked so tests are fast +
deterministic. Verifies:

  - Tool-call payload parses into SynthesizedCandidate cleanly
  - Empty candidates list is valid (the prompt explicitly allows it)
  - LLM exception / unparseable / wrong-tool returns [] not raises
  - Hard cap of 3 candidates even if model emits more
  - Malformed individual candidate dropped, valid siblings preserved
"""
from __future__ import annotations


def _minimal_input():
    from engine.agents.papers_curator.synthesis import (
        SynthesisInput, PaperSummaryRef, SleeveStateRef,
        RecentEventRef, DoctrineHit,
    )
    return SynthesisInput(
        recent_summaries  = (
            PaperSummaryRef(
                paper_id="p1", title="EM bond carry refresh",
                authors_short="Asness et al",
                thesis="EM sovereign QMJ delivers Sharpe 0.74",
                testable_hypothesis="CARRY/EM_SOV",
                why_matters_for_us="orthogonal to deployed equity",
                risk_flags_short=("data not free",),
                recommended_action="INGEST",
            ),
        ),
        deployed_sleeves  = (
            SleeveStateRef(
                sleeve_id="carry_g10", family="CARRY", status="DEPLOYED",
                ann_sharpe_live=0.83, months_since_deploy=6,
                last_decay_alert=None,
            ),
        ),
        recent_events     = (
            RecentEventRef(
                event_id="ev1", event_type="factor_verdict_filed",
                subject_id="auto_xyz", family="PROFITABILITY",
                verdict="RED", summary="GP/A decayed post-pub",
                ts="2026-06-05T12:30:00Z",
            ),
        ),
        doctrine_snippets = (
            DoctrineHit(
                memory_file_id="project-cross-asset-breadth-focus-2026-05-28",
                headline="equity single-name exhausted",
                snippet="12+ RED categories, do not pursue more equity single-name",
            ),
        ),
        snapshot_ts="2026-06-06T13:00:00Z",
    )


def _valid_candidate_dict():
    return {
        "claim": "QMJ on EM sovereign bonds delivers risk-adjusted excess",
        "mechanism_family": "carry",
        "mechanism_subtype": "qmj_em_sovereign",
        "predicted_direction": "positive",
        "predicted_magnitude": "Sharpe 0.5+ OOS",
        "required_data": ["EM sovereign bond returns", "fiscal indicators"],
        "test_methodology": "long-short decile sort on composite quality",
        "synthesizes_paper_ids": ["p1"],
        "synthesizes_event_ids": ["ev1"],
        "addresses_decay_in": None,
        "cochrane_frame": "risk",
        "novelty_vs_known": "extension_to_em_sov",
        "estimated_n_trials_in_family": 5,
        "graveyard_conflicts": [],
        "doctrine_conflicts": [],
        "expected_outcome_prior": "marginal_per_HXZ_with_some_replication",
    }


def _mock_llm_result(tool_call_input=None, text="", raise_exc=None):
    """Build a LLMCallResult-shaped object the synthesizer can consume.
    Returns from the closure when raise_exc is set."""
    from engine.llm.call import LLMCallResult, ToolCall

    def _fake_call(**kw):
        if raise_exc is not None:
            raise raise_exc
        tool_calls = ()
        if tool_call_input is not None:
            tool_calls = (ToolCall(id="tc1", name="emit_synthesis",
                                     input=tool_call_input),)
        return LLMCallResult(
            text=text, tool_calls=tool_calls, stop_reason="tool_use",
            model="claude-sonnet-4-6", provider="anthropic",
            cost_usd=0.08, latency_ms=4200, cache_read_tokens=0,
            raw_usage={},
        )
    return _fake_call


# ──────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────
def test_run_synthesis_parses_valid_candidate(monkeypatch):
    from engine.agents.papers_curator import synthesis as sm
    monkeypatch.setattr(sm, "llm_call",
                          _mock_llm_result(tool_call_input={
                              "candidates": [_valid_candidate_dict()],
                          }))
    out = sm.run_synthesis(_minimal_input())
    assert len(out) == 1
    c = out[0]
    assert c.cochrane_frame == "risk"
    assert c.synthesizes_paper_ids == ("p1",)
    assert c.synthesizes_event_ids == ("ev1",)
    assert c.addresses_decay_in is None
    assert c.estimated_n_trials_in_family == 5
    assert c.model == "claude-sonnet-4-6"


def test_run_synthesis_empty_candidates_is_valid(monkeypatch):
    """The prompt explicitly says 'empty list is valid and preferred
    over weak candidates' — the call must return [] not raise."""
    from engine.agents.papers_curator import synthesis as sm
    monkeypatch.setattr(sm, "llm_call",
                          _mock_llm_result(tool_call_input={"candidates": []}))
    out = sm.run_synthesis(_minimal_input())
    assert out == []


# ──────────────────────────────────────────────────────────────────
# Failure modes
# ──────────────────────────────────────────────────────────────────
def test_run_synthesis_llm_exception_returns_empty(monkeypatch):
    from engine.agents.papers_curator import synthesis as sm
    monkeypatch.setattr(sm, "llm_call",
                          _mock_llm_result(raise_exc=RuntimeError("api down")))
    assert sm.run_synthesis(_minimal_input()) == []


def test_run_synthesis_no_tool_call_returns_empty(monkeypatch):
    """Model returned text instead of calling the tool — fail-safe to []."""
    from engine.agents.papers_curator import synthesis as sm
    monkeypatch.setattr(sm, "llm_call",
                          _mock_llm_result(tool_call_input=None,
                                            text="I am not going to use the tool"))
    assert sm.run_synthesis(_minimal_input()) == []


def test_run_synthesis_wrong_tool_name_returns_empty(monkeypatch):
    """Model called a tool with a different name — fail-safe to []."""
    from engine.agents.papers_curator import synthesis as sm
    from engine.llm.call import LLMCallResult, ToolCall

    def _fake(**kw):
        return LLMCallResult(
            text="", tool_calls=(ToolCall(id="x", name="some_other_tool",
                                            input={"candidates": []}),),
            stop_reason="tool_use", model="claude-sonnet-4-6",
            provider="anthropic", cost_usd=0.08, latency_ms=4200,
            cache_read_tokens=0, raw_usage={},
        )
    monkeypatch.setattr(sm, "llm_call", _fake)
    assert sm.run_synthesis(_minimal_input()) == []


# ──────────────────────────────────────────────────────────────────
# Quality / safety constraints
# ──────────────────────────────────────────────────────────────────
def test_run_synthesis_caps_at_3_candidates(monkeypatch):
    """Even if the model emits 5 candidates, we keep only 3 per the
    spec budget (hard cap)."""
    from engine.agents.papers_curator import synthesis as sm
    monkeypatch.setattr(sm, "llm_call", _mock_llm_result(tool_call_input={
        "candidates": [_valid_candidate_dict() for _ in range(5)],
    }))
    out = sm.run_synthesis(_minimal_input())
    assert len(out) == 3


def test_run_synthesis_drops_malformed_keeps_valid(monkeypatch):
    """One candidate missing a required field — drop it, keep the
    valid siblings. Prevents one bad LLM row from killing the rest."""
    from engine.agents.papers_curator import synthesis as sm
    bad = dict(_valid_candidate_dict())
    del bad["claim"]                # missing required field
    monkeypatch.setattr(sm, "llm_call", _mock_llm_result(tool_call_input={
        "candidates": [_valid_candidate_dict(), bad, _valid_candidate_dict()],
    }))
    out = sm.run_synthesis(_minimal_input())
    assert len(out) == 2


def test_run_synthesis_candidates_list_not_a_list_returns_empty(monkeypatch):
    """Model returned candidates as a string or dict — fail-safe to []."""
    from engine.agents.papers_curator import synthesis as sm
    monkeypatch.setattr(sm, "llm_call", _mock_llm_result(tool_call_input={
        "candidates": "this should have been a list",
    }))
    assert sm.run_synthesis(_minimal_input()) == []


# ──────────────────────────────────────────────────────────────────
# Prompt formatting — content checks (so prompt regressions are loud)
# ──────────────────────────────────────────────────────────────────
def test_format_input_includes_all_sections():
    """The user message MUST surface all 4 store sections so the LLM
    can synthesize across them. Missing a section = regression."""
    from engine.agents.papers_curator.synthesis import _format_input
    msg = _format_input(_minimal_input())
    assert "RECENT PAPER SUMMARIES" in msg
    assert "DEPLOYED SLEEVES" in msg
    assert "RECENT EVENTS" in msg
    assert "DOCTRINE SNIPPETS" in msg
    # The doctrine should propagate verbatim so the LLM sees the
    # graveyard ban context
    assert "equity single-name" in msg


def test_system_prompt_carries_load_bearing_priors():
    """The prompt's stated constraints (HXZ 65%, Cochrane frame, prefer
    empty over weak) MUST be present. If a future edit silently drops
    them, the LLM stops behaving as designed."""
    from engine.agents.papers_curator.synthesis import _SYSTEM_PROMPT
    assert "Hou-Xue-Zhang" in _SYSTEM_PROMPT
    assert "65%" in _SYSTEM_PROMPT
    assert "PREFER zero output over weak output" in _SYSTEM_PROMPT
    assert "behavioral, risk, friction" in _SYSTEM_PROMPT
    assert "graveyard_conflicts honestly" in _SYSTEM_PROMPT
    assert "doctrine_conflicts honestly" in _SYSTEM_PROMPT


def test_system_prompt_orthogonality_gate_present():
    """Anti-mental-rut Stage B (2026-06-07): the orthogonality gate
    explicitly downgrades comfortable-repetition candidates. If this
    block is silently dropped, A loses its anti-rut discipline."""
    from engine.agents.papers_curator.synthesis import _SYSTEM_PROMPT
    assert "ORTHOGONALITY GATE" in _SYSTEM_PROMPT
    assert "solo" in _SYSTEM_PROMPT
    assert "deployed sleeve" in _SYSTEM_PROMPT
    assert "70%" in _SYSTEM_PROMPT
    assert "DOWNGRADE" in _SYSTEM_PROMPT
    # The block is tightening, not softening — make that explicit
    assert "tightening, not a softening" in _SYSTEM_PROMPT
