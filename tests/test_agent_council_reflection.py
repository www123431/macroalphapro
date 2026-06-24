"""Tests for Frontier 1 (2026-06-01) — structured reflection round.

Goal: each critic gets ONE shot to read the peer's verdict and either
confirm or revise. Parallel, single-turn, bounded. Pattern 6 (NOT
Pattern 5 autonomous debate).

LLM calls are mocked so tests are deterministic + fast (no network,
no $ cost). The mocks return canned JSON verdicts matching the schema.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest import mock

import pytest

from engine.research.agent_council import (
    AgentVerdict,
    CouncilVerdict,
    ProposalDict,
    _run_one_reflector,
    aggregate_verdicts,
    critique_council,
)


@pytest.fixture
def proposal():
    return ProposalDict(
        title="test_proposal_alpha",
        family="test_family",
        parent_family="test_parent",
        proposed_role="alpha_seeker",
        economics_text="A test mechanism.",
        required_data=["test_data"],
        motivation="testing reflection",
    )


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    """Redirect the council ledger to a tmp file so tests don't pollute
    the real data/research/council_runs.jsonl."""
    fake = tmp_path / "council_runs.jsonl"
    monkeypatch.setattr(
        "engine.research.agent_council.COUNCIL_RUNS_LEDGER", fake,
    )
    return fake


def _make_verdict(name: str, verdict: str, conf: float = 0.7,
                  concerns: list[str] | None = None) -> AgentVerdict:
    return AgentVerdict(
        agent_name=name,
        verdict=verdict,
        confidence=conf,
        rationale=f"{name} round-1 rationale",
        material_concerns=concerns or [],
    )


# ── reflector unit tests ─────────────────────────────────────────────────


def test_reflector_confirmed_when_llm_returns_same_verdict(proposal):
    own = _make_verdict("behavioral_theorist", "WARN", 0.6)
    peer = _make_verdict("empirical_devils_advocate", "WARN", 0.55)

    fake_response = json.dumps({
        "verdict":     "WARN",
        "confidence":  0.62,
        "rationale":   "peer raised a stat concern but it doesn't change "
                       "my behavioral critique; concerns coexist.",
        "fatal_red_flags":   [],
        "material_concerns": ["overreaction-vs-attention ambiguity"],
        "reflection_action": "confirmed",
    })
    with mock.patch(
        "engine.research.agent_council.run_agent_with_tools",
        return_value=(fake_response, []),
    ):
        out = _run_one_reflector(
            agent_name="behavioral_theorist",
            system_prompt="...",
            allowed_tools=[],
            proposal=proposal,
            own_round_1=own,
            peer_round_1=peer,
            api_key="fake",
        )
    assert out.verdict == "WARN"
    assert out.reflection_action == "confirmed"
    assert out.round_1_verdict == "WARN"
    assert out.round_1_confidence == 0.6


def test_reflector_infers_revised_down_action_on_verdict_drop(proposal):
    """If LLM omits reflection_action, helper infers from verdict delta."""
    own = _make_verdict("empirical_devils_advocate", "PASS", 0.8)
    peer = _make_verdict("behavioral_theorist", "FAIL", 0.9,
                          concerns=["mechanism story doesn't hold"])

    fake_response = json.dumps({
        "verdict":     "WARN",
        "confidence":  0.5,
        "rationale":   "peer's behavioral concern made me reconsider",
        # no reflection_action field
    })
    with mock.patch(
        "engine.research.agent_council.run_agent_with_tools",
        return_value=(fake_response, []),
    ):
        out = _run_one_reflector(
            agent_name="empirical_devils_advocate",
            system_prompt="...",
            allowed_tools=[],
            proposal=proposal,
            own_round_1=own,
            peer_round_1=peer,
            api_key="fake",
        )
    assert out.verdict == "WARN"
    assert out.reflection_action == "revised_down"  # PASS → WARN


def test_reflector_revised_up_on_verdict_lift(proposal):
    own = _make_verdict("behavioral_theorist", "FAIL", 0.7)
    peer = _make_verdict("empirical_devils_advocate", "PASS", 0.8)

    fake_response = json.dumps({
        "verdict":     "WARN",
        "confidence":  0.5,
        "rationale":   "peer's stat evidence weakens my fatal concern",
        "reflection_action": "revised_up",
    })
    with mock.patch(
        "engine.research.agent_council.run_agent_with_tools",
        return_value=(fake_response, []),
    ):
        out = _run_one_reflector(
            agent_name="behavioral_theorist",
            system_prompt="...",
            allowed_tools=[],
            proposal=proposal,
            own_round_1=own,
            peer_round_1=peer,
            api_key="fake",
        )
    assert out.verdict == "WARN"
    assert out.reflection_action == "revised_up"


def test_reflector_failure_does_not_lose_round_1(proposal):
    """If reflection LLM call raises, round-1 verdict survives."""
    own = _make_verdict("empirical_devils_advocate", "WARN", 0.6)
    peer = _make_verdict("behavioral_theorist", "PASS", 0.7)

    with mock.patch(
        "engine.research.agent_council.run_agent_with_tools",
        side_effect=RuntimeError("anthropic 503"),
    ):
        out = _run_one_reflector(
            agent_name="empirical_devils_advocate",
            system_prompt="...",
            allowed_tools=[],
            proposal=proposal,
            own_round_1=own,
            peer_round_1=peer,
            api_key="fake",
        )
    assert out.verdict == "WARN"
    assert out.reflection_action == "confirmed"
    assert "reflection skipped" in out.rationale


# ── critique_council integration with reflection ─────────────────────────


def test_critique_council_no_reflection_path(proposal, isolated_ledger):
    """Without enable_reflection, behavior matches the original path."""
    canned = json.dumps({
        "verdict": "WARN", "confidence": 0.6, "rationale": "ok-ish",
        "fatal_red_flags": [], "material_concerns": ["test concern"],
    })
    with mock.patch(
        "engine.research.agent_council.run_agent_with_tools",
        return_value=(canned, []),
    ):
        council = asyncio.run(
            critique_council(proposal, api_key="fake"),
        )
    assert council.reflection_enabled is False
    assert council.round_1_consensus == council.consensus
    assert council.consensus == "NEEDS_REVISION"
    assert all(v.reflection_action is None for v in council.verdicts)


def test_critique_council_with_reflection_runs_4_llm_calls(
    proposal, isolated_ledger,
):
    """With reflection, we expect 2 round-1 calls + 2 reflection calls."""
    canned = json.dumps({
        "verdict": "WARN", "confidence": 0.6,
        "rationale": "deterministic mock",
        "fatal_red_flags": [], "material_concerns": ["mock concern"],
        "reflection_action": "confirmed",
    })
    with mock.patch(
        "engine.research.agent_council.run_agent_with_tools",
        return_value=(canned, []),
    ) as mocked:
        council = asyncio.run(
            critique_council(proposal, api_key="fake",
                             enable_reflection=True),
        )
    # 2 critics × (1 round-1 + 1 reflection) = 4 calls
    assert mocked.call_count == 4
    assert council.reflection_enabled is True
    # Round-1 + final should both be NEEDS_REVISION (verdicts unchanged)
    assert council.round_1_consensus == "NEEDS_REVISION"
    assert council.consensus == "NEEDS_REVISION"
    # All critics carry round-1 snapshot + reflection_action
    for v in council.verdicts:
        assert v.round_1_verdict == "WARN"
        assert v.reflection_action == "confirmed"


def test_critique_council_reflection_can_flip_consensus(
    proposal, isolated_ledger,
):
    """Round-1 WARN+WARN can flip to PASS+PASS after reflection."""

    call_sequence = []

    def fake_run(*, agent_name: str, **kwargs):
        call_sequence.append(agent_name)
        # Round-1 calls have agent_name like "behavioral_theorist" (no
        # .reflection suffix); round-2 calls have ".reflection" suffix.
        if ".reflection" in agent_name:
            payload = json.dumps({
                "verdict": "PASS", "confidence": 0.75,
                "rationale": "peer's evidence resolved my concern",
                "fatal_red_flags": [], "material_concerns": [],
                "reflection_action": "revised_up",
            })
        else:
            payload = json.dumps({
                "verdict": "WARN", "confidence": 0.55,
                "rationale": "concerned but uncertain",
                "fatal_red_flags": [],
                "material_concerns": ["initial uncertainty"],
            })
        return payload, []

    with mock.patch(
        "engine.research.agent_council.run_agent_with_tools",
        side_effect=fake_run,
    ):
        council = asyncio.run(
            critique_council(proposal, api_key="fake",
                             enable_reflection=True),
        )
    assert council.round_1_consensus == "NEEDS_REVISION"
    assert council.consensus == "APPROVE"
    for v in council.verdicts:
        assert v.verdict == "PASS"
        assert v.reflection_action == "revised_up"
        assert v.round_1_verdict == "WARN"


def test_ledger_records_reflection_metadata(proposal, isolated_ledger):
    canned = json.dumps({
        "verdict": "PASS", "confidence": 0.8,
        "rationale": "clean", "fatal_red_flags": [],
        "material_concerns": [], "reflection_action": "confirmed",
    })
    with mock.patch(
        "engine.research.agent_council.run_agent_with_tools",
        return_value=(canned, []),
    ):
        asyncio.run(
            critique_council(proposal, api_key="fake",
                             enable_reflection=True),
        )
    assert isolated_ledger.exists()
    lines = isolated_ledger.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["reflection_enabled"] is True
    assert row["consensus"] == "APPROVE"
    assert row["round_1_consensus"] == "APPROVE"
    assert row["reflection_actions"] == ["confirmed", "confirmed"]
