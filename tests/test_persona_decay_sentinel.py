"""tests/test_persona_decay_sentinel.py — Decay Sentinel persona wiring + doctrine.

Does NOT call the live LLM (chat_turn). Asserts the persona is well-formed, routed,
allow-listed, and that its prompt encodes the 0-LLM-in-DECISION rule + role-aware
judging; plus the owned read-only tool returns a valid report.
"""
import json

import pytest

from engine.agents.persona import DECAY_SENTINEL
from engine.agents.persona.base import AgentPersona
from engine.agents.persona.tools import execute_tool
from engine.llm.call import _WORKLOAD_ROUTING
from engine.llm_cost_ledger import ALLOWED_AGENT_IDS


def test_persona_is_well_formed_and_routed():
    assert isinstance(DECAY_SENTINEL, AgentPersona)
    assert DECAY_SENTINEL.agent_id == "decay_sentinel"
    assert DECAY_SENTINEL.agent_id in ALLOWED_AGENT_IDS         # cost ledger key
    assert DECAY_SENTINEL.workload in _WORKLOAD_ROUTING         # provider routing
    assert _WORKLOAD_ROUTING[DECAY_SENTINEL.workload][0] == "anthropic"


def test_persona_owns_the_decay_tool():
    names = [t["name"] for t in DECAY_SENTINEL.tools]
    assert "read_decay_sentinel_report" in names               # owns the report
    assert "lookup_strategy_status" in names                   # shared
    # read-only: must NOT expose any mutate/trade tool
    assert all("trade" not in n and "mutate" not in n for n in names)


def test_prompt_encodes_zero_llm_decision_doctrine():
    p = DECAY_SENTINEL.system_prompt
    assert "0-LLM-in-DECISION" in p or "math decides" in p.lower()
    assert "NEVER compute your own decay verdict" in p
    # role-aware judging must be spelled out
    for role in ("alpha", "insurance", "trend", "regime_premium"):
        assert role in p
    assert "crisis-payoff" in p.lower() and "signal-ic" in p.lower()
    assert "NO EMOJIS" in p


def test_owned_tool_returns_valid_report():
    out, is_err = execute_tool("read_decay_sentinel_report", {})
    assert is_err is False
    d = json.loads(out)
    assert d["overall"] in ("HEALTHY", "WATCH", "ACTION")
    assert d["n_mechanisms"] >= 1
    # every mechanism carries a role (the basis the persona must respect)
    assert all("role" in m for m in d["mechanisms"].values())


def test_exported_from_persona_package():
    import engine.agents.persona as pkg
    assert "DECAY_SENTINEL" in pkg.__all__
    assert pkg.DECAY_SENTINEL is DECAY_SENTINEL
