"""tests/test_authority.py — least-privilege enforcement at the tool executor."""
from types import SimpleNamespace

from engine.agents.governance.authority import (
    capability_set, check_authority, enforce_tool_call,
)


def _spy_persona():
    calls = []

    def spy(name, tool_input):
        calls.append(name)
        return ("OK", False)

    p = SimpleNamespace(agent_id="test_agent",
                        tools=[{"name": "allowed_tool"}],
                        tool_executor=spy)
    return p, calls


# ── capability set + check ────────────────────────────────────────────────────
def test_capability_set_is_the_declared_palette():
    p, _ = _spy_persona()
    assert capability_set(p) == frozenset({"allowed_tool"})


def test_check_authority_in_and_out_of_palette():
    p, _ = _spy_persona()
    assert check_authority(p, "allowed_tool")[0] is True
    assert check_authority(p, "forbidden_tool")[0] is False


# ── enforcement modes ────────────────────────────────────────────────────────
def test_enforce_blocks_out_of_palette_and_does_not_run_tool():
    p, calls = _spy_persona()
    out, is_err = enforce_tool_call(p, "forbidden_tool", {}, mode="enforce")
    assert is_err is True and "authority denied" in out
    assert calls == []                         # tool was NOT executed


def test_enforce_allows_in_palette():
    p, calls = _spy_persona()
    out, is_err = enforce_tool_call(p, "allowed_tool", {}, mode="enforce")
    assert (out, is_err) == ("OK", False) and calls == ["allowed_tool"]


def test_warn_mode_executes_but_flags():
    p, calls = _spy_persona()
    out, is_err = enforce_tool_call(p, "forbidden_tool", {}, mode="warn")
    assert (out, is_err) == ("OK", False) and calls == ["forbidden_tool"]   # ran anyway


def test_off_mode_passthrough():
    p, calls = _spy_persona()
    enforce_tool_call(p, "forbidden_tool", {}, mode="off")
    assert calls == ["forbidden_tool"]


# ── cross-agent least-privilege on REAL personas ────────────────────────────
def test_decay_sentinel_cannot_call_risk_manager_tool():
    from engine.agents.persona import DECAY_SENTINEL
    caps = capability_set(DECAY_SENTINEL)
    assert "read_decay_sentinel_report" in caps          # its own tool
    assert "query_recent_alerts" not in caps             # RM's tool
    assert check_authority(DECAY_SENTINEL, "query_recent_alerts")[0] is False
    # blocked at runtime, RM's tool never runs
    out, is_err = enforce_tool_call(DECAY_SENTINEL, "query_recent_alerts", {}, mode="enforce")
    assert is_err is True and "authority denied" in out
