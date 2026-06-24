"""Tests for Frontier 4 (2026-06-01) — multi-step research chains.

Chain runner is tested deterministically with fake tool_dispatcher.
Reference chains are tested for correct step structure + guard wiring.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from engine.research.research_chain import (
    Chain, ChainRun, Step, StepResult,
    _eval_guard, _resolve_path, _resolve_template,
    read_recent_chain_runs, run_chain,
)


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    fake = tmp_path / "chain_runs.jsonl"
    monkeypatch.setattr(
        "engine.research.research_chain.CHAIN_RUNS_LEDGER", fake,
    )
    return fake


# ── _resolve_path ────────────────────────────────────────────────────


def test_resolve_path_dict():
    ctx = {"a": {"b": {"c": 42}}}
    assert _resolve_path("a.b.c", ctx) == 42


def test_resolve_path_missing_returns_none():
    ctx = {"a": {}}
    assert _resolve_path("a.b.c", ctx) is None


def test_resolve_path_list_index():
    ctx = {"xs": [{"n": 1}, {"n": 2}]}
    assert _resolve_path("xs.1.n", ctx) == 2


# ── _resolve_template ────────────────────────────────────────────────


def test_resolve_template_full_ref_preserves_type():
    """A string that is ENTIRELY one template ref should return the
    resolved object with its native type."""
    ctx = {"x": {"y": [1, 2, 3]}}
    out = _resolve_template("{{x.y}}", ctx)
    assert out == [1, 2, 3]


def test_resolve_template_mixed_string_coerces():
    ctx = {"name": "alpha", "n": 5}
    out = _resolve_template("hello {{name}} count {{n}}", ctx)
    assert out == "hello alpha count 5"


def test_resolve_template_recursive_into_dict_and_list():
    ctx = {"v": 7}
    args = {"a": "{{v}}", "b": ["x", "{{v}}", "y"]}
    out = _resolve_template(args, ctx)
    assert out == {"a": 7, "b": ["x", 7, "y"]}


def test_resolve_template_missing_ref_is_empty_in_mixed_string():
    ctx = {}
    assert _resolve_template("a {{missing}} b", ctx) == "a  b"


# ── _eval_guard ──────────────────────────────────────────────────────


def test_guard_none_is_truthy():
    assert _eval_guard(None, {}) is True


def test_guard_empty_string_is_truthy_too():
    """Empty guard string = no guard = run."""
    assert _eval_guard("", {}) is True


def test_guard_resolves_template_then_truthy():
    assert _eval_guard("{{x}}", {"x": "ok"}) is True
    assert _eval_guard("{{x}}", {"x": ""}) is False
    assert _eval_guard("{{x}}", {"x": None}) is False
    assert _eval_guard("{{x}}", {"x": 0}) is False


# ── run_chain ────────────────────────────────────────────────────────


def test_run_chain_simple_sequence(isolated_ledger):
    chain = Chain(
        chain_id="simple",
        description="t",
        steps=[
            Step(name="s1", tool="dummy",
                  args={"x": "{{initial.input}}"}),
            Step(name="s2", tool="dummy",
                  args={"prev": "{{steps.s1.result.echo}}"}),
        ],
    )

    def fake_dispatcher(name, args):
        return {"echo": args.get("x") or args.get("prev")}

    run = run_chain(chain, initial_context={"input": "hello"},
                     tool_dispatcher=fake_dispatcher)
    assert run.status == "completed"
    assert len(run.steps) == 2
    assert run.steps[0].result == {"echo": "hello"}
    assert run.steps[1].result == {"echo": "hello"}


def test_run_chain_guard_skips_step(isolated_ledger):
    chain = Chain(
        chain_id="guarded",
        description="t",
        steps=[
            Step(name="optional", tool="dummy",
                  guard="{{initial.enable_optional}}"),
            Step(name="always", tool="dummy"),
        ],
    )
    run = run_chain(chain, initial_context={"enable_optional": False},
                     tool_dispatcher=lambda n, a: {"ok": True})
    assert run.steps[0].status == "skipped"
    assert run.steps[1].status == "ok"


def test_run_chain_halts_on_failure_when_default(isolated_ledger):
    chain = Chain(
        chain_id="halt_test",
        description="t",
        steps=[
            Step(name="s1", tool="dummy"),
            Step(name="s2", tool="dummy"),  # default halt
            Step(name="s3", tool="dummy"),
        ],
    )
    def fake(name, args):
        if "s2" in name or False:
            pass
        raise RuntimeError("boom")
    run = run_chain(chain, tool_dispatcher=fake)
    assert run.status == "halted"
    # s1 fails (halt), s2 + s3 never run
    assert run.steps[0].status == "failed"
    assert len(run.steps) == 1


def test_run_chain_continues_on_failure_when_marked(isolated_ledger):
    chain = Chain(
        chain_id="continue_test",
        description="t",
        steps=[
            Step(name="optional", tool="dummy", on_failure="continue"),
            Step(name="critical", tool="dummy"),
        ],
    )
    calls = []
    def fake(name, args):
        calls.append(name)
        if name == "optional_tool":
            raise RuntimeError("network down")
        return {"ok": True}
    # The tool name is just "dummy" since we don't differentiate
    # tool-name from step-name here. Use a different fake.
    def fake2(name, args):
        calls.append(args)
        if not calls:  # first call
            raise RuntimeError("boom")
        return {"ok": True}
    # Simpler: make failure depend on step ordering via mock
    state = {"first": True}
    def fake3(name, args):
        if state["first"]:
            state["first"] = False
            raise RuntimeError("first call fails")
        return {"ok": True}
    run = run_chain(chain, tool_dispatcher=fake3)
    assert run.status == "completed"
    assert run.steps[0].status == "failed"
    assert run.steps[1].status == "ok"


def test_run_chain_writes_ledger(isolated_ledger):
    chain = Chain(
        chain_id="ledger_test", description="t",
        steps=[Step(name="s1", tool="dummy")],
    )
    run = run_chain(chain, tool_dispatcher=lambda n, a: {"ok": True})
    assert isolated_ledger.exists()
    lines = isolated_ledger.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["chain_id"] == "ledger_test"
    assert row["run_id"] == run.run_id
    assert row["status"] == "completed"


def test_read_recent_chain_runs_filters_by_chain_id(isolated_ledger):
    for cid in ("A", "B", "A", "C"):
        chain = Chain(chain_id=cid, description="t",
                       steps=[Step(name="s", tool="dummy")])
        run_chain(chain, tool_dispatcher=lambda n, a: {})
    rows = read_recent_chain_runs(chain_id="A")
    assert len(rows) == 2
    assert all(r["chain_id"] == "A" for r in rows)


# ── chain_library ────────────────────────────────────────────────────


def test_chain_library_registry_includes_canonical_chains():
    from engine.research.chain_library import (
        CHAINS, PAPER_TO_CANDIDATE, FAMILY_AUDIT, list_chains,
    )
    assert "paper_to_candidate" in CHAINS
    assert "family_audit"       in CHAINS
    summaries = list_chains()
    assert {s["chain_id"] for s in summaries} == \
            {"paper_to_candidate", "family_audit"}


def test_chain_library_get_chain_unknown_raises():
    from engine.research.chain_library import get_chain
    with pytest.raises(KeyError, match="unknown chain_id"):
        get_chain("not_a_chain")


def test_family_audit_runs_against_real_tools(isolated_ledger):
    """Smoke test against the real tool dispatcher — confirms our
    reference chain's tool names + arg shapes line up with the
    actual llm_tools registry. Doesn't assert on contents (which
    depend on live data state)."""
    from engine.research.chain_library import FAMILY_AUDIT
    run = run_chain(FAMILY_AUDIT,
                     initial_context={"family": "earnings_drift"})
    # The chain shouldn't fully halt — at least the deployed_in_family
    # step should succeed since query_library is always available
    statuses = {s.name: s.status for s in run.steps}
    assert statuses.get("deployed_in_family") == "ok"
    # graveyard_in_family is also generally available
    assert statuses.get("graveyard_in_family") in ("ok", "failed")
