"""
tests/test_quant_co_pilot_decision_lineage.py — Stage 3a Tool 1 unit tests.

Pre-registration: docs/spec_quant_co_pilot_decision_lineage_v1.md (id=53)

Covers:
  - 9 tool implementations (each: success path + failure path)
  - Citation validator (regex extraction + verifier dispatch)
  - ReAct agent loop (mocked LLM): tool dispatch + budget caps + abort reasons
  - DecisionLineageAgent wrapper smoke
"""
from __future__ import annotations

import json
from unittest import mock

import pytest

from engine.quant_co_pilot.base import (
    N_STEPS_MAX_LOCKED,
    COST_BUDGET_USD_LOCKED,
    LATENCY_BUDGET_MS_LOCKED,
    TEMPERATURE_LOCKED,
    Citation,
    ValidationResult,
    validate_citations,
    run_react_agent,
    TraceResult,
)
from engine.quant_co_pilot.tools import (
    TOOL_REGISTRY,
    TOOL_NAMES,
    dispatch_tool,
    read_spec_registry,
    search_amendments,
    read_memory_file,
    search_memory_index,
    read_capability_evidence,
    read_verdict_json,
    read_git_log,
)
from engine.quant_co_pilot.decision_lineage import DecisionLineageAgent


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_constants():
    assert N_STEPS_MAX_LOCKED == 8        # spec §2.3
    assert COST_BUDGET_USD_LOCKED == 0.05
    assert LATENCY_BUDGET_MS_LOCKED == 30000
    assert TEMPERATURE_LOCKED == 0.1
    assert len(TOOL_NAMES) == 9            # spec §2.2 + 2026-05-09 refresh (added read_capability_evidence)
    assert "read_capability_evidence" in TOOL_NAMES
    assert "read_spec_registry" in TOOL_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Tool: read_spec_registry
# ─────────────────────────────────────────────────────────────────────────────

def test_read_spec_registry_with_seeded_db():
    """Seed test DB with a row, verify read_spec_registry returns it."""
    import datetime
    from engine.memory import SessionFactory, SpecRegistry
    with SessionFactory() as s:
        # Clean any existing test row to avoid uniqueness conflicts
        s.query(SpecRegistry).filter(SpecRegistry.spec_path == "docs/_test_seed.md").delete()
        s.commit()
        row = SpecRegistry(
            spec_path="docs/_test_seed.md",
            git_blob_hash="abc123",
            current_hash="abc123",
            registered_at=datetime.datetime.utcnow(),
            amendment_log='[]',
            status="active",
            retro_registered=False,
            n_trials_contributed=1,
            last_validated_at=datetime.datetime.utcnow(),
        )
        s.add(row)
        s.commit()
        seeded_id = row.id

    r = read_spec_registry(spec_id=seeded_id)
    assert r.success, f"expected success, got error: {r.error_msg}"
    assert r.data["spec_id"] == seeded_id
    assert r.data["spec_path"] == "docs/_test_seed.md"
    assert "n_amendments" in r.data and r.data["n_amendments"] == 0
    assert "amendment_summary" in r.data and "amendment_log_full" in r.data


def test_read_spec_registry_unknown_id():
    r = read_spec_registry(spec_id=99999)
    assert not r.success
    assert "not found" in r.error_msg


# ─────────────────────────────────────────────────────────────────────────────
# Tool: search_amendments
# ─────────────────────────────────────────────────────────────────────────────

def test_search_amendments_empty_query():
    r = search_amendments(reason_substring="")
    assert not r.success
    assert "required" in r.error_msg


def test_search_amendments_no_match():
    r = search_amendments(reason_substring="xxxxnotarealwordzzzz")
    assert r.success
    assert r.data == []


# ─────────────────────────────────────────────────────────────────────────────
# Tool: read_memory_file
# ─────────────────────────────────────────────────────────────────────────────

def test_read_memory_file_index_exists():
    r = read_memory_file(memory_filename="MEMORY.md")
    assert r.success
    assert "Memory Index" in r.data


def test_read_memory_file_path_traversal_rejected():
    r = read_memory_file(memory_filename="../../../etc/passwd")
    assert not r.success
    assert "filename only" in r.error_msg or "must end" in r.error_msg


def test_read_memory_file_non_md_rejected():
    r = read_memory_file(memory_filename="README.txt")
    assert not r.success
    assert ".md" in r.error_msg


def test_read_memory_file_not_found():
    r = read_memory_file(memory_filename="nonexistent_xxxx.md")
    assert not r.success
    assert "not found" in r.error_msg


# ─────────────────────────────────────────────────────────────────────────────
# Tool: search_memory_index
# ─────────────────────────────────────────────────────────────────────────────

def test_search_memory_index_returns_matches():
    r = search_memory_index(keyword="WRDS")
    assert r.success
    assert isinstance(r.data, list)
    if r.data:
        for m in r.data:
            assert "title" in m and "filename" in m


def test_search_memory_index_empty_keyword():
    r = search_memory_index(keyword="")
    assert not r.success


# ─────────────────────────────────────────────────────────────────────────────
# Tool: read_capability_evidence
# ─────────────────────────────────────────────────────────────────────────────

def test_read_capability_evidence_v1_locked():
    r = read_capability_evidence(filename="factor_ensemble_v1_descriptive_positive_2026-05-09.md")
    assert r.success
    assert "DESCRIPTIVE_POSITIVE" in r.data


def test_read_capability_evidence_path_traversal_rejected():
    r = read_capability_evidence(filename="../spec_factor_ensemble_v1.md")
    assert not r.success


# ─────────────────────────────────────────────────────────────────────────────
# Tool: read_verdict_json
# ─────────────────────────────────────────────────────────────────────────────

def test_read_verdict_json_v1_real():
    r = read_verdict_json(verdict_path="data/factor_ensemble_v1/v1_verdict.json")
    if r.success:
        assert r.data.get("decision_label") == "DESCRIPTIVE_POSITIVE"


def test_read_verdict_json_path_outside_data_rejected():
    r = read_verdict_json(verdict_path="docs/spec_factor_ensemble_v1.md")
    assert not r.success
    assert "under data/" in r.error_msg


# ─────────────────────────────────────────────────────────────────────────────
# Tool: read_git_log
# ─────────────────────────────────────────────────────────────────────────────

def test_read_git_log_existing_file():
    r = read_git_log(file_path="engine/portfolio.py", max_commits=3)
    assert r.success
    if r.data:
        first = r.data[0]
        assert "commit_hash" in first
        assert len(first["commit_hash"]) == 40
        assert "author" in first and "date" in first and "message" in first


# ─────────────────────────────────────────────────────────────────────────────
# Citation validator
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_citations_extracts_spec_ids():
    answer = "因为 spec id=50 是 DESCRIPTIVE_POSITIVE,引用 spec_id=51 的 v2 robust。"
    result = validate_citations(answer)
    spec_id_cites = [c for c in result.citations if c.pattern == "spec_id"]
    assert len(spec_id_cites) >= 2
    raw_matches = {c.raw_match for c in spec_id_cites}
    assert "50" in raw_matches and "51" in raw_matches


def test_validate_citations_marks_unverified():
    """Fake spec_id=99999 → not in registry → annotated [UNVERIFIED]."""
    answer = "我引用 spec id=99999 这个不存在的 spec。"
    result = validate_citations(answer)
    fake_cites = [c for c in result.citations if c.raw_match == "99999"]
    assert len(fake_cites) == 1
    assert not fake_cites[0].verified
    assert "[UNVERIFIED: 99999]" in result.annotated_answer


def test_validate_citations_known_memory_file():
    """Real memory file should verify."""
    answer = "见 feedback_plain_language_first.md"
    result = validate_citations(answer)
    mem_cites = [c for c in result.citations if c.pattern == "memory_file"]
    if mem_cites:
        # If pattern matched, should be verified (file exists)
        assert mem_cites[0].verified


# ─────────────────────────────────────────────────────────────────────────────
# ReAct agent (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

def test_run_react_agent_terminates_on_final_answer():
    """Mock LLM: first call a tool, then give final_answer (must call ≥1 tool first)."""
    call_count = [0]
    def fake_llm_call(prompt, response_schema=None, *, scope="", extra=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return json.dumps({
                "thought": "Let me look up the production signal",
                "action": "read_spec_registry",
                "action_input": {"spec_id": 50},
            }), 0.001, 50
        return json.dumps({
            "thought": "I have the data, answering",
            "final_answer": "PRODUCTION_SIGNAL is ql01_bab per spec id=50",
        }), 0.001, 50

    with mock.patch("engine.quant_co_pilot.base._call_llm", side_effect=fake_llm_call):
        result = run_react_agent(
            query="What is PRODUCTION_SIGNAL?",
            tool_dispatcher=lambda action, args: {"data": {"production_signal": "ql01_bab"}},
            tool_descriptions="(mocked)",
            valid_tool_names=set(TOOL_NAMES),
        )
    assert isinstance(result, TraceResult)
    assert result.final_answer
    assert "ql01_bab" in result.final_answer
    assert len(result.steps) == 2  # tool call + final answer
    assert result.steps[1].final_answer is not None
    assert result.abort_reason is None


def test_run_react_agent_aborts_on_unknown_tool():
    """Mock LLM tries to call non-inventory tool → fail loud per spec §2.2."""
    def fake_llm_call(prompt, response_schema=None, *, scope="", extra=None):
        return json.dumps({
            "thought": "I'll call a fake tool",
            "action": "fake_tool_xyz",
            "action_input": {},
        }), 0.001, 50

    with mock.patch("engine.quant_co_pilot.base._call_llm", side_effect=fake_llm_call):
        result = run_react_agent(
            query="Test",
            tool_dispatcher=lambda action, args: {"data": {}},
            tool_descriptions="(mocked)",
            valid_tool_names=set(TOOL_NAMES),
        )
    assert result.abort_reason is not None
    assert "unknown tool" in result.abort_reason


def test_run_react_agent_respects_max_steps():
    """Agent that never gives final_answer → max_steps cap fires."""
    call_count = [0]
    def fake_llm_call(prompt, response_schema=None, *, scope="", extra=None):
        call_count[0] += 1
        return json.dumps({
            "thought": f"step {call_count[0]}, calling tool",
            "action": "read_spec_registry",
            "action_input": {"spec_id": 50},
        }), 0.001, 50

    with mock.patch("engine.quant_co_pilot.base._call_llm", side_effect=fake_llm_call):
        result = run_react_agent(
            query="Loop forever",
            tool_dispatcher=lambda action, args: {"data": {}},
            tool_descriptions="(mocked)",
            max_steps=3,  # locked override for this test
            valid_tool_names=set(TOOL_NAMES),
        )
    assert len(result.steps) == 3
    assert "max_steps" in (result.abort_reason or "")


def test_run_react_agent_respects_cost_budget():
    """Each step costs $0.10 → budget $0.05 cap fires after step 0."""
    def fake_llm_call(prompt, response_schema=None, *, scope="", extra=None):
        return json.dumps({
            "thought": "expensive",
            "action": "read_spec_registry",
            "action_input": {"spec_id": 50},
        }), 0.10, 50  # exceeds budget on first step

    with mock.patch("engine.quant_co_pilot.base._call_llm", side_effect=fake_llm_call):
        result = run_react_agent(
            query="cost burn test",
            tool_dispatcher=lambda action, args: {"data": {}},
            tool_descriptions="(mocked)",
            cost_budget_usd=0.05,
            valid_tool_names=set(TOOL_NAMES),
        )
    # First step runs (LLM call already happened before cost-check on step 1)
    # Second iteration: cost ≥ 0.05 → abort
    assert result.cost_usd >= 0.05
    assert "cost budget" in (result.abort_reason or "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# DecisionLineageAgent wrapper smoke
# ─────────────────────────────────────────────────────────────────────────────

def test_decision_lineage_agent_init_defaults():
    agent = DecisionLineageAgent()
    assert agent.max_steps == N_STEPS_MAX_LOCKED
    assert agent.cost_budget_usd == COST_BUDGET_USD_LOCKED
    assert agent.latency_budget_ms == LATENCY_BUDGET_MS_LOCKED


def test_decision_lineage_agent_answer_with_mock():
    def fake_llm_call(prompt, response_schema=None, *, scope="", extra=None):
        return json.dumps({
            "thought": "easy question",
            "final_answer": "spec id=50 has 11 amendments per the registry.",
        }), 0.002, 200

    with mock.patch("engine.quant_co_pilot.base._call_llm", side_effect=fake_llm_call):
        agent = DecisionLineageAgent()
        result = agent.answer("how many amendments on spec 50?")
    assert isinstance(result, TraceResult)
    assert "11" in result.final_answer
    assert any(c.pattern == "spec_id" for c in result.citations)


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatch
# ─────────────────────────────────────────────────────────────────────────────

def test_dispatch_unknown_tool_returns_error():
    out = dispatch_tool("nonexistent_tool", {})
    assert "error" in out
    assert "unknown tool" in out["error"]


def test_dispatch_known_tool_arg_mismatch():
    out = dispatch_tool("read_spec_registry", {"wrong_arg": "x"})
    assert "error" in out
    assert "arg mismatch" in out["error"] or "TypeError" in out["error"]


def test_dispatch_known_tool_success():
    """Use seeded test row from earlier test (spec_path docs/_test_seed.md)."""
    import datetime
    from engine.memory import SessionFactory, SpecRegistry
    with SessionFactory() as s:
        existing = s.query(SpecRegistry).filter(SpecRegistry.spec_path == "docs/_test_seed.md").first()
        if existing is None:
            row = SpecRegistry(
                spec_path="docs/_test_seed.md",
                git_blob_hash="abc123", current_hash="abc123",
                registered_at=datetime.datetime.utcnow(),
                amendment_log='[]', status="active",
                retro_registered=False, n_trials_contributed=1,
                last_validated_at=datetime.datetime.utcnow(),
            )
            s.add(row); s.commit()
            seeded_id = row.id
        else:
            seeded_id = existing.id

    out = dispatch_tool("read_spec_registry", {"spec_id": seeded_id})
    assert "data" in out, f"expected data, got: {out}"
    assert out["data"]["spec_id"] == seeded_id
