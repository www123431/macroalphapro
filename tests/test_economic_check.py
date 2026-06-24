"""Tests for engine.research.economic_check (Phase 5 F).

Critical properties:
1. Deterministic fallback when no API key → returns valid PlausibilityCheck
2. Schema conformance: all expected fields present
3. Score in [0, 1] range
4. Fidelity recommendation in valid enum
5. Log file appended on log=True
6. Log file untouched on log=False
7. LLM failure → graceful fallback (no exception)
8. JSON parse helper handles edge cases
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research import economic_check as EC


@pytest.fixture
def mock_mechanism():
    return {
        "id": "test_mech_v1",
        "family": "earnings_underreaction",
        "parent_family": "equity_factor",
        "canonical_paper_id": "bernard_thomas_1989_jar",
        "typical_sample": "1990-2024",
        "required_data": ["SUE_panel", "ann_dates"],
        "mechanism_economics":
            "Investors underreact to earnings surprises. Drift persists 60d post-announcement.",
        "mechanism_break_conditions": ["Decimalization 2001", "Reg FD 2000"],
        "status_in_our_book": "UNTESTED",
    }


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    monkeypatch.setattr(EC, "PLAUSIBILITY_LOG",
                          tmp_path / "economic_plausibility_log.jsonl")


def test_deterministic_fallback_returns_valid_check(mock_mechanism, monkeypatch):
    monkeypatch.setattr(EC, "_read_anthropic_key", lambda: None)
    result = EC.check_economic_plausibility(mock_mechanism, use_llm=True, log=False)
    assert isinstance(result, EC.PlausibilityCheck)
    assert result.mode == "deterministic_fallback"
    assert 0.0 <= result.plausibility_score <= 1.0
    assert result.mechanism_id == "test_mech_v1"


def test_use_llm_false_returns_fallback(mock_mechanism):
    result = EC.check_economic_plausibility(mock_mechanism, use_llm=False, log=False)
    assert result.mode == "deterministic_fallback"


def test_required_schema_fields_present(mock_mechanism):
    result = EC.check_economic_plausibility(mock_mechanism, use_llm=False, log=False)
    d = result.to_dict()
    for field in ("mechanism_id", "plausibility_score", "economic_intuition",
                    "concerns", "regime_assessment", "cousin_with_deployed",
                    "fidelity_recommendation", "mode", "cost_usd", "ts"):
        assert field in d


def test_score_bounded_in_unit_interval(mock_mechanism):
    result = EC.check_economic_plausibility(mock_mechanism, use_llm=False, log=False)
    assert 0.0 <= result.plausibility_score <= 1.0


def test_fidelity_recommendation_in_enum(mock_mechanism):
    result = EC.check_economic_plausibility(mock_mechanism, use_llm=False, log=False)
    assert result.fidelity_recommendation in ("literal", "adapted", "inspired")


def test_log_written_when_log_true(mock_mechanism):
    EC.check_economic_plausibility(mock_mechanism, use_llm=False, log=True)
    assert EC.PLAUSIBILITY_LOG.exists()
    rows = [json.loads(l) for l in EC.PLAUSIBILITY_LOG.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["mechanism_id"] == "test_mech_v1"


def test_log_not_written_when_log_false(mock_mechanism):
    EC.check_economic_plausibility(mock_mechanism, use_llm=False, log=False)
    assert not EC.PLAUSIBILITY_LOG.exists()


def test_llm_failure_falls_back_gracefully(mock_mechanism, monkeypatch):
    """Mock anthropic to raise; check that we fallback cleanly."""

    class _MockAnthropic:
        def __init__(self, **kw):
            pass
        @property
        def messages(self):
            class _M:
                def create(self, **kw):
                    raise RuntimeError("simulated LLM failure")
            return _M()

    monkeypatch.setattr(EC, "_read_anthropic_key", lambda: "fake-key")
    monkeypatch.setitem(
        __import__("sys").modules, "anthropic",
        type("M", (), {"Anthropic": _MockAnthropic}),
    )
    result = EC.check_economic_plausibility(mock_mechanism, use_llm=True, log=False)
    # Even with simulated failure, we get a fallback result
    assert result.mode == "deterministic_fallback"


def test_read_plausibility_log_recent_first(mock_mechanism):
    EC.check_economic_plausibility(
        {**mock_mechanism, "id": "a"}, use_llm=False, log=True)
    EC.check_economic_plausibility(
        {**mock_mechanism, "id": "b"}, use_llm=False, log=True)
    EC.check_economic_plausibility(
        {**mock_mechanism, "id": "c"}, use_llm=False, log=True)
    rows = EC.read_plausibility_log(limit=10)
    assert [r["mechanism_id"] for r in rows] == ["c", "b", "a"]


def test_parse_json_handles_no_braces():
    assert EC._parse_json("no json here") is None


def test_parse_json_handles_valid():
    text = 'Some preamble {"plausibility_score": 0.7, "concerns": []} trailing'
    parsed = EC._parse_json(text)
    assert parsed == {"plausibility_score": 0.7, "concerns": []}


def test_parse_json_handles_nested():
    text = '{"a": {"b": 1}, "c": [1,2,3]}'
    parsed = EC._parse_json(text)
    assert parsed["a"]["b"] == 1
    assert parsed["c"] == [1, 2, 3]


def test_deployed_summaries_default_handles_missing(monkeypatch, tmp_path):
    """When library has no DEPLOYED entries, returns []."""
    monkeypatch.setattr(EC, "REPO_ROOT", tmp_path)
    summaries = EC._deployed_summaries_default()
    assert summaries == []
