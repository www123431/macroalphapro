"""
Offline tests for engine.agents.spec_drafter — no LLM calls.

Live LLM eval is in scripts/eval_spec_drafter.py (cost-aware; opt-in
because it spends real LLM budget on every run).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    from engine.agents.spec_drafter import (  # noqa: F401
        draft_spec, SpecDraft, SPEC_DRAFT_RESPONSE_SCHEMA,
        _safety_gate, _render_markdown,
        _FORBIDDEN_PATH_PATTERNS, _ALLOWED_CITATIONS,
    )


def test_response_schema_required_fields():
    from engine.agents.spec_drafter import SPEC_DRAFT_RESPONSE_SCHEMA
    required = SPEC_DRAFT_RESPONSE_SCHEMA["required"]
    assert len(required) >= 13
    for field in [
        "title", "tldr", "hypothesis", "decision_rule", "n_trials_impact",
        "data_requirements", "predictions", "implementation_steps",
        "success_criteria", "failure_modes", "out_of_scope",
        "literature_anchors", "risks_and_caveats",
    ]:
        assert field in required, f"missing required field: {field}"


def test_empty_hypothesis_rejected_no_llm():
    from engine.agents.spec_drafter import draft_spec
    r = draft_spec("")
    assert r.status == "rejected"
    assert r.cost_usd == 0.0
    assert "empty" in (r.error_msg or "").lower()


def test_safety_gate_forbidden_path():
    from engine.agents.spec_drafter import _safety_gate
    metadata = {
        "title": "x", "tldr": "x",
        "hypothesis": {"h0": "a", "h1": "b"},
        "decision_rule": {"ship_criteria": "x", "fail_criteria": "y",
                           "literature_conditional_exemption": False},
        "n_trials_impact": {"n_added": 1, "rationale": "x"},
        "implementation_steps": ["modify engine/auto_audit_rules.py"],
        "literature_anchors": ["Frazzini & Pedersen 2014"],
    }
    findings = _safety_gate(metadata, "")
    assert any("forbidden" in f.lower() for f in findings)


def test_safety_gate_no_citation():
    from engine.agents.spec_drafter import _safety_gate
    metadata = {
        "implementation_steps": ["normal step"],
        "literature_anchors": [],
        "n_trials_impact": {"n_added": 1, "rationale": "x"},
        "decision_rule": {"literature_conditional_exemption": False},
    }
    findings = _safety_gate(metadata, "")
    assert any("citation" in f.lower() for f in findings)


def test_safety_gate_n_trials_zero():
    from engine.agents.spec_drafter import _safety_gate
    metadata = {
        "implementation_steps": ["normal"],
        "literature_anchors": ["Frazzini & Pedersen 2014"],
        "n_trials_impact": {"n_added": 0, "rationale": "x"},
        "decision_rule": {"literature_conditional_exemption": False},
    }
    findings = _safety_gate(metadata, "")
    assert any("n_trials_impact.n_added" in f for f in findings)


def test_safety_gate_lit_conditional_without_anchor():
    """Literature-conditional exemption requires strong factor-finance citation."""
    from engine.agents.spec_drafter import _safety_gate
    metadata = {
        "implementation_steps": ["normal"],
        "literature_anchors": ["Hamilton 1989"],   # not factor-finance
        "n_trials_impact": {"n_added": 1, "rationale": "x"},
        "decision_rule": {"literature_conditional_exemption": True},
    }
    findings = _safety_gate(metadata, "")
    assert any("literature_conditional_exemption" in f for f in findings)


def test_render_markdown_has_required_sections():
    from engine.agents.spec_drafter import _render_markdown
    metadata = {
        "title": "Spec — Test",
        "tldr": "Testing.",
        "hypothesis": {"h0": "null", "h1": "alt"},
        "decision_rule": {"ship_criteria": "x", "fail_criteria": "y",
                           "literature_conditional_exemption": False},
        "n_trials_impact": {"n_added": 1, "rationale": "x"},
        "data_requirements": ["A", "B"],
        "predictions": ["pred1", "pred2"],
        "implementation_steps": ["step1", "step2"],
        "success_criteria": "x",
        "failure_modes": ["fail1"],
        "out_of_scope": ["scope1"],
        "literature_anchors": ["Frazzini-Pedersen 2014"],
        "risks_and_caveats": "caveats",
    }
    md = _render_markdown(metadata, "test", "docs/spec_test.md")
    for section in [
        "## TL;DR", "## 1. Hypothesis", "## 2. Decision rule",
        "## 3. Multiple-testing impact", "## 4. Data requirements",
        "## 5. Predictions", "## 6. Implementation steps",
        "## 7. Success criteria", "## 8. Pre-registered failure modes",
        "## 9. Out of scope", "## 10. Literature anchors",
        "## 11. Risks & caveats", "Reviewer checklist",
    ]:
        assert section in md, f"missing section: {section}"
