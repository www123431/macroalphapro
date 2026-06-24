"""DeepSeek external audit adapter tests."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from engine.llm.providers import deepseek_external_audit_provider as ds_audit


def test_severity_critical_keywords():
    out = ds_audit._parse_severity_from_response(
        "This verdict is likely WRONG. Principal should not trust."
    )
    assert out == "critical"


def test_severity_concern_keywords():
    out = ds_audit._parse_severity_from_response(
        "There is a meaningful caveat about the spanning."
    )
    assert out == "concern"


def test_severity_no_issue_keywords():
    out = ds_audit._parse_severity_from_response(
        "The methodology looks sound and appropriate for this case."
    )
    assert out == "no_issue"


def test_severity_critical_wins_over_no_issue():
    """When both critical AND no_issue keywords present, critical wins."""
    out = ds_audit._parse_severity_from_response(
        "Looks fine at first but methodology is actually WRONG."
    )
    assert out == "critical"


def test_flagged_categories_extracts_known_tags():
    response = (
        "Concern: the spanning anchor uses CAPM but should use FF5 "
        "factor model. Also potential look-ahead bias in the PIT data."
    )
    cats = ds_audit._parse_flagged_categories(response)
    assert "spanning" in cats
    assert "PIT" in cats


def test_flagged_categories_empty_when_no_known_tags():
    response = "Generic comment with no specific failure mode."
    cats = ds_audit._parse_flagged_categories(response)
    assert cats == []


def test_adapter_registered_on_import():
    """The auto-install at module load registers 'deepseek' provider."""
    from engine.research.external_audit import _PROVIDER_REGISTRY
    assert "deepseek" in _PROVIDER_REGISTRY


def test_adapter_handles_deepseek_call_failure_gracefully():
    """When call_deepseek raises, adapter returns skipped severity."""
    provider = ds_audit.DeepSeekExternalAuditProvider()
    with patch("engine.llm.providers.deepseek_provider.call_deepseek",
                 side_effect=RuntimeError("network down")):
        response, severity, cats, cost = provider.adversarial_audit(
            subject_payload={"event_id": "t1"},
            prompt="test prompt",
        )
        assert severity == "skipped"
        assert "call_failed" in response
        assert cost == 0.0


def test_adapter_records_cost_from_usage():
    """Mock a DeepSeek response with usage stats; adapter computes cost."""
    provider = ds_audit.DeepSeekExternalAuditProvider()
    # Use a plain object with explicit attrs (not MagicMock — MagicMock
    # returns truthy stubs for any attribute access which would break
    # getattr defaults).
    class _FakeResult:
        text = "Concern: spanning anchor questionable."
        output_tokens = 100
        input_tokens  = 500
        raw_usage     = {}
    with patch("engine.llm.providers.deepseek_provider.call_deepseek",
                 return_value=_FakeResult()):
        response, severity, cats, cost = provider.adversarial_audit(
            subject_payload={"event_id": "t1"},
            prompt="test prompt",
        )
        assert severity == "concern"
        assert "spanning" in cats
        # 100 output tokens × $0.20/M = $0.00002
        # 500 input tokens × $0.05/M = $0.000025
        # Total ≈ $0.000045
        assert 0.0 < cost < 0.001
