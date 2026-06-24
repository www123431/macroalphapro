"""Tests for engine.research.discovery.llm_feature_extractor."""
from __future__ import annotations

from unittest import mock

import pytest

from engine.research.discovery.llm_feature_extractor import (
    LLMFeatureExtraction, compute_hybrid_confidence,
    extract_boolean_features,
)


# ── LLMFeatureExtraction dataclass ───────────────────────────────────────

def test_extraction_defaults_all_false():
    ext = LLMFeatureExtraction()
    assert ext.estimates_sharpe_or_alpha is False
    assert ext.reports_tstatistic is False
    assert ext.extraction_ok is False
    assert ext.feature_count() == 0


def test_extraction_feature_count():
    ext = LLMFeatureExtraction(
        estimates_sharpe_or_alpha=True,
        reports_tstatistic=True,
        specifies_long_short=True,
    )
    assert ext.feature_count() == 3


def test_extraction_to_dict_round_trip():
    ext = LLMFeatureExtraction(
        estimates_sharpe_or_alpha=True,
        specifies_universe=True,
        extraction_ok=True,
        cost_usd=0.0012,
    )
    d = ext.to_dict()
    assert d["estimates_sharpe_or_alpha"] is True
    assert d["specifies_universe"] is True
    assert d["extraction_ok"] is True
    assert d["cost_usd"] == 0.0012


# ── extract_boolean_features (mocked LLM) ────────────────────────────────

def test_extract_returns_empty_when_no_text():
    """Empty title + abstract → empty extraction (no LLM call)."""
    ext = extract_boolean_features("", "")
    assert ext.extraction_ok is False
    assert ext.cost_usd == 0.0


def test_extract_returns_empty_when_no_key(monkeypatch):
    """No API key → empty extraction (graceful fallback)."""
    monkeypatch.setattr(
        "engine.research.discovery.llm_feature_extractor._read_anthropic_key",
        lambda: None,
    )
    ext = extract_boolean_features("Carry paper", "Some abstract")
    assert ext.extraction_ok is False


def test_extract_happy_path_mocked():
    """Mock Anthropic SDK to return a valid JSON response."""
    mock_anthropic = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.usage.input_tokens = 200
    mock_response.usage.output_tokens = 30
    mock_text_block = mock.MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = (
        '{"estimates_sharpe_or_alpha": true,'
        ' "reports_tstatistic": true,'
        ' "specifies_long_short": true,'
        ' "specifies_holding_period": true,'
        ' "specifies_universe": true,'
        ' "specifies_sample_window": false,'
        ' "proposes_tradable_mechanism": true}'
    )
    mock_response.content = [mock_text_block]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.Anthropic.return_value = mock_client

    with mock.patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        with mock.patch(
            "engine.research.discovery.llm_feature_extractor._read_anthropic_key",
            return_value="sk-ant-fake",
        ):
            ext = extract_boolean_features(
                "Carry Across Asset Classes",
                "Long-short carry portfolio Sharpe 1.5 monthly rebal NYSE 1990-2012",
            )
    assert ext.extraction_ok is True
    assert ext.estimates_sharpe_or_alpha is True
    assert ext.reports_tstatistic is True
    assert ext.specifies_long_short is True
    assert ext.specifies_sample_window is False
    assert ext.feature_count() == 6
    assert ext.cost_usd > 0


def test_extract_malformed_json_response():
    """LLM returns non-JSON → extraction_ok=False but doesn't crash."""
    mock_anthropic = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 10
    mock_text_block = mock.MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "I cannot extract features for this paper."
    mock_response.content = [mock_text_block]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.Anthropic.return_value = mock_client

    with mock.patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        with mock.patch(
            "engine.research.discovery.llm_feature_extractor._read_anthropic_key",
            return_value="sk-fake",
        ):
            ext = extract_boolean_features("X", "Y")
    assert ext.extraction_ok is False
    assert ext.cost_usd > 0     # cost was incurred even on failure


def test_extract_anthropic_exception(monkeypatch):
    """Anthropic SDK raises → empty extraction, no crash."""
    mock_anthropic = mock.MagicMock()
    mock_client = mock.MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("timeout")
    mock_anthropic.Anthropic.return_value = mock_client

    with mock.patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        with mock.patch(
            "engine.research.discovery.llm_feature_extractor._read_anthropic_key",
            return_value="sk-fake",
        ):
            ext = extract_boolean_features("X", "Y")
    assert ext.extraction_ok is False


# ── Hybrid confidence ────────────────────────────────────────────────────

def test_hybrid_no_llm_returns_base_only():
    """enable_llm=False, llm_features=None → returns regex base only."""
    result = compute_hybrid_confidence(
        "Carry paper",
        "Long-short Sharpe 1.5 monthly 1990-2020 CRSP",
        family_guess="carry",
    )
    assert result["base_confidence"] > 0
    assert result["hybrid_confidence"] == result["base_confidence"]
    assert result["rescued_features"] == []
    assert result["llm_extraction_ok"] is False


def test_hybrid_llm_rescues_missing_features():
    """LLM identifies markers regex missed → hybrid > base."""
    # A paper where regex misses Sharpe (e.g. "outperforms 1.5x risk-
    # adjusted") but LLM correctly flags estimates_sharpe_or_alpha=True
    llm = LLMFeatureExtraction(
        estimates_sharpe_or_alpha=True,
        reports_tstatistic=True,
        specifies_long_short=True,
        extraction_ok=True,
        cost_usd=0.0008,
    )
    result = compute_hybrid_confidence(
        "Some paper",
        "We document outperformance on risk-adjusted basis through "
        "carry strategies. Statistical significance above 3 sigma.",
        llm_features=llm,
    )
    assert result["hybrid_confidence"] > result["base_confidence"]
    assert len(result["rescued_features"]) > 0
    assert result["llm_extraction_ok"] is True


def test_hybrid_does_not_double_credit_already_caught_features():
    """If regex AND LLM both find Sharpe, only count weight once."""
    # Abstract clearly mentions "Sharpe ratio of 1.5" so regex catches it
    llm = LLMFeatureExtraction(
        estimates_sharpe_or_alpha=True,    # LLM agrees
        extraction_ok=True,
    )
    base_result = compute_hybrid_confidence(
        "x", "Sharpe ratio of 1.5 from monthly long-short portfolio.",
        family_guess="unknown",
    )
    hybrid_result = compute_hybrid_confidence(
        "x", "Sharpe ratio of 1.5 from monthly long-short portfolio.",
        family_guess="unknown",
        llm_features=llm,
    )
    # Since regex already caught Sharpe, no rescue for that feature
    rescued_names = [r["llm_feature"] for r in hybrid_result["rescued_features"]]
    assert "estimates_sharpe_or_alpha" not in rescued_names


def test_hybrid_clipped_to_unit_interval():
    """Even if LLM credits all 6 features, hybrid ≤ 1.0."""
    llm = LLMFeatureExtraction(
        estimates_sharpe_or_alpha=True,
        reports_tstatistic=True,
        specifies_long_short=True,
        specifies_holding_period=True,
        specifies_universe=True,
        specifies_sample_window=True,
        proposes_tradable_mechanism=True,
        extraction_ok=True,
    )
    result = compute_hybrid_confidence(
        "Strong paper",
        "Sharpe 1.5 t=4 long-short monthly CRSP 1990-2020.",
        required_data_tokens=["crsp_dsf"],
        family_guess="carry",
        llm_features=llm,
    )
    assert result["hybrid_confidence"] <= 1.0


def test_hybrid_llm_failed_extraction_no_bonus():
    """When llm.extraction_ok=False, no bonus applied."""
    llm = LLMFeatureExtraction(
        estimates_sharpe_or_alpha=True,    # all flags ignored
        reports_tstatistic=True,
        extraction_ok=False,
    )
    result = compute_hybrid_confidence(
        "X", "Some text",
        llm_features=llm,
    )
    assert result["hybrid_confidence"] == result["base_confidence"]
    assert result["llm_extraction_ok"] is False


def test_hybrid_rescue_records_weight():
    """rescued_features carries the weight of each rescued feature
    so caller can audit."""
    llm = LLMFeatureExtraction(
        specifies_universe=True,   # regex misses if no CRSP/Bloomberg keyword
        extraction_ok=True,
    )
    result = compute_hybrid_confidence(
        "X", "We use top-tier exchange-listed equities for our study.",
        llm_features=llm,
    )
    if result["rescued_features"]:
        for r in result["rescued_features"]:
            assert "llm_feature" in r
            assert "regex_feature" in r
            assert "weight" in r
            assert r["weight"] > 0
