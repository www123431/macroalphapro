"""Tests for engine.agents.papers_curator.claim_type_router."""
from __future__ import annotations

import pytest

from engine.agents.papers_curator.claim_type_router import (
    classify,
    classify_summary,
)
from engine.hypothesis_spec.enums import ClaimType


def test_empty_input_returns_unknown():
    v = classify("", "")
    assert v.claim_type == ClaimType.UNKNOWN
    assert v.confidence == 0.0


def test_factor_hypothesis_clear_signal():
    v = classify(
        title    = "A new factor in the cross-section of returns",
        abstract = "We construct a long-short portfolio sorted on X. "
                   "The decile portfolio yields a 0.5% monthly alpha "
                   "against the Fama-French five-factor model.",
    )
    assert v.claim_type == ClaimType.FACTOR_HYPOTHESIS
    assert v.confidence > 0.5


def test_methodology_clear_signal():
    v = classify(
        title    = "Correcting for data-snooping bias in factor research",
        abstract = "We develop a framework for testing multiple hypotheses "
                   "under FDR control with block bootstrap.",
    )
    assert v.claim_type == ClaimType.METHODOLOGY


def test_decay_study_clear_signal():
    v = classify(
        title    = "Post-publication decay of anomaly returns",
        abstract = "We replicate 95 anomalies and document factor decay "
                   "out-of-sample after publication.",
    )
    assert v.claim_type == ClaimType.DECAY_STUDY


def test_capacity_clear_signal():
    v = classify(
        title    = "Capacity constraints of momentum strategies",
        abstract = "We use Almgren-Chriss market impact to estimate AUM "
                   "ceiling and break-even fund size for momentum.",
    )
    assert v.claim_type == ClaimType.CAPACITY


def test_microstructure_clear_signal():
    v = classify(
        title    = "Bid-ask spread dynamics in the limit-order book",
        abstract = "Effective spread and Amihud illiquidity over the trading day.",
    )
    assert v.claim_type == ClaimType.MICROSTRUCTURE


def test_factor_structure_clear_signal():
    v = classify(
        title    = "Is the value factor redundant under the FF5 model?",
        abstract = "Spanning regression evidence: HML is subsumed by RMW + CMA.",
    )
    assert v.claim_type == ClaimType.FACTOR_STRUCTURE


def test_domain_fact_clear_signal():
    v = classify(
        title    = "Customer-supplier links and stock returns",
        abstract = "We document that input-output linkages predict cross-firm "
                   "return correlations. Stylized fact, not a tradable signal.",
    )
    assert v.claim_type == ClaimType.DOMAIN_FACT


def test_confidence_sums_to_at_most_one():
    v = classify(
        title    = "Long-short portfolio of anomaly returns and data-snooping",
        abstract = "We propose a method to control for multiple testing in "
                   "factor research, then construct decile portfolios.",
    )
    # Has both FACTOR_HYPOTHESIS and METHODOLOGY hits → confidence < 1.0
    assert 0.0 < v.confidence < 1.0
    assert sum(v.scores.values()) > 0


def test_scores_dict_keyed_by_string():
    v = classify("anomaly returns", "long-short portfolio of decile sorted on X")
    assert "FACTOR_HYPOTHESIS" in v.scores
    assert v.scores["FACTOR_HYPOTHESIS"] >= 2


def test_classify_summary_convenience():
    # W6-rigor-A-router-v2 (2026-06-22): require multiple keyword hits.
    # This summary has 2 distinct FACTOR_HYPOTHESIS hits
    # ("long-short portfolio" + "we form portfolios").
    summary_row = {
        "thesis":    "Momentum profitability is conditional on market state.",
        "mechanism": "We form long-short portfolios in high-state months "
                     "with the cross-section of returns sorted on volatility.",
    }
    v = classify_summary(summary_row)
    assert v.claim_type == ClaimType.FACTOR_HYPOTHESIS


def test_hits_field_records_matched_keywords():
    # W6-rigor-A-router-v2 (2026-06-22): "anomaly" alone is no longer a
    # keyword (multi-word "anomaly returns" required). Test now uses 2
    # FACTOR_HYPOTHESIS hits that survive the min_total_score=2 floor.
    v = classify("decile portfolio of anomaly returns",
                  "we construct a long-short portfolio of high-minus-low ranked stocks")
    assert v.claim_type == ClaimType.FACTOR_HYPOTHESIS
    assert "long-short portfolio" in v.hits["FACTOR_HYPOTHESIS"]
    assert any("anomaly returns" in kw or "decile portfolio" in kw
                  or "high minus low" in kw or "high-minus-low" in kw
                  for kw in v.hits["FACTOR_HYPOTHESIS"])


def test_ambiguous_single_word_no_longer_triggers_w6_router_v2():
    """W6-rigor-A-router-v2: bare 'anomaly' / 'premia' no longer keywords."""
    v1 = classify(
        "Off the Labor Supply Curve: Wage Premia in Large Firms",
        "We document that employer size has zero effect on wage premia.",
    )
    v2 = classify(
        "From Data to Action: Accelerating Refinery Optimization with AI",
        "Our anomaly detection algorithm identifies process inefficiencies.",
    )
    v3 = classify(
        "A Contemporary Survey on GNSS Spoofing Attacks",
        "Anomaly-based intrusion detection for satellite navigation.",
    )
    # Principle: NONE of these should be tagged FACTOR_HYPOTHESIS by
    # the deterministic router (LLM fallback may further classify).
    # Each is allowed to be UNKNOWN, DOMAIN_FACT, METHODOLOGY, or
    # any non-FACTOR_HYPOTHESIS class — the FP we're fixing is the
    # mis-classification AS FACTOR_HYPOTHESIS specifically.
    for v, label in [(v1, "labor"), (v2, "refinery"), (v3, "GNSS")]:
        assert v.claim_type != ClaimType.FACTOR_HYPOTHESIS, (
            f"{label}: still mis-classified as FACTOR_HYPOTHESIS"
        )


# ── W4-piece-2 LLM fallback tests ───────────────────────────────


def test_classify_hybrid_skips_llm_when_det_confident():
    """If deterministic classifier returns non-UNKNOWN, LLM is NOT invoked."""
    from engine.agents.papers_curator.claim_type_router import classify_hybrid

    v = classify_hybrid(
        title    = "A new factor in the cross-section of returns",
        abstract = "We construct a long-short decile portfolio and document alpha.",
        llm_fallback = True,  # would invoke LLM if det returned UNKNOWN
    )
    # Pure-keyword path; hits has no llm_rationale
    assert v.claim_type == ClaimType.FACTOR_HYPOTHESIS
    assert "llm_rationale" not in v.hits


def test_classify_hybrid_no_fallback_returns_unknown_on_blank():
    """When det returns UNKNOWN + llm_fallback=False, UNKNOWN propagates."""
    from engine.agents.papers_curator.claim_type_router import classify_hybrid

    v = classify_hybrid(title="", abstract="", llm_fallback=False)
    assert v.claim_type == ClaimType.UNKNOWN
    assert v.confidence == 0.0


def test_classify_llm_swallows_failure_returns_unknown(monkeypatch):
    """If engine.llm.call.call raises, classify_llm returns UNKNOWN
    so the pipeline never blocks."""
    from engine.agents.papers_curator import claim_type_router as r

    def boom(**kwargs):
        raise RuntimeError("API down")

    # Patch the module's lazy import of engine.llm.call
    import sys
    _llm_mod = sys.modules["engine.llm.call"]
    monkeypatch.setattr(_llm_mod, "call", boom)

    v = r.classify_llm(
        title    = "Anything",
        abstract = "Anything",
        agent_id = "papers_curator_filter",
    )
    assert v.claim_type == ClaimType.UNKNOWN
    assert v.confidence == 0.0


def test_classify_llm_handles_invalid_label(monkeypatch):
    """If LLM returns a label not in ClaimType enum, fall back to UNKNOWN."""
    from engine.agents.papers_curator import claim_type_router as r

    class _FakeResult:
        text = '{"claim_type": "NOT_A_REAL_LABEL", "rationale": "x"}'

    import sys
    _llm_mod = sys.modules["engine.llm.call"]
    monkeypatch.setattr(_llm_mod, "call", lambda **kw: _FakeResult())

    v = r.classify_llm(
        title="x", abstract="y", agent_id="papers_curator_filter",
    )
    assert v.claim_type == ClaimType.UNKNOWN


def test_classify_llm_strips_code_fences(monkeypatch):
    """Tolerate ```json ... ``` fenced responses."""
    from engine.agents.papers_curator import claim_type_router as r

    class _FakeResult:
        text = '```json\n{"claim_type": "FACTOR_HYPOTHESIS", "rationale": "x"}\n```'

    import sys
    _llm_mod = sys.modules["engine.llm.call"]
    monkeypatch.setattr(_llm_mod, "call", lambda **kw: _FakeResult())

    v = r.classify_llm(
        title="x", abstract="y", agent_id="papers_curator_filter",
    )
    assert v.claim_type == ClaimType.FACTOR_HYPOTHESIS
    assert v.confidence == 1.0
