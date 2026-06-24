"""burn-1b-followup tests for hypothesis_type classifier."""
from __future__ import annotations

import pytest

from engine.research_store.hypothesis.classifier import (
    classify_hypothesis_type,
    hypothesis_type_breakdown,
)


def _h(**kw) -> dict:
    """Build a hypothesis-row dict with sensible defaults."""
    d = {
        "hypothesis_id":   kw.get("hid", "h-test"),
        "claim":           kw.get("claim", ""),
        "tags":            kw.get("tags", []),
        "created_by":      kw.get("created_by", "engine.agents.hypothesis_extractor"),
        "extraction_method": kw.get("extraction_method", "llm_extract"),
        "synthesizes_event_ids": kw.get("synth_event_ids", []),
        "mechanism_family": kw.get("family", "PROFITABILITY"),
        "addresses_decay_in": kw.get("addresses_decay_in"),
    }
    return d


# ── Tag-based: SLEEVE_IMPROVEMENT ──────────────────────────────────


def test_doctrine_signal_tag_classifies_sleeve_improvement():
    h = _h(tags=["source:doctrine_signal", "pattern:family_red_cluster"],
            claim="Family PROFITABILITY shows N RED verdicts")
    assert classify_hypothesis_type(h) == "sleeve_improvement"


def test_sleeve_fix_proposer_creator_classifies_sleeve_improvement():
    h = _h(created_by="engine.agents.strengthener.sleeve_fix_proposer",
            claim="Modify gp_at sleeve")
    assert classify_hypothesis_type(h) == "sleeve_improvement"


def test_active_b_sleeve_scan_tag_classifies_sleeve_improvement():
    """Phase 1 (2026-06-11): sleeve_strengthen_scan-produced hypotheses
    were silently falling through to forward — must classify as enhance."""
    h = _h(tags=["source:active_b_sleeve_scan", "sleeve:gp_at_2025",
                  "improvement_kind:cost_robustness"],
            claim="Replace 13bp cost assumption with regime-conditional model",
            extraction_method="llm_synthesis")
    assert classify_hypothesis_type(h) == "sleeve_improvement"


def test_sleeve_strengthen_scan_creator_classifies_sleeve_improvement():
    h = _h(created_by="engine.agents.strengthener.sleeve_strengthen_scan",
            claim="Tighten gp_at quintile breakpoints")
    assert classify_hypothesis_type(h) == "sleeve_improvement"


def test_addresses_decay_in_classifies_sleeve_improvement():
    """addresses_decay_in non-null is the strongest enhance signal —
    only enhance proposers set this field."""
    h = _h(claim="Improve carry_g10_fx with rebalancing frequency change")
    h["addresses_decay_in"] = "carry_g10_fx"
    assert classify_hypothesis_type(h) == "sleeve_improvement"


# ── Claim-pattern: METHODOLOGY ─────────────────────────────────────


@pytest.mark.parametrize("claim", [
    "Given the large number of factors already tested, a new factor needs to clear a t-ratio threshold of at least 3.0",
    "Most claimed research findings on cross-sectional return factors are false positives because the conventional t-ratio is insufficient",
    "Multiple testing adjustment via the BHY procedure discovers more factors",
    "Applying the Bonferroni correction to the set of 316 published factors",
    "The minimum t-ratio for 5% significance after multiple-testing correction is X",
    "Publication bias in factor research inflates the false discovery rate",
])
def test_methodology_patterns_classified(claim):
    h = _h(claim=claim)
    assert classify_hypothesis_type(h) == "methodology"


# ── Claim-pattern: FACTOR_ANALYSIS ─────────────────────────────────


@pytest.mark.parametrize("claim", [
    "When profitability (RMW) and investment (CMA) factors are added to the Fama-French three-factor model, the value factor HML becomes redundant",
    "Gross profitability subsumes the predictive power of earnings-to-book equity",
    "The value premium loses its significance once we control for profitability",
    "Momentum is fully captured by the four-factor model of Carhart",
    "SMB is no longer significant once QMJ is added to the right-hand side",
    "Size effect is statistically redundant in the FF5 framework",
    "An alternative four-factor model using industry-adjusted variables",
    "Returns drop to zero once the investment factor is controlled",
    # B.1 microcap-critique style
    "Microcap stocks are the primary driver of anomaly profits in the literature",
    "Liquidity variables are especially susceptible to microcap-driven inflation",
    "95 out of 102 (93%) trading frictions variables are insignificant after microcap exclusion",
    "Anomaly profits disappear when small-cap-driven returns are excluded",
])
def test_factor_analysis_patterns_classified(claim):
    h = _h(claim=claim)
    assert classify_hypothesis_type(h) == "factor_analysis"


# ── Default: FACTOR_PROPOSAL ───────────────────────────────────────


@pytest.mark.parametrize("claim", [
    "Stocks with higher gross profit to assets ratio earn higher subsequent returns",
    "Time-series momentum strategy delivered positive returns in 8 of 10 worst drawdowns",
    "A 20% allocation to TSMOM in a 60/40 portfolio reduces maximum drawdown",
    "The cross-section of expected returns depends on the book-to-market ratio",
    "A long-short portfolio sorted on operating profitability has alpha t-stat of 2.95",
])
def test_factor_proposal_default(claim):
    h = _h(claim=claim)
    assert classify_hypothesis_type(h) == "factor_proposal"


# ── Edge cases ─────────────────────────────────────────────────────


def test_empty_claim_returns_unknown():
    h = _h(claim="")
    assert classify_hypothesis_type(h) == "unknown"


def test_llm_synthesis_without_event_ids_returns_unknown():
    # Synthesis with no claim pattern + no provenance — conservative
    h = _h(claim="A general claim with no matching patterns",
            extraction_method="llm_synthesis",
            synth_event_ids=[])
    assert classify_hypothesis_type(h) == "unknown"


def test_llm_synthesis_with_event_ids_falls_through_to_proposal():
    # Synthesis with event provenance and no pattern → treat as proposal
    h = _h(claim="Stocks with X earn Y excess return",
            extraction_method="llm_synthesis",
            synth_event_ids=["abc-123"])
    assert classify_hypothesis_type(h) == "factor_proposal"


# ── Breakdown utility ──────────────────────────────────────────────


def test_breakdown_counts():
    rows = [
        _h(claim="Stocks with X earn higher returns"),                     # proposal
        _h(claim="HML becomes redundant under FF5"),                       # analysis
        _h(claim="Minimum t-ratio threshold should be 3.0"),               # methodology
        _h(tags=["source:doctrine_signal"], claim="..."),                  # sleeve_improvement
        _h(claim=""),                                                       # unknown
    ]
    out = hypothesis_type_breakdown(rows)
    assert out == {
        "factor_analysis":    1,
        "factor_proposal":    1,
        "methodology":        1,
        "sleeve_improvement": 1,
        "unknown":            1,
    }
