"""Tests for engine.research.discovery.confidence_calculator.

Calibration tests verify that known factor-paper abstracts score high
and known non-factor abstracts (theory / survey / lab) score low.
"""
from __future__ import annotations

import pytest

from engine.research.discovery.confidence_calculator import (
    NEGATIVE_WEIGHTS, POSITIVE_WEIGHTS,
    compute_confidence, explain_confidence,
)


# ── Individual feature detection ──────────────────────────────────────────

def test_return_prediction_feature_fires():
    res = compute_confidence(
        "Momentum and Long-Short Returns",
        "We find that long-short decile portfolios predict returns.",
    )
    assert "return_prediction_claim" in res.positives_hit


def test_sharpe_ratio_feature_fires():
    res = compute_confidence(
        "Carry",
        "The strategy generates a Sharpe ratio of 1.5 over the sample.",
    )
    assert "sharpe_or_alpha_number" in res.positives_hit


def test_alpha_number_feature_fires():
    res = compute_confidence(
        "X",
        "Our strategy delivers an alpha of 8.5% per annum.",
    )
    assert "sharpe_or_alpha_number" in res.positives_hit


def test_tstat_feature_fires():
    res = compute_confidence(
        "X",
        "We document a t = 4.2 alpha after controls.",
    )
    assert "tstat_pattern" in res.positives_hit


def test_holding_period_feature_fires():
    res = compute_confidence(
        "X",
        "Monthly rebalancing of the long-short portfolio.",
    )
    assert "holding_period" in res.positives_hit


def test_universe_specifier_feature_fires():
    res = compute_confidence(
        "X",
        "Sample includes all NYSE and NASDAQ stocks from CRSP.",
    )
    assert "universe_specifier" in res.positives_hit


def test_sample_window_feature_fires():
    res = compute_confidence(
        "X",
        "Sample period: 1990-2020 monthly data.",
    )
    assert "sample_window" in res.positives_hit


def test_sample_window_under_5_years_does_not_fire():
    res = compute_confidence(
        "X",
        "Sample 2018-2020 only.",
    )
    assert "sample_window" not in res.positives_hit


def test_required_data_tokens_feature_fires():
    res = compute_confidence(
        "X", "abstract",
        required_data_tokens=["crsp_dsf", "compustat_funda"],
    )
    assert "required_data_extracted" in res.positives_hit


def test_family_recognized_feature_fires():
    res = compute_confidence(
        "X", "abstract",
        family_guess="momentum",
    )
    assert "family_recognized" in res.positives_hit


def test_family_unknown_does_not_fire():
    for unknown in (None, "", "unknown", "Unknown"):
        res = compute_confidence("X", "abstract", family_guess=unknown)
        assert "family_recognized" not in res.positives_hit


# ── Negative features ────────────────────────────────────────────────────

def test_pure_theory_feature_fires():
    res = compute_confidence(
        "Asset Pricing Theory",
        "We derive a general equilibrium asset pricing model with rational expectations.",
    )
    assert "pure_theory" in res.negatives_hit


def test_survey_feature_fires():
    res = compute_confidence(
        "Anomaly Survey",
        "This paper surveys the literature on cross-sectional asset pricing anomalies.",
    )
    assert "survey_or_review" in res.negatives_hit


def test_no_data_source_fires_when_absent():
    res = compute_confidence(
        "X",
        "Some abstract with no specific data mentioned.",
    )
    assert "no_data_source_mentioned" in res.negatives_hit


def test_data_source_neutralizes_no_data():
    res = compute_confidence(
        "X",
        "We use CRSP data from 1990-2020 monthly.",
    )
    assert "no_data_source_mentioned" not in res.negatives_hit


def test_behavioral_lab_feature_fires():
    res = compute_confidence(
        "Behavioral Finance Lab Study",
        "Subjects were recruited from undergraduate population for the lab experiment.",
    )
    assert "behavioral_lab" in res.negatives_hit


# ── Calibration: known factor papers score high ──────────────────────────

def test_carry_paper_scores_high():
    """Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere"-style."""
    title = "Carry Across Asset Classes"
    abstract = (
        "We document predictable returns from long-short carry strategies "
        "across currencies, commodities, bonds, and equities. The cross-"
        "sectional Sharpe ratio is 1.5 over 1990-2012 from monthly "
        "rebalanced portfolios using futures and Bloomberg currency data."
    )
    res = compute_confidence(title, abstract,
                                required_data_tokens=["fx_futures"],
                                family_guess="carry")
    assert res.confidence >= 0.55, f"got {res.confidence:.3f}: {res.positives_hit}"


def test_fama_french_factor_paper_scores_high():
    """FF 2015 5-factor model abstract."""
    title = "A Five-Factor Asset Pricing Model"
    abstract = (
        "A five-factor model that adds profitability and investment "
        "factors to the three-factor model captures cross-sectional "
        "returns. The model is tested on CRSP NYSE/AMEX/NASDAQ stocks "
        "from 1963-2013 with monthly rebalancing of long-short decile "
        "portfolios. Average t-statistic 5.2."
    )
    res = compute_confidence(title, abstract,
                                required_data_tokens=["crsp_dsf"],
                                family_guess="factor_model")
    assert res.confidence >= 0.65, f"got {res.confidence:.3f}: {res.positives_hit}"


def test_pead_paper_scores_high():
    """Bernard-Thomas 1989-style PEAD."""
    title = "Post-Earnings-Announcement Drift"
    abstract = (
        "We find post-earnings-announcement drift in CRSP stock prices "
        "consistent with underreaction to earnings news. Long-short "
        "decile portfolios formed on standardized unexpected earnings "
        "(SUE) generate alpha = 12% per annum with t = 5.4 over 1974-2004. "
        "Sample includes all NYSE, AMEX, and NASDAQ stocks."
    )
    res = compute_confidence(title, abstract,
                                required_data_tokens=["crsp_dsf", "ibes_summary"],
                                family_guess="pead")
    assert res.confidence >= 0.65


# ── Calibration: non-factor papers score low ────────────────────────────

def test_pure_theory_paper_scores_low():
    """Pricing kernel derivation paper."""
    title = "Asset Pricing in General Equilibrium with Heterogeneous Agents"
    abstract = (
        "We derive a general equilibrium asset pricing model with "
        "heterogeneous agents and incomplete markets. The stochastic "
        "discount factor reflects the wealth-weighted marginal utility "
        "of heterogeneous consumers. We prove existence and uniqueness "
        "of equilibrium under standard regularity conditions."
    )
    res = compute_confidence(title, abstract)
    assert res.confidence <= 0.30, f"got {res.confidence:.3f}"


def test_literature_survey_scores_low():
    title = "Anomalies in Asset Pricing: A Survey"
    abstract = (
        "This paper surveys the literature on cross-sectional asset "
        "pricing anomalies. We review over 100 published findings."
    )
    res = compute_confidence(title, abstract)
    assert res.confidence <= 0.30


def test_lab_experiment_scores_low():
    title = "Risk Preferences in the Lab"
    abstract = (
        "We ran an experiment where subjects completed risk-preference "
        "elicitation tasks. Participants completed 40 trials in the lab."
    )
    res = compute_confidence(title, abstract)
    assert res.confidence <= 0.30


# ── Range invariants ──────────────────────────────────────────────────────

def test_confidence_always_in_unit_interval():
    """Even with all positive features hitting, score should be ≤ 1.0."""
    title = "Momentum"
    abstract = (
        "Long-short decile portfolios from CRSP NYSE stocks predict "
        "returns. Sharpe ratio of 1.5 over 1990-2020. Monthly rebalanced. "
        "t-statistic of 6.0 on alpha."
    )
    res = compute_confidence(title, abstract,
                                required_data_tokens=["crsp_dsf"],
                                family_guess="momentum")
    assert 0.0 <= res.confidence <= 1.0
    assert 0.0 <= res.pos_score <= sum(POSITIVE_WEIGHTS.values()) + 0.001
    assert 0.0 <= res.neg_score <= sum(NEGATIVE_WEIGHTS.values()) + 0.001


def test_confidence_zero_for_empty():
    res = compute_confidence("", "")
    # Empty text → no positive features but ALL negative features fire
    # (no data source mentioned + others). Score = max(0, 0 - X) = 0.
    assert res.confidence == 0.0


def test_weights_sum_to_documented_totals():
    """POSITIVE_WEIGHTS sum = 0.90 (max pos_score); NEGATIVE_WEIGHTS = 0.85.
    confidence = max(0, pos - neg) saturates at 1.0 by clip in formula,
    but raw pos_score can't exceed 0.90."""
    assert sum(POSITIVE_WEIGHTS.values()) == pytest.approx(0.90, abs=0.001)
    assert sum(NEGATIVE_WEIGHTS.values()) == pytest.approx(0.85, abs=0.001)


# ── Auditability ─────────────────────────────────────────────────────────

def test_explain_confidence_returns_readable_text():
    out = explain_confidence(
        "Carry",
        "Sharpe ratio 1.5 from monthly rebal CRSP 1990-2020.",
        required_data_tokens=["crsp_dsf"],
        family_guess="carry",
    )
    assert "Title:" in out
    assert "CONFIDENCE:" in out
    assert "Positives" in out


def test_to_dict_audit_trail():
    res = compute_confidence("X", "monthly rebalanced CRSP",
                                  required_data_tokens=["crsp_dsf"],
                                  family_guess="momentum")
    d = res.to_dict()
    for k in ("confidence", "pos_score", "neg_score",
                "positives_hit", "negatives_hit", "feature_weights"):
        assert k in d


# ── Reproducibility ──────────────────────────────────────────────────────

def test_same_input_same_score():
    """STRICT: same inputs → same score (no LLM stochasticity)."""
    title = "Momentum"
    abstract = "Long-short CRSP decile 1990-2020 Sharpe 1.5."
    res1 = compute_confidence(title, abstract,
                                   required_data_tokens=["crsp_dsf"],
                                   family_guess="momentum")
    res2 = compute_confidence(title, abstract,
                                   required_data_tokens=["crsp_dsf"],
                                   family_guess="momentum")
    assert res1.confidence == res2.confidence
    assert res1.positives_hit == res2.positives_hit
    assert res1.negatives_hit == res2.negatives_hit
