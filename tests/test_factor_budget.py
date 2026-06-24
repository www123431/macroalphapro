"""Tests for engine.risk.factor_budget."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.risk import factor_budget as fb


@pytest.fixture
def synthetic_setup():
    """Build (factors, 3 sleeves, weights) for testing."""
    np.random.seed(7)
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    factors = pd.DataFrame({
        "MKT": np.random.randn(120) * 0.04,
        "SMB": np.random.randn(120) * 0.03,
        "MOM": np.random.randn(120) * 0.04,
    }, index=idx)
    # Sleeve A: 0.5 MOM + alpha + noise
    np.random.seed(11)
    a = (0.01 + 0.5 * factors["MOM"]
         + np.random.randn(120) * 0.005)
    # Sleeve B: 0.3 MKT + noise (no alpha)
    b = (0.3 * factors["MKT"]
         + np.random.RandomState(13).randn(120) * 0.005)
    # Sleeve C: pure noise alpha
    c = pd.Series(np.random.RandomState(17).randn(120) * 0.01,
                       index=idx)
    sleeves = {"A": a, "B": b, "C": c}
    weights = {"A": 0.7, "B": 0.25, "C": 0.05}
    return factors, sleeves, weights


# -- compute_factor_budget ----------------------------------------------

def test_factor_budget_returns_report(synthetic_setup):
    factors, sleeves, weights = synthetic_setup
    r = fb.compute_factor_budget(sleeves, weights, factors=factors)
    assert isinstance(r, fb.FactorBudgetReport)
    assert r.book_vol_annualized > 0
    assert 0 <= r.pct_factor <= 1
    assert 0 <= r.pct_idio <= 1
    assert abs((r.pct_factor + r.pct_idio) - 1.0) < 1e-6


def test_factor_budget_top5_present(synthetic_setup):
    factors, sleeves, weights = synthetic_setup
    r = fb.compute_factor_budget(sleeves, weights, factors=factors)
    assert len(r.top_5_factors_by_risk) <= 5
    # Order is by |pct|
    pcts = [abs(p) for _, p in r.top_5_factors_by_risk]
    assert pcts == sorted(pcts, reverse=True)


def test_factor_budget_mom_dominates_when_a_loads_strong(synthetic_setup):
    """Sleeve A loads heavily on MOM at 70% weight; MOM should be top
    factor by contribution."""
    factors, sleeves, weights = synthetic_setup
    r = fb.compute_factor_budget(sleeves, weights, factors=factors)
    top_name, _ = r.top_5_factors_by_risk[0]
    assert top_name == "MOM"


def test_factor_budget_idio_attributed_to_alpha_sleeves(synthetic_setup):
    """Sleeve A has constant alpha; A should have larger idio contribution
    than B (which is pure factor)."""
    factors, sleeves, weights = synthetic_setup
    r = fb.compute_factor_budget(sleeves, weights, factors=factors)
    pct_a = r.sleeve_idio_contrib_pct["A"]
    pct_b = r.sleeve_idio_contrib_pct["B"]
    assert pct_a > pct_b


def test_factor_budget_weights_warn_if_not_summing(synthetic_setup, caplog):
    factors, sleeves, _ = synthetic_setup
    weights_bad = {"A": 0.5, "B": 0.25, "C": 0.05}    # sums to 0.8
    fb.compute_factor_budget(sleeves, weights_bad, factors=factors)
    assert any("sleeve weights sum" in rec.message for rec in caplog.records)


def test_factor_budget_excludes_zero_weight_sleeves(synthetic_setup):
    factors, sleeves, _ = synthetic_setup
    weights = {"A": 1.0, "B": 0.0, "C": 0.0}
    r = fb.compute_factor_budget(sleeves, weights, factors=factors)
    # B + C should contribute ~0 to factor exposures
    # Sleeve A's MOM beta ~0.5 → book MOM exposure ~0.5
    assert 0.4 < r.factor_exposures["MOM"] < 0.6


# -- factor_orthogonality_score ----------------------------------------

def test_orthogonality_aligned_candidate_has_positive_cosine(synthetic_setup):
    factors, sleeves, weights = synthetic_setup
    book_r = fb.compute_factor_budget(sleeves, weights, factors=factors)
    aligned = {"MKT": 0.0, "SMB": 0.0, "MOM": 0.5}  # same direction as book MOM
    s = fb.factor_orthogonality_score(aligned, book_r)
    assert s["cosine_to_book_risk"] > 0.5
    assert s["risk_diversifying_score"] < 0


def test_orthogonality_opposite_candidate_has_negative_cosine(synthetic_setup):
    factors, sleeves, weights = synthetic_setup
    book_r = fb.compute_factor_budget(sleeves, weights, factors=factors)
    opposite = {"MKT": 0.0, "SMB": 0.0, "MOM": -0.5}  # anti-MOM
    s = fb.factor_orthogonality_score(opposite, book_r)
    assert s["cosine_to_book_risk"] < -0.5
    assert s["risk_diversifying_score"] > 0.5


def test_orthogonality_diversifier_lists_correct(synthetic_setup):
    factors, sleeves, weights = synthetic_setup
    book_r = fb.compute_factor_budget(sleeves, weights, factors=factors)
    candidate = {"MKT": -0.3, "SMB": +0.2, "MOM": -0.4}
    s = fb.factor_orthogonality_score(candidate, book_r)
    # MOM is the strongest book exposure → candidate's -0.4 MOM should
    # show in diversifiers (sign opposite to book +0.371)
    div_names = [n for n, _ in s["candidate_top_3_diversifiers"]]
    assert "MOM" in div_names


def test_orthogonality_empty_candidate():
    book_r = fb.FactorBudgetReport(
        book_vol_annualized=0.1, factor_vol_annualized=0.08,
        idio_vol_annualized=0.06, pct_factor=0.6, pct_idio=0.4,
        factor_exposures={"MKT": 0.1, "MOM": 0.3},
        factor_var_contrib_pct={}, sleeve_idio_contrib_pct={},
        top_5_factors_by_risk=[], n_months_used=100,
    )
    s = fb.factor_orthogonality_score({}, book_r)
    assert "error" in s


def test_orthogonality_zero_candidate(synthetic_setup):
    factors, sleeves, weights = synthetic_setup
    book_r = fb.compute_factor_budget(sleeves, weights, factors=factors)
    zero = {c: 0.0 for c in book_r.factor_exposures}
    s = fb.factor_orthogonality_score(zero, book_r)
    assert s["cosine_to_book_risk"] == 0.0
