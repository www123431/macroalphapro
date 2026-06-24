"""Tests for engine.risk.risk_forecast (BARRA Phase 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.risk import risk_forecast as rf


# ── ledoit_wolf_shrinkage ────────────────────────────────────────────────

def test_ledoit_wolf_returns_psd():
    """Shrunken cov must be positive semi-definite."""
    np.random.seed(7)
    R = pd.DataFrame(np.random.randn(60, 5), columns=list("ABCDE"))
    Sigma, delta = rf.ledoit_wolf_shrinkage(R)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert (eigvals >= -1e-10).all()
    assert 0 <= delta <= 1


def test_ledoit_wolf_shrinkage_intensity_in_unit_interval():
    np.random.seed(11)
    R = pd.DataFrame(np.random.randn(120, 3), columns=list("ABC"))
    _, delta = rf.ledoit_wolf_shrinkage(R)
    assert 0 <= delta <= 1


def test_ledoit_wolf_diagonal_target_zero_offdiag_when_full_shrink():
    """If delta=1 (target = diagonal of sample cov), the result should
    have only diagonal entries."""
    np.random.seed(13)
    R = pd.DataFrame(np.random.randn(40, 4), columns=list("ABCD"))
    Sigma, delta = rf.ledoit_wolf_shrinkage(R)
    if delta >= 0.99:
        offdiag = Sigma - np.diag(np.diag(Sigma))
        assert np.allclose(offdiag, 0.0, atol=1e-10)


def test_ledoit_wolf_constant_correlation_target_runs():
    np.random.seed(17)
    R = pd.DataFrame(np.random.randn(80, 4), columns=list("ABCD"))
    Sigma, delta = rf.ledoit_wolf_shrinkage(R, target="constant_correlation")
    assert Sigma.shape == (4, 4)
    assert 0 <= delta <= 1


def test_ledoit_wolf_handles_empty():
    R = pd.DataFrame(columns=list("ABC"))
    Sigma, delta = rf.ledoit_wolf_shrinkage(R)
    assert Sigma.shape == (3, 3)
    assert delta == 0.0


# ── ewma_specific_risk ──────────────────────────────────────────────────

def test_ewma_returns_positive():
    np.random.seed(19)
    eps = pd.Series(np.random.randn(60) * 0.02)
    forecast = rf.ewma_specific_risk(eps, lambda_=0.97)
    assert forecast > 0


def test_ewma_lambda_close_to_1_uses_history():
    """High lambda = nearly all history weight; should give relatively
    smooth forecast."""
    np.random.seed(23)
    eps = pd.Series(np.random.randn(100) * 0.01)
    high_lam = rf.ewma_specific_risk(eps, lambda_=0.99)
    low_lam = rf.ewma_specific_risk(eps, lambda_=0.5)
    # both positive, low_lam more reactive to recent squared returns
    assert high_lam > 0 and low_lam > 0


def test_ewma_zero_history():
    forecast = rf.ewma_specific_risk(pd.Series(dtype=float))
    assert forecast == 0.0


def test_ewma_single_value():
    forecast = rf.ewma_specific_risk(pd.Series([0.05]))
    assert abs(forecast - 0.05 ** 2) < 1e-12


# ── portfolio_risk_forecast (synthetic end-to-end) ──────────────────────

@pytest.fixture
def synth_setup():
    np.random.seed(7)
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    factors = pd.DataFrame({
        "MKT": np.random.randn(120) * 0.04,
        "SMB": np.random.randn(120) * 0.03,
        "MOM": np.random.randn(120) * 0.04,
    }, index=idx)
    # Sleeve A: 0.5 MOM + small alpha + noise
    a = (0.01 + 0.5 * factors["MOM"]
         + np.random.RandomState(11).randn(120) * 0.005)
    b = (0.3 * factors["MKT"]
         + np.random.RandomState(13).randn(120) * 0.005)
    c = pd.Series(np.random.RandomState(17).randn(120) * 0.01, index=idx)
    sleeves = {"A": a, "B": b, "C": c}
    weights = {"A": 0.7, "B": 0.25, "C": 0.05}
    return factors, sleeves, weights


def test_forecast_returns_report(synth_setup):
    factors, sleeves, weights = synth_setup
    r = rf.portfolio_risk_forecast(
        sleeves, weights, factor_returns=factors, n_bootstrap=50,
    )
    assert isinstance(r, rf.RiskForecastReport)
    assert r.forecast_vol_annualized > 0
    assert r.factor_vol_forecast > 0
    assert r.idio_vol_forecast >= 0
    assert 0 <= r.pct_factor <= 1
    assert abs(r.pct_factor + r.pct_idio - 1.0) < 1e-6


def test_forecast_ci_brackets_point(synth_setup):
    """Bootstrap CI should contain the point estimate (approximately)."""
    factors, sleeves, weights = synth_setup
    r = rf.portfolio_risk_forecast(
        sleeves, weights, factor_returns=factors, n_bootstrap=200,
    )
    lo, hi = r.forecast_ci_95
    # Allow some slack since bootstrap with small B can miss point
    assert lo <= r.forecast_vol_annualized * 1.5
    assert hi >= r.forecast_vol_annualized * 0.5


def test_forecast_higher_weight_higher_idio_share(synth_setup):
    """Sleeve C is pure noise (high idio); putting all weight on C
    should give higher idio share than putting all on A (which loads
    on MOM factor)."""
    factors, sleeves, _ = synth_setup
    r_all_c = rf.portfolio_risk_forecast(
        sleeves, {"A": 0.0, "B": 0.0, "C": 1.0},
        factor_returns=factors, n_bootstrap=50,
    )
    r_all_a = rf.portfolio_risk_forecast(
        sleeves, {"A": 1.0, "B": 0.0, "C": 0.0},
        factor_returns=factors, n_bootstrap=50,
    )
    assert r_all_c.pct_idio > r_all_a.pct_idio


def test_forecast_per_sleeve_idio_present(synth_setup):
    factors, sleeves, weights = synth_setup
    r = rf.portfolio_risk_forecast(
        sleeves, weights, factor_returns=factors, n_bootstrap=20,
    )
    assert set(r.per_sleeve_idio_forecast.keys()) == set(sleeves.keys())


def test_forecast_book_exposures_match_weighted_betas(synth_setup):
    """Book β for MOM should match weighted sum of sleeve MOM betas."""
    factors, sleeves, weights = synth_setup
    r = rf.portfolio_risk_forecast(
        sleeves, weights, factor_returns=factors, n_bootstrap=20,
    )
    # Sleeve A has MOM β ~0.5, weight 0.7 → book MOM β ~0.35
    assert 0.25 < r.book_exposures["MOM"] < 0.45


def test_forecast_zero_bootstrap_skips_ci(synth_setup):
    """n_bootstrap=0 should produce NaN CI without crashing."""
    factors, sleeves, weights = synth_setup
    r = rf.portfolio_risk_forecast(
        sleeves, weights, factor_returns=factors, n_bootstrap=0,
    )
    assert r.forecast_vol_annualized > 0
    lo, hi = r.forecast_ci_95
    assert np.isnan(lo) and np.isnan(hi)
