"""Tests for engine.risk.barra_lite (BARRA Phase 1 — MKT/SMB/MOM)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.risk import barra_lite as bl


# -- _hac_se --------------------------------------------------------------

def test_hac_se_iid_matches_ols():
    """For uncorrelated residuals, HAC ~ OLS heteroscedasticity-robust SE."""
    np.random.seed(7)
    n = 200
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    beta_true = np.array([1.0, 2.0])
    eps = np.random.randn(n) * 0.5
    y = X @ beta_true + eps
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat
    se_hac = bl._hac_se(X, resid, lags=4)
    # Sanity: SEs positive and finite
    assert np.all(np.isfinite(se_hac))
    assert np.all(se_hac > 0)
    # Beta intercept SE roughly sigma / sqrt(n)
    assert abs(se_hac[0] - 0.5 / np.sqrt(n)) < 0.1


# -- regress_sleeve_on_factors -------------------------------------------

@pytest.fixture
def synthetic_factors():
    np.random.seed(11)
    idx = pd.date_range("2015-01-31", periods=72, freq="ME")
    return pd.DataFrame({
        "MKT": np.random.randn(72) * 0.04,
        "SMB": np.random.randn(72) * 0.03,
        "MOM": np.random.randn(72) * 0.04,
    }, index=idx)


def test_regress_pure_alpha(synthetic_factors):
    """A sleeve with constant 1%/mo alpha and no factor exposure -> alpha_m ~ 0.01."""
    np.random.seed(13)
    sleeve = pd.Series(
        0.01 + np.random.randn(72) * 0.005,
        index=synthetic_factors.index,
        name="pure_alpha",
    )
    r = bl.regress_sleeve_on_factors(sleeve, synthetic_factors, sleeve_name="pure")
    assert abs(r.alpha_monthly - 0.01) < 0.003
    # All factor betas should be near zero
    for k in ["MKT", "SMB", "MOM"]:
        assert abs(r.betas[k]) < 0.3


def test_regress_picks_up_known_beta(synthetic_factors):
    """A sleeve constructed as 0.5*MKT + 0.3*MOM should recover those betas."""
    sleeve = (0.5 * synthetic_factors["MKT"]
                + 0.3 * synthetic_factors["MOM"]
                + np.random.RandomState(17).randn(72) * 0.005)
    sleeve.name = "constructed"
    r = bl.regress_sleeve_on_factors(sleeve, synthetic_factors, sleeve_name="cs")
    assert abs(r.betas["MKT"] - 0.5) < 0.1
    assert abs(r.betas["MOM"] - 0.3) < 0.1
    assert abs(r.betas["SMB"]) < 0.15
    assert abs(r.alpha_monthly) < 0.003


def test_regress_short_sample_raises():
    factors = pd.DataFrame({
        "MKT": [0.01, 0.02], "SMB": [0.0, 0.01], "MOM": [0.0, 0.01],
    }, index=pd.date_range("2024-01-31", periods=2, freq="ME"))
    sleeve = pd.Series([0.01, 0.02], index=factors.index)
    with pytest.raises(ValueError, match="too few"):
        bl.regress_sleeve_on_factors(sleeve, factors)


def test_report_to_dict(synthetic_factors):
    sleeve = pd.Series(np.random.randn(72) * 0.02,
                          index=synthetic_factors.index)
    r = bl.regress_sleeve_on_factors(sleeve, synthetic_factors, sleeve_name="x")
    d = r.to_dict()
    assert "alpha_monthly" in d
    assert "betas" in d
    assert "t_stats_hac" in d
    assert "r_squared" in d
    assert "verdict" in d


# -- Factor construction smoke (requires data cache) ---------------------

def test_build_mkt_factor_runs():
    """Smoke test: MKT factor from cached vwretd should produce a Series."""
    if not bl.VWRETD_PATH.exists():
        pytest.skip("CRSP vwretd cache not present")
    mkt = bl.build_mkt_factor()
    assert isinstance(mkt, pd.Series)
    assert len(mkt) > 24
    # MKT is monthly compounded; values should be in plausible monthly range
    assert mkt.std() < 0.20
    assert abs(mkt.mean()) < 0.05


def test_verdict_picks_up_significant_mom():
    """A sleeve with strong MOM loading should produce a verdict flagging it."""
    idx = pd.date_range("2015-01-31", periods=72, freq="ME")
    np.random.seed(31)
    mom = np.random.randn(72) * 0.04
    factors = pd.DataFrame({
        "MKT": np.random.randn(72) * 0.04,
        "SMB": np.random.randn(72) * 0.03,
        "MOM": mom,
    }, index=idx)
    sleeve = pd.Series(0.7 * mom + np.random.randn(72) * 0.005, index=idx)
    r = bl.regress_sleeve_on_factors(sleeve, factors, sleeve_name="momentum")
    assert "MOM" in r.verdict
    assert abs(r.t_stats_hac["MOM"]) >= 2.0
