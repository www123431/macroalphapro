"""Tests for engine.research.templates.factor_quartile."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research.templates.factor_quartile import run_factor_quartile
from engine.research.templates import TEMPLATES


@pytest.fixture
def synth_panels():
    rng = np.random.RandomState(2026)
    n_months, n_tickers = 60, 50
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rets = rng.randn(n_months, n_tickers) * 0.06
    prices = pd.DataFrame(
        np.cumprod(1 + rets, axis=0) * 100.0, index=dates, columns=tickers,
    )
    # Factor: rolling 12m volatility (lower vol = "high quality" in BAB sense)
    factor = (pd.DataFrame(rets, index=dates, columns=tickers)
              .rolling(12).std())
    return {"price_panel": prices, "factor_panel": factor}


def test_factor_quartile_registered():
    assert "factor_quartile" in TEMPLATES


def test_run_factor_quartile_returns_series(synth_panels):
    ls = run_factor_quartile(
        factor_panel=synth_panels["factor_panel"],
        price_panel=synth_panels["price_panel"],
        top_frac=0.2, bottom_frac=0.2, factor_sign=1,
        vol_target=0.10, vol_target_lookback=12,
    )
    assert isinstance(ls, pd.Series)
    # Must have at least some non-NaN months after warmup
    assert ls.dropna().shape[0] > 0


def test_factor_sign_flip_inverts(synth_panels):
    """factor_sign=+1 vs -1 should produce strongly ANTI-CORRELATED returns.

    Note: not exactly bit-equal negation because pandas pct rank tie-breaking
    produces rank(-x) + rank(x) ≠ 1.0 exactly (off by 1/n). This 1/n
    discrepancy at the quantile cutoff can shift one ticker between
    long/short sets across the sign flip. The semantic intent (sign flip
    flips L/S) is captured by correlation ≈ -1.
    """
    ls_pos = run_factor_quartile(
        factor_panel=synth_panels["factor_panel"],
        price_panel=synth_panels["price_panel"],
        top_frac=0.2, bottom_frac=0.2, factor_sign=1,
        vol_target=None, cost_bps_per_side=0,
    )
    ls_neg = run_factor_quartile(
        factor_panel=synth_panels["factor_panel"],
        price_panel=synth_panels["price_panel"],
        top_frac=0.2, bottom_frac=0.2, factor_sign=-1,
        vol_target=None, cost_bps_per_side=0,
    )
    overlap = ls_pos.dropna().index.intersection(ls_neg.dropna().index)
    assert len(overlap) > 5
    # Post-optimization (rank-once-swap-masks): should be EXACTLY negated
    diff = (ls_pos.loc[overlap] + ls_neg.loc[overlap]).abs().max()
    assert diff < 1e-10, f"sign-flip should bit-exactly negate; got max diff={diff:.2e}"


def test_invalid_factor_sign_raises(synth_panels):
    with pytest.raises(ValueError):
        run_factor_quartile(
            factor_panel=synth_panels["factor_panel"],
            price_panel=synth_panels["price_panel"],
            factor_sign=0,
        )


def test_invalid_weighting_raises(synth_panels):
    with pytest.raises(NotImplementedError):
        run_factor_quartile(
            factor_panel=synth_panels["factor_panel"],
            price_panel=synth_panels["price_panel"],
            weighting="value_weight",
        )


def test_anti_look_ahead_lag(synth_panels):
    """First period (no lag fillable) should be NaN."""
    ls = run_factor_quartile(
        factor_panel=synth_panels["factor_panel"],
        price_panel=synth_panels["price_panel"],
        vol_target=None,
    )
    # First period after factor window completes should be NaN due to lag
    # (factor has 11 NaN, then real values from t=11; lagged factor has NaN
    # at t=11; rank/membership at t=11 = NaN; long-short return at t=11 = NaN)
    assert pd.isna(ls.iloc[11]) or pd.isna(ls.iloc[12])


def test_dsl_runner_routes_to_factor_quartile(synth_panels):
    """End-to-end via DSL runner."""
    from engine.research.strategy_dsl_runner import run_proposal
    proposal = {
        "mechanism_id": "low_vol_bab",
        "execution_template": {
            "template_id": "factor_quartile",
            "binding": {
                "top_frac": 0.2, "bottom_frac": 0.2,
                "factor_sign": -1, "weighting": "equal_weight",
                "rebal_freq": "monthly", "cost_bps_per_side": 12.0,
                "microcap_price_threshold": 5.0,
                "vol_target": 0.10, "vol_target_lookback": 12,
            },
        },
    }
    ls = run_proposal(proposal,
                      factor_panel=synth_panels["factor_panel"],
                      price_panel=synth_panels["price_panel"])
    assert isinstance(ls, pd.Series)
    assert ls.dropna().shape[0] > 0
