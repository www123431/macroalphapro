"""Tests for engine.research.cost_model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research import cost_model as cm


@pytest.fixture
def simple_weights():
    """Two-period long-only weights, 3 assets. Both DataFrames share the
    SAME index (representing the trade-date) since the subtract operation
    needs alignment to produce |Δw|. In production, the runner would
    pre-shift the prior-period weights to match the current trade date
    before calling the cost model."""
    cols = ["A", "B", "C"]
    idx = pd.DatetimeIndex(["2024-02-29"])
    w_tminus1 = pd.DataFrame([[0.3, 0.3, 0.4]], index=idx, columns=cols)
    w_t = pd.DataFrame([[0.5, 0.2, 0.3]], index=idx, columns=cols)
    return w_t, w_tminus1


# ── simple_bps ────────────────────────────────────────────────────────────

def test_simple_bps_known_turnover():
    """100% turnover, 10 bps/side → 20 bps cost (round-trip)."""
    turnover = pd.Series([1.0, 1.0, 1.0])
    params = cm.CostModelParams(model="simple_bps", cost_bps_per_side=10.0)
    cost = cm.simple_bps_cost(turnover, params)
    assert all(c == 20.0 for c in cost)


def test_simple_bps_zero_turnover_zero_cost():
    cost = cm.simple_bps_cost(pd.Series([0.0, 0.0]),
                                  cm.CostModelParams())
    assert all(c == 0 for c in cost)


def test_simple_bps_partial_turnover():
    """50% turnover, 12 bps/side → 12 bps cost (0.5 × 2 × 12)."""
    cost = cm.simple_bps_cost(pd.Series([0.5]),
                                  cm.CostModelParams(cost_bps_per_side=12.0))
    assert cost.iloc[0] == 12.0


# ── linear_spread ─────────────────────────────────────────────────────────

def test_linear_spread_zero_change_zero_cost(simple_weights):
    w_t, w_tminus1 = simple_weights
    # If weights identical → no cost
    cost = cm.linear_spread_cost(
        w_tminus1, w_tminus1, cm.CostModelParams(spread_bps_per_side=2.5),
    )
    assert cost.iloc[0] == 0


def test_linear_spread_known_delta(simple_weights):
    """|Δw| sum = 0.2 + 0.1 + 0.1 = 0.4. Half-spread 2.5 → 0.4 × 2 × 2.5 = 2.0 bps."""
    w_t, w_tminus1 = simple_weights
    cost = cm.linear_spread_cost(
        w_t, w_tminus1, cm.CostModelParams(spread_bps_per_side=2.5),
    )
    # Cost is computed on w_t's index
    assert cost.iloc[0] == pytest.approx(2.0, abs=0.001)


# ── almgren_chriss ────────────────────────────────────────────────────────

def test_almgren_chriss_zero_change_zero_cost(simple_weights):
    w_t, w_tminus1 = simple_weights
    cost = cm.almgren_chriss_cost(
        w_tminus1, w_tminus1, None, None,
        cm.CostModelParams(spread_bps_per_side=2.5),
    )
    assert cost.iloc[0] == 0


def test_almgren_chriss_uses_defaults_when_panels_none(simple_weights):
    """When σ and ADV not provided → uses default mega-cap values."""
    w_t, w_tminus1 = simple_weights
    cost = cm.almgren_chriss_cost(
        w_t, w_tminus1, None, None,
        cm.CostModelParams(
            spread_bps_per_side=2.5, impact_coef=0.5,
            portfolio_aum=100_000_000,
            default_adv_dollars=200_000_000,
            default_daily_sigma=0.015,
        ),
    )
    # Spread component: 0.4 × 2 × 2.5 = 2.0 bps
    # Impact component should be a small positive number
    assert cost.iloc[0] > 2.0    # at least spread
    assert cost.iloc[0] < 50.0   # reasonable


def test_almgren_chriss_larger_aum_larger_impact_cost(simple_weights):
    """With same Δw, bigger AUM → bigger notional → bigger √ impact cost."""
    w_t, w_tminus1 = simple_weights
    p_small = cm.CostModelParams(
        portfolio_aum=10_000_000, default_adv_dollars=200_000_000,
        default_daily_sigma=0.015, spread_bps_per_side=2.5, impact_coef=0.5,
    )
    p_big = cm.CostModelParams(
        portfolio_aum=10_000_000_000,
        default_adv_dollars=200_000_000,
        default_daily_sigma=0.015, spread_bps_per_side=2.5, impact_coef=0.5,
    )
    cost_small = cm.almgren_chriss_cost(w_t, w_tminus1, None, None, p_small)
    cost_big = cm.almgren_chriss_cost(w_t, w_tminus1, None, None, p_big)
    assert cost_big.iloc[0] > cost_small.iloc[0]


def test_almgren_chriss_participation_capped(simple_weights):
    """When notional > ADV (participation rate hit 1), cost shouldn't blow up."""
    w_t, w_tminus1 = simple_weights
    # AUM >> ADV — should still produce finite cost
    p = cm.CostModelParams(
        portfolio_aum=10_000_000_000_000,   # absurd $10 trillion
        default_adv_dollars=1_000_000,       # tiny $1M ADV
        default_daily_sigma=0.015, spread_bps_per_side=2.5, impact_coef=0.5,
    )
    cost = cm.almgren_chriss_cost(w_t, w_tminus1, None, None, p)
    assert np.isfinite(cost.iloc[0])
    assert cost.iloc[0] > 0


# ── compute_cost_bps dispatch ────────────────────────────────────────────

def test_dispatch_simple_bps_needs_turnover():
    with pytest.raises(ValueError):
        cm.compute_cost_bps(model="simple_bps")


def test_dispatch_linear_spread_needs_weights():
    with pytest.raises(ValueError):
        cm.compute_cost_bps(model="linear_spread")


def test_dispatch_unknown_model():
    with pytest.raises(ValueError):
        cm.compute_cost_bps(model="not_a_real_model",  # type: ignore
                               turnover_series=pd.Series([1.0]))


def test_dispatch_simple_bps_via_weights(simple_weights):
    """simple_bps without explicit turnover_series → derives from weights."""
    w_t, w_tminus1 = simple_weights
    p = cm.CostModelParams(model="simple_bps", cost_bps_per_side=12.0)
    cost = cm.compute_cost_bps(
        model="simple_bps",
        weights_t=w_t, weights_tminus1=w_tminus1, params=p,
    )
    # |Δw| sum = 0.4 → 0.4 × 2 × 12 = 9.6 bps
    assert cost.iloc[0] == pytest.approx(9.6, abs=0.001)


# ── apply_cost_to_returns ────────────────────────────────────────────────

def test_apply_cost_subtracts_bps_correctly():
    gross = pd.Series([0.01, 0.02, 0.005],
                      index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    cost_bps = pd.Series([10.0, 20.0, 5.0], index=gross.index)
    net = cm.apply_cost_to_returns(gross, cost_bps)
    # 10 bps = 0.001 → 0.01 - 0.001 = 0.009
    assert net.iloc[0] == pytest.approx(0.009, abs=1e-6)
    assert net.iloc[1] == pytest.approx(0.018, abs=1e-6)
    assert net.iloc[2] == pytest.approx(0.0045, abs=1e-6)


def test_apply_cost_handles_missing_dates():
    """If cost series has fewer dates than gross, missing dates default to 0."""
    gross = pd.Series([0.01, 0.02],
                      index=pd.date_range("2024-01-31", periods=2, freq="ME"))
    cost_bps = pd.Series([10.0],
                          index=pd.date_range("2024-01-31", periods=1, freq="ME"))
    net = cm.apply_cost_to_returns(gross, cost_bps)
    # First period: cost 10 bps → 0.009
    # Second period: no cost → 0.02
    assert net.iloc[0] == pytest.approx(0.009, abs=1e-6)
    assert net.iloc[1] == pytest.approx(0.02, abs=1e-6)


# ── Helper functions ─────────────────────────────────────────────────────

def test_rolling_daily_sigma():
    """Confirm rolling vol computation is sensible."""
    rng = np.random.default_rng(42)
    ret_panel = pd.DataFrame(
        rng.standard_normal((100, 3)) * 0.02,    # ~2% daily vol
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
        columns=["A", "B", "C"],
    )
    sigma = cm.rolling_daily_sigma(ret_panel, window=60)
    # After 60 days, should be near 0.02
    mean_late_sigma = sigma.iloc[-1].mean()
    assert 0.015 < mean_late_sigma < 0.025


def test_adv_dollars_from_price_volume():
    prc = pd.DataFrame(
        [[100, 200, 300]] * 100,
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
        columns=["A", "B", "C"],
    )
    vol = pd.DataFrame(
        [[1_000_000, 500_000, 100_000]] * 100,
        index=prc.index, columns=prc.columns,
    )
    adv = cm.adv_dollars_from_price_volume(prc, vol, window=60)
    # Last row: A should be 100 × 1M = $100M
    assert adv.iloc[-1]["A"] == pytest.approx(100_000_000, abs=1.0)
    assert adv.iloc[-1]["B"] == pytest.approx(100_000_000, abs=1.0)
    assert adv.iloc[-1]["C"] == pytest.approx(30_000_000, abs=1.0)
