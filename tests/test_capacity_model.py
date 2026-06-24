"""Tests for engine.research.capacity_model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research import capacity_model as cap


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def equal_weights():
    """24-month, 10-asset equal-weight long portfolio."""
    cols = [f"T{i:02d}" for i in range(10)]
    idx = pd.date_range("2022-01-31", periods=24, freq="ME")
    w = pd.DataFrame(0.1, index=idx, columns=cols)
    return w


@pytest.fixture
def adv_panel_uniform():
    """Each asset has $200M ADV uniformly."""
    cols = [f"T{i:02d}" for i in range(10)]
    idx = pd.date_range("2022-01-31", periods=24, freq="ME")
    return pd.DataFrame(200_000_000.0, index=idx, columns=cols)


# ── hard_capacity_usd ────────────────────────────────────────────────────

def test_hard_capacity_empty_weights():
    cap_usd, *rest = cap.hard_capacity_usd(pd.DataFrame(), None)
    assert cap_usd == float("inf")


def test_hard_capacity_known_math(equal_weights, adv_panel_uniform):
    """10 names × 10% weight × $200M ADV @ 5% participation:
    capacity_per_pos = $200M × 0.05 / 0.10 = $100M."""
    cap_usd, asset, adv_b, w_b = cap.hard_capacity_usd(
        equal_weights, adv_panel_uniform, max_participation=0.05,
    )
    assert cap_usd == pytest.approx(100_000_000.0, abs=1.0)
    assert asset is not None
    assert adv_b == 200_000_000.0
    assert w_b == pytest.approx(0.1, abs=0.001)


def test_hard_capacity_lower_participation_lower_capacity(equal_weights, adv_panel_uniform):
    cap_5pct, *_ = cap.hard_capacity_usd(equal_weights, adv_panel_uniform,
                                              max_participation=0.05)
    cap_1pct, *_ = cap.hard_capacity_usd(equal_weights, adv_panel_uniform,
                                              max_participation=0.01)
    assert cap_1pct < cap_5pct


def test_hard_capacity_no_adv_uses_default():
    cols = ["A", "B"]
    idx = pd.date_range("2024-01-31", periods=2, freq="ME")
    w = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=idx, columns=cols)
    cap_usd, *_ = cap.hard_capacity_usd(w, None,
                                              max_participation=0.05,
                                              default_adv_usd=100_000_000)
    # 0.5 weight @ $100M ADV @ 5% participation = $100M × 0.05 / 0.5 = $10M
    assert cap_usd == pytest.approx(10_000_000.0, abs=1.0)


def test_hard_capacity_concentrated_in_illiquid(equal_weights):
    """If one asset has much smaller ADV, it becomes the binding constraint."""
    cols = equal_weights.columns
    idx = equal_weights.index
    adv = pd.DataFrame(200_000_000.0, index=idx, columns=cols)
    adv["T00"] = 10_000_000.0    # illiquid name
    cap_usd, asset, *_ = cap.hard_capacity_usd(equal_weights, adv,
                                                    max_participation=0.05)
    # Binding asset = T00 = $10M × 0.05 / 0.1 = $5M
    assert asset == "T00"
    assert cap_usd == pytest.approx(5_000_000.0, abs=1.0)


# ── estimate_half_life_aum ───────────────────────────────────────────────

def test_half_life_zero_gross_returns_none():
    out = cap.estimate_half_life_aum(
        gross_sharpe=0.0, monthly_turnover_usd_at_test_aum=1_000_000,
        test_aum_usd=10_000_000, universe_adv_usd=100_000_000,
    )
    assert out is None


def test_half_life_zero_turnover_returns_none():
    out = cap.estimate_half_life_aum(
        gross_sharpe=1.0, monthly_turnover_usd_at_test_aum=0,
        test_aum_usd=10_000_000, universe_adv_usd=100_000_000,
    )
    assert out is None


def test_half_life_scales_with_aum():
    """At higher gross Sharpe, the half-life AUM should be higher (more
    headroom)."""
    base = cap.estimate_half_life_aum(
        gross_sharpe=0.5, monthly_turnover_usd_at_test_aum=1_000_000,
        test_aum_usd=10_000_000, universe_adv_usd=1_000_000_000,
    )
    higher = cap.estimate_half_life_aum(
        gross_sharpe=2.0, monthly_turnover_usd_at_test_aum=1_000_000,
        test_aum_usd=10_000_000, universe_adv_usd=1_000_000_000,
    )
    assert higher > base


def test_half_life_lower_for_thinner_universe():
    """Same Sharpe + turnover but thinner universe ADV → lower half-life."""
    big = cap.estimate_half_life_aum(
        gross_sharpe=1.0, monthly_turnover_usd_at_test_aum=1_000_000,
        test_aum_usd=10_000_000, universe_adv_usd=10_000_000_000,
    )
    small = cap.estimate_half_life_aum(
        gross_sharpe=1.0, monthly_turnover_usd_at_test_aum=1_000_000,
        test_aum_usd=10_000_000, universe_adv_usd=100_000_000,
    )
    assert small < big


# ── capacity_report ──────────────────────────────────────────────────────

def test_capacity_report_empty():
    r = cap.capacity_report(pd.DataFrame(), None)
    assert r.hard_capacity_usd == 0.0
    assert r.n_periods_analyzed == 0


def test_capacity_report_complete_shape(equal_weights, adv_panel_uniform):
    r = cap.capacity_report(
        equal_weights, adv_panel_uniform,
        test_aum_usd=100_000_000, gross_sharpe=1.0,
    )
    assert r.hard_capacity_usd > 0
    assert r.n_periods_analyzed == 24
    assert r.test_aum_usd == 100_000_000
    assert r.n_names_average == 10
    assert r.binding_constraint_asset is not None


def test_capacity_report_no_turnover_no_half_life(adv_panel_uniform):
    """A static portfolio (no rebal) → no turnover → no half-life estimate."""
    cols = adv_panel_uniform.columns
    idx = adv_panel_uniform.index
    static_w = pd.DataFrame(0.1, index=idx, columns=cols)
    r = cap.capacity_report(static_w, adv_panel_uniform,
                                test_aum_usd=100_000_000, gross_sharpe=1.0)
    assert r.monthly_turnover_usd == 0.0
    assert r.half_life_aum_usd is None


def test_capacity_report_turnover_signal():
    """Verify turnover computation matches expectation."""
    cols = ["A", "B"]
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    w = pd.DataFrame([[0.5, 0.5], [0.7, 0.3], [0.3, 0.7]],
                       index=idx, columns=cols)
    r = cap.capacity_report(w, None,
                                test_aum_usd=100_000_000, gross_sharpe=None)
    # Period turnover: 0 (first), 0.4 (Δ both 0.2), 0.8 (Δ both 0.4) → avg ~0.4
    # At $100M AUM → ~$40M average monthly turnover
    assert 30_000_000 < r.monthly_turnover_usd < 50_000_000


def test_capacity_report_dataclass_serializable(equal_weights, adv_panel_uniform):
    r = cap.capacity_report(equal_weights, adv_panel_uniform,
                                gross_sharpe=1.0)
    d = r.to_dict()
    assert "hard_capacity_usd" in d
    assert "half_life_aum_usd" in d
    assert "monthly_turnover_usd" in d
    assert "binding_constraint_asset" in d
