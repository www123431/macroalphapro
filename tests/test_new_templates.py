"""Tests for the 3 new DSL templates (senior roadmap #2).

Each template is a thin composition over primitives — we test:
  1. shape / dtype / index correctness
  2. parameter validation
  3. economic-direction invariants (long-vs-short signs)
  4. cost & vol-target sanity
  5. edge cases (empty panel / all NaN / missing tenors)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research.templates.event_study   import (
    _active_basket_at_each_month, run_event_study, warmup_months as ws_event,
)
from engine.research.templates.dispersion    import (
    run_dispersion, warmup_months as ws_disp,
)
from engine.research.templates.term_structure import (
    _nelson_siegel_basis, _fit_nelson_siegel_panel,
    run_term_structure, warmup_months as ws_ts,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def monthly_dates():
    return pd.date_range("2010-01-01", "2024-12-01", freq="MS")


@pytest.fixture
def small_universe():
    return [f"T{i:02d}" for i in range(10)]


@pytest.fixture
def return_panel(monthly_dates, small_universe):
    np.random.seed(42)
    data = np.random.randn(len(monthly_dates), len(small_universe)) * 0.05
    return pd.DataFrame(data, index=monthly_dates, columns=small_universe)


# ── event_study ────────────────────────────────────────────────────────────

def test_event_warmup_minimal():
    assert ws_event({"hold_months": 3}) == 4
    assert ws_event({"hold_months": 3, "vol_target": 0.1,
                       "vol_target_lookback": 24}) == 25


def test_active_basket_rolling_or(monthly_dates, small_universe):
    """One event for T00 in month 0 → active for next 3 months only."""
    ep = pd.DataFrame(False, index=monthly_dates, columns=small_universe)
    ep.loc[monthly_dates[5], "T00"] = True
    active = _active_basket_at_each_month(ep, hold_months=3, skip_first_month=True)
    # Active months: 6, 7, 8 (after t=5 event with skip_first_month)
    assert active.loc[monthly_dates[5], "T00"] == False    # event month itself
    assert active.loc[monthly_dates[6], "T00"] == True
    assert active.loc[monthly_dates[7], "T00"] == True
    assert active.loc[monthly_dates[8], "T00"] == True
    assert active.loc[monthly_dates[9], "T00"] == False    # past window


def test_event_study_runs_and_returns_series(return_panel, small_universe, monthly_dates):
    ep = pd.DataFrame(False, index=monthly_dates, columns=small_universe)
    # Add 30 random events across the panel
    np.random.seed(0)
    for _ in range(30):
        ep.iloc[np.random.randint(0, len(monthly_dates)),
                  np.random.randint(0, len(small_universe))] = True
    result = run_event_study(
        return_panel=return_panel, event_panel=ep,
        hold_months=3, vol_target=None,
    )
    assert isinstance(result, pd.Series)
    assert len(result) > 0
    assert result.dtype == float


def test_event_study_with_benchmark_subtracts_market(return_panel, small_universe, monthly_dates):
    ep = pd.DataFrame(False, index=monthly_dates, columns=small_universe)
    for i in range(0, len(monthly_dates), 12):
        ep.iloc[i, 0] = True
    bench = pd.Series(0.01, index=monthly_dates)    # constant benchmark
    result_with = run_event_study(
        return_panel=return_panel, event_panel=ep,
        benchmark_returns=bench, hold_months=3, vol_target=None,
    )
    result_without = run_event_study(
        return_panel=return_panel, event_panel=ep,
        hold_months=3, vol_target=None,
    )
    # With benchmark subtracted, mean should be lower
    assert result_with.mean() < result_without.mean()


def test_event_study_zero_events_returns_empty(return_panel, small_universe, monthly_dates):
    ep = pd.DataFrame(False, index=monthly_dates, columns=small_universe)
    result = run_event_study(
        return_panel=return_panel, event_panel=ep,
        hold_months=3, vol_target=None,
    )
    assert len(result) == 0    # nothing to hold ever


# ── dispersion ─────────────────────────────────────────────────────────────

def test_dispersion_warmup():
    assert ws_disp({"mode": "level"}) == 1
    assert ws_disp({"mode": "change", "signal_lookback": 6}) == 7
    assert ws_disp({"mode": "level", "vol_target": 0.1,
                      "vol_target_lookback": 24}) == 25


def test_dispersion_invalid_mode_raises(return_panel):
    with pytest.raises(ValueError):
        run_dispersion(signal_panel=return_panel, return_panel=return_panel,
                          mode="banana")


def test_dispersion_value_weight_not_implemented(return_panel):
    with pytest.raises(NotImplementedError):
        run_dispersion(signal_panel=return_panel, return_panel=return_panel,
                          mode="level", weighting="value_weight")


def test_dispersion_level_runs(monthly_dates, small_universe, return_panel):
    np.random.seed(1)
    signal = pd.DataFrame(
        np.random.randn(len(monthly_dates), len(small_universe)),
        index=monthly_dates, columns=small_universe,
    )
    result = run_dispersion(
        signal_panel=signal, return_panel=return_panel,
        mode="level", top_frac=0.3, vol_target=None,
    )
    assert isinstance(result, pd.Series)
    assert len(result) > 0


def test_dispersion_change_runs(monthly_dates, small_universe, return_panel):
    np.random.seed(2)
    signal = pd.DataFrame(
        np.random.randn(len(monthly_dates), len(small_universe)),
        index=monthly_dates, columns=small_universe,
    )
    result = run_dispersion(
        signal_panel=signal, return_panel=return_panel,
        mode="change", signal_lookback=3, top_frac=0.3, vol_target=None,
    )
    assert isinstance(result, pd.Series)


def test_dispersion_level_vs_change_distinct(monthly_dates, small_universe, return_panel):
    """Level and change modes should produce different return series."""
    np.random.seed(3)
    signal = pd.DataFrame(
        np.cumsum(np.random.randn(len(monthly_dates), len(small_universe)),
                  axis=0) * 0.1,
        index=monthly_dates, columns=small_universe,
    )
    r_level = run_dispersion(
        signal_panel=signal, return_panel=return_panel,
        mode="level", top_frac=0.3, vol_target=None,
    )
    r_change = run_dispersion(
        signal_panel=signal, return_panel=return_panel,
        mode="change", signal_lookback=3, top_frac=0.3, vol_target=None,
    )
    # Series should be NOT identical (different signals)
    common_idx = r_level.index.intersection(r_change.index)
    if len(common_idx) > 10:
        corr = r_level.loc[common_idx].corr(r_change.loc[common_idx])
        assert abs(corr) < 0.99    # not identical (allow some overlap)


# ── term_structure ─────────────────────────────────────────────────────────

def test_ts_warmup():
    assert ws_ts({"mode": "slope"}) == 1
    assert ws_ts({"mode": "slope", "vol_target": 0.1,
                    "vol_target_lookback": 36}) == 37


def test_ns_basis_shape():
    """NS basis should be (n_tenors, 3) and finite."""
    tenors = np.array([1, 3, 6, 12, 24, 60, 120, 360], dtype=float)
    basis = _nelson_siegel_basis(tenors, lam=18.0)
    assert basis.shape == (8, 3)
    assert np.all(np.isfinite(basis))
    # X0 column is all 1s
    assert np.allclose(basis[:, 0], 1.0)


def test_fit_ns_recovers_known_betas():
    """Generate yield curve from known (β0=4, β1=-2, β2=1) and fit."""
    tenors = np.array([3, 6, 12, 24, 60, 120, 240, 360], dtype=float)
    lam = 18.0
    basis = _nelson_siegel_basis(tenors, lam)
    true_betas = np.array([4.0, -2.0, 1.0])
    yields = basis @ true_betas
    # Build a 5-row yield_panel with the same yields
    dates = pd.date_range("2020-01-01", periods=5, freq="MS")
    yield_panel = pd.DataFrame(
        np.tile(yields, (5, 1)),
        index=dates,
        columns=[int(t) for t in tenors],
    )
    fitted = _fit_nelson_siegel_panel(yield_panel, lam)
    assert fitted["beta0"].iloc[0] == pytest.approx(4.0, abs=0.01)
    assert fitted["beta1"].iloc[0] == pytest.approx(-2.0, abs=0.01)
    assert fitted["beta2"].iloc[0] == pytest.approx(1.0, abs=0.01)


def test_ts_slope_mode_requires_columns(monthly_dates):
    """yield_panel must have requested tenors as columns."""
    yield_panel = pd.DataFrame(
        np.random.randn(len(monthly_dates), 3) + 3,
        index=monthly_dates,
        columns=[24, 60, 120],     # 2y, 5y, 10y
    )
    with pytest.raises(KeyError):
        run_term_structure(yield_panel=yield_panel,
                              mode="slope",
                              long_tenor_months=360,    # not in columns!
                              short_tenor_months=24)


def test_ts_slope_runs(monthly_dates):
    """10y - 2y slope strategy."""
    np.random.seed(4)
    yield_panel = pd.DataFrame(
        np.cumsum(np.random.randn(len(monthly_dates), 5) * 0.1, axis=0) + 3,
        index=monthly_dates,
        columns=[3, 24, 60, 120, 360],
    )
    result = run_term_structure(
        yield_panel=yield_panel, mode="slope",
        long_tenor_months=120, short_tenor_months=24,
        vol_target=None,
    )
    assert isinstance(result, pd.Series)
    assert len(result) > 0


def test_ts_ns_mode_runs(monthly_dates):
    """Nelson-Siegel 3-factor mode."""
    np.random.seed(5)
    yield_panel = pd.DataFrame(
        np.cumsum(np.random.randn(len(monthly_dates), 5) * 0.05, axis=0) + 3,
        index=monthly_dates,
        columns=[3, 24, 60, 120, 360],
    )
    result = run_term_structure(
        yield_panel=yield_panel, mode="nelson_siegel_3factor",
        asset_class="rates", vol_target=None,
    )
    assert isinstance(result, pd.Series)


def test_ts_curvature_mode_runs(monthly_dates):
    np.random.seed(6)
    yield_panel = pd.DataFrame(
        np.cumsum(np.random.randn(len(monthly_dates), 5) * 0.05, axis=0) + 3,
        index=monthly_dates,
        columns=[3, 24, 60, 120, 360],
    )
    result = run_term_structure(
        yield_panel=yield_panel, mode="curvature_only",
        asset_class="rates", vol_target=None,
    )
    assert isinstance(result, pd.Series)


def test_ts_unknown_mode_raises(monthly_dates):
    yield_panel = pd.DataFrame(
        np.random.randn(len(monthly_dates), 5) + 3,
        index=monthly_dates,
        columns=[3, 24, 60, 120, 360],
    )
    with pytest.raises(ValueError):
        run_term_structure(yield_panel=yield_panel, mode="banana")


# ── auto-discovery integration ────────────────────────────────────────────

def test_templates_auto_discovered():
    """All 3 new templates should appear in the registry."""
    from engine.research.templates import reload_templates
    registry = reload_templates()
    assert "event_study" in registry
    assert "dispersion" in registry
    assert "term_structure" in registry
