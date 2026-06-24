"""Smoke tests confirming the 3 new templates (senior #2) feed
the existing strict gate (engine.research.pipeline.run_gate).

run_gate is generic — it consumes any monthly L/S net-of-cost
return series and produces a verdict dict. We verify each new
template produces a compatible series and the gate processes it
without error.

This proves the new templates are FULLY wired into the same
strict-gate machinery that PEAD / equity_xsmom / cross_asset_carry
already use; no per-template gate logic needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research.pipeline import run_gate
from engine.research.templates.event_study   import run_event_study
from engine.research.templates.dispersion    import run_dispersion
from engine.research.templates.term_structure import run_term_structure


@pytest.fixture
def long_monthly_dates():
    """Enough months for run_gate's 24-month floor + halves."""
    return pd.date_range("2000-01-01", "2024-12-01", freq="MS")


@pytest.fixture
def synthetic_universe(long_monthly_dates):
    """50 tickers, 25 years monthly data."""
    n_t = len(long_monthly_dates)
    tickers = [f"T{i:03d}" for i in range(50)]
    np.random.seed(11)
    return_panel = pd.DataFrame(
        np.random.randn(n_t, 50) * 0.05,
        index=long_monthly_dates, columns=tickers,
    )
    return tickers, return_panel


# ── event_study → run_gate ────────────────────────────────────────────────

def test_event_study_feeds_gate(long_monthly_dates, synthetic_universe):
    tickers, rp = synthetic_universe
    # Generate ~200 events sprinkled across panel
    event_panel = pd.DataFrame(False, index=long_monthly_dates, columns=tickers)
    np.random.seed(12)
    for _ in range(200):
        i = np.random.randint(12, len(long_monthly_dates) - 6)
        j = np.random.randint(0, 50)
        event_panel.iloc[i, j] = True
    series = run_event_study(
        return_panel=rp, event_panel=event_panel,
        hold_months=3, vol_target=None,
    )
    assert len(series) >= 24
    # n_trials=1 since this is a smoke (real callers would track)
    verdict = run_gate(series, name="event_study_smoke",
                          mechanism="generic post-event drift",
                          n_trials=1, pead_control=False, log=False)
    assert verdict["available"] is True
    assert "verdict" in verdict
    assert verdict["verdict"] in ("GREEN", "YELLOW", "RED")
    assert verdict["n_months"] >= 24


# ── dispersion → run_gate ─────────────────────────────────────────────────

def test_dispersion_level_feeds_gate(long_monthly_dates, synthetic_universe):
    tickers, rp = synthetic_universe
    np.random.seed(13)
    signal = pd.DataFrame(
        np.random.randn(len(long_monthly_dates), 50),
        index=long_monthly_dates, columns=tickers,
    )
    series = run_dispersion(
        signal_panel=signal, return_panel=rp,
        mode="level", top_frac=0.2, vol_target=None,
    )
    verdict = run_gate(series, name="dispersion_level_smoke",
                          mechanism="cross-sectional dispersion (DMS-style)",
                          n_trials=1, pead_control=False, log=False)
    assert verdict["available"] is True
    assert verdict["verdict"] in ("GREEN", "YELLOW", "RED")


def test_dispersion_change_feeds_gate(long_monthly_dates, synthetic_universe):
    tickers, rp = synthetic_universe
    np.random.seed(14)
    signal = pd.DataFrame(
        np.cumsum(np.random.randn(len(long_monthly_dates), 50) * 0.1, axis=0),
        index=long_monthly_dates, columns=tickers,
    )
    series = run_dispersion(
        signal_panel=signal, return_panel=rp,
        mode="change", signal_lookback=3, top_frac=0.2, vol_target=None,
    )
    verdict = run_gate(series, name="dispersion_change_smoke",
                          mechanism="resolution-of-uncertainty",
                          n_trials=1, pead_control=False, log=False)
    assert verdict["available"] is True


# ── term_structure → run_gate ─────────────────────────────────────────────

def test_term_structure_slope_feeds_gate(long_monthly_dates):
    np.random.seed(15)
    yield_panel = pd.DataFrame(
        np.cumsum(np.random.randn(len(long_monthly_dates), 5) * 0.05, axis=0) + 3,
        index=long_monthly_dates,
        columns=[3, 24, 60, 120, 360],
    )
    series = run_term_structure(
        yield_panel=yield_panel, mode="slope",
        long_tenor_months=120, short_tenor_months=24,
        vol_target=None,
    )
    verdict = run_gate(series, name="term_structure_slope_smoke",
                          mechanism="10y-2y slope timing",
                          n_trials=1, pead_control=False, log=False)
    assert verdict["available"] is True


def test_term_structure_ns_feeds_gate(long_monthly_dates):
    np.random.seed(16)
    yield_panel = pd.DataFrame(
        np.cumsum(np.random.randn(len(long_monthly_dates), 5) * 0.05, axis=0) + 3,
        index=long_monthly_dates,
        columns=[3, 24, 60, 120, 360],
    )
    series = run_term_structure(
        yield_panel=yield_panel, mode="nelson_siegel_3factor",
        asset_class="rates", vol_target=None,
    )
    verdict = run_gate(series, name="term_structure_ns_smoke",
                          mechanism="Nelson-Siegel β1 timing",
                          n_trials=1, pead_control=False, log=False)
    assert verdict["available"] is True


# ── auto-discovery confirms the gate sees them ────────────────────────────

def test_all_3_new_templates_in_registry():
    """Verify protocol_executor can dispatch into the new templates
    via the auto-discovered registry."""
    from engine.research.templates import reload_templates
    reg = reload_templates()
    assert "event_study" in reg
    assert "dispersion" in reg
    assert "term_structure" in reg
    assert callable(reg["event_study"])
    assert callable(reg["dispersion"])
    assert callable(reg["term_structure"])
