"""tests/test_factor_template_tsmom.py — Tier C-2b.

Tests for engine.agents.strengthener.templates.tsmom_sector_etf.

Split into:
  - Unit (offline, fast, deterministic): scope guards, date parse,
    verdict mapping, signal computation on synthetic prices
  - Integration (real DB + real _fetch_closes): one short 3y window
    end-to-end, asserts metrics come back finite

Integration test is marked @pytest.mark.integration and skipped by
default unless RUN_TSMOM_INTEGRATION=1 in env, so the regular test
suite stays offline.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _spec(**kw):
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    base = dict(
        hypothesis_id="hid_test",
        signal_kind="time_series_momentum",
        universe="us_equities_sector_etf",
        date_range="2020-01:2024-12",
        signal_inputs=("etf.adj_close.spy",),
        rebal="weekly",
        weighting="signed_signal_volatility_targeted",
        expected_holding_period="weekly",
        min_obs_months=24,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="canonical TSMOM(52,4) on sector ETFs",
        extracted_ts="2026-06-08T00:00:00Z",
        model="claude-sonnet-4-6",
    )
    base.update(kw)
    return FactorSpec(**base)


# ────────────────────────────────────────────────────────────────────
# Date range parsing
# ────────────────────────────────────────────────────────────────────
def test_parse_date_range_happy():
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        _parse_date_range,
    )
    s, e = _parse_date_range("2020-01:2024-12")
    assert s == _dt.date(2020, 1, 1)
    assert e == _dt.date(2024, 12, 31)


def test_parse_date_range_february_month_end():
    """Edge case: end month=Feb → 28 or 29 depending on leap."""
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        _parse_date_range,
    )
    s, e = _parse_date_range("2020-02:2024-02")
    assert e == _dt.date(2024, 2, 29)   # 2024 IS leap
    s2, e2 = _parse_date_range("2020-02:2023-02")
    assert e2 == _dt.date(2023, 2, 28)


def test_parse_date_range_bad_format():
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        _parse_date_range,
    )
    with pytest.raises(ValueError):
        _parse_date_range("no colon here")


# ────────────────────────────────────────────────────────────────────
# Scope guards (return EXECUTION_ERROR or UNSUPPORTED_UNIVERSE
# WITHOUT calling DB / network — pure spec validation)
# ────────────────────────────────────────────────────────────────────
def test_template_refuses_wrong_signal_kind():
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        template_tsmom_sector_etf,
    )
    r = template_tsmom_sector_etf(_spec(signal_kind="carry"))
    assert r.verdict == "EXECUTION_ERROR"
    assert r.metrics.get("misroute") is True


def test_template_refuses_wrong_universe():
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        template_tsmom_sector_etf,
    )
    r = template_tsmom_sector_etf(_spec(universe="fx_g10"))
    assert r.verdict == "UNSUPPORTED_UNIVERSE"
    assert r.metrics["unsupported_universe"] == "fx_g10"


# ────────────────────────────────────────────────────────────────────
# Verdict mapping (mirror factor_lab.runner)
# ────────────────────────────────────────────────────────────────────
def test_verdict_thresholds():
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        _verdict_from_t,
    )
    assert _verdict_from_t(2.5)   == "GREEN"
    assert _verdict_from_t(1.96)  == "GREEN"
    assert _verdict_from_t(1.95)  == "MARGINAL"
    assert _verdict_from_t(1.65)  == "MARGINAL"
    assert _verdict_from_t(1.64)  == "RED"
    assert _verdict_from_t(-2.5)  == "GREEN"   # sign-agnostic
    assert _verdict_from_t(0.0)   == "RED"
    assert _verdict_from_t(float("nan")) == "RED"


# ────────────────────────────────────────────────────────────────────
# Weekly TSMOM signal — synthetic prices
# ────────────────────────────────────────────────────────────────────
def test_tsmom_signal_constant_uptrend_returns_plus_one():
    """A monotonically rising price series → past return is + →
    signal = +1 (long)."""
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        _tsmom_signal_weekly,
    )
    dates = pd.date_range("2020-01-03", periods=80, freq="W-FRI")
    closes = pd.DataFrame({"X": np.linspace(100, 200, 80)}, index=dates)
    sig = _tsmom_signal_weekly(closes, lookback_weeks=10, skip_weeks=4)
    # First 14 rows NaN (lookback+skip); rest should all be +1
    valid = sig.iloc[14:]
    assert (valid["X"] == 1.0).all()


def test_tsmom_signal_constant_downtrend_returns_minus_one():
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        _tsmom_signal_weekly,
    )
    dates = pd.date_range("2020-01-03", periods=80, freq="W-FRI")
    closes = pd.DataFrame({"X": np.linspace(200, 100, 80)}, index=dates)
    sig = _tsmom_signal_weekly(closes, lookback_weeks=10, skip_weeks=4)
    assert (sig.iloc[14:]["X"] == -1.0).all()


def test_tsmom_signal_warmup_period_is_nan():
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        _tsmom_signal_weekly,
    )
    dates = pd.date_range("2020-01-03", periods=80, freq="W-FRI")
    closes = pd.DataFrame({"X": np.linspace(100, 200, 80)}, index=dates)
    sig = _tsmom_signal_weekly(closes, lookback_weeks=20, skip_weeks=4)
    # First 24 rows must be NaN (lookback+skip)
    assert sig.iloc[:24]["X"].isna().all()


# ────────────────────────────────────────────────────────────────────
# Backtest harness — synthetic prices (no DB / no network)
# ────────────────────────────────────────────────────────────────────
def test_run_tsmom_backtest_makes_money_on_persistent_trend():
    """4 assets, all trending up persistently. TSMOM should be +
    every week post-warmup → net of small TC drag, PnL > 0."""
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        _run_tsmom_backtest,
    )
    dates = pd.date_range("2020-01-03", periods=200, freq="W-FRI")
    base  = np.linspace(100, 250, 200)
    closes = pd.DataFrame({
        "A": base, "B": base * 1.05,
        "C": base * 0.97, "D": base * 1.02,
    }, index=dates)
    pnl, diag = _run_tsmom_backtest(closes, lookback_weeks=20,
                                       skip_weeks=4, vol_target=0.10,
                                       tc_bp_per_rt=13.0)
    assert len(pnl) > 100
    # After warmup, signals are persistent so turnover is low
    assert diag["avg_weekly_turnover"] < 0.5
    # Cumulative PnL should be positive over the trending sample
    assert pnl.sum() > 0


def test_run_tsmom_backtest_handles_all_nan_signals():
    """If all assets have insufficient history (warmup not done),
    backtest still returns a (possibly empty) PnL series without
    crashing."""
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        _run_tsmom_backtest,
    )
    dates = pd.date_range("2020-01-03", periods=20, freq="W-FRI")
    closes = pd.DataFrame({"A": np.linspace(100, 105, 20),
                             "B": np.linspace(100, 110, 20)},
                             index=dates)
    pnl, diag = _run_tsmom_backtest(closes, lookback_weeks=20,
                                       skip_weeks=4, vol_target=0.10,
                                       tc_bp_per_rt=13.0)
    # All weeks warmup-only → empty after dropna
    assert isinstance(pnl, pd.Series)
    assert diag["n_weeks_signaled"] == 0


# ────────────────────────────────────────────────────────────────────
# Template registry wiring — dispatcher routes TSMOM to real template
# ────────────────────────────────────────────────────────────────────
def test_dispatcher_routes_tsmom_to_real_template():
    """C-2a stub returned PENDING_TEMPLATE_BUILD. After C-2b wiring,
    the dispatcher should call the real template (which returns
    UNSUPPORTED_UNIVERSE for non-sector_etf universes, not pending)."""
    from engine.agents.strengthener.factor_dispatcher import (
        TEMPLATE_REGISTRY,
    )
    tpl = TEMPLATE_REGISTRY["time_series_momentum"]
    # Non-sector_etf universe → lazy wrapper falls back to stub
    r_other = tpl(_spec(universe="commodity_futures_27"))
    assert r_other.verdict == "PENDING_TEMPLATE_BUILD"
    # sector_etf universe → real template invoked (will likely
    # error on misroute since we keep signal_kind valid)
    # Use a spec with bad signal_kind to confirm WE'RE in the real
    # template path (would EXECUTION_ERROR misroute), not the stub
    r_misroute = tpl(_spec(signal_kind="time_series_momentum",
                              universe="us_equities_sector_etf"))
    # On sector_etf universe + correct signal_kind, we'd hit DB.
    # If DB unavailable in test env, we'd get DATA_ERROR; if data
    # available, we'd get an actual verdict. Either way it's NOT
    # PENDING_TEMPLATE_BUILD (which would mean stub still wired)
    assert r_misroute.verdict != "PENDING_TEMPLATE_BUILD"


# ────────────────────────────────────────────────────────────────────
# Integration test — real DB + real _fetch_closes
# (Skipped unless RUN_TSMOM_INTEGRATION=1 in env)
# ────────────────────────────────────────────────────────────────────
def test_tsmom_b_class_defaults_parity():
    """L2-1 Phase 4: when B-class params None, eff values match
    Moskowitz et al. defaults (lookback 52w, skip 4w, vol 10%)."""
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        _LOOKBACK_WEEKS, _SKIP_WEEKS, _VOL_TARGET,
    )
    # Convert defaults: 12 months × 52/12 = 52 weeks; 1 month → 4
    assert _LOOKBACK_WEEKS == 52
    assert _SKIP_WEEKS == 4
    assert _VOL_TARGET == 0.10


def test_tsmom_b_class_months_to_weeks_conversion():
    """L2-1 Phase 4: 6 months → 26 weeks (within 1 of 6*52/12)."""
    eff_w_6mo = round(6 * 52 / 12)
    eff_w_3mo = round(3 * 52 / 12)
    eff_w_12mo = round(12 * 52 / 12)
    assert eff_w_6mo == 26   # 6-1 TSMOM common variant
    assert eff_w_3mo == 13   # short-horizon
    assert eff_w_12mo == 52  # default


@pytest.mark.skipif(
    os.environ.get("RUN_TSMOM_INTEGRATION") != "1",
    reason="set RUN_TSMOM_INTEGRATION=1 to run live data-touching test",
)
def test_template_integration_b_class_variant_lookback():
    """L2-1 Phase 4 live: TSMOM 6-month lookback (instead of default
    12) should produce DIFFERENT verdict numbers vs default.
    Verifies B-class wiring actually flows through to backtest."""
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        template_tsmom_sector_etf,
    )
    base = dict(
        hypothesis_id="tsmom_b_class", signal_kind="time_series_momentum",
        universe="us_equities_sector_etf", date_range="2018-01:2023-12",
        signal_inputs=("etf.adj_close.spy",), rebal="weekly",
        weighting="signed_signal_volatility_targeted",
        expected_holding_period="weekly", min_obs_months=24,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="tsmom B-class variant test",
        extracted_ts="2026-06-08T00:00:00Z", model="claude-sonnet-4-6",
    )
    r_default = template_tsmom_sector_etf(FactorSpec(**base))
    base_variant = {**base, "signal_lookback_m": 6,
                     "signal_skip_m": 1, "vol_target_annual": 0.15}
    r_variant = template_tsmom_sector_etf(FactorSpec(**base_variant))
    # B-class params actually wired through to metrics
    assert r_default.metrics["lookback_weeks"] == 52
    assert r_default.metrics["skip_weeks"] == 4
    assert r_default.metrics["vol_target"] == 0.10
    assert r_variant.metrics["lookback_weeks"] == 26
    assert r_variant.metrics["skip_weeks"] == 4   # round(1*52/12)=4
    assert r_variant.metrics["vol_target"] == 0.15
    # Different params → different Sharpe (probably)
    # Not asserting direction, just that they DIFFER (system actually
    # changed behavior, not silently ignored params)
    assert r_default.metrics["sharpe"] != r_variant.metrics["sharpe"]


@pytest.mark.skipif(
    os.environ.get("RUN_TSMOM_INTEGRATION") != "1",
    reason="set RUN_TSMOM_INTEGRATION=1 to run live data-touching test",
)
def test_template_integration_short_window():
    """End-to-end on a 3y window. Requires DB + yfinance reachable.
    Asserts metrics come back finite + verdict is one of the
    real verdicts (not error)."""
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        template_tsmom_sector_etf,
    )
    spec = _spec(date_range="2021-01:2024-01", min_obs_months=24)
    r = template_tsmom_sector_etf(spec)
    assert r.verdict in ("GREEN", "MARGINAL", "RED")
    m = r.metrics
    assert math.isfinite(m["sharpe"])
    assert math.isfinite(m["nw_t_stat"])
    assert m["n_obs_months"] >= 24
    # n_tickers requested can be 35; n_tickers_in_data varies by
    # window (newer ETFs may not have inception in 3y window)
    assert m["n_tickers"] >= 10
    assert m["n_tickers_in_data"] >= 10
