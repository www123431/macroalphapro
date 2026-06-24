"""tests/test_factor_template_carry_g10_fx.py — Tier C-2f.

Tests for engine.agents.strengthener.templates.carry_g10_fx.

Mirrors test_factor_template_tsmom + test_factor_template_cross_sec
split:
  - Unit (offline, fast, deterministic): scope guards, date parse,
    verdict mapping, drawdown computation
  - Integration (live cached FX/rates parquets): end-to-end dispatch
    asserting finite Sharpe + finite t and verdict ∈ {GREEN, MARGINAL, RED}

Integration test runs ALWAYS when the cached parquets exist — no env
flag — because the parquets are LOCAL (committed during LRV chain),
no network or DB call needed.
"""
from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _spec(**kw):
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    base = dict(
        hypothesis_id="carry_test_hid",
        signal_kind="carry",
        universe="fx_g10",
        date_range="2002-04:2024-12",
        signal_inputs=("fred.fx_spot_g10",),
        rebal="monthly",
        weighting="tercile_long_short_dollar_neutral",
        expected_holding_period="monthly",
        min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="LRV 2011 HML_FX cross-sec carry on G10 FX",
        extracted_ts="2026-06-09T00:00:00Z",
        model="claude-sonnet-4-6",
    )
    base.update(kw)
    return FactorSpec(**base)


# ────────────────────────────────────────────────────────────────────
# Date range parsing
# ────────────────────────────────────────────────────────────────────
def test_parse_date_range_happy():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _parse_date_range,
    )
    s, e = _parse_date_range("2002-04:2024-12")
    assert s == _dt.date(2002, 4, 1)
    assert e == _dt.date(2024, 12, 31)


def test_parse_date_range_bad_format():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _parse_date_range,
    )
    with pytest.raises(ValueError):
        _parse_date_range("no_colon")


# ────────────────────────────────────────────────────────────────────
# Scope guards (no FX/rate parquet touched — pure spec validation)
# ────────────────────────────────────────────────────────────────────
def test_template_refuses_wrong_signal_kind():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        template_carry_g10_fx,
    )
    r = template_carry_g10_fx(_spec(signal_kind="time_series_momentum"))
    assert r.verdict == "EXECUTION_ERROR"
    assert r.metrics.get("misroute") is True


def test_template_refuses_wrong_universe():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        template_carry_g10_fx,
    )
    r = template_carry_g10_fx(_spec(universe="us_equities_top_3000"))
    assert r.verdict == "UNSUPPORTED_UNIVERSE"
    assert r.metrics["unsupported_universe"] == "us_equities_top_3000"


# ────────────────────────────────────────────────────────────────────
# Verdict mapping mirrors cross_sec + tsmom
# ────────────────────────────────────────────────────────────────────
def test_verdict_thresholds():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _verdict_from_t,
    )
    assert _verdict_from_t(2.5)   == "GREEN"
    assert _verdict_from_t(1.96)  == "GREEN"
    assert _verdict_from_t(1.95)  == "MARGINAL"
    assert _verdict_from_t(1.65)  == "MARGINAL"
    assert _verdict_from_t(1.64)  == "RED"
    assert _verdict_from_t(-2.5)  == "GREEN"  # sign-agnostic
    assert _verdict_from_t(0.0)   == "RED"
    assert _verdict_from_t(float("nan")) == "RED"


def test_cost_stress_verdict_is_sign_aware():
    """Cost-stress verdict should NOT call sign-flipped negative GREEN."""
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _cost_stress_verdict_from_t,
    )
    assert _cost_stress_verdict_from_t(2.5)  == "GREEN"
    assert _cost_stress_verdict_from_t(-2.5) == "RED"   # sign matters
    assert _cost_stress_verdict_from_t(1.96) == "GREEN"
    assert _cost_stress_verdict_from_t(-1.96) == "RED"


# ────────────────────────────────────────────────────────────────────
# Drawdown metrics — synthetic series
# ────────────────────────────────────────────────────────────────────
def test_drawdown_zero_when_monotonic_up():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _compute_drawdown_metrics,
    )
    pnl = pd.Series([0.01] * 60)
    out = _compute_drawdown_metrics(pnl)
    assert out["max_drawdown_pct"] == 0.0 or out["max_drawdown_pct"] == 0
    assert out["max_underwater_months"] == 0


def test_drawdown_reflects_single_big_dip():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _compute_drawdown_metrics,
    )
    # 12 months of zero, one -20%, then back to zero
    pnl = pd.Series([0.0] * 24 + [-0.20] + [0.0] * 24)
    out = _compute_drawdown_metrics(pnl)
    assert out["max_drawdown_pct"] is not None
    assert out["max_drawdown_pct"] < -0.15


def test_drawdown_short_series_returns_nones():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _compute_drawdown_metrics,
    )
    out = _compute_drawdown_metrics(pd.Series([0.01] * 5))
    assert out["max_drawdown_pct"] is None
    assert out["calmar_ratio"] is None


# ────────────────────────────────────────────────────────────────────
# Turnover construction — synthetic sort keys
# ────────────────────────────────────────────────────────────────────
def test_turnover_zero_when_sort_key_constant():
    """If the lagged rdiff is identical every month, the sort is
    stable → no portfolio rebalancing → turnover ≈ 0 after month 1."""
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _build_turnover_series,
    )
    from engine.research.fx_carry_anchors import G10_CURRENCIES

    idx = pd.date_range("2010-01-31", periods=24, freq="ME")
    fixed_keys = pd.DataFrame(
        {c: [float(i)] * 24 for i, c in enumerate(G10_CURRENCIES)},
        index=idx,
    )
    t = _build_turnover_series(fixed_keys, n_buckets=3,
                                  g10_currencies=G10_CURRENCIES)
    # First month sets up holdings (one-way magnitude ≈ 1 gross/2)
    # Subsequent months: no change → tiny / zero
    assert t.iloc[0] > 0
    assert (t.iloc[1:] < 0.05).all()


def test_turnover_spikes_when_sort_key_flips():
    """When the bottom currency suddenly becomes top, holdings flip
    completely → high turnover on that month."""
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _build_turnover_series,
    )
    from engine.research.fx_carry_anchors import G10_CURRENCIES

    idx = pd.date_range("2010-01-31", periods=24, freq="ME")
    # First 12 months: increasing order by currency index;
    # last 12: REVERSED order
    rows = []
    n = len(G10_CURRENCIES)
    for month_idx in range(24):
        if month_idx < 12:
            row = {c: float(i) for i, c in enumerate(G10_CURRENCIES)}
        else:
            row = {c: float(n - i) for i, c in enumerate(G10_CURRENCIES)}
        rows.append(row)
    keys = pd.DataFrame(rows, index=idx)
    t = _build_turnover_series(keys, n_buckets=3,
                                  g10_currencies=G10_CURRENCIES)
    # The month-12 flip should produce notable turnover
    assert t.iloc[12] > 0.3


# ────────────────────────────────────────────────────────────────────
# Cost stress — synthetic gross/turnover
# ────────────────────────────────────────────────────────────────────
def test_cost_stress_higher_cost_reduces_sharpe():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _compute_cost_stress,
    )
    rng = np.random.default_rng(7)
    n = 120
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    pnl_gross = pd.Series(0.006 + rng.normal(0, 0.03, n), index=idx)
    turnover  = pd.Series(rng.uniform(0.1, 0.3, n), index=idx)
    out = _compute_cost_stress(pnl_gross, turnover)
    # All four levels reported
    for k in ("0bp", "8bp", "16bp", "24bp"):
        assert k in out
    # Monotonic Sharpe decline with cost (or at least non-increasing)
    sharpe_seq = [out[k]["sharpe"] for k in ("0bp", "8bp", "16bp", "24bp")]
    assert all(s is not None for s in sharpe_seq)
    # 24bp Sharpe must be ≤ 0bp Sharpe (with finite turnover > 0)
    assert sharpe_seq[-1] <= sharpe_seq[0] + 1e-9


# ────────────────────────────────────────────────────────────────────
# Replication mode — synthetic PnL with overlap to a paper window
# ────────────────────────────────────────────────────────────────────
def test_replication_returns_no_data_on_bad_window_str():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _compute_replication_subsample,
    )
    pnl = pd.Series([0.01] * 60,
                       index=pd.date_range("2010-01-31", periods=60, freq="ME"))
    out = _compute_replication_subsample(pnl, "garbage", paper_reported_t=2.8)
    assert out["status"] == "NO_DATA"


def test_replication_returns_insufficient_overlap_when_window_disjoint():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        _compute_replication_subsample,
    )
    pnl = pd.Series([0.01] * 60,
                       index=pd.date_range("2020-01-31", periods=60, freq="ME"))
    out = _compute_replication_subsample(
        pnl, "1983-11:2009-12", paper_reported_t=2.8,
    )
    assert out["status"] == "INSUFFICIENT_OVERLAP"


# ────────────────────────────────────────────────────────────────────
# Template contract registration
# ────────────────────────────────────────────────────────────────────
def test_contract_registered_for_carry_fx_g10():
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    c = contract_for_scope("carry", "fx_g10")
    assert c is not None
    assert c.template_name == "carry_g10_fx"
    assert c.is_fresh()
    # Paper benchmark wired
    assert c.canonical_paper_id == "lustig_roussanov_verdelhan_2011"
    assert c.canonical_paper_t is not None


def test_contract_rejects_carry_on_unsupported_universe():
    """commodity_futures_27 + carry has no contract yet → caller's
    Gate #10 should refuse, NOT silently route."""
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    assert contract_for_scope("carry", "commodity_futures_27") is None
    assert contract_for_scope("carry", "us_treasury_curve") is None


# ────────────────────────────────────────────────────────────────────
# Dispatcher TEMPLATE_REGISTRY wiring
# ────────────────────────────────────────────────────────────────────
def test_dispatcher_routes_carry_to_real_template():
    """Before C-2f, carry was wired to _template_pending_build (stub).
    After C-2f, dispatcher should route to the real carry template."""
    from engine.agents.strengthener.factor_dispatcher import TEMPLATE_REGISTRY

    tpl = TEMPLATE_REGISTRY["carry"]
    # Non-fx_g10 universe → lazy wrapper falls back to stub
    r_other = tpl(_spec(universe="commodity_futures_27"))
    assert r_other.verdict == "PENDING_TEMPLATE_BUILD"
    # fx_g10 universe → real template path (might DATA_ERROR if
    # parquets not cached locally, but NOT PENDING_TEMPLATE_BUILD)
    r_fx = tpl(_spec(universe="fx_g10"))
    assert r_fx.verdict != "PENDING_TEMPLATE_BUILD"


# ────────────────────────────────────────────────────────────────────
# Live integration — requires cached LRV FX + rates parquets
# (No env flag — these are LOCAL parquets shipped via earlier commits)
# ────────────────────────────────────────────────────────────────────
def _parquets_cached() -> bool:
    repo = Path(__file__).resolve().parents[1]
    p1 = repo / "data" / "anchor_library" / "fx_spot_g10_monthly.parquet"
    p2 = repo / "data" / "anchor_library" / "g10_short_rates_monthly.parquet"
    return p1.exists() and p2.exists()


@pytest.mark.skipif(not _parquets_cached(),
                     reason="LRV FX/rates parquets not cached")
def test_integration_live_carry_returns_finite_metrics():
    from engine.agents.strengthener.templates.carry_g10_fx import (
        template_carry_g10_fx,
    )
    spec = _spec(date_range="2002-04:2024-12", min_obs_months=60)
    r = template_carry_g10_fx(spec)
    assert r.verdict in ("GREEN", "MARGINAL", "RED"), (
        f"unexpected verdict={r.verdict!r}; summary={r.summary!r}"
    )
    m = r.metrics
    assert math.isfinite(m["sharpe"])
    assert math.isfinite(m["nw_t_stat"])
    assert m["n_months"] >= 60
    assert m["n_currencies"] == 10
    assert m["n_buckets"] == 3
    # pnl_series_df artifact present + correct shape
    assert "pnl_series_df" in r.artifacts
    df = r.artifacts["pnl_series_df"]
    assert "pnl_gross" in df.columns
    assert "turnover"  in df.columns


@pytest.mark.skipif(not _parquets_cached(),
                     reason="LRV FX/rates parquets not cached")
def test_integration_b_class_n_buckets_2_changes_metrics():
    """B-class wiring: n_buckets=2 (halves L/S) MUST produce different
    Sharpe than default n_buckets=3 (terciles). If they match, the
    B-class param is being silently ignored."""
    from engine.agents.strengthener.templates.carry_g10_fx import (
        template_carry_g10_fx,
    )
    r_default = template_carry_g10_fx(_spec(date_range="2002-04:2024-12"))
    r_n2      = template_carry_g10_fx(_spec(date_range="2002-04:2024-12",
                                                n_buckets=2))
    # n_buckets=2 needs each bucket = 4-5 ccys (9/2). Should compute.
    assert r_n2.verdict in ("GREEN", "MARGINAL", "RED"), (
        f"n_buckets=2 unexpected verdict={r_n2.verdict!r}"
    )
    assert r_default.metrics["n_buckets"] == 3
    assert r_n2.metrics["n_buckets"] == 2
    # Sharpe differs (different portfolio construction)
    assert r_default.metrics["sharpe"] != r_n2.metrics["sharpe"]
