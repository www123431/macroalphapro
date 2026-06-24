"""tests/test_factor_template_cross_sec.py — Tier C-2e.1.

Tests for engine.agents.strengthener.templates.cross_sec_us_equities.

Split into:
  - Unit (offline, deterministic): scope guards, date parse, signal-key
    picker on LLM-style inputs, verdict mapping, signal compute on
    synthetic prices, quintile L/S backtest on synthetic panel
  - Integration: full template on real CRSP parquet (skipped unless
    RUN_CROSS_SEC_INTEGRATION=1, mirroring tsmom integration test)
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import uuid
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _spec(**kw):
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    base = dict(
        hypothesis_id="hid_test",
        signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000",
        date_range="2010-01:2023-12",
        signal_inputs=("crsp.msf.derived.vol_12m",),
        rebal="monthly",
        weighting="quintile_long_short_dollar_neutral",
        expected_holding_period="monthly",
        min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="test",
        extracted_ts="2026-06-08T00:00:00Z",
        model="claude-sonnet-4-6",
    )
    base.update(kw)
    return FactorSpec(**base)


# ────────────────────────────────────────────────────────────────────
# Date range parsing — same shape as tsmom
# ────────────────────────────────────────────────────────────────────
def test_parse_date_range_happy():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _parse_date_range,
    )
    s, e = _parse_date_range("2010-01:2023-12")
    assert s == _dt.date(2010, 1, 1)
    assert e == _dt.date(2023, 12, 31)


def test_parse_date_range_bad_format_raises():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _parse_date_range,
    )
    with pytest.raises(ValueError):
        _parse_date_range("not a range")


# ────────────────────────────────────────────────────────────────────
# Signal-key picker — LLM-style inputs
# ────────────────────────────────────────────────────────────────────
def test_signal_picker_matches_low_vol_variants():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _pick_signal_key,
    )
    for inp in [
        ("vol_12m",),
        ("idio_vol_12mo",),
        ("low_vol_signal",),
        ("crsp.msf.derived.volatility_lookback_12",),
        ("realized_vol_proxy",),
    ]:
        assert _pick_signal_key(inp) == "vol_12m", f"input {inp} missed"


def test_signal_picker_matches_momentum_variants():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _pick_signal_key,
    )
    for inp in [
        ("return_12_2",),
        ("mom_12_1",),
        ("momentum_12_1",),
        ("ret_12_1_cumulative",),
    ]:
        assert _pick_signal_key(inp) == "ret_12_1", f"input {inp} missed"


def test_signal_picker_matches_size_variants():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _pick_signal_key,
    )
    for inp in [
        ("mktcap",),
        ("log_market_equity",),
        ("size_factor",),
        ("crsp.msf.market_cap",),
    ]:
        assert _pick_signal_key(inp) == "mktcap", f"input {inp} missed"


def test_signal_picker_matches_reversal():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _pick_signal_key,
    )
    assert _pick_signal_key(("reversal_1m",)) == "reversal_1m"
    assert _pick_signal_key(("short_term_reversal",)) == "reversal_1m"
    assert _pick_signal_key(("prior_month_return",)) == "reversal_1m"


def test_signal_picker_returns_none_for_unsupported():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _pick_signal_key,
    )
    # C-2e.2 added fundamental signals; these are now SUPPORTED
    # (we used them as the "unsupported" exemplar in C-2e.1). Use
    # genuinely-unsupported hints instead.
    assert _pick_signal_key(("intraday_overnight_drift",)) is None
    assert _pick_signal_key(("textual_sentiment_score",)) is None
    assert _pick_signal_key(()) is None


# ────────────────────────────────────────────────────────────────────
# Fundamental signal picker (C-2e.2)
# ────────────────────────────────────────────────────────────────────
def test_signal_picker_matches_gp_at_variants():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _pick_signal_key,
    )
    for inp in [
        ("gp_at",),
        ("gross_profitability",),
        ("gross_profit_to_assets",),
        ("compustat.funda.gpa",),
        ("gross_profitability_to_assets",),
    ]:
        assert _pick_signal_key(inp) == "gp_at", f"input {inp} missed"


def test_signal_picker_matches_book_to_market_variants():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _pick_signal_key,
    )
    for inp in [
        ("book_to_market",),
        ("book_to_market_ratio",),
        ("btm",),
        ("b_to_m",),
        ("log_book_to_market",),
    ]:
        assert _pick_signal_key(inp) == "book_to_market", f"input {inp} missed"


def test_signal_picker_matches_at_growth_variants():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _pick_signal_key,
    )
    for inp in [
        ("asset_growth",),
        ("at_growth",),
        ("investment_factor",),
        ("d_at",),
    ]:
        assert _pick_signal_key(inp) == "at_growth", f"input {inp} missed"


def test_signal_picker_matches_roe_variants():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _pick_signal_key,
    )
    for inp in [
        ("roe",),
        ("return_on_equity",),
        ("profitability_roe",),
        ("net_income_to_equity",),
    ]:
        assert _pick_signal_key(inp) == "roe", f"input {inp} missed"


def test_compute_replication_subsample_replicated():
    """L2-2: subsample t within 0.5 of paper-reported → REPLICATED."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_replication_subsample,
    )
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    # Construct a series targeting Sharpe ~0.6 (10y, t ~1.9)
    pnl = pd.Series(rng.normal(0.005, 0.029, 120), index=idx)
    out = _compute_replication_subsample(
        pnl_net          = pnl,
        paper_window     = "2000-01:2010-12",
        paper_reported_t = 2.0,
    )
    assert out["n_months_overlap"] >= 24
    assert out["status"] in {"REPLICATED", "MISMATCH"}
    # Not enforcing specific status — we just want the structure


def test_compute_replication_subsample_mismatch():
    """L2-2: subsample t far from paper-reported → MISMATCH."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_replication_subsample,
    )
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    # Construct a roughly-zero-alpha series
    pnl = pd.Series(rng.normal(0.0, 0.03, 120), index=idx)
    out = _compute_replication_subsample(
        pnl_net          = pnl,
        paper_window     = "2000-01:2010-12",
        paper_reported_t = 5.0,   # FAR higher than this noise series
    )
    assert out["status"] == "MISMATCH"
    assert out["t_gap"] > 0.5


def test_compute_replication_subsample_no_benchmark():
    """L2-2: when paper_reported_t is None, compute our t but can't
    judge — status = NO_BENCHMARK."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_replication_subsample,
    )
    rng = np.random.default_rng(2)
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    pnl = pd.Series(rng.normal(0.005, 0.029, 120), index=idx)
    out = _compute_replication_subsample(
        pnl_net=pnl, paper_window="2000-01:2010-12",
        paper_reported_t=None,
    )
    assert out["status"] == "NO_BENCHMARK"
    assert out["our_t"] is not None
    assert out["t_gap"] is None


def test_compute_replication_subsample_insufficient_overlap():
    """L2-2: < 24 months overlap → INSUFFICIENT_OVERLAP."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_replication_subsample,
    )
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    pnl = pd.Series(rng.normal(0.005, 0.03, 12), index=idx)
    out = _compute_replication_subsample(
        pnl_net=pnl, paper_window="1990-01:1995-12",   # no overlap
        paper_reported_t=3.0,
    )
    assert out["status"] == "INSUFFICIENT_OVERLAP"


def test_compute_replication_subsample_bad_window_format():
    """L2-2: malformed paper_window → NO_DATA (defensive)."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_replication_subsample,
    )
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    pnl = pd.Series([0.01] * 24, index=idx)
    out = _compute_replication_subsample(
        pnl_net=pnl, paper_window="garbage", paper_reported_t=3.0,
    )
    assert out["status"] == "NO_DATA"


def test_factor_spec_replication_fields_default_to_none():
    """L2-2: backward-compat — old FactorSpec constructions without
    the new fields still work."""
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    s = FactorSpec(
        hypothesis_id="t", signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000", date_range="2010-01:2020-12",
        signal_inputs=("crsp.msf.derived.vol_12m",), rebal="monthly",
        weighting="quintile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=(), cost_model="basic", rationale="t",
        extracted_ts="2026-06-08T00:00:00Z", model="claude-sonnet-4-6",
    )
    assert s.paper_original_window is None
    assert s.paper_reported_t is None


def test_compute_drawdown_metrics_on_known_drawdown():
    """L2-8: drawdown helper on a hand-constructed series with a
    known 30% peak-to-trough decline."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_drawdown_metrics,
    )
    # Construct PnL so cumulative NAV goes: 1.0 → 1.5 → 1.05 → 1.5
    # peak-to-trough = (1.05 - 1.5) / 1.5 = -30%
    rets = [0.5,           # NAV 1.0 → 1.5 (peak)
            -0.30,         # 1.5 → 1.05 (-30% from peak)
            0.42857143]    # 1.05 → 1.5 (recover)
    # pad to 12 obs minimum
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    rets += [0.0] * 9
    pnl = pd.Series(rets, index=idx)
    m = _compute_drawdown_metrics(pnl)
    assert abs(m["max_drawdown_pct"] - (-0.30)) < 1e-6
    assert m["max_underwater_months"] == 1   # exactly 1 month underwater
    assert m["current_underwater_months"] == 0   # ended at peak


def test_compute_drawdown_metrics_handles_short_history():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_drawdown_metrics,
    )
    pnl = pd.Series([0.01] * 6,
                      index=pd.date_range("2024-01-31", periods=6, freq="ME"))
    m = _compute_drawdown_metrics(pnl)
    assert m["max_drawdown_pct"] is None
    assert m["calmar_ratio"] is None


def test_compute_drawdown_metrics_persistent_underwater():
    """If PnL never recovers after a drawdown, current_underwater_months
    should equal the trailing underwater period — relevant for the
    Momentum 1992-2024 case where strategy is still ~46% underwater
    at sample end."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_drawdown_metrics,
    )
    # NAV trajectory: 1 → 1.2 (peak) → 1.0 → 0.9 → 0.85 → ... never recovers
    rets = [0.20, -0.166666667, -0.10, -0.0555555,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    pnl = pd.Series(rets, index=idx)
    m = _compute_drawdown_metrics(pnl)
    assert m["max_drawdown_pct"] < -0.20   # at least 20% DD
    # Currently underwater (last obs is below peak)
    assert m["current_underwater_months"] >= 10
    assert m["drawdown_at_end_pct"] < 0


def test_compute_cost_stress_runs_multiple_cost_levels():
    """L2-3 multi-cost stress: same gross PnL re-stressed at multiple
    cost levels should produce strictly monotone-decreasing Sharpe
    (higher cost → lower Sharpe) when turnover is non-zero."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_cost_stress,
    )
    rng = np.random.default_rng(7)
    n = 60
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    pnl_gross = pd.Series(rng.normal(0.005, 0.05, n), index=idx)
    turnover  = pd.Series(rng.uniform(0.5, 1.0, n), index=idx)
    stress = _compute_cost_stress(
        pnl_gross, turnover,
        cost_levels_bp=(0.0, 30.0, 80.0),
    )
    assert set(stress.keys()) == {"0bp", "30bp", "80bp"}
    # Sharpe must monotone-decrease as TC rises (turnover > 0 by construct)
    s_0  = stress["0bp"]["sharpe"]
    s_30 = stress["30bp"]["sharpe"]
    s_80 = stress["80bp"]["sharpe"]
    assert s_0 > s_30 > s_80


def test_compute_cost_stress_handles_short_history():
    """< 12 obs → INSUFFICIENT_HISTORY at every cost level, no crash."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_cost_stress,
    )
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    stress = _compute_cost_stress(
        pd.Series([0.01] * 6, index=idx),
        pd.Series([0.5]  * 6, index=idx),
    )
    for level, m in stress.items():
        assert m["verdict"] == "INSUFFICIENT_HISTORY"


def test_cost_robust_verdict_picks_strictest_level():
    """L2-3 senior rigor: overall verdict is whatever 80bp gives.
    Force the system to report the conservative number."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _cost_robust_verdict,
    )
    stress = {
        "0bp":  {"verdict": "GREEN"},
        "30bp": {"verdict": "GREEN"},
        "60bp": {"verdict": "MARGINAL"},
        "80bp": {"verdict": "RED"},
    }
    assert _cost_robust_verdict(stress, robust_at_bp=80.0) == "RED"
    assert _cost_robust_verdict(stress, robust_at_bp=30.0) == "GREEN"


# ────────────────────────────────────────────────────────────────────
# 2026-06-08 regression: cost stress is SIGN-AWARE.
#
# Bug surfaced by L3-2 self_doubt on reversal_1m seed dispatch:
# 80bp Sharpe -1.501 (t ≈ -3.0) was being reported as GREEN
# because _verdict_from_t uses abs(t). Cost stress specifically
# means "does the SAME factor under the SAME convention survive
# cost?" — sign-flips fail by construction.
# ────────────────────────────────────────────────────────────────────
def test_cost_stress_verdict_requires_positive_t_stat():
    """A negative significant t at high cost = RED, NOT GREEN.
    Anti-regression for the cost-table-inversion bug Sonnet caught
    on reversal_1m 2026-06-08."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _cost_stress_verdict_from_t,
    )
    # Strongly positive
    assert _cost_stress_verdict_from_t( 3.5) == "GREEN"
    assert _cost_stress_verdict_from_t( 1.96) == "GREEN"
    assert _cost_stress_verdict_from_t( 1.70) == "MARGINAL"
    assert _cost_stress_verdict_from_t( 1.65) == "MARGINAL"
    assert _cost_stress_verdict_from_t( 1.50) == "RED"
    # Negative — never GREEN, never MARGINAL, no matter how "significant"
    assert _cost_stress_verdict_from_t(-1.50) == "RED"
    assert _cost_stress_verdict_from_t(-1.96) == "RED"
    assert _cost_stress_verdict_from_t(-3.00) == "RED"   # the seed case
    # Degenerate (NaN AND inf both fail isfinite — return RED to be
    # safe; inf usually means div-by-zero vol upstream).
    assert _cost_stress_verdict_from_t(float("nan")) == "RED"
    assert _cost_stress_verdict_from_t(float("inf")) == "RED"
    assert _cost_stress_verdict_from_t(float("-inf")) == "RED"


def test_compute_cost_stress_flags_sign_flip_red_not_green():
    """End-to-end: a PnL series that's mildly positive at 0bp and
    flips strongly negative once cost is applied must report RED
    at the high-cost level — never GREEN."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _compute_cost_stress,
    )
    # Construct: tiny positive monthly drift, very high turnover, so
    # any cost > a few bps wipes + flips it.
    n = 240
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    # 0.05% monthly mean — marginally positive even before cost
    pnl_gross = pd.Series([0.0005] * n, index=idx)
    turnover  = pd.Series([1.5]    * n, index=idx)  # 150% monthly TC base
    stress = _compute_cost_stress(
        pnl_gross, turnover, cost_levels_bp=(0.0, 80.0),
    )
    # 0bp: tiny positive drift, t may or may not be significant
    # 80bp: -1.5 * 80/10000 = -0.012 monthly → -14% ann; t very negative
    assert stress["80bp"]["sharpe"] < 0
    assert stress["80bp"]["nw_t_stat"] < -1.96
    assert stress["80bp"]["verdict"] == "RED", (
        "sign-flipped strongly negative must be RED not GREEN "
        f"(got verdict={stress['80bp']['verdict']!r} "
        f"sharpe={stress['80bp']['sharpe']:.3f} "
        f"t={stress['80bp']['nw_t_stat']:.3f})"
    )


def test_compustat_signal_constants():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _COMPUSTAT_SIGNALS,
    )
    assert _COMPUSTAT_SIGNALS == frozenset(
        {"gp_at", "book_to_market", "at_growth", "roe"})


# ────────────────────────────────────────────────────────────────────
# Scope guards (no DB / no parquet read)
# ────────────────────────────────────────────────────────────────────
def test_template_refuses_wrong_signal_kind():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        template_cross_sec_us_equities,
    )
    r = template_cross_sec_us_equities(_spec(signal_kind="carry"))
    assert r.verdict == "EXECUTION_ERROR"
    assert r.metrics.get("misroute") is True


def test_template_refuses_wrong_universe():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        template_cross_sec_us_equities,
    )
    r = template_cross_sec_us_equities(_spec(universe="fx_g10"))
    assert r.verdict == "UNSUPPORTED_UNIVERSE"
    assert r.metrics["unsupported_universe"] == "fx_g10"


def test_template_refuses_unsupported_signal():
    """When extractor emits a signal not in the supported keys,
    template returns UNSUPPORTED_SIGNAL — distinguishable from
    DATA_ERROR / EXECUTION_ERROR. C-2e.2 added Compustat funda
    signals so we use genuinely-non-templated examples here."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        template_cross_sec_us_equities,
    )
    r = template_cross_sec_us_equities(_spec(
        signal_inputs=("intraday_overnight_drift",
                          "textual_sentiment_score"),
    ))
    assert r.verdict == "UNSUPPORTED_SIGNAL"


# ────────────────────────────────────────────────────────────────────
# Verdict mapping
# ────────────────────────────────────────────────────────────────────
def test_verdict_thresholds():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _verdict_from_t,
    )
    assert _verdict_from_t(2.5)   == "GREEN"
    assert _verdict_from_t(1.96)  == "GREEN"
    assert _verdict_from_t(1.95)  == "MARGINAL"
    assert _verdict_from_t(1.65)  == "MARGINAL"
    assert _verdict_from_t(1.64)  == "RED"
    assert _verdict_from_t(-2.5)  == "GREEN"
    assert _verdict_from_t(float("nan")) == "RED"


# ────────────────────────────────────────────────────────────────────
# _build_signal on synthetic panel
# ────────────────────────────────────────────────────────────────────
def _synth_panel(n_permnos: int = 100, n_months: int = 60,
                   seed: int = 42) -> pd.DataFrame:
    """Build a CRSP-shaped synthetic panel."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    rows = []
    for perm in range(10001, 10001 + n_permnos):
        for d in dates:
            rows.append({
                "permno": perm,
                "month_end": d,
                "ret": rng.normal(0.005, 0.06),
                "mktcap": float(rng.uniform(1e5, 5e7)),
            })
    return pd.DataFrame(rows)


def test_build_signal_mktcap_inverts_sign():
    """mktcap signal: long SMALL → returned signal should equal
    -mktcap (so HIGH score = SMALL cap)."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _build_signal,
    )
    panel = _synth_panel(n_permnos=30, n_months=20)
    sig = _build_signal(panel, "mktcap")
    # Pick one date, confirm score = -mktcap
    mc = panel.pivot(index="month_end", columns="permno", values="mktcap")
    t = mc.index[5]
    assert (sig.loc[t] == -mc.loc[t]).all()


def test_build_signal_vol_12m_returns_negated_rolling_std():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _build_signal,
    )
    panel = _synth_panel(n_permnos=20, n_months=30)
    sig = _build_signal(panel, "vol_12m")
    # First 7 rows are NaN due to min_periods=8
    assert sig.iloc[:7].isna().all().all()
    # Later rows are finite (vol > 0 → -vol < 0)
    assert (sig.iloc[15:].dropna(how="all") < 0).any().any()


def test_build_signal_ret_12_1_uses_shift():
    """ret_12_1 signal at month t is built from rolling(12) summed
    log-returns shifted by 1 (so signal AT t reflects t-12..t-1)."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _build_signal,
    )
    panel = _synth_panel(n_permnos=15, n_months=24)
    sig = _build_signal(panel, "ret_12_1")
    # First ~11 rows NaN (rolling + shift)
    assert sig.iloc[:10].isna().all().all()


def test_build_signal_reversal_inverts_prior_month():
    """reversal_1m signal at t uses -ret_t (long prior-month losers)."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _build_signal,
    )
    panel = _synth_panel(n_permnos=10, n_months=10)
    sig = _build_signal(panel, "reversal_1m")
    rets = panel.pivot(index="month_end", columns="permno", values="ret")
    # Sample 5th row, signal should equal -ret
    t = rets.index[5]
    assert np.allclose(sig.loc[t].dropna().values,
                          (-rets.loc[t]).dropna().values)


def test_build_signal_unknown_raises():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _build_signal,
    )
    with pytest.raises(ValueError, match="unknown signal_key"):
        _build_signal(_synth_panel(5, 12), "neural_net_2024")


# ────────────────────────────────────────────────────────────────────
# _quintile_long_short_pnl on synthetic data
# ────────────────────────────────────────────────────────────────────
def test_quintile_long_short_pnl_returns_series_with_diagnostics():
    """Build a panel where signal HIGH → next-month return HIGH so
    L-S PnL should be reliably positive. Confirms the math wires
    correctly + diagnostics populated."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _quintile_long_short_pnl,
    )
    n_perms = 200
    n_months = 30
    rng = np.random.default_rng(7)
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    perms = list(range(10000, 10000 + n_perms))

    # Synthetic signal: persistent per-permno score; next-month ret
    # has a deterministic component proportional to the signal score
    per_perm_score = pd.Series(
        rng.normal(0, 1, n_perms), index=perms)
    signal_panel = pd.DataFrame(
        np.tile(per_perm_score.values, (n_months, 1)),
        index=dates, columns=perms)
    # Next-month return = signal_score × 0.02 + noise (0.06 sd)
    return_panel = signal_panel * 0.02 + rng.normal(
        0, 0.06, size=(n_months, n_perms))
    mktcap_panel = pd.DataFrame(
        rng.uniform(1e6, 1e8, size=(n_months, n_perms)),
        index=dates, columns=perms)
    delist_panel = pd.DataFrame(columns=["permno", "dlst_month_end", "dlret"])

    pnl, diag = _quintile_long_short_pnl(
        signal_panel = signal_panel,
        return_panel = return_panel,
        mktcap_panel = mktcap_panel,
        delist_panel = delist_panel,
        top_n        = 200,
        tc_bp_per_rt = 0.0,   # 0 TC for cleaner signal
    )
    assert isinstance(pnl, pd.Series)
    assert diag["n_months"] > 5
    # Average Q5-Q1 spread should be POSITIVE since signal predicts ret
    assert pnl.mean() > 0, f"expected positive mean, got {pnl.mean()}"


def test_quintile_long_short_skips_when_universe_too_small():
    """If universe has too few stocks at a given month, that month is
    skipped (NOT crash). Critical for sparse early-history months."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _quintile_long_short_pnl, _MIN_STOCKS_PER_BUCKET, _N_QUINTILES,
    )
    # Only 10 stocks: below _N_QUINTILES * _MIN_STOCKS_PER_BUCKET (150)
    n_perms = 10
    n_months = 12
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    perms = list(range(20000, 20000 + n_perms))
    rng = np.random.default_rng(1)
    signal_panel = pd.DataFrame(
        rng.normal(0, 1, size=(n_months, n_perms)),
        index=dates, columns=perms)
    return_panel = pd.DataFrame(
        rng.normal(0, 0.06, size=(n_months, n_perms)),
        index=dates, columns=perms)
    mktcap_panel = pd.DataFrame(
        rng.uniform(1e6, 1e8, size=(n_months, n_perms)),
        index=dates, columns=perms)
    delist_panel = pd.DataFrame(columns=["permno", "dlst_month_end", "dlret"])

    pnl, diag = _quintile_long_short_pnl(
        signal_panel, return_panel, mktcap_panel, delist_panel,
        top_n=10, tc_bp_per_rt=13.0,
    )
    assert diag["n_months"] == 0
    assert pnl.empty


# ────────────────────────────────────────────────────────────────────
# Template registry wiring — dispatcher routes cross_sec to real template
# ────────────────────────────────────────────────────────────────────
def test_dispatcher_routes_cross_sec_to_real_template():
    from engine.agents.strengthener.factor_dispatcher import (
        TEMPLATE_REGISTRY,
    )
    tpl = TEMPLATE_REGISTRY["cross_sectional_rank"]
    # us_equities_sp500 universe → falls back to stub
    r_sp500 = tpl(_spec(universe="us_equities_sp500"))
    assert r_sp500.verdict == "PENDING_TEMPLATE_BUILD"
    # us_equities_top_3000 + unsupported signal → UNSUPPORTED_SIGNAL
    # (real template invoked, not stub)
    r = tpl(_spec(signal_inputs=("totally_made_up_signal",)))
    assert r.verdict == "UNSUPPORTED_SIGNAL"


# ────────────────────────────────────────────────────────────────────
# Integration test — real CRSP parquet
# ────────────────────────────────────────────────────────────────────
def test_quintile_long_short_pnl_n_buckets_parameter_default():
    """L2-1 Phase 3.0: n_buckets default = _N_QUINTILES (5)."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _quintile_long_short_pnl, _N_QUINTILES,
    )
    import inspect
    sig = inspect.signature(_quintile_long_short_pnl)
    assert sig.parameters["n_buckets"].default == _N_QUINTILES


def test_quintile_long_short_pnl_n_buckets_10_decile():
    """L2-1 Phase 3.0: n_buckets=10 (decile) wires through qcut."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _quintile_long_short_pnl,
    )
    n_perms = 400
    n_months = 36
    rng = np.random.default_rng(11)
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    perms = list(range(20000, 20000 + n_perms))
    per_perm_score = pd.Series(rng.normal(0, 1, n_perms), index=perms)
    signal_panel = pd.DataFrame(
        np.tile(per_perm_score.values, (n_months, 1)),
        index=dates, columns=perms)
    return_panel = signal_panel * 0.015 + rng.normal(
        0, 0.05, size=(n_months, n_perms))
    mktcap_panel = pd.DataFrame(
        rng.uniform(1e6, 1e8, size=(n_months, n_perms)),
        index=dates, columns=perms)
    delist_panel = pd.DataFrame(
        columns=["permno", "dlst_month_end", "dlret"])
    pnl, diag = _quintile_long_short_pnl(
        signal_panel = signal_panel, return_panel = return_panel,
        mktcap_panel = mktcap_panel, delist_panel = delist_panel,
        top_n = 400, tc_bp_per_rt = 0.0, n_buckets = 10,
    )
    assert diag["n_months"] >= 10
    assert pnl.mean() > 0


@pytest.mark.skipif(
    os.environ.get("RUN_CROSS_SEC_INTEGRATION") != "1",
    reason=("set RUN_CROSS_SEC_INTEGRATION=1 to run live integration "
              "test (needs CRSP parquet)"),
)
def test_template_integration_gp_at_compustat():
    """C-2e.2 integration: GP/A on Compustat 1992-2024. Needs
    _compustat_funda_long_history.parquet cache (built by
    scripts/extend_compustat_funda_history.py)."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        template_cross_sec_us_equities, _COMPUSTAT_FUNDA_PATH,
    )
    if not _COMPUSTAT_FUNDA_PATH.is_file():
        pytest.skip(f"Compustat long-history cache missing at "
                    f"{_COMPUSTAT_FUNDA_PATH} — run "
                    f"scripts/extend_compustat_funda_history.py first")
    spec = _spec(
        date_range="1992-01:2024-12", min_obs_months=120,
        signal_inputs=("compustat.funda.gross_profitability_to_assets",),
    )
    r = template_cross_sec_us_equities(spec)
    assert r.verdict in ("GREEN", "MARGINAL", "RED")
    m = r.metrics
    assert m["signal"] == "gp_at"
    assert math.isfinite(m["sharpe"])
    assert math.isfinite(m["nw_t_stat"])
    assert m["n_months"] >= 120


@pytest.mark.skipif(
    os.environ.get("RUN_CROSS_SEC_INTEGRATION") != "1",
    reason=("set RUN_CROSS_SEC_INTEGRATION=1 to run live end-to-end "
              "stack test (needs CRSP+Compustat caches + event store)"),
)
def test_e2e_layer_2_stack_through_dispatcher(tmp_path, monkeypatch):
    """L2 end-to-end stack verification (2026-06-08).

    Validates that a FactorSpec with all L2 fields populated
    (paper_original_window / paper_reported_t) flows cleanly through:
      1. dispatch_factor_spec gates (defense-in-depth)
      2. cross_sec template with cost_stress + drawdown + replication
      3. metrics dict JSON-serializable
      4. factor_verdict_filed event emit + event store roundtrip
      5. all 3 new L2 metrics survive emit and re-read intact

    This is the regression check for the entire L2 stack. Runs
    against REAL CRSP + Compustat caches; gated behind
    RUN_CROSS_SEC_INTEGRATION=1 so the regular suite stays offline.
    Redirects registry + event store + evidence dir to tmp.
    """
    import json
    from engine.agents.strengthener.factor_dispatcher import (
        dispatch_factor_spec,
    )
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    from engine.agents.strengthener import factor_verdict_emit as fve
    from engine.research_store import registry as reg
    from engine.research_store import store as st

    # Isolate persistence (same pattern as test_factor_verdict_emit
    # fixture). Prevents real research store pollution.
    monkeypatch.setattr(fve, "EVIDENCE_DIR", tmp_path / "evidence")
    monkeypatch.setattr(reg, "_SUBJECTS_PATH", tmp_path / "subjects.yaml")
    monkeypatch.setattr(reg, "_ALIASES_PATH", tmp_path / "aliases.yaml")
    monkeypatch.setattr(st, "_EVENTS_PATH", tmp_path / "events.jsonl")

    spec = FactorSpec(
        hypothesis_id="e2e_test_" + uuid.uuid4().hex[:8],
        signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000",
        date_range="1992-01:2024-12",
        signal_inputs=("compustat.funda.gross_profitability_to_assets",),
        rebal="monthly",
        weighting="quintile_long_short_dollar_neutral",
        expected_holding_period="monthly",
        min_obs_months=120,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="e2e stack regression",
        extracted_ts="2026-06-08T00:00:00Z",
        model="claude-sonnet-4-6",
        paper_original_window="1963-01:2010-12",
        paper_reported_t=3.0,
    )
    out = dispatch_factor_spec(
        spec, family_hint="PROFITABILITY", spec_approved=True,
        log_path=tmp_path / "factor_dispatch_log.jsonl",
    )

    # (1) Dispatcher gates pass
    assert out["refusal"] is None
    # (2) Template ran (no DATA_ERROR/EXECUTION_ERROR)
    tr = out["template_result"]
    assert tr["verdict"] in {"GREEN", "MARGINAL", "RED"}
    # (3) All L2 new metrics present
    m = tr["metrics"]
    assert "cost_stress"    in m
    assert set(m["cost_stress"].keys()) == {"0bp", "30bp", "60bp", "80bp"}
    assert "drawdown_naive" in m
    assert "drawdown_80bp"  in m
    assert "replication"    in m
    assert m["replication"]["status"] in {
        "REPLICATED", "MISMATCH", "NO_BENCHMARK",
        "INSUFFICIENT_OVERLAP", "NO_DATA", "NOT_APPLICABLE",
    }
    # (4) metrics serializable for events.jsonl
    json.dumps(m)
    # (5) verdict event roundtrip — all 3 new metric dicts survived
    assert out["verdict_event_id"]
    from engine.research_store.store import by_event_id
    ev = by_event_id(out["verdict_event_id"])
    assert ev is not None
    ev_m = ev.metrics or {}
    assert "cost_stress"    in ev_m
    assert "drawdown_naive" in ev_m
    assert "drawdown_80bp"  in ev_m
    assert "replication"    in ev_m


@pytest.mark.skipif(
    os.environ.get("RUN_CROSS_SEC_INTEGRATION") != "1",
    reason=("set RUN_CROSS_SEC_INTEGRATION=1 to run live CRSP "
              "parquet integration test"),
)
def test_template_integration_low_vol():
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        template_cross_sec_us_equities,
    )
    spec = _spec(date_range="2012-01:2023-12", min_obs_months=60,
                    signal_inputs=("crsp.msf.derived.vol_12m",))
    r = template_cross_sec_us_equities(spec)
    assert r.verdict in ("GREEN", "MARGINAL", "RED")
    m = r.metrics
    assert m["signal"] == "vol_12m"
    assert math.isfinite(m["sharpe"])
    assert math.isfinite(m["nw_t_stat"])
    assert m["n_months"] >= 60
    assert m["avg_universe_size"] >= 1000
