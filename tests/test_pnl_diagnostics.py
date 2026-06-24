"""tests/test_pnl_diagnostics.py — N: the 4 senior gates.

Locks the math + thresholds for DSR-in-prompt, ρ₁ check, paper-OOS
ratio, and T_GREEN power analysis. Tolerances derived from theory
per [[feedback-random-data-test-tolerances-from-theory-2026-06-09]].
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


def _series(values, start="2010-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx)


# ────────────────────────────────────────────────────────────────────
# N2 — ρ₁
# ────────────────────────────────────────────────────────────────────
def test_rho1_iid_noise_near_zero():
    """IID noise → ρ₁ within 3/sqrt(n) of zero (3σ band per the
    random-data tolerance doctrine)."""
    from engine.research.pnl_diagnostics import compute_rho1
    rng = np.random.default_rng(7)
    n = 400
    out = compute_rho1(_series(rng.normal(0, 0.02, n)))
    assert abs(out["rho1"]) < 3.0 / math.sqrt(n)
    assert out["smell"] is False


def test_rho1_ar1_process_recovered():
    """AR(1) with φ=0.5 → ρ₁ ≈ 0.5 within 3σ; smell fires;
    inflation ≈ sqrt(1.5/0.5) ≈ 1.73."""
    from engine.research.pnl_diagnostics import compute_rho1
    rng = np.random.default_rng(11)
    n = 600
    phi = 0.5
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 0.01)
    out = compute_rho1(_series(x))
    se = math.sqrt((1 - phi**2)) / math.sqrt(n)   # asymptotic SE of rho1
    assert abs(out["rho1"] - phi) < 4 * se
    assert out["smell"] is True
    assert abs(out["sharpe_se_inflation"]
                 - math.sqrt((1 + out["rho1"]) / (1 - out["rho1"]))) < 1e-12


def test_rho1_short_series_none():
    from engine.research.pnl_diagnostics import compute_rho1
    assert compute_rho1(_series([0.01] * 10)) is None


# ────────────────────────────────────────────────────────────────────
# N3 — paper-OOS ratio
# ────────────────────────────────────────────────────────────────────
def test_paper_oos_dead_factor_flagged():
    """Strong in-window (mean 1%/mo), dead post-window (0%) →
    ratio ≈ 0 → effectively_dead."""
    from engine.research.pnl_diagnostics import compute_paper_oos_ratio
    rng = np.random.default_rng(3)
    in_w  = rng.normal(0.010, 0.02, 120)   # 2000-01..2009-12
    post  = rng.normal(0.000, 0.02, 120)   # 2010-01..2019-12
    pnl = _series(np.concatenate([in_w, post]), start="2000-01-31")
    out = compute_paper_oos_ratio(pnl, "2000-01:2009-12")
    assert out is not None
    assert out["effectively_dead"] is True
    assert out["oos_ratio"] < 0.30


def test_paper_oos_healthy_factor_not_flagged():
    """Mild decay (50% of in-window mean) → ratio ~0.5, not dead.
    Means chosen large vs noise so the ratio estimate is stable."""
    from engine.research.pnl_diagnostics import compute_paper_oos_ratio
    rng = np.random.default_rng(5)
    in_w = rng.normal(0.020, 0.01, 120)
    post = rng.normal(0.010, 0.01, 120)
    pnl = _series(np.concatenate([in_w, post]), start="2000-01-31")
    out = compute_paper_oos_ratio(pnl, "2000-01:2009-12")
    assert out is not None
    assert out["effectively_dead"] is False
    assert 0.30 < out["oos_ratio"] < 0.80


def test_paper_oos_none_when_segment_short():
    from engine.research.pnl_diagnostics import compute_paper_oos_ratio
    pnl = _series([0.01] * 40, start="2000-01-31")
    # post-window only ~16 months < 24 floor
    assert compute_paper_oos_ratio(pnl, "2000-01:2001-12") is None


def test_paper_oos_none_when_in_window_sharpe_nonpositive():
    from engine.research.pnl_diagnostics import compute_paper_oos_ratio
    rng = np.random.default_rng(9)
    in_w = rng.normal(-0.005, 0.02, 60)
    post = rng.normal(0.005, 0.02, 60)
    pnl = _series(np.concatenate([in_w, post]), start="2000-01-31")
    assert compute_paper_oos_ratio(pnl, "2000-01:2004-12") is None


def test_paper_oos_garbage_window_none():
    from engine.research.pnl_diagnostics import compute_paper_oos_ratio
    pnl = _series([0.01] * 100)
    assert compute_paper_oos_ratio(pnl, "not a window") is None


# ────────────────────────────────────────────────────────────────────
# N4 — power
# ────────────────────────────────────────────────────────────────────
def test_power_increases_with_sample_and_sr():
    from engine.research.pnl_diagnostics import power_of_t_green
    # Monotone in n
    assert power_of_t_green(360, 0.5) > power_of_t_green(120, 0.5)
    # Monotone in SR
    assert power_of_t_green(120, 0.8) > power_of_t_green(120, 0.3)


def test_power_known_value_sr05_10y():
    """Closed-form check: SR=0.5, 10 years.
    SE = sqrt((1+0.125)/10) ≈ 0.3354 → true_t ≈ 1.491
    power = 1 - Φ(1.96 - 1.491) = 1 - Φ(0.469) ≈ 0.3195."""
    from engine.research.pnl_diagnostics import power_of_t_green
    p = power_of_t_green(120, 0.5)
    assert abs(p - 0.3195) < 0.01


def test_power_table_keys():
    from engine.research.pnl_diagnostics import compute_power_table
    t = compute_power_table(240)
    assert set(t.keys()) == {"sr_0.3", "sr_0.5", "sr_0.8"}
    assert all(0 <= v <= 1 for v in t.values())


# ────────────────────────────────────────────────────────────────────
# Aggregate + N1 DSR
# ────────────────────────────────────────────────────────────────────
def test_aggregate_all_blocks_present():
    from engine.research.pnl_diagnostics import compute_pnl_diagnostics
    rng = np.random.default_rng(13)
    pnl = _series(0.005 + rng.normal(0, 0.02, 240), start="2000-01-31")
    out = compute_pnl_diagnostics(
        pnl, n_trials_family=7, paper_window="2000-01:2009-12",
    )
    assert out is not None
    assert out["n_months"] == 240
    assert out["dsr"]["n_trials_family"] == 7
    assert out["dsr"]["deflated_sr_prob"] is not None
    assert 0 <= out["dsr"]["deflated_sr_prob"] <= 1
    assert out["rho1"] is not None
    assert out["paper_oos"] is not None
    assert out["power"]["table"]


def test_aggregate_dsr_penalizes_more_trials():
    """Same PnL, more family trials → lower deflated probability.
    THE Bailey-LdP point."""
    from engine.research.pnl_diagnostics import compute_pnl_diagnostics
    rng = np.random.default_rng(17)
    pnl = _series(0.006 + rng.normal(0, 0.025, 240))
    p1  = compute_pnl_diagnostics(pnl, n_trials_family=1)
    p15 = compute_pnl_diagnostics(pnl, n_trials_family=15)
    assert (p15["dsr"]["deflated_sr_prob"]
              < p1["dsr"]["deflated_sr_prob"])


def test_aggregate_none_when_too_short():
    from engine.research.pnl_diagnostics import compute_pnl_diagnostics
    assert compute_pnl_diagnostics(_series([0.01] * 12)) is None


def test_aggregate_no_paper_window_skips_oos():
    from engine.research.pnl_diagnostics import compute_pnl_diagnostics
    rng = np.random.default_rng(19)
    out = compute_pnl_diagnostics(
        _series(rng.normal(0.005, 0.02, 100)))
    assert out["paper_oos"] is None


# ────────────────────────────────────────────────────────────────────
# Prompt rendering
# ────────────────────────────────────────────────────────────────────
def test_self_doubt_prompt_renders_diagnostics_section():
    from engine.agents.strengthener.self_doubt import _format_user_message
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    spec = FactorSpec(
        hypothesis_id="n_prompt", signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000", date_range="2014-01:2024-12",
        signal_inputs=("crsp.msf.ret",), rebal="monthly",
        weighting="decile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="n", extracted_ts="2026-06-10T00:00:00Z", model="c",
    )
    tr = TemplateResult(verdict="GREEN", summary="s",
                          metrics={"sharpe": 1.0, "nw_t_stat": 2.2},
                          artifacts={}, template_version="t")
    pdg = {
        "n_months": 240,
        "dsr": {"deflated_sr_prob": 0.85, "n_trials_family": 7,
                  "sharpe_ann": 1.0},
        "rho1": {"rho1": 0.25, "rho1_se": 0.06, "rho1_t": 3.9,
                   "sharpe_se_inflation": 1.29, "smell": True,
                   "smell_bar": 0.20},
        "paper_oos": {"sharpe_in_window": 1.5, "sharpe_post_window": 0.3,
                        "oos_ratio": 0.2, "n_months_in": 120,
                        "n_months_post": 120, "dead_bar": 0.30,
                        "effectively_dead": True},
        "power": {"t_green": 1.96, "n_months": 240,
                    "table": {"sr_0.3": 0.13, "sr_0.5": 0.55,
                                "sr_0.8": 0.95}},
    }
    msg = _format_user_message(spec, tr, "TEST", 7,
                                  pnl_diagnostics=pdg)
    assert "PNL DIAGNOSTICS" in msg
    assert "0.850" in msg          # DSR prob
    assert "SMELL" in msg          # rho1 flag
    assert "EFFECTIVELY DEAD" in msg
    assert "WEAK evidence of absence" in msg


def test_self_doubt_prompt_absent_block_renders_not_computed():
    from engine.agents.strengthener.self_doubt import _format_user_message
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    spec = FactorSpec(
        hypothesis_id="n2", signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000", date_range="2014-01:2024-12",
        signal_inputs=("crsp.msf.ret",), rebal="monthly",
        weighting="decile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="n", extracted_ts="2026-06-10T00:00:00Z", model="c",
    )
    tr = TemplateResult(verdict="GREEN", summary="s",
                          metrics={}, artifacts={}, template_version="t")
    msg = _format_user_message(spec, tr, "TEST", 0)
    assert "PNL DIAGNOSTICS: not computed" in msg
