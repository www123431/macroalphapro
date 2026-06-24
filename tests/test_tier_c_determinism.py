"""tests/test_tier_c_determinism.py — 4th test blind spot.

Same inputs → byte-identical outputs. Statistical inference is
meaningless if the pipeline is non-deterministic: a verdict that
flips between runs on identical data is a bug, full stop. Sources
of accidental non-determinism this guards against:
  - dict/set iteration order leaking into numerics
  - unseeded randomness in any lens
  - float accumulation order changes (e.g. parallelism)
  - pandas version-dependent groupby ordering

Strategy: run each lens TWICE on the same in-memory inputs and
deep-compare the outputs after JSON canonicalization (sorted keys,
repr-stable floats). Uses the real GP/A parquet when cached.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


GP_A_PARQUET = (Path(__file__).resolve().parents[1]
                  / "data" / "research_store" / "tier_c_pnl"
                  / "dc4cf6beaa247880_GREEN.parquet")


def _canon(obj) -> str:
    """Canonical JSON: sorted keys, NaN-safe, float repr-stable."""
    def _default(v):
        if isinstance(v, float):
            return repr(v)
        return str(v)
    return json.dumps(obj, sort_keys=True, default=_default)


@pytest.fixture
def gp_a_artifacts():
    if not GP_A_PARQUET.is_file():
        pytest.skip("GP/A fixture not cached")
    df = pd.read_parquet(GP_A_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return {
        "pnl_series_df":   df,
        "pnl_default_col": "pnl_net_13bp",
        "pnl_gross_col":   "pnl_gross",
    }


def test_anchor_regression_deterministic(gp_a_artifacts):
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    a = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts)
    b = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts)
    assert _canon(a) == _canon(b)


def test_subsample_stability_deterministic(gp_a_artifacts):
    from engine.research.subsample_stability import (
        compute_for_tier_c_pnl_series,
    )
    a = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], n_splits=4,
        artifacts=gp_a_artifacts)
    b = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], n_splits=4,
        artifacts=gp_a_artifacts)
    assert _canon(a) == _canon(b)


def test_industry_extension_deterministic(gp_a_artifacts):
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series as ar,
    )
    from engine.research.industry_attribution import (
        compute_for_tier_c_with_stage1_residual as ix,
    )
    s1 = ar(gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts)
    a = ix(s1, gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts)
    b = ix(s1, gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts)
    assert _canon(a) == _canon(b)


def test_cross_asset_extension_deterministic(gp_a_artifacts):
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series as ar,
    )
    from engine.research.cross_asset_attribution import (
        compute_for_tier_c_with_macro as xa,
    )
    s1 = ar(gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts)
    a = xa(s1, None, gp_a_artifacts["pnl_series_df"],
            include_industry=True, artifacts=gp_a_artifacts)
    b = xa(s1, None, gp_a_artifacts["pnl_series_df"],
            include_industry=True, artifacts=gp_a_artifacts)
    assert _canon(a) == _canon(b)


def test_decay_evaluation_deterministic():
    from engine.research.decay_watch_trigger import (
        evaluate_subsample_for_decay,
    )
    sub = {
        "n_splits": 4, "n_total_months": 240,
        "windows": [
            {"start": "2008-01", "end": "2012-12", "n_months": 60,
              "sharpe_ann": 1.5, "nw_t_stat": 4.0,
              "ann_return": 0.06, "ann_vol": 0.04},
            {"start": "2013-01", "end": "2017-12", "n_months": 60,
              "sharpe_ann": 0.2, "nw_t_stat": 0.5,
              "ann_return": 0.008, "ann_vol": 0.04},
        ],
        "worst_best_sharpe_ratio": 0.13,
        "institutional_stable": False,
        "monotone_decay": True, "monotone_growth": False,
        "decay_slope_per_year": None, "decay_slope_t": None,
    }
    assert _canon(evaluate_subsample_for_decay(sub)) == \
           _canon(evaluate_subsample_for_decay(sub))


def test_spec_robustness_deterministic():
    """Stubbed template (no randomness) → identical ablation output."""
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    spec = FactorSpec(
        hypothesis_id="det", signal_kind="time_series_momentum",
        universe="us_equities_sector_etf", date_range="2014-01:2024-12",
        signal_inputs=("etf.adj_close.spy",), rebal="weekly",
        weighting="signed_signal_volatility_targeted",
        expected_holding_period="weekly", min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="det", extracted_ts="2026-06-10T00:00:00Z",
        model="c", investment_role="alpha",
        statistical_role="directional", asset_class="equity",
        mechanism="momentum", horizon="monthly",
        capacity_tier="100m_to_1b", data_dependency_type="market",
        regime_sensitivity="known_regime_break",
        signal_lookback_m=12, signal_skip_m=1, vol_target_annual=0.10,
    )
    def _tmpl(s):
        sharpe = 1.5 - abs((s.signal_lookback_m or 12) - 12) * 0.15
        return TemplateResult(
            verdict="GREEN", summary="d",
            metrics={"sharpe": sharpe, "nw_t_stat": sharpe * 2,
                      "n_months": 120},
            artifacts={}, template_version="d")
    base = _tmpl(spec)
    a = compute_specification_robustness(spec, _tmpl, base)
    b = compute_specification_robustness(spec, _tmpl, base)
    assert _canon(a) == _canon(b)
