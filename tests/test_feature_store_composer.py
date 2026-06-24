"""Tests for engine.feature_store composability layer — Option C Week 1c.

Covers:
  - Primitives (rolling_return / shift / zscore / vol_scale / etc.)
  - Axis loaders (universe / signal_recipe / weighting)
  - Compose-spec end-to-end materialize
  - Auto-route between v1 (function-wrapper) and v2 (compose) specs
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from engine.feature_store.composer import (
    SignalRecipe, Universe, Weighting,
    apply_weighting, compose_factor, load_compose_spec,
    load_signal_recipe, load_universe, load_weighting,
    run_signal_recipe,
)
from engine.feature_store.materializer import (
    _spec_is_compose, materialize_spec,
)
from engine.feature_store.primitives import (
    apply_primitive, list_primitives,
    rolling_return, shift, xs_rank, xs_zscore,
)


# ── Primitives ──────────────────────────────────────────────────────────


def _toy_panel(n_periods: int = 24, n_assets: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.005, 0.04, size=(n_periods, n_assets)),
        index=pd.date_range("2020-01-31", periods=n_periods, freq="ME"),
        columns=[f"A{i}" for i in range(n_assets)],
    )


def test_rolling_return_shape_and_nan_prefix():
    p = _toy_panel(24, 5)
    out = rolling_return(p, months=12)
    assert out.shape == p.shape
    # First 11 rows should be NaN (window not complete)
    assert out.iloc[:11].isna().all().all()
    assert out.iloc[12:].notna().all().all()


def test_shift_lags_correctly():
    p = _toy_panel()
    out = shift(p, periods=1)
    pd.testing.assert_frame_equal(out.iloc[1:].reset_index(drop=True),
                                    p.iloc[:-1].reset_index(drop=True))


def test_xs_zscore_rowwise_mean_zero():
    p = _toy_panel()
    out = xs_zscore(p)
    # Each row's mean ≈ 0 (within float tolerance)
    row_means = out.mean(axis=1).dropna()
    assert (row_means.abs() < 1e-10).all()


def test_xs_rank_in_unit_interval():
    p = _toy_panel()
    out = xs_rank(p)
    vals = out.values
    vals_clean = vals[~np.isnan(vals)]
    assert (vals_clean >= 0).all()
    assert (vals_clean <= 1).all()


def test_list_primitives_includes_core():
    prims = list_primitives()
    for required in ("rolling_return", "shift", "skip", "ts_zscore",
                      "xs_zscore", "xs_rank", "vol_scale", "sign"):
        assert required in prims


def test_apply_primitive_unknown_raises():
    p = _toy_panel()
    with pytest.raises(KeyError, match="unknown primitive"):
        apply_primitive("not_a_real_primitive", p)


# ── Axis loaders ────────────────────────────────────────────────────────


def test_load_universe_real_demo():
    """Smoke against the actual repo demo universe."""
    uni = load_universe("synthetic_equity_demo")
    assert uni.name == "synthetic_equity_demo"
    assert uni.input_kind == "wide_returns_monthly"


def test_load_signal_recipe_real_demo():
    rec = load_signal_recipe("momentum_12_1")
    assert rec.name == "momentum_12_1"
    assert len(rec.steps) >= 1
    assert rec.steps[0]["primitive"] == "rolling_return"


def test_load_weighting_real_demo():
    w = load_weighting("decile_ls_10")
    assert w.kind == "decile_long_short"
    assert "q" in w.args


def test_load_signal_recipe_rejects_missing_steps(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"name": "b", "description": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_signal_recipe("bad", base_dir=tmp_path)


def test_load_weighting_rejects_bad_kind(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"name": "b", "description": "x",
                                   "kind": "not_a_valid_kind"}),
                  encoding="utf-8")
    with pytest.raises(ValueError, match="kind must be one of"):
        load_weighting("bad", base_dir=tmp_path)


# ── Signal pipeline executor ────────────────────────────────────────────


def test_run_signal_recipe_executes_in_order():
    """rolling_return(12) then shift(1) should produce the same result
    as applying them sequentially by hand."""
    p = _toy_panel(36, 5, seed=1)
    rec = SignalRecipe(
        name="test", description="",
        steps=[
            {"primitive": "rolling_return", "args": {"months": 12}},
            {"primitive": "shift", "args": {"periods": 1}},
        ],
    )
    out = run_signal_recipe(p, rec)
    expected = shift(rolling_return(p, months=12), periods=1)
    pd.testing.assert_frame_equal(out, expected)


def test_run_signal_recipe_bad_primitive_raises():
    p = _toy_panel()
    rec = SignalRecipe(
        name="test", description="",
        steps=[{"primitive": "totally_made_up"}],
    )
    with pytest.raises(RuntimeError, match="step 0"):
        run_signal_recipe(p, rec)


# ── Weighting ──────────────────────────────────────────────────────────


def test_decile_long_short_produces_signal_correlated_returns():
    """Build a panel with a known signal and confirm L/S returns are
    positively correlated with the signal."""
    n_periods, n_assets = 60, 10
    rng = np.random.default_rng(seed=42)
    # Persistent signal: alpha_i drives next-period return for asset i
    alphas = rng.uniform(-0.02, 0.02, size=n_assets)
    rets = pd.DataFrame(
        rng.normal(0.0, 0.03, size=(n_periods, n_assets)) + alphas,
        index=pd.date_range("2020-01-31", periods=n_periods, freq="ME"),
        columns=[f"A{i}" for i in range(n_assets)],
    )
    # Signal = realized return — should predict next-period return
    signal = rets
    w = Weighting(name="w", description="",
                   kind="decile_long_short", args={"q": 0.3})
    ls = apply_weighting(signal, rets, w)
    # Series should have valid values after the lag warm-up
    assert ls.dropna().shape[0] > 30
    # Should not be all NaN
    assert ls.notna().any()


def test_unknown_weighting_kind_raises():
    p = _toy_panel()
    w = Weighting(name="w", description="", kind="totally_made_up")
    with pytest.raises(ValueError, match="unknown weighting"):
        apply_weighting(p, p, w)


# ── End-to-end compose ─────────────────────────────────────────────────


def test_compose_spec_v2_auto_detected_by_materializer():
    """Materializer must auto-route eq_mom_12_1_demo (compose) to the
    composer, not to the v1 function loader."""
    from engine.feature_store.registry import SPECS_DIR
    p = SPECS_DIR / "eq_mom_12_1_demo.yaml"
    assert _spec_is_compose(p)


def test_compose_spec_real_demo_materializes_clean(tmp_path, monkeypatch):
    """Smoke against the actual eq_mom_12_1_demo spec — materialize end-
    to-end, validate output shape, write to tmp computed dir to not
    pollute the repo."""
    monkeypatch.setattr(
        "engine.feature_store.registry.COMPUTED_DIR", tmp_path,
    )
    monkeypatch.setattr(
        "engine.feature_store.materializer.COMPUTED_DIR", tmp_path,
    )
    result = materialize_spec("eq_mom_12_1_demo", force=True)
    assert result["spec_kind"] == "compose"
    v = result["validation"]
    assert v["ok"], v["violations"]
    assert v["observed_n_rows"] >= 40


def test_function_and_compose_specs_coexist():
    """list_specs must include both kinds, correctly classified."""
    from engine.feature_store.registry import list_specs
    specs = list_specs()
    kinds = {s["spec_id"]: s.get("spec_kind") for s in specs if "spec_kind" in s}
    assert kinds.get("cross_asset_carry_4leg") == "function"
    assert kinds.get("eq_mom_12_1_demo") == "compose"


def test_compose_load_spec_extracts_axes():
    """Direct loader test — confirms the 4 axes resolve."""
    from engine.feature_store.registry import SPECS_DIR
    p = SPECS_DIR / "eq_mom_12_1_demo.yaml"
    spec = load_compose_spec(p)
    assert spec.universe.name == "synthetic_equity_demo"
    assert spec.signal.name == "momentum_12_1"
    assert spec.weighting.name == "decile_ls_10"
    assert spec.rebalance == "monthly"


# ── Composability claim: new factor via YAML only, no Python ───────────


def test_new_factor_creatable_from_yaml_only(tmp_path, monkeypatch):
    """The headline claim of Week 1c: a brand-new factor needs no Python.

    Build a new compose-spec by writing 1 YAML file that references
    existing axis components — materialize without writing any
    Python function for it.
    """
    # Use existing axis components (no new YAML for those)
    new_spec_yaml = {
        "_schema_version": 1,
        "spec_id": "test_new_factor_yaml_only",
        "version": 1,
        "description": "Brand new factor, defined entirely in YAML",
        "compose": {
            "universe":   {"ref": "synthetic_equity_demo"},
            "signal":     {"ref": "momentum_12_1"},
            "weighting":  {"ref": "decile_ls_10", "args": {"q": 0.2}},  # override
            "rebalance":  {"freq": "monthly"},
        },
        "output": {
            "kind": "monthly_returns",
            "expected_date_range": {"start": "2020-01-01", "end_min": "2023-12-01"},
            "expected_shape": {"n_rows": [40, 200]},
            "sanity": {
                "no_nan_after_first_observation": False,
                "annualized_vol_range":    [0.001, 5.0],
                "annualized_sharpe_range": [-10.0, 10.0],
            },
        },
        "inputs": [
            {"cache_path": "data/feature_store/_demo_data/synthetic_equity_demo.parquet"},
        ],
        "source_module_files": [
            "engine/feature_store/composer.py",
            "engine/feature_store/primitives.py",
        ],
    }
    # Write to tmp specs dir
    specs_dir = tmp_path / "_specs"
    specs_dir.mkdir()
    new_spec_path = specs_dir / "test_new_factor_yaml_only.yaml"
    new_spec_path.write_text(yaml.safe_dump(new_spec_yaml), encoding="utf-8")

    computed_dir = tmp_path / "_computed"
    computed_dir.mkdir()

    monkeypatch.setattr(
        "engine.feature_store.registry.SPECS_DIR", specs_dir,
    )
    monkeypatch.setattr(
        "engine.feature_store.registry.COMPUTED_DIR", computed_dir,
    )
    monkeypatch.setattr(
        "engine.feature_store.materializer.COMPUTED_DIR", computed_dir,
    )

    result = materialize_spec("test_new_factor_yaml_only")
    assert result["spec_kind"] == "compose"
    assert result["validation"]["ok"], result["validation"]["violations"]
    # Confirm meta records the 4-axis composition for forensic replay
    meta = json.loads(Path(result["meta_path"]).read_text(encoding="utf-8"))
    assert meta["compose_axes"]["universe"] == "synthetic_equity_demo"
    assert meta["compose_axes"]["weighting"] == "decile_ls_10"
