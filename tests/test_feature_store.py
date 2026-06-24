"""Tests for engine.feature_store — Option C Week 1.

Schema validation + hashing + materialize end-to-end. The end-to-end
test uses an in-test synthetic spec pointing at a fixture function so
we don't depend on WRDS data being available.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml

from engine.feature_store.materializer import (
    _validate_output,
    compute_input_hash,
    materialize_spec,
    verify_spec,
)
from engine.feature_store.registry import (
    FeatureSpec, _validate_spec, list_specs, load_spec,
)


# ── Schema validation ───────────────────────────────────────────────────


def _minimal_spec_dict() -> dict:
    """Schema-valid minimal spec dict, modifiable per test."""
    return {
        "spec_id": "test_spec", "version": 1,
        "description": "test",
        "materialize": {"module": "mod", "function": "fn"},
        "output": {
            "kind": "monthly_returns",
            "expected_date_range": {"start": "2020-01-01", "end_min": "2024-01-01"},
            "expected_shape": {"n_rows": [10, 100]},
            "sanity": {
                "no_nan_after_first_observation": True,
                "annualized_vol_range":    [0.0, 1.0],
                "annualized_sharpe_range": [-2.0, 3.0],
            },
        },
        "inputs": [{"cache_path": "data/cache/x.parquet"}],
        "source_module_files": ["engine/foo.py"],
    }


def test_validate_spec_accepts_minimal(tmp_path):
    p = tmp_path / "spec.yaml"
    _validate_spec(_minimal_spec_dict(), p)  # no raise


def test_validate_spec_rejects_missing_top_level(tmp_path):
    spec = _minimal_spec_dict()
    del spec["materialize"]
    with pytest.raises(ValueError, match="missing top-level"):
        _validate_spec(spec, tmp_path / "spec.yaml")


def test_validate_spec_rejects_bad_kind(tmp_path):
    spec = _minimal_spec_dict()
    spec["output"]["kind"] = "not_a_kind"
    with pytest.raises(ValueError, match="output.kind"):
        _validate_spec(spec, tmp_path / "spec.yaml")


def test_validate_spec_returns_requires_sanity_ranges(tmp_path):
    spec = _minimal_spec_dict()
    del spec["output"]["sanity"]["annualized_vol_range"]
    with pytest.raises(ValueError, match="annualized_vol_range"):
        _validate_spec(spec, tmp_path / "spec.yaml")


def test_validate_spec_bad_n_rows_shape(tmp_path):
    spec = _minimal_spec_dict()
    spec["output"]["expected_shape"]["n_rows"] = 42  # not a list
    with pytest.raises(ValueError, match="n_rows"):
        _validate_spec(spec, tmp_path / "spec.yaml")


def test_load_spec_round_trip(tmp_path):
    spec_dict = _minimal_spec_dict()
    p = tmp_path / "test_spec.yaml"
    p.write_text(yaml.safe_dump(spec_dict), encoding="utf-8")
    spec = load_spec(p)
    assert spec.spec_id == "test_spec"
    assert spec.materialize_module == "mod"
    assert spec.materialize_function == "fn"


# ── Hashing ─────────────────────────────────────────────────────────────


def _spec_obj(tmp_path, **overrides) -> FeatureSpec:
    spec_dict = _minimal_spec_dict()
    spec_dict.update(overrides)
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec_dict), encoding="utf-8")
    return load_spec(p)


def test_hash_is_deterministic(tmp_path):
    spec = _spec_obj(tmp_path)
    h1 = compute_input_hash(spec)
    h2 = compute_input_hash(spec)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_hash_changes_when_spec_content_changes(tmp_path):
    spec_a = _spec_obj(tmp_path, description="a")
    h_a = compute_input_hash(spec_a)
    spec_b = _spec_obj(tmp_path, description="completely different text here")
    h_b = compute_input_hash(spec_b)
    assert h_a != h_b


# ── Output validation ───────────────────────────────────────────────────


def test_validate_output_accepts_clean_series(tmp_path):
    spec = _spec_obj(tmp_path)
    s = pd.Series(
        [0.01, -0.02, 0.005, 0.01, -0.005] * 12,
        index=pd.date_range("2020-01-01", periods=60, freq="ME"),
    )
    rep = _validate_output(spec, s)
    assert rep.ok, rep.violations


def test_validate_output_rejects_wrong_type(tmp_path):
    spec = _spec_obj(tmp_path)
    rep = _validate_output(spec, "not a series")
    assert not rep.ok
    assert any("Series" in v for v in rep.violations)


def test_validate_output_rejects_too_few_rows(tmp_path):
    spec = _spec_obj(tmp_path)
    s = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2020-01-01", periods=3, freq="ME"),
    )
    rep = _validate_output(spec, s)
    assert not rep.ok
    assert any("n_rows" in v for v in rep.violations)


def test_validate_output_rejects_late_start(tmp_path):
    spec = _spec_obj(tmp_path)
    # Spec wants start <= 2020-01-01; observation starts 2022 → violation
    s = pd.Series(
        [0.01] * 30,
        index=pd.date_range("2022-01-01", periods=30, freq="ME"),
    )
    rep = _validate_output(spec, s)
    assert not rep.ok
    assert any("start" in v for v in rep.violations)


def test_validate_output_rejects_nan_after_first_obs(tmp_path):
    spec = _spec_obj(tmp_path)
    s = pd.Series(
        [None, 0.01, None, 0.02] + [0.01] * 30,
        index=pd.date_range("2020-01-01", periods=34, freq="ME"),
    )
    rep = _validate_output(spec, s)
    assert not rep.ok
    assert any("NaN" in v for v in rep.violations)


def test_validate_output_rejects_extreme_vol(tmp_path):
    spec = _spec_obj(tmp_path)
    # Spec allows ann_vol in [0, 1.0]; build a series with 200% ann vol
    s = pd.Series(
        [0.3, -0.3] * 30,   # huge monthly swings
        index=pd.date_range("2020-01-01", periods=60, freq="ME"),
    )
    rep = _validate_output(spec, s)
    assert not rep.ok
    assert any("ann_vol" in v for v in rep.violations)


# ── End-to-end materialize ──────────────────────────────────────────────


@pytest.fixture
def synthetic_fixture_module():
    """Register a synthetic module on sys.modules so the materializer
    can import it. Returns the module so the test can call the fn too."""
    mod = types.ModuleType("_test_feature_store_fixture")

    def _fake_build(n_rows: int = 60, scale: float = 0.01):
        return pd.Series(
            [scale, -scale, 0.005, 0.0, scale * 0.5] * (n_rows // 5),
            index=pd.date_range("2020-01-01", periods=n_rows, freq="ME"),
        )

    mod.fake_build = _fake_build
    sys.modules["_test_feature_store_fixture"] = mod
    yield mod
    del sys.modules["_test_feature_store_fixture"]


@pytest.fixture
def isolated_spec(tmp_path, monkeypatch):
    """Build a spec pointing at the fixture module + redirect SPECS_DIR
    and COMPUTED_DIR to tmp."""
    specs_dir = tmp_path / "_specs"
    computed_dir = tmp_path / "_computed"
    specs_dir.mkdir()
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

    spec_dict = _minimal_spec_dict()
    spec_dict["spec_id"] = "fixture_spec"
    spec_dict["materialize"] = {
        "module":   "_test_feature_store_fixture",
        "function": "fake_build",
        "args":     {"n_rows": 60, "scale": 0.01},
    }
    spec_dict["inputs"] = []
    spec_dict["source_module_files"] = []

    p = specs_dir / "fixture_spec.yaml"
    p.write_text(yaml.safe_dump(spec_dict), encoding="utf-8")
    return p, computed_dir


def test_materialize_end_to_end(synthetic_fixture_module, isolated_spec):
    spec_path, computed_dir = isolated_spec
    spec = load_spec(spec_path)
    result = materialize_spec(spec)
    assert result["validation"]["ok"], result["validation"]["violations"]
    assert result["cached"] is False
    assert Path(result["output_path"]).is_absolute() or \
            (Path.cwd() / result["output_path"]).is_file() or \
            any(computed_dir.glob("*.parquet"))


def test_materialize_caches_by_input_hash(synthetic_fixture_module, isolated_spec):
    spec_path, _ = isolated_spec
    spec = load_spec(spec_path)
    r1 = materialize_spec(spec)
    r2 = materialize_spec(spec)
    assert r2["cached"] is True
    assert r2["input_hash"] == r1["input_hash"]


def test_materialize_force_rebuilds(synthetic_fixture_module, isolated_spec):
    spec_path, _ = isolated_spec
    spec = load_spec(spec_path)
    materialize_spec(spec)
    r = materialize_spec(spec, force=True)
    assert r["cached"] is False


def test_materialize_writes_meta_sidecar(synthetic_fixture_module, isolated_spec):
    spec_path, computed_dir = isolated_spec
    spec = load_spec(spec_path)
    result = materialize_spec(spec)
    meta_files = list(computed_dir.glob("*.meta.json"))
    assert len(meta_files) == 1
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert meta["spec_id"] == "fixture_spec"
    assert "input_hash" in meta
    assert "materialized_at" in meta
    assert meta["validation"]["ok"] is True


def test_materialize_strict_sanity_raises_on_violation(
    synthetic_fixture_module, isolated_spec, monkeypatch,
):
    spec_path, _ = isolated_spec
    # Modify the spec to have an impossible sanity range
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    raw["output"]["sanity"]["annualized_vol_range"] = [10.0, 20.0]
    spec_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    spec = load_spec(spec_path)
    with pytest.raises(ValueError, match="validation failed"):
        materialize_spec(spec, strict_sanity=True)


def test_verify_spec_detects_no_drift_when_clean(
    synthetic_fixture_module, isolated_spec,
):
    spec_path, _ = isolated_spec
    spec = load_spec(spec_path)
    result = verify_spec(spec)
    assert result["ok"] is True
    assert result["hash_match"] is True


# ── Registry discovery ─────────────────────────────────────────────────


def test_list_specs_discovers_real_repo_specs():
    """Smoke test against the actual repo — confirms our 3 real specs load."""
    specs = list_specs()
    spec_ids = {s.get("spec_id") for s in specs}
    expected = {"cross_asset_carry_4leg", "tsmom_5leg", "tail_hedge_put_spread"}
    assert expected.issubset(spec_ids), \
        f"missing specs in registry: {expected - spec_ids}"


def test_list_specs_reports_errors_inline(tmp_path, monkeypatch):
    """Bad specs in the directory should not crash list_specs — they
    should surface with an error field instead."""
    monkeypatch.setattr(
        "engine.feature_store.registry.SPECS_DIR", tmp_path,
    )
    (tmp_path / "broken.yaml").write_text("not: a valid spec",
                                            encoding="utf-8")
    specs = list_specs()
    assert len(specs) == 1
    assert "error" in specs[0]
