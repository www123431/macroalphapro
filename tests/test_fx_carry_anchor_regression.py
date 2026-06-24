"""tests/test_fx_carry_anchor_regression.py — B.1 FX carry lens.

Tests for engine.research.fx_carry_anchor_regression.

Mirrors test_anchor_regression structure:
  - Unit (synthetic data): residual α recovery, β identification,
    column auto-detection, joint F-test, output shape
  - Integration (real cached LRV parquet): self-regression of HML_FX
    against itself should yield β_HML_FX ≈ 1, α ≈ 0; auto-discovery
    via lens registry; live dispatch through carry template
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────
@pytest.fixture
def idx_240():
    return pd.date_range("2005-01-31", periods=240, freq="ME")


@pytest.fixture
def synth_anchors(idx_240):
    """Synthetic HML_FX + DOL panel, independent."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "HML_FX": rng.normal(0.003, 0.03, 240),
        "DOL":    rng.normal(0.001, 0.025, 240),
    }, index=idx_240)


# ────────────────────────────────────────────────────────────────────
# Core compute_fx_carry_residual_alpha
# ────────────────────────────────────────────────────────────────────
def test_pure_alpha_recovered(idx_240, synth_anchors):
    """factor = true_α + pure noise (no anchor loading) → recover α."""
    from engine.research.fx_carry_anchor_regression import (
        compute_fx_carry_residual_alpha,
    )
    rng = np.random.default_rng(101)
    true_alpha = 0.005
    factor = pd.Series(true_alpha + rng.normal(0, 0.02, 240),
                          index=idx_240)
    out = compute_fx_carry_residual_alpha(factor, synth_anchors)
    assert out is not None
    # 3-sigma band per random-data tolerance doctrine
    assert abs(out["alpha_monthly"] - true_alpha) < 0.005
    # Both betas near zero (independent regressors)
    assert abs(out["betas"]["HML_FX"]) < 0.20
    assert abs(out["betas"]["DOL"])    < 0.20


def test_beta_hml_fx_identified_when_factor_loads(idx_240, synth_anchors):
    """factor = 0.8 · HML_FX + noise → β_HML_FX ≈ 0.8 within 3σ band."""
    from engine.research.fx_carry_anchor_regression import (
        compute_fx_carry_residual_alpha,
    )
    rng = np.random.default_rng(31)
    factor = pd.Series(
        0.8 * synth_anchors["HML_FX"].values
        + rng.normal(0, 0.005, 240),
        index=idx_240,
    )
    out = compute_fx_carry_residual_alpha(factor, synth_anchors)
    assert out is not None
    assert abs(out["betas"]["HML_FX"] - 0.8) < 0.05
    assert abs(out["betas"]["DOL"]) < 0.05
    # Residual α should be near zero (factor IS HML_FX, no orthogonal alpha)
    assert abs(out["alpha_monthly"]) < 0.002


def test_self_regression_degenerate_case(idx_240, synth_anchors):
    """factor = HML_FX exactly → β_HML_FX = 1.0, β_DOL = 0, α = 0,
    R² = 1.0. The "your factor is a textbook restatement" canary."""
    from engine.research.fx_carry_anchor_regression import (
        compute_fx_carry_residual_alpha,
    )
    factor = synth_anchors["HML_FX"].copy()
    out = compute_fx_carry_residual_alpha(factor, synth_anchors)
    assert out is not None
    assert abs(out["betas"]["HML_FX"] - 1.0) < 1e-6
    assert abs(out["betas"]["DOL"]) < 1e-6
    assert abs(out["alpha_monthly"]) < 1e-9
    assert out["r2"] > 0.999


def test_insufficient_overlap_returns_none(synth_anchors):
    from engine.research.fx_carry_anchor_regression import (
        compute_fx_carry_residual_alpha,
    )
    # 10 months of factor — well below MIN_OVERLAP_MONTHS_DEFAULT=24
    idx = pd.date_range("2010-01-31", periods=10, freq="ME")
    factor = pd.Series(np.zeros(10), index=idx)
    # Anchors panel has 240 obs but overlap = 10 → too short
    assert compute_fx_carry_residual_alpha(factor, synth_anchors) is None


def test_empty_inputs_return_none(synth_anchors):
    from engine.research.fx_carry_anchor_regression import (
        compute_fx_carry_residual_alpha,
    )
    assert compute_fx_carry_residual_alpha(
        pd.Series(dtype=float), synth_anchors,
    ) is None


# ────────────────────────────────────────────────────────────────────
# (Column-name resolution moved to engine.research.lens_helpers in B.2
# — see tests/test_lens_helpers.py for the contract tests.)
# ────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────
# Tier C wiring helper — pnl_series_df → AnchorRegressionOutput dict
# ────────────────────────────────────────────────────────────────────
def test_tier_c_wiring_accepts_carry_template_columns(
    idx_240, synth_anchors,
):
    """carry_g10_fx template emits pnl_gross + pnl_net_8bp + pnl_net_24bp.
    Wiring helper must handle that naming (not just pnl_net_13bp)."""
    from engine.research.fx_carry_anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    # Use a DIFFERENT seed from synth_anchors fixture (seed 7) — same
    # seed would make the "noise" perfectly correlated with HML_FX
    # itself and bias the recovered β.
    rng = np.random.default_rng(91)
    pnl_gross = 0.5 * synth_anchors["HML_FX"].values + rng.normal(0, 0.005, 240)
    turnover  = np.full(240, 0.07)
    df = pd.DataFrame({
        "pnl_gross":   pnl_gross,
        "pnl_net_8bp": pnl_gross - 0.07 * (8 / 10_000.0),
        "pnl_net_24bp": pnl_gross - 0.07 * (24 / 10_000.0),
        "turnover":    turnover,
    }, index=idx_240)
    out = compute_for_tier_c_pnl_series(df, anchors=synth_anchors)
    assert out is not None
    assert out["anchor_library"] == "lrv_fx_carry"
    assert "HML_FX" in out["betas"]
    assert "DOL" in out["betas"]
    assert abs(out["betas"]["HML_FX"] - 0.5) < 0.05
    # Joint F-test populated
    assert out["joint_loading_f_test"] is not None
    assert "f_pvalue" in out["joint_loading_f_test"]
    # Gross block present (template has pnl_gross column)
    assert out["gross"] is not None
    assert "alpha_nw_t" in out["gross"]


def test_tier_c_wiring_accepts_plain_series(idx_240, synth_anchors):
    """Backwards compat with anchor_regression contract — Series input."""
    from engine.research.fx_carry_anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    rng = np.random.default_rng(11)
    series = pd.Series(0.003 + rng.normal(0, 0.02, 240), index=idx_240)
    out = compute_for_tier_c_pnl_series(series, anchors=synth_anchors)
    assert out is not None
    assert "alpha_nw_t" in out
    # No gross block when input is Series (only net)
    assert out["gross"] is None


def test_tier_c_wiring_returns_none_when_anchors_missing(
    monkeypatch, idx_240,
):
    from engine.research import fx_carry_anchor_regression as far
    monkeypatch.setattr(far, "load_fx_carry_anchors", lambda: None)
    series = pd.Series(np.zeros(240), index=idx_240)
    assert far.compute_for_tier_c_pnl_series(series) is None


# ────────────────────────────────────────────────────────────────────
# Parquet loader + SHA helpers
# ────────────────────────────────────────────────────────────────────
def test_load_returns_none_when_parquet_missing(tmp_path):
    from engine.research.fx_carry_anchor_regression import (
        load_fx_carry_anchors,
    )
    bad = str(tmp_path / "nope.parquet")
    assert load_fx_carry_anchors(bad) is None


def test_sha_is_64_hex_when_present():
    from engine.research.fx_carry_anchor_regression import (
        _fx_carry_parquet_sha256, _FX_CARRY_PARQUET,
    )
    sha = _fx_carry_parquet_sha256()
    if not _FX_CARRY_PARQUET.exists():
        assert sha == ""
        return
    assert len(sha) == 64


# ────────────────────────────────────────────────────────────────────
# Lens registry — auto-discovered + applicable to FX only
# ────────────────────────────────────────────────────────────────────
def test_lens_auto_discovered_in_registry():
    from engine.research.lens_registry import discover_lenses
    reg = discover_lenses()
    assert "fx_carry_anchor_regression" in reg
    decl = reg["fx_carry_anchor_regression"]
    assert decl.output_protocol == "AnchorRegressionOutput"
    # Strictly FX for this commit
    assert decl.applicable_to["asset_class"] == ("fx",)
    assert "alpha" in decl.applicable_to["investment_role"]


def test_lens_does_not_apply_to_equity():
    """An equity FactorSpec must NOT pick up the FX-carry lens. Same
    discipline as anchor_regression NOT picking up FX."""
    from engine.research.lens_registry import (
        discover_lenses, applicable_lenses,
    )
    from engine.agents.strengthener.factor_spec_extractor import (
        FactorSpec, infer_legacy_axes,
    )
    spec = FactorSpec(
        hypothesis_id="dummy_eq",
        signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000",
        date_range="2010-01:2024-12",
        signal_inputs=("compustat.funda.gp_at",),
        rebal="monthly",
        weighting="decile_long_short_dollar_neutral",
        expected_holding_period="monthly",
        min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="equity test",
        extracted_ts="2026-06-09T00:00:00Z",
        model="claude-sonnet-4-6",
        investment_role="alpha",
        statistical_role="directional",
        asset_class="equity",
        mechanism="behavioral",
        horizon="monthly",
        capacity_tier="100m_to_1b",
        data_dependency_type="fundamental",
        regime_sensitivity="known_regime_break",
    )
    reg = discover_lenses()
    applicable = applicable_lenses(reg, spec, infer_legacy_axes(spec))
    names = {l.name for l in applicable}
    assert "fx_carry_anchor_regression" not in names
    assert "anchor_regression" in names  # equity lens correctly applies


def test_lens_applies_to_fx_carry():
    from engine.research.lens_registry import (
        discover_lenses, applicable_lenses,
    )
    from engine.agents.strengthener.factor_spec_extractor import (
        FactorSpec, infer_legacy_axes,
    )
    spec = FactorSpec(
        hypothesis_id="dummy_fx",
        signal_kind="carry",
        universe="fx_g10",
        date_range="2002-05:2024-12",
        signal_inputs=("fred.fx_spot_g10",),
        rebal="monthly",
        weighting="tercile_long_short_dollar_neutral",
        expected_holding_period="monthly",
        min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="fx test",
        extracted_ts="2026-06-09T00:00:00Z",
        model="claude-sonnet-4-6",
        investment_role="alpha",
        statistical_role="directional",
        asset_class="fx",
        mechanism="risk_premium",
        horizon="monthly",
        capacity_tier="100m_to_1b",
        data_dependency_type="fundamental",
        regime_sensitivity="regime_dependent",
    )
    reg = discover_lenses()
    applicable = applicable_lenses(reg, spec, infer_legacy_axes(spec))
    names = {l.name for l in applicable}
    assert "fx_carry_anchor_regression" in names
    # Equity lenses correctly DO NOT apply to FX
    assert "anchor_regression" not in names


# ────────────────────────────────────────────────────────────────────
# Live integration — real cached LRV parquet
# ────────────────────────────────────────────────────────────────────
def _parquet_cached() -> bool:
    from engine.research.fx_carry_anchor_regression import _FX_CARRY_PARQUET
    return _FX_CARRY_PARQUET.is_file()


@pytest.mark.skipif(not _parquet_cached(),
                     reason="LRV anchor parquet not cached")
def test_integration_real_hml_fx_self_regression():
    """HML_FX regressed against itself MUST recover β=1, α=0, R²=1.
    The "your lens math works" smoke test."""
    from engine.research.fx_carry_anchor_regression import (
        load_fx_carry_anchors, compute_fx_carry_residual_alpha,
    )
    anchors = load_fx_carry_anchors()
    assert anchors is not None
    # Regress HML_FX against (HML_FX, DOL) — should pin β_HML=1, β_DOL=0
    factor = anchors["HML_FX"].copy()
    out = compute_fx_carry_residual_alpha(factor, anchors)
    assert out is not None
    assert abs(out["betas"]["HML_FX"] - 1.0) < 1e-6
    assert abs(out["betas"]["DOL"]) < 1e-6
    assert out["r2"] > 0.999


@pytest.mark.skipif(not _parquet_cached(),
                     reason="LRV anchor parquet not cached")
def test_integration_dispatcher_runs_fx_carry_lens_on_carry_spec():
    """End-to-end: dispatch a carry/fx_g10 FactorSpec through the
    Phase 1 lens registry. Routing decisions must show
    fx_carry_anchor_regression EXECUTED (not skipped_inapplicable)."""
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import self_doubt as sd_mod
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    spec = FactorSpec(
        hypothesis_id="b1_lens_smoke",
        signal_kind="carry",
        universe="fx_g10",
        date_range="2002-05:2024-12",
        signal_inputs=("fred.fx_spot_g10",),
        rebal="monthly",
        weighting="tercile_long_short_dollar_neutral",
        expected_holding_period="monthly",
        min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="B.1 lens smoke",
        extracted_ts="2026-06-09T00:00:00Z",
        model="claude-sonnet-4-6",
        investment_role="alpha",
        statistical_role="directional",
        asset_class="fx",
        mechanism="risk_premium",
        horizon="monthly",
        capacity_tier="100m_to_1b",
        data_dependency_type="fundamental",
        regime_sensitivity="regime_dependent",
    )

    def _llm_spy(**kw):
        return SimpleNamespace(text="", tool_calls=(),
                                  model="claude-sonnet-4-6")

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "dispatch.jsonl"
        with patch.object(fd, "_family_n_trials_now", lambda fam: 0), \
             patch.object(sd_mod, "llm_call", _llm_spy):
            out = fd.dispatch_factor_spec(
                spec, family_hint="CARRY",
                spec_approved=True, log_path=log,
            )

    routing = out.get("routing_decisions") or []
    # Find the fx_carry_anchor_regression decision
    fx_decisions = [r for r in routing
                       if r.get("lens") == "fx_carry_anchor_regression"]
    assert len(fx_decisions) == 1, (
        f"expected exactly 1 decision for fx_carry_anchor_regression, "
        f"got {len(fx_decisions)}: {routing}"
    )
    decision = fx_decisions[0]
    assert decision["action"] == "executed", (
        f"expected fx_carry_anchor_regression to execute, "
        f"got action={decision['action']!r} reason={decision.get('reason')!r}"
    )
