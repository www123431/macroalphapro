"""tests/test_specification_robustness.py — B of senior施工建议.

Tests for engine.research.specification_robustness — neighborhood
ablation of B-class params with stability scoring.

CRITICAL invariant the test locks: n_trials_increment is ALWAYS 0.
The ablation cells are robustness checks of one hypothesis, NOT N
hypotheses. self_doubt prompt + factor_verdict_emit must respect
this; downstream consumers must not double-count.
"""
from __future__ import annotations

import dataclasses as _dc

import pytest


def _spec(**kw):
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    base = dict(
        hypothesis_id="spec_rob",
        signal_kind="time_series_momentum",
        universe="us_equities_sector_etf",
        date_range="2014-01:2024-12",
        signal_inputs=("etf.adj_close.spy",),
        rebal="weekly",
        weighting="signed_signal_volatility_targeted",
        expected_holding_period="weekly",
        min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="test",
        extracted_ts="2026-06-09T00:00:00Z",
        model="claude-sonnet-4-6",
        investment_role="alpha",
        statistical_role="directional",
        asset_class="equity",
        mechanism="momentum",
        horizon="monthly",
        capacity_tier="100m_to_1b",
        data_dependency_type="market",
        regime_sensitivity="known_regime_break",
        signal_lookback_m=12,
        signal_skip_m=1,
        vol_target_annual=0.10,
    )
    base.update(kw)
    return FactorSpec(**base)


def _tr(verdict="GREEN", sharpe=1.5):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    return TemplateResult(
        verdict=verdict, summary="t",
        metrics={"sharpe": sharpe, "nw_t_stat": sharpe*2.0,
                  "n_months": 120, "avg_turnover": 0.2},
        artifacts={}, template_version="t",
    )


# ────────────────────────────────────────────────────────────────────
# Neighborhood generator
# ────────────────────────────────────────────────────────────────────
def test_neighborhood_size_full_spec():
    """flex-4: deltas derived from B_CLASS_RANGES steps (±1·step,
    ±2·step, clipped). Spec sets lookback=12 (step 1 → 10,11,13,14 =
    4), skip=1 (step 1, min 0 → -1 clipped → 0,2,3 = 3), vol=0.10
    (step 0.02 → 0.06,0.08,0.12,0.14 all in [0.03,0.30] = 4).
    Total 11 (was 8 pre-derivation — skip/vol gained ±2 cells)."""
    from engine.research.specification_robustness import (
        build_neighborhood_specs,
    )
    spec = _spec()
    variants = build_neighborhood_specs(spec)
    assert len(variants) == 11


def test_neighborhood_clips_to_B_CLASS_RANGES():
    """signal_lookback_m=2 → +1 OK, +2 OK, -1 hits range floor=1 OK,
    -2 → 0 outside floor=1 → clipped (skipped)."""
    from engine.research.specification_robustness import (
        build_neighborhood_specs,
    )
    spec = _spec(signal_lookback_m=2, signal_skip_m=None, vol_target_annual=None)
    variants = build_neighborhood_specs(spec)
    # Should produce 3 (10, 11, 13 — wait, no: from base=2 the deltas
    # are -2,-1,+1,+2 → 0,1,3,4 → 0 clipped, 1+3+4 kept = 3 variants)
    assert len(variants) == 3
    new_values = [v[2] for v in variants]
    assert 0 not in new_values   # below range floor
    assert all(v >= 1 for v in new_values)


def test_neighborhood_skips_unset_params():
    """When a spec param is None, no neighborhood variant is generated
    for it (template uses internal default; user opted out of varying)."""
    from engine.research.specification_robustness import (
        build_neighborhood_specs,
    )
    spec = _spec(signal_lookback_m=None, signal_skip_m=None,
                    vol_target_annual=0.10)
    variants = build_neighborhood_specs(spec)
    # Only vol_target_annual variants: ±0.02, ±0.04 = 4 (flex-4)
    assert len(variants) == 4
    assert all(f == "vol_target_annual" for f, *_ in variants)


def test_neighborhood_resolves_cgs_f_gap_2():
    """THE F-gap-2 acceptance: the exact CGS-2008 shape (n_buckets=10
    at range max + universe_size=2000) had ZERO variation room
    pre-flex-4 (INSUFFICIENT_VARIATION in session 7326492e). Now:
    n_buckets → 8,9 (2 cells, +1/+2 clipped at max=10);
    universe_size step 250 → 1500,1750,2250,2500 (4 cells).
    Total 6 ≥ MIN_VARIANTS=3 → the ablation RUNS."""
    from engine.research.specification_robustness import (
        MIN_VARIANTS, build_neighborhood_specs,
    )
    spec = _spec(signal_lookback_m=None, signal_skip_m=None,
                    vol_target_annual=None,
                    n_buckets=10, universe_size=2000)
    variants = build_neighborhood_specs(spec)
    assert len(variants) >= MIN_VARIANTS
    fields = {f for f, *_ in variants}
    assert "universe_size" in fields    # the previously-impossible axis
    us_values = sorted(v for f, _, v, _ in variants
                          if f == "universe_size")
    assert us_values == [1500, 1750, 2250, 2500]


# ────────────────────────────────────────────────────────────────────
# Verdict mapping
# ────────────────────────────────────────────────────────────────────
def test_verdict_robust_above_bar():
    from engine.research.specification_robustness import _verdict_from_score
    assert _verdict_from_score(0.85) == "ROBUST"
    assert _verdict_from_score(0.60) == "ROBUST"


def test_verdict_marginal_in_band():
    from engine.research.specification_robustness import _verdict_from_score
    assert _verdict_from_score(0.55) == "MARGINAL_OVERFIT"
    assert _verdict_from_score(0.40) == "MARGINAL_OVERFIT"


def test_verdict_likely_overfit_below_marginal():
    from engine.research.specification_robustness import _verdict_from_score
    assert _verdict_from_score(0.30) == "LIKELY_OVERFIT"
    assert _verdict_from_score(0.10) == "LIKELY_OVERFIT"


def test_verdict_untestable_when_none():
    from engine.research.specification_robustness import _verdict_from_score
    assert _verdict_from_score(None) == "UNTESTABLE"
    assert _verdict_from_score(float("nan")) == "UNTESTABLE"


# ────────────────────────────────────────────────────────────────────
# Lens gates
# ────────────────────────────────────────────────────────────────────
def test_lens_skips_RED_verdict():
    """RED already rejected — no point burning compute on ablation."""
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    spec = _spec()
    red_tr = _tr(verdict="RED", sharpe=0.4)
    out = compute_specification_robustness(spec, lambda s: red_tr, red_tr)
    assert out is None


def test_lens_runs_on_GREEN():
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    spec = _spec()
    green_tr = _tr(verdict="GREEN", sharpe=1.5)
    out = compute_specification_robustness(
        spec, lambda s: _tr(verdict="GREEN", sharpe=1.5), green_tr,
    )
    assert out is not None
    assert out["status"] == "COMPLETE"


def test_lens_runs_on_MARGINAL():
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    spec = _spec()
    marg_tr = _tr(verdict="MARGINAL", sharpe=0.9)
    out = compute_specification_robustness(
        spec, lambda s: _tr(verdict="MARGINAL", sharpe=0.9), marg_tr,
    )
    assert out is not None
    assert out["status"] == "COMPLETE"


# ────────────────────────────────────────────────────────────────────
# CRITICAL INVARIANT — n_trials_increment ALWAYS 0
# ────────────────────────────────────────────────────────────────────
def test_n_trials_increment_always_zero_on_complete():
    """Bailey-LdP DSR n_trials MUST NOT inflate from this lens output.
    Cells are robustness checks of one hypothesis, not N hypotheses.
    Asness 2017 + HXZ 2020 convention."""
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    spec = _spec()
    out = compute_specification_robustness(
        spec, lambda s: _tr("GREEN", sharpe=1.0), _tr("GREEN", 1.0),
    )
    assert out["n_trials_increment"] == 0


def test_n_trials_increment_zero_on_insufficient_variation():
    """Even degenerate paths must respect n_trials_increment=0
    contract — downstream consumers don't have to special-case."""
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    # n_buckets at max=10 → both +1, +2 clipped; only -1, -2 survive,
    # which is < MIN_VARIANTS=3 alone, but together with signal_lookback
    # variants it's not. Force insufficient by unsetting everything else.
    spec = _spec(signal_lookback_m=None, signal_skip_m=None,
                    vol_target_annual=None, n_buckets=10)
    out = compute_specification_robustness(
        spec, lambda s: _tr("GREEN", 1.5), _tr("GREEN", 1.5),
    )
    assert out is not None
    assert out["status"] == "INSUFFICIENT_VARIATION"
    assert out["n_trials_increment"] == 0


# ────────────────────────────────────────────────────────────────────
# Stability scoring
# ────────────────────────────────────────────────────────────────────
def test_robust_when_template_sharpe_stable():
    """Flat-Sharpe template → median ≈ max → stability_score ≈ 1.0."""
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    spec = _spec()
    out = compute_specification_robustness(
        spec, lambda s: _tr("GREEN", 1.5), _tr("GREEN", 1.5),
    )
    assert out["verdict"] == "ROBUST"
    assert out["stability_score"] >= 0.99


def test_likely_overfit_when_only_default_peaks():
    """Cherry-picked template: only base spec gives Sharpe 1.8;
    every neighborhood variant collapses to 0.3 → stability ≈ 0.17."""
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    spec = _spec()
    def _peaky(s):
        is_default = (s.signal_lookback_m == 12 and s.signal_skip_m == 1
                        and abs((s.vol_target_annual or 0.10) - 0.10) < 1e-6)
        return _tr("GREEN", 1.8 if is_default else 0.3)
    base = _peaky(spec)
    out = compute_specification_robustness(spec, _peaky, base)
    assert out["verdict"] == "LIKELY_OVERFIT"
    assert out["stability_score"] < 0.40


def test_marginal_overfit_when_partial_decay():
    """All neighborhood variants fall to mid-Sharpe, only the EXACT
    base config peaks → median ≈ 0.45 of max → MARGINAL_OVERFIT band."""
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    spec = _spec()
    def _partial(s):
        # Detect ANY deviation from base config → demote to 0.45
        is_default = (
            s.signal_lookback_m == 12
            and s.signal_skip_m == 1
            and abs((s.vol_target_annual or 0.10) - 0.10) < 1e-6
        )
        return _tr("GREEN", 1.0 if is_default else 0.45)
    base = _partial(spec)
    out = compute_specification_robustness(spec, _partial, base)
    assert out["verdict"] == "MARGINAL_OVERFIT"
    # base 1.0 + 8 variants of 0.45 → median 0.45, max 1.0, score 0.45
    assert 0.40 <= out["stability_score"] < 0.60


# ────────────────────────────────────────────────────────────────────
# Lens auto-discovery
# ────────────────────────────────────────────────────────────────────
def test_lens_auto_discovered_in_registry():
    from engine.research.lens_registry import discover_lenses
    reg = discover_lenses()
    assert "specification_robustness" in reg
    decl = reg["specification_robustness"]
    assert decl.output_protocol == "SpecificationRobustnessOutput"
    # alpha + overlay; asset_class-agnostic
    assert decl.applicable_to.get("investment_role") == ("alpha", "overlay")
    assert "asset_class" not in decl.applicable_to


def test_lens_skips_for_insurance_role():
    """Insurance routes to Tier D — never reaches Tier C lens stack.
    But applicable_to filter must still NOT match insurance."""
    from engine.research.lens_registry import (
        discover_lenses, applicable_lenses,
    )
    from engine.agents.strengthener.factor_spec_extractor import (
        infer_legacy_axes,
    )
    spec = _spec(investment_role="insurance")
    reg = discover_lenses()
    applicable = applicable_lenses(reg, spec, infer_legacy_axes(spec))
    names = {l.name for l in applicable}
    assert "specification_robustness" not in names


# ────────────────────────────────────────────────────────────────────
# Per-cell breakdown shape
# ────────────────────────────────────────────────────────────────────
def test_cells_tested_includes_base_first():
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    spec = _spec()
    out = compute_specification_robustness(
        spec, lambda s: _tr("GREEN", 1.5), _tr("GREEN", 1.5),
    )
    cells = out["cells_tested"]
    assert cells[0]["label"] == "base"
    assert cells[0]["param_changed"] is None
    # Subsequent cells label like "signal_lookback_m=10"
    for c in cells[1:]:
        assert "=" in c["label"]
        assert c["param_changed"] is not None


def test_neighborhood_size_in_output():
    """neighborhood_size = number of variants (not counting base)."""
    from engine.research.specification_robustness import (
        compute_specification_robustness,
    )
    spec = _spec()
    out = compute_specification_robustness(
        spec, lambda s: _tr("GREEN", 1.5), _tr("GREEN", 1.5),
    )
    # flex-4: 11 derived variants (lookback 4 + skip 3 + vol 4)
    assert out["neighborhood_size"] == 11
    # successful_cells includes base
    assert out["successful_cells"] == 12
