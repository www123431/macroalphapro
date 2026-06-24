"""tests/test_lens_registry.py — Phase 1 Commit 2.

Tests the lens registry + DAG resolution + applicability filter.
Uses synthetic LensDeclaration fixtures + the real 4 lens modules
that declare LENS_DECLARATION in this commit.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.research.lens_registry import (
    LensDeclaration,
    discover_lenses,
    applicable_lenses,
    resolve_lens_dag,
    should_execute,
    validate_registry,
    CircularLensDependency,
)


# ────────────────────────────────────────────────────────────────────
# Synthetic fixtures
# ────────────────────────────────────────────────────────────────────
def _decl(name, input_protocols=(), output_protocol="AnchorRegressionOutput",
            applicable_to=None, conditional_on=None,
            consumed_by=(), runner=None):
    return LensDeclaration(
        name=name,
        version="test_v1",
        applicable_to=applicable_to or {},
        input_protocols=input_protocols,
        output_protocol=output_protocol,
        conditional_on=conditional_on,
        fallback_chain=(),
        output_schema={"primary": "alpha_nw_t", "secondary": ()},
        consumed_by=consumed_by,
        runner=runner or (lambda spec, tr, prior: None),
    )


def _spec(**kw):
    """A lightweight SimpleNamespace masquerading as FactorSpec for
    matches_spec() tests. Real FactorSpec is dataclass; this is
    enough for routing logic."""
    base = dict(
        investment_role=None, statistical_role=None,
        asset_class=None, mechanism=None,
        horizon=None, capacity_tier=None,
        data_dependency_type=None, regime_sensitivity=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ────────────────────────────────────────────────────────────────────
# matches_spec — applicability filter
# ────────────────────────────────────────────────────────────────────
def test_empty_applicable_to_matches_anything():
    d = _decl("any")
    assert d.matches_spec(_spec(investment_role="alpha"))
    assert d.matches_spec(_spec(investment_role="insurance"))


def test_applicable_to_filters_on_role():
    d = _decl("alpha_only", applicable_to={"investment_role": ("alpha",)})
    assert d.matches_spec(_spec(investment_role="alpha"))
    assert not d.matches_spec(_spec(investment_role="insurance"))


def test_applicable_to_filters_on_multiple_axes():
    d = _decl("alpha_equity", applicable_to={
        "investment_role": ("alpha",),
        "asset_class":     ("equity", "multi_asset"),
    })
    assert d.matches_spec(_spec(investment_role="alpha",
                                  asset_class="equity"))
    assert d.matches_spec(_spec(investment_role="alpha",
                                  asset_class="multi_asset"))
    assert not d.matches_spec(_spec(investment_role="alpha",
                                       asset_class="fx"))
    assert not d.matches_spec(_spec(investment_role="insurance",
                                       asset_class="equity"))


def test_applicable_to_uses_fallback_when_spec_is_none():
    d = _decl("alpha_only", applicable_to={"investment_role": ("alpha",)})
    spec = _spec(investment_role=None)
    # Without fallback: None is treated as wildcard match
    assert d.matches_spec(spec)
    # With fallback: alpha matches
    assert d.matches_spec(spec, fallback_axes={"investment_role": "alpha"})
    # With fallback: insurance doesn't match
    assert not d.matches_spec(spec,
                                  fallback_axes={"investment_role": "insurance"})


# ────────────────────────────────────────────────────────────────────
# DAG resolution
# ────────────────────────────────────────────────────────────────────
def test_dag_topological_sort_simple_chain():
    """A → B → C: A has no deps, B depends on A's output,
    C depends on B's output. Order must be A, B, C."""
    a = _decl("A", input_protocols=(),
                  output_protocol="AnchorRegressionOutput")
    b = _decl("B", input_protocols=("AnchorRegressionOutput",),
                  output_protocol="IndustryExtensionOutput")
    c = _decl("C", input_protocols=("IndustryExtensionOutput",),
                  output_protocol="CrossAssetExtensionOutput")
    ordered = resolve_lens_dag([c, a, b])  # input order shouldn't matter
    assert [l.name for l in ordered] == ["A", "B", "C"]


def test_dag_parallel_lenses_stable_alphabetic_order():
    """Two lenses with no deps → both ready at start; alphabetic
    order tiebreaker for determinism."""
    a = _decl("z_lens", output_protocol="AnchorRegressionOutput")
    b = _decl("a_lens", output_protocol="IndustryExtensionOutput")
    ordered = resolve_lens_dag([a, b])
    assert [l.name for l in ordered] == ["a_lens", "z_lens"]


def test_dag_raises_on_circular_dependency():
    """A → B → A creates a cycle."""
    a = _decl("A", input_protocols=("IndustryExtensionOutput",),
                  output_protocol="AnchorRegressionOutput")
    b = _decl("B", input_protocols=("AnchorRegressionOutput",),
                  output_protocol="IndustryExtensionOutput")
    with pytest.raises(CircularLensDependency):
        resolve_lens_dag([a, b])


def test_dag_handles_unsatisfiable_dependency_gracefully():
    """Lens B depends on Industry output, but no lens in the set
    produces Industry. B's dep list is empty (no satisfiable
    producer), so B runs first (no actual upstream).

    This is by design: if an upstream lens is filtered OUT by
    applicability, downstream still runs (and may produce
    nothing if it really needed the upstream output)."""
    b = _decl("B", input_protocols=("IndustryExtensionOutput",),
                  output_protocol="CrossAssetExtensionOutput")
    ordered = resolve_lens_dag([b])
    assert [l.name for l in ordered] == ["B"]


# ────────────────────────────────────────────────────────────────────
# should_execute — conditional skip
# ────────────────────────────────────────────────────────────────────
def test_should_execute_no_conditional_always_runs():
    d = _decl("X")
    proceed, reason = should_execute(d, {})
    assert proceed
    assert reason is None


def test_should_execute_conditional_unmet_skips():
    d = _decl("X", conditional_on={
        "lens": "Y",
        "condition": lambda out: out.get("alpha_nw_t", 0) >= 1.96,
        "skip_reason_if_unmet": "Y alpha below threshold",
    })
    proceed, reason = should_execute(d, {"Y": {"alpha_nw_t": 0.5}})
    assert not proceed
    assert "below threshold" in reason


def test_should_execute_conditional_met_proceeds():
    d = _decl("X", conditional_on={
        "lens": "Y",
        "condition": lambda out: out.get("alpha_nw_t", 0) >= 1.96,
        "skip_reason_if_unmet": "Y alpha below threshold",
    })
    proceed, _ = should_execute(d, {"Y": {"alpha_nw_t": 3.5}})
    assert proceed


def test_should_execute_skips_when_conditional_lens_missing():
    d = _decl("X", conditional_on={
        "lens": "Y",
        "condition": lambda out: True,
    })
    proceed, reason = should_execute(d, {})  # Y not in prior_outputs
    assert not proceed
    assert "produced no output" in reason


# ────────────────────────────────────────────────────────────────────
# applicable_lenses — filter
# ────────────────────────────────────────────────────────────────────
def test_applicable_lenses_filters_registry():
    reg = {
        "alpha_only": _decl(
            "alpha_only",
            applicable_to={"investment_role": ("alpha",)}),
        "insurance_only": _decl(
            "insurance_only",
            applicable_to={"investment_role": ("insurance",)}),
        "anything": _decl("anything", applicable_to={}),
    }
    out = applicable_lenses(reg, _spec(investment_role="alpha"))
    names = {l.name for l in out}
    assert names == {"alpha_only", "anything"}


# ────────────────────────────────────────────────────────────────────
# Real-module discovery: verify the 4 lens modules declare correctly
# ────────────────────────────────────────────────────────────────────
def test_discover_finds_all_four_lens_modules():
    """anchor_regression, subsample_stability, industry_attribution,
    cross_asset_attribution should all declare LENS_DECLARATION."""
    reg = discover_lenses()
    assert "anchor_regression" in reg
    assert "subsample_stability" in reg
    assert "industry_extension" in reg
    assert "cross_asset_extension" in reg


def test_real_registry_validates_clean():
    reg = discover_lenses()
    errors = validate_registry(reg)
    assert errors == [], f"validation errors: {errors}"


def test_real_registry_dag_resolves():
    reg = discover_lenses()
    ordered = resolve_lens_dag(list(reg.values()))
    # anchor must come before industry / cross_asset
    pos = {l.name: i for i, l in enumerate(ordered)}
    assert pos["anchor_regression"] < pos["industry_extension"]
    assert pos["anchor_regression"] < pos["cross_asset_extension"]


def test_real_registry_alpha_equity_applicability():
    """For an alpha+equity spec, anchor + subsample + industry +
    cross_asset should all be applicable. Insurance/etc. wouldn't
    have all of these."""
    reg = discover_lenses()
    spec = _spec(investment_role="alpha", asset_class="equity")
    applicable = applicable_lenses(reg, spec)
    names = {l.name for l in applicable}
    assert {"anchor_regression", "subsample_stability",
            "industry_extension", "cross_asset_extension"} <= names


def test_real_registry_cross_asset_skips_industry():
    """A cross-asset spec should NOT include industry_extension
    (US-equity industries mis-specified per A4 amendment)."""
    reg = discover_lenses()
    spec = _spec(investment_role="alpha", asset_class="cross_asset")
    applicable = applicable_lenses(reg, spec)
    names = {l.name for l in applicable}
    assert "industry_extension" not in names
    # But anchor + cross-asset still apply
    assert "anchor_regression" in names
    assert "cross_asset_extension" in names


def test_real_registry_insurance_excludes_all_tier_c_lenses():
    """Per A3 amendment: insurance/diversifier sleeves get routed
    to Tier D entirely. NO Tier C lenses should apply."""
    reg = discover_lenses()
    spec = _spec(investment_role="insurance", asset_class="equity")
    applicable = applicable_lenses(reg, spec)
    names = {l.name for l in applicable}
    assert "anchor_regression"      not in names
    assert "industry_extension"     not in names
    assert "cross_asset_extension"  not in names
    assert "subsample_stability"    not in names
    # All 4 Tier C lenses correctly filtered out → empty set
    assert names == set()


def test_real_registry_diversifier_also_excluded_from_tier_c():
    """A3 amendment applies to diversifier too."""
    reg = discover_lenses()
    spec = _spec(investment_role="diversifier", asset_class="cross_asset")
    applicable = applicable_lenses(reg, spec)
    assert applicable == []


def test_conditional_skip_industry_when_anchor_alpha_below_threshold():
    """Industry extension declares conditional_on anchor α t ≥ 1.0.
    Verify the predicate works on a synthetic prior output."""
    reg = discover_lenses()
    ix_decl = reg["industry_extension"]
    assert ix_decl.conditional_on is not None
    # Synthetic anchor output with low α
    proceed, reason = should_execute(ix_decl,
                                          {"anchor_regression": {"alpha_nw_t": 0.5}})
    assert not proceed
    assert "below 1.0" in reason or "uninformative" in reason
    # High α → run
    proceed2, _ = should_execute(ix_decl,
                                       {"anchor_regression": {"alpha_nw_t": 3.5}})
    assert proceed2
