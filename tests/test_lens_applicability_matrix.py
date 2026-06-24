"""tests/test_lens_applicability_matrix.py — O.2 + O.3.

O.2 — Lens applicability matrix: enumerates asset_class ×
investment_role combinations and asserts EXACTLY which lenses apply
for each. Pre-O.2 we had verified only 2 of ~16 common combinations
(equity-alpha, equity-insurance); FX-alpha was verified by hand
during B.1 but never locked. This matrix locks the full topology.

O.3 — Late-binding monkey-patch lock: A.2's factory runner does
getattr(wiring_module, "compute_for_tier_c_pnl_series") per
invocation so external patches on the module-level public surface
keep working. Verified with a quick smoke during the chain audit;
this locks it as a standing regression test.

WHY THE MATRIX MATTERS
======================
Routing topology is the load-bearing contract between FactorSpec
metadata and which rigor checks run. A silent applicable_to typo
(e.g. dropping "multi_asset" from anchor_regression) would mean
multi-asset sleeves silently skip the FF5+MOM check — no error,
no warning, just less rigor. The matrix makes every such change a
visible, reviewed diff in this file.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _spec(asset_class: str, investment_role: str):
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    return FactorSpec(
        hypothesis_id=f"matrix_{asset_class}_{investment_role}",
        signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000",
        date_range="2014-01:2024-12",
        signal_inputs=("crsp.msf.ret",),
        rebal="monthly",
        weighting="decile_long_short_dollar_neutral",
        expected_holding_period="monthly",
        min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="matrix test",
        extracted_ts="2026-06-10T00:00:00Z",
        model="claude-sonnet-4-6",
        investment_role=investment_role,
        statistical_role="directional",
        asset_class=asset_class,
        mechanism="behavioral",
        horizon="monthly",
        capacity_tier="100m_to_1b",
        data_dependency_type="fundamental",
        regime_sensitivity="known_regime_break",
    )


def _applicable_names(asset_class: str, investment_role: str) -> set[str]:
    from engine.research.lens_registry import (
        discover_lenses, applicable_lenses,
    )
    from engine.agents.strengthener.factor_spec_extractor import (
        infer_legacy_axes,
    )
    spec = _spec(asset_class, investment_role)
    reg = discover_lenses()
    return {l.name for l in applicable_lenses(reg, spec,
                                                  infer_legacy_axes(spec))}


# ────────────────────────────────────────────────────────────────────
# O.2 — The locked matrix. Every cell is the EXACT expected set.
# Changing routing topology requires editing this file — that's the
# point: topology changes become visible, reviewed diffs.
# ────────────────────────────────────────────────────────────────────

# All 6 lenses for reference:
#   anchor_regression           equity/multi_asset/cross_asset × alpha/overlay
#   fx_carry_anchor_regression  fx × alpha/overlay
#   industry_extension          equity/multi_asset × alpha
#   cross_asset_extension       (any asset) × alpha
#   subsample_stability         (any asset) × alpha/overlay
#   specification_robustness    (any asset) × alpha/overlay

EXPECTED_MATRIX: dict[tuple[str, str], set[str]] = {
    # ── alpha ──
    ("equity", "alpha"): {
        "anchor_regression", "industry_extension",
        "cross_asset_extension", "subsample_stability",
        "specification_robustness",
    },
    ("fx", "alpha"): {
        "fx_carry_anchor_regression",
        "cross_asset_extension", "subsample_stability",
        "specification_robustness",
    },
    ("multi_asset", "alpha"): {
        "anchor_regression", "industry_extension",
        "cross_asset_extension", "subsample_stability",
        "specification_robustness",
    },
    ("cross_asset", "alpha"): {
        "anchor_regression",
        "cross_asset_extension", "subsample_stability",
        "specification_robustness",
    },

    # ── overlay (industry + cross_asset are alpha-only) ──
    ("equity", "overlay"): {
        "anchor_regression", "subsample_stability",
        "specification_robustness",
    },
    ("fx", "overlay"): {
        "fx_carry_anchor_regression", "subsample_stability",
        "specification_robustness",
    },
    ("multi_asset", "overlay"): {
        "anchor_regression", "subsample_stability",
        "specification_robustness",
    },
    ("cross_asset", "overlay"): {
        "anchor_regression", "subsample_stability",
        "specification_robustness",
    },

    # ── insurance / diversifier: Tier D pre-routing means these
    #    never REACH the lens registry in production, but the
    #    registry filter itself must also exclude every lens —
    #    defense in depth (spec §15.A3). ──
    ("equity", "insurance"):      set(),
    ("fx", "insurance"):          set(),
    ("multi_asset", "insurance"): set(),
    ("cross_asset", "insurance"): set(),
    ("equity", "diversifier"):      set(),
    ("fx", "diversifier"):          set(),
    ("multi_asset", "diversifier"): set(),
    ("cross_asset", "diversifier"): set(),
}


@pytest.mark.parametrize(
    "asset_class,investment_role",
    sorted(EXPECTED_MATRIX.keys()),
)
def test_lens_applicability_matrix(asset_class, investment_role):
    expected = EXPECTED_MATRIX[(asset_class, investment_role)]
    actual = _applicable_names(asset_class, investment_role)
    assert actual == expected, (
        f"Routing topology drift for ({asset_class}, {investment_role}):\n"
        f"  expected: {sorted(expected)}\n"
        f"  actual:   {sorted(actual)}\n"
        f"  missing:  {sorted(expected - actual)}\n"
        f"  extra:    {sorted(actual - expected)}"
    )


def test_matrix_covers_all_lenses():
    """Every discovered lens must appear in at least one matrix cell —
    otherwise we shipped a lens that never fires (dead code) or the
    matrix is stale."""
    from engine.research.lens_registry import discover_lenses
    all_lenses = set(discover_lenses().keys())
    lenses_in_matrix = set().union(*EXPECTED_MATRIX.values())
    assert all_lenses == lenses_in_matrix, (
        f"Lens registry vs matrix drift:\n"
        f"  discovered but not in matrix: {sorted(all_lenses - lenses_in_matrix)}\n"
        f"  in matrix but not discovered: {sorted(lenses_in_matrix - all_lenses)}"
    )


def test_fx_and_equity_anchor_lenses_mutually_exclusive():
    """B.1 invariant: the two anchor lenses share the
    anchor_orthogonality output slot in the dispatcher — they MUST
    never both apply to the same spec, or the dispatcher's `or`
    union would silently pick one."""
    for (ac, role), expected in EXPECTED_MATRIX.items():
        both = {"anchor_regression",
                  "fx_carry_anchor_regression"} <= expected
        assert not both, (
            f"({ac}, {role}) routes BOTH anchor lenses — slot collision"
        )


# ────────────────────────────────────────────────────────────────────
# O.3 — Late-binding monkey-patch lock (A.2 backwards-compat contract)
# ────────────────────────────────────────────────────────────────────
def _tmpl_result():
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    import numpy as np
    n = 60
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(7)
    return TemplateResult(
        verdict="GREEN", summary="o3",
        metrics={"sharpe": 1.0, "nw_t_stat": 2.0, "n_months": n},
        artifacts={
            "pnl_series_df": pd.DataFrame(
                {"pnl_net_13bp": rng.normal(0, 0.01, n),
                  "pnl_gross":    rng.normal(0, 0.01, n)},
                index=idx),
            "pnl_default_col": "pnl_net_13bp",
        },
        template_version="o3",
    )


def test_equity_lens_runner_honors_module_level_patch(monkeypatch):
    """A.2 contract: patching anchor_regression.compute_for_tier_c_
    pnl_series at the MODULE level must redirect the factory-built
    lens runner. 3 dispatcher integration tests depend on this; this
    test locks the mechanism itself."""
    from engine.research import anchor_regression as ar
    sentinel = {"anchor_library": "patched_sentinel"}
    monkeypatch.setattr(ar, "compute_for_tier_c_pnl_series",
                          lambda *a, **kw: sentinel)
    out = ar.LENS_DECLARATION.runner(None, _tmpl_result(), {})
    assert out is sentinel, (
        "late-binding broken: module-level patch did not reach the "
        "factory lens runner (A.2 wiring_module contract)"
    )


def test_fx_lens_runner_honors_module_level_patch(monkeypatch):
    """Same contract for the FX lens module."""
    from engine.research import fx_carry_anchor_regression as fxa
    sentinel = {"anchor_library": "patched_fx_sentinel"}
    monkeypatch.setattr(fxa, "compute_for_tier_c_pnl_series",
                          lambda *a, **kw: sentinel)
    out = fxa.LENS_DECLARATION.runner(None, _tmpl_result(), {})
    assert out is sentinel


def test_unpatched_runner_uses_real_helper(monkeypatch):
    """Sanity inverse: without a patch, the runner produces a real
    regression result (or None for missing anchors) — NOT a stale
    closure capture of some older function."""
    from engine.research import anchor_regression as ar
    out = ar.LENS_DECLARATION.runner(None, _tmpl_result(), {})
    # Real FF5+MOM parquet is cached in this repo → expect real output
    if out is not None:
        assert out["anchor_library"] == "ken_french_ff5_mom"
        assert "alpha_nw_t" in out
