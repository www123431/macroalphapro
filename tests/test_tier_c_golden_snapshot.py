"""tests/test_tier_c_golden_snapshot.py — O.1.

Golden-snapshot parity test for the Tier C lens stack. Locks numerical
output of every shipped lens on a known-good PnL series (GP/A — the
canonical equity sleeve). Detects silent numerical drift introduced
by future refactors of any of:

  - engine.research.anchor_regression (FF5+MOM math)
  - engine.research.subsample_stability (window split math)
  - engine.research.industry_attribution (FF12 joint OLS)
  - engine.research.cross_asset_attribution (macro joint OLS)
  - engine.research.specification_robustness (B-class ablation)
  - engine.research.residual_alpha_lens (A.2 factory)
  - engine.research.lens_helpers (B.2 contract)
  - engine.research.anchor_library_registry (A.1 catalog)

Senior-審計 lesson from 2026-06-09 chain audit (commit f1fe6df1
discovered subsample_stability had a stray _nw_lag duplicate that
A.1 missed). Without this snapshot, that kind of drift would land
silently. With it, any refactor that perturbs a beta by 0.001 fires
a regression test immediately.

CONTRACT
========
Each snapshot stores rounded floats (4-6 decimals depending on the
metric's sensitivity). Pinned thresholds match observed values from
the GP/A audit (see audit_deployed_sleeves_rigor report + memory
project_gpa_candidate_alpha_factor_2026-06-08.md):

  anchor_regression net:
    α t-stat        ≈ 1.88
    R²              ≈ 0.252
    β_RMW           ≈ +0.667  (the senior finding: ~67% RMW-spanned)

  subsample_stability:
    worst/best      ≈ 0.17
    institutional_stable = False
    monotone_decay  = False
    n_splits = 4, all 4 windows have finite stats

  industry_extension (joint FF5+MOM + FF12):
    α_full t        ≈ -1.38  (negative — α is absorbed by industries)
    industry joint F p ≈ 2.7e-31 (extreme orthogonality rejection)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


GP_A_PARQUET = (Path(__file__).resolve().parents[1]
                  / "data" / "research_store" / "tier_c_pnl"
                  / "dc4cf6beaa247880_GREEN.parquet")


@pytest.fixture
def gp_a_artifacts():
    if not GP_A_PARQUET.is_file():
        pytest.skip("GP/A snapshot fixture not cached")
    df = pd.read_parquet(GP_A_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return {
        "pnl_series_df":   df,
        "pnl_default_col": "pnl_net_13bp",
        "pnl_gross_col":   "pnl_gross",
    }


# ────────────────────────────────────────────────────────────────────
# anchor_regression — FF5+MOM
# ────────────────────────────────────────────────────────────────────
def test_anchor_regression_pinned_alpha_t(gp_a_artifacts):
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    out = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts,
    )
    assert out is not None
    # α t-stat — pinned to 0.01 tolerance
    assert abs(out["alpha_nw_t"] - 1.8797) < 0.01, (
        f"α t-stat drift: {out['alpha_nw_t']} (expected ~1.88)"
    )


def test_anchor_regression_pinned_r2(gp_a_artifacts):
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    out = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts,
    )
    assert abs(out["r2"] - 0.2522) < 0.001
    assert abs(out["r2_adj"] - 0.2407) < 0.001


def test_anchor_regression_pinned_rmw_beta(gp_a_artifacts):
    """The senior finding: GP/A loads ~67% on RMW. Numerical pin
    locks this — if A.1/A.2 refactors silently broke unit conversion
    or anchor selection, this fires."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    out = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts,
    )
    assert abs(out["betas"]["RMW"] - 0.6675) < 0.005, (
        f"RMW β drift: {out['betas']['RMW']} (expected ~0.667; sentinel for "
        "anchor unit conversion + library registry integrity)"
    )
    assert abs(out["beta_nw_t"]["RMW"] - 10.04) < 0.05


def test_anchor_regression_pinned_full_beta_signature(gp_a_artifacts):
    """All 6 FF5+MOM betas pinned to 3-decimal precision. Catches
    any silent reordering, sign flip, or unit conversion bug."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    out = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts,
    )
    expected = {
        "MKT_RF": +0.135,
        "SMB":    +0.118,
        "HML":    -0.346,
        "RMW":    +0.667,
        "CMA":    +0.165,
        "MOM":    -0.005,
    }
    for k, v in expected.items():
        assert abs(out["betas"][k] - v) < 0.003, (
            f"β_{k} drift: {out['betas'][k]} (expected ~{v})"
        )


def test_anchor_regression_n_overlap_locked(gp_a_artifacts):
    """The GP/A PnL parquet covers 1992-2024 → 395 months. Locked
    so we know if a date-range bug ever crops up."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    out = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts,
    )
    assert out["n_overlap"] == 395
    assert out["window"] == "1992-02:2024-12"


def test_anchor_regression_anchor_library_tag_unchanged(gp_a_artifacts):
    """A.1 registry contract: equity path tags as ken_french_ff5_mom."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    out = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts,
    )
    assert out["anchor_library"] == "ken_french_ff5_mom"


# ────────────────────────────────────────────────────────────────────
# subsample_stability
# ────────────────────────────────────────────────────────────────────
def test_subsample_stability_pinned_worst_best(gp_a_artifacts):
    from engine.research.subsample_stability import (
        compute_for_tier_c_pnl_series,
    )
    out = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], n_splits=4,
        artifacts=gp_a_artifacts,
    )
    assert out is not None
    assert abs(out["worst_best_sharpe_ratio"] - 0.1736) < 0.005
    assert out["institutional_stable"] is False
    assert out["monotone_decay"] is False


def test_subsample_stability_pinned_window_2_dominance(gp_a_artifacts):
    """The GP/A audit lesson: window 2 (2000-2008) Sharpe ~1.45,
    other 3 windows essentially weak. Pin to 0.05 tolerance — if
    window math drifts, this fires."""
    from engine.research.subsample_stability import (
        compute_for_tier_c_pnl_series,
    )
    out = compute_for_tier_c_pnl_series(
        gp_a_artifacts["pnl_series_df"], n_splits=4,
        artifacts=gp_a_artifacts,
    )
    sharpes = [w["sharpe_ann"] for w in out["windows"]]
    assert len(sharpes) == 4
    # Window 2 ≈ 1.45 (the strong year window)
    assert abs(sharpes[1] - 1.447) < 0.05
    # Other 3 are all weaker than window 2 (the senior finding)
    assert sharpes[1] > max(sharpes[0], sharpes[2], sharpes[3])


# ────────────────────────────────────────────────────────────────────
# industry_extension (JOINT FF5+MOM + FF12)
# ────────────────────────────────────────────────────────────────────
def test_industry_extension_pinned_alpha_collapses(gp_a_artifacts):
    """The B.2 + L4 audit finding: GP/A α t = +1.88 standalone, but
    once you add FF12 to a JOINT model, α t goes NEGATIVE.
    Industries fully absorb the alpha. This is the senior finding
    that should NEVER drift silently."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series as ar,
    )
    from engine.research.industry_attribution import (
        compute_for_tier_c_with_stage1_residual as ix,
    )
    s1 = ar(gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts)
    out = ix(s1, gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts)
    assert out is not None
    # α_full NW t-stat is NEGATIVE (~-1.38) — the absorption finding
    assert out["alpha_full_nw_t"] < -1.0
    assert abs(out["alpha_full_nw_t"] - (-1.3794)) < 0.05


def test_industry_extension_pinned_joint_f_rejects(gp_a_artifacts):
    """industry joint F p-value ≈ 2.7e-31 — extreme. Pin to log scale
    to detect if joint F math drifts in either direction."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series as ar,
    )
    from engine.research.industry_attribution import (
        compute_for_tier_c_with_stage1_residual as ix,
    )
    import math
    s1 = ar(gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts)
    out = ix(s1, gp_a_artifacts["pnl_series_df"], artifacts=gp_a_artifacts)
    jf = out["industry_joint_f_test"]
    # joint F p << 0.001
    assert jf["f_pvalue"] < 1e-20, (
        f"industry joint F p drift: {jf['f_pvalue']}"
    )
    # log scale within tolerance
    assert -35 < math.log10(jf["f_pvalue"]) < -25


# ────────────────────────────────────────────────────────────────────
# A.1 registry parity invariant
# ────────────────────────────────────────────────────────────────────
def test_registry_libraries_present():
    """A.1 invariant: the registered library set is an explicit,
    reviewed list. Adding one is intentional (edit this test);
    removing one is a regression.
    B.3 (2026-06-10): msss_gfx_vol added — 5 libraries."""
    from engine.research.anchor_library_registry import ANCHOR_LIBRARIES
    expected = {"ken_french_ff5_mom", "lrv_fx_carry",
                  "ff12_us_industry", "macro_us", "msss_gfx_vol"}
    assert set(ANCHOR_LIBRARIES.keys()) == expected


def test_registry_units_pinned():
    """A.1 silent-100x-bug protection: locked units for every
    library. If someone changes LRV from percent to decimal
    (or vice versa) silently, the regression fires immediately."""
    from engine.research.anchor_library_registry import ANCHOR_LIBRARIES
    assert ANCHOR_LIBRARIES["ken_french_ff5_mom"].units == "decimal"
    assert ANCHOR_LIBRARIES["lrv_fx_carry"].units == "percent"
    assert ANCHOR_LIBRARIES["ff12_us_industry"].units == "decimal"
    assert ANCHOR_LIBRARIES["macro_us"].units == "decimal"
    assert ANCHOR_LIBRARIES["msss_gfx_vol"].units == "decimal"
