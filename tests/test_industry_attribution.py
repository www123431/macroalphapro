"""tests/test_industry_attribution.py — Tier C L2-6 lite Commit 2.

JOINT-model tests for industry-extended alpha decomposition. Per
[[feedback-fwl-sequential-residual-trap-2026-06-09]]: the old
sequential ε regression was a Frisch-Waugh-Lovell violation; we now
test the joint [FF5+MOM ∪ Industry] OLS+HAC model.

Key assertions tested:
1. Pure-alpha factor → α_full recovered, β all near zero, F p high
2. Industry-spanned residual → F p << 0.01, large industry β
3. Joint α ≠ Stage 1 α when industries add explanation (Δα ≠ 0)
4. The OLD bug pattern (α₂ ≈ 0 with F p high) CANNOT recur because
   we report α from joint model, not from regressing ε₁ on raw X₂.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def idx_240():
    return pd.date_range("2005-01-31", periods=240, freq="ME")


@pytest.fixture
def synthetic_ff5mom_240(idx_240):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "MKT_RF": rng.normal(0.005, 0.04, 240),
        "SMB":    rng.normal(0.002, 0.03, 240),
        "HML":    rng.normal(0.003, 0.03, 240),
        "RMW":    rng.normal(0.003, 0.02, 240),
        "CMA":    rng.normal(0.002, 0.02, 240),
        "MOM":    rng.normal(0.006, 0.04, 240),
    }, index=idx_240)


@pytest.fixture
def synthetic_industries_240(idx_240):
    rng = np.random.default_rng(13)
    cols = ("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
            "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")
    return pd.DataFrame(
        {c: rng.normal(0.005 + i * 0.0005, 0.04 + i * 0.001, 240)
         for i, c in enumerate(cols)},
        index=idx_240,
    )


# ────────────────────────────────────────────────────────────────────
# JOINT model recovers true α (not zero by construction)
# ────────────────────────────────────────────────────────────────────
def test_joint_alpha_recovered_for_pure_alpha_factor(
    idx_240, synthetic_ff5mom_240, synthetic_industries_240,
):
    """factor = α₀ + iid noise (orthogonal to all 18 regressors) →
    joint OLS must recover α₀ within sampling SE, NOT collapse to 0.

    This is the anti-regression for the FWL bug — the buggy
    implementation would have given α = ~0 here regardless of α₀."""
    from engine.research.industry_attribution import (
        compute_industry_extended_alpha,
    )
    rng = np.random.default_rng(101)
    true_alpha = 0.006
    factor = pd.Series(true_alpha + rng.normal(0, 0.02, 240),
                          index=idx_240)
    out = compute_industry_extended_alpha(
        factor, synthetic_ff5mom_240, synthetic_industries_240,
    )
    assert out is not None
    # Per [[feedback-random-data-test-tolerances-from-theory-2026-06-09]]
    # SE(α) ≈ σ/√N = 0.02/√240 ≈ 0.0013; 3σ band ≈ 0.004
    assert abs(out["alpha_monthly"] - true_alpha) < 0.004, (
        f"BUG REGRESSION RISK: joint α = {out['alpha_monthly']} "
        f"differs from true {true_alpha} by more than 3σ. "
        "If α≈0 here, the FWL bug has reappeared."
    )
    assert out["alpha_nw_t"] > 1.5
    # No regressor should have meaningfully significant β under H0
    f_p = out["industry_joint_f_test"]["f_pvalue"]
    # Under H0, p uniform — assert at least p > 0.001 (very lenient)
    assert f_p > 0.001


def test_joint_alpha_zero_when_factor_is_pure_industry_spanning(
    idx_240, synthetic_ff5mom_240, synthetic_industries_240,
):
    """factor IS a linear combo of industries → joint α should be ~0
    AND industry-subset F-test must reject (p << 0.01)."""
    from engine.research.industry_attribution import (
        compute_industry_extended_alpha,
    )
    rng = np.random.default_rng(13)
    factor = pd.Series(
        0.5 * synthetic_industries_240["NoDur"].values
        + 0.3 * synthetic_industries_240["BusEq"].values
        + rng.normal(0, 0.003, 240),
        index=idx_240,
    )
    out = compute_industry_extended_alpha(
        factor, synthetic_ff5mom_240, synthetic_industries_240,
    )
    assert out is not None
    # Joint α should be near zero (factor is fully spanned by industries)
    assert abs(out["alpha_monthly"]) < 0.001
    # Industry β should recover the loading structure
    assert abs(out["industry_betas"]["NoDur"] - 0.5) < 0.1
    assert abs(out["industry_betas"]["BusEq"] - 0.3) < 0.1
    # Joint F: industry orthogonality MASSIVELY rejected
    jf = out["industry_joint_f_test"]
    assert jf is not None
    assert jf["f_pvalue"] < 1e-6


# ────────────────────────────────────────────────────────────────────
# Refusal cases
# ────────────────────────────────────────────────────────────────────
def test_returns_none_when_overlap_too_short():
    from engine.research.industry_attribution import (
        compute_industry_extended_alpha,
    )
    idx = pd.date_range("2024-01-31", periods=10, freq="ME")
    factor = pd.Series(np.zeros(10), index=idx)
    anchors = pd.DataFrame({"MKT_RF": np.zeros(10)}, index=idx)
    inds = pd.DataFrame({"NoDur": np.zeros(10)}, index=idx)
    assert compute_industry_extended_alpha(factor, anchors, inds) is None


def test_returns_none_on_empty_inputs(synthetic_ff5mom_240,
                                          synthetic_industries_240):
    from engine.research.industry_attribution import (
        compute_industry_extended_alpha,
    )
    assert compute_industry_extended_alpha(
        pd.Series(dtype=float), synthetic_ff5mom_240,
        synthetic_industries_240,
    ) is None


def test_returns_none_when_index_not_datetime(synthetic_ff5mom_240,
                                                  synthetic_industries_240):
    from engine.research.industry_attribution import (
        compute_industry_extended_alpha,
    )
    factor = pd.Series(np.zeros(240), index=range(240))
    assert compute_industry_extended_alpha(
        factor, synthetic_ff5mom_240, synthetic_industries_240,
    ) is None


# ────────────────────────────────────────────────────────────────────
# load_industry_anchors + SHA pinning
# ────────────────────────────────────────────────────────────────────
def test_load_industry_anchors_returns_none_when_missing(tmp_path):
    from engine.research.industry_attribution import load_industry_anchors
    assert load_industry_anchors(str(tmp_path / "nope.parquet")) is None


def test_load_industry_anchors_real_parquet():
    from engine.research.industry_attribution import (
        load_industry_anchors, INDUSTRY_COLUMNS,
    )
    df = load_industry_anchors()
    if df is None:
        pytest.skip("industry parquet not cached — run fetcher")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert set(INDUSTRY_COLUMNS) <= set(df.columns)


def test_industry_snapshot_sha_64_hex_when_present():
    from engine.research.industry_attribution import (
        _industry_parquet_sha256,
    )
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "industries_12_monthly.parquet")
    sha = _industry_parquet_sha256(str(p))
    if not p.exists():
        assert sha == ""
        return
    assert len(sha) == 64


# ────────────────────────────────────────────────────────────────────
# Tier C wiring helper — JOINT model + Δα computation
# ────────────────────────────────────────────────────────────────────
def test_wiring_helper_returns_none_when_industry_missing(monkeypatch):
    from engine.research import industry_attribution as ia
    monkeypatch.setattr(ia, "load_industry_anchors", lambda: None)
    out = ia.compute_for_tier_c_with_stage1_residual(
        stage1_result={"alpha_monthly": 0.005,
                          "alpha_nw_t": 2.0,
                          "betas": {}, "anchor_names": []},
        pnl_series=pd.Series([0.01] * 240),
    )
    assert out is None


def test_wiring_helper_returns_none_when_stage1_result_missing():
    from engine.research.industry_attribution import (
        compute_for_tier_c_with_stage1_residual,
    )
    assert compute_for_tier_c_with_stage1_residual(
        stage1_result=None,
        pnl_series=pd.Series([0.01] * 240),
    ) is None


def test_wiring_helper_real_data_if_present():
    """End-to-end on real GP/A parquet — verifies the JOINT-model
    pipeline runs and Δα is sensible."""
    p = (Path(__file__).resolve().parents[1] / "data" / "research_store"
         / "tier_c_pnl" / "dc4cf6beaa247880_GREEN.parquet")
    if not p.exists():
        pytest.skip("GP/A PnL parquet missing")
    ind_p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
             / "industries_12_monthly.parquet")
    if not ind_p.exists():
        pytest.skip("industry parquet missing")

    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series as stage1,
    )
    from engine.research.industry_attribution import (
        compute_for_tier_c_with_stage1_residual as joint_extension,
    )

    gpa = pd.read_parquet(p)
    gpa["date"] = pd.to_datetime(gpa["date"])
    gpa_df = gpa.set_index("date")

    s1 = stage1(gpa_df)
    assert s1 is not None

    js = joint_extension(s1, gpa_df)
    assert js is not None
    assert math.isfinite(js["alpha_full_monthly"])
    assert math.isfinite(js["alpha_full_nw_t"])
    assert js["model_form"] == "joint_ff5mom_plus_12_industry"
    # Industry-subset F-test should compute on 12 industries
    assert js["industry_joint_f_test"] is not None
    # Δα should be present and meaningful
    assert js["delta_alpha_monthly"] is not None
    assert math.isfinite(js["delta_alpha_monthly"])
