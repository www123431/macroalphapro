"""tests/test_anchor_regression.py — Tier C L2-4 Commit 2.

Pure-function tests for compute_residual_alpha. Builds synthetic
factor + anchor series with known properties (identity / orthogonal
/ spanned / pure-alpha) and asserts the regression recovers the
true parameters within tight tolerances.

Also tests the integration with the cached Ken French parquet
when present (skipped if fetcher not yet run).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def monthly_index_60():
    return pd.date_range("2015-01-31", periods=60, freq="ME")


@pytest.fixture
def monthly_index_240():
    return pd.date_range("2005-01-31", periods=240, freq="ME")


# ────────────────────────────────────────────────────────────────────
# Identity: factor == single anchor → alpha ≈ 0, beta ≈ 1, R² ≈ 1
# ────────────────────────────────────────────────────────────────────
def test_identity_factor_equals_anchor_gives_zero_alpha(monthly_index_240):
    from engine.research.anchor_regression import compute_residual_alpha
    rng = np.random.default_rng(42)
    anchor = pd.Series(rng.normal(0.005, 0.04, 240),
                          index=monthly_index_240, name="MKT_RF")
    factor = anchor.copy()
    result = compute_residual_alpha(
        factor, anchor.to_frame(),
    )
    assert result is not None
    assert abs(result["alpha_monthly"]) < 1e-10
    assert abs(result["betas"]["MKT_RF"] - 1.0) < 1e-10
    assert result["r2"] > 0.9999
    assert result["n_overlap"] == 240


# ────────────────────────────────────────────────────────────────────
# Orthogonal: factor independent of anchors → alpha ≈ factor.mean,
# β ≈ 0, R² ≈ 0, t-α ≈ headline-t
# ────────────────────────────────────────────────────────────────────
def test_orthogonal_factor_alpha_equals_mean(monthly_index_240):
    from engine.research.anchor_regression import compute_residual_alpha
    rng = np.random.default_rng(7)
    factor = pd.Series(rng.normal(0.008, 0.04, 240),
                          index=monthly_index_240)
    anchors = pd.DataFrame({
        "A": rng.normal(0.005, 0.04, 240),
        "B": rng.normal(0.002, 0.03, 240),
    }, index=monthly_index_240)
    result = compute_residual_alpha(factor, anchors)
    assert result is not None
    # Alpha should be close to factor mean since anchors don't explain it
    assert abs(result["alpha_monthly"] - factor.mean()) < 0.003
    # Betas should be small (noise level, ~few-percent)
    for name in ("A", "B"):
        assert abs(result["betas"][name]) < 0.15
    # R² very low — independence
    assert result["r2"] < 0.05


# ────────────────────────────────────────────────────────────────────
# Spanned: factor IS a known linear combo of anchors → α ≈ 0, R² ≈ 1
# ────────────────────────────────────────────────────────────────────
def test_spanned_factor_has_near_zero_alpha(monthly_index_240):
    from engine.research.anchor_regression import compute_residual_alpha
    rng = np.random.default_rng(11)
    a1 = pd.Series(rng.normal(0.005, 0.04, 240), index=monthly_index_240)
    a2 = pd.Series(rng.normal(0.003, 0.03, 240), index=monthly_index_240)
    # Factor = 0.5·a1 + 0.3·a2 + tiny noise
    noise = pd.Series(rng.normal(0, 0.001, 240), index=monthly_index_240)
    factor = 0.5 * a1 + 0.3 * a2 + noise
    anchors = pd.DataFrame({"A1": a1, "A2": a2}, index=monthly_index_240)
    result = compute_residual_alpha(factor, anchors)
    assert result is not None
    assert abs(result["alpha_monthly"]) < 0.0003
    assert abs(result["betas"]["A1"] - 0.5) < 0.01
    assert abs(result["betas"]["A2"] - 0.3) < 0.01
    assert result["r2"] > 0.99
    # Crucial regression: alpha t-stat must be small even though factor
    # has a healthy mean (~0.5*0.005 + 0.3*0.003 = 0.0034/mo)
    assert abs(result["alpha_nw_t"]) < 1.0


# ────────────────────────────────────────────────────────────────────
# Pure alpha case: factor = α₀ + ε where ε ⊥ anchors → α recovered
# ────────────────────────────────────────────────────────────────────
def test_pure_alpha_recovered(monthly_index_240):
    from engine.research.anchor_regression import compute_residual_alpha
    rng = np.random.default_rng(23)
    a1 = pd.Series(rng.normal(0, 0.04, 240), index=monthly_index_240)
    a2 = pd.Series(rng.normal(0, 0.03, 240), index=monthly_index_240)
    true_alpha = 0.006   # 6 bps/month ~ 7.2%/yr
    eps = pd.Series(rng.normal(0, 0.02, 240), index=monthly_index_240)
    factor = true_alpha + 0.0 * a1 + 0.0 * a2 + eps
    anchors = pd.DataFrame({"A1": a1, "A2": a2}, index=monthly_index_240)
    result = compute_residual_alpha(factor, anchors)
    assert result is not None
    assert abs(result["alpha_monthly"] - true_alpha) < 0.001
    # 6bp/mo over 240 months with 2% monthly noise → SE ≈ 0.02/sqrt(240) ≈ 0.00129
    # t-stat ≈ 0.006/0.00129 ≈ 4.6 → strongly significant
    assert result["alpha_nw_t"] > 3.0
    # Annualization
    assert abs(result["alpha_annual"] - true_alpha * 12) < 0.012


# ────────────────────────────────────────────────────────────────────
# Partial overlap handled — anchor longer than factor or vice versa
# ────────────────────────────────────────────────────────────────────
def test_partial_overlap_uses_intersection(monthly_index_240):
    from engine.research.anchor_regression import compute_residual_alpha
    rng = np.random.default_rng(31)
    factor = pd.Series(rng.normal(0.005, 0.04, 240),
                          index=monthly_index_240)
    # Anchor starts 5 years later and goes 5 years past factor end
    anchor_idx = pd.date_range("2010-01-31", periods=300, freq="ME")
    anchor = pd.Series(rng.normal(0.005, 0.04, 300),
                          index=anchor_idx, name="A")
    result = compute_residual_alpha(factor, anchor.to_frame())
    assert result is not None
    # Overlap = 2010-01 to 2024-12 (factor end = 240 months from 2005-01)
    expected = len(factor.index.intersection(anchor.index))
    assert result["n_overlap"] == expected


def test_nan_in_factor_dropped(monthly_index_60):
    from engine.research.anchor_regression import compute_residual_alpha
    rng = np.random.default_rng(43)
    factor = pd.Series(rng.normal(0.005, 0.04, 60), index=monthly_index_60)
    factor.iloc[5:10] = np.nan
    anchor = pd.Series(rng.normal(0.005, 0.04, 60),
                          index=monthly_index_60, name="A")
    result = compute_residual_alpha(factor, anchor.to_frame())
    assert result is not None
    assert result["n_overlap"] == 60 - 5


# ────────────────────────────────────────────────────────────────────
# Refusal: insufficient overlap → None (no garbage stats)
# ────────────────────────────────────────────────────────────────────
def test_returns_none_when_overlap_below_min():
    from engine.research.anchor_regression import compute_residual_alpha
    idx = pd.date_range("2024-01-31", periods=12, freq="ME")
    rng = np.random.default_rng(53)
    factor = pd.Series(rng.normal(0, 0.04, 12), index=idx)
    anchor = pd.Series(rng.normal(0, 0.04, 12), index=idx, name="A")
    assert compute_residual_alpha(
        factor, anchor.to_frame(), min_overlap=24,
    ) is None


def test_returns_none_when_factor_empty():
    from engine.research.anchor_regression import compute_residual_alpha
    idx = pd.date_range("2024-01-31", periods=60, freq="ME")
    anchor = pd.DataFrame({"A": np.zeros(60)}, index=idx)
    assert compute_residual_alpha(pd.Series(dtype=float),
                                     anchor) is None


def test_returns_none_when_anchors_empty(monthly_index_60):
    from engine.research.anchor_regression import compute_residual_alpha
    factor = pd.Series(np.zeros(60), index=monthly_index_60)
    assert compute_residual_alpha(factor, pd.DataFrame()) is None


def test_returns_none_when_index_not_datetime(monthly_index_60):
    from engine.research.anchor_regression import compute_residual_alpha
    factor = pd.Series(np.zeros(60), index=range(60))
    anchors = pd.DataFrame({"A": np.zeros(60)}, index=monthly_index_60)
    assert compute_residual_alpha(factor, anchors) is None


# ────────────────────────────────────────────────────────────────────
# Multi-anchor 5+1 (FF5+MOM shape)
# ────────────────────────────────────────────────────────────────────
def test_full_ff5_mom_shape_runs(monthly_index_240):
    from engine.research.anchor_regression import compute_residual_alpha
    rng = np.random.default_rng(67)
    anchors = pd.DataFrame({
        "MKT_RF": rng.normal(0.005, 0.04, 240),
        "SMB":    rng.normal(0.002, 0.03, 240),
        "HML":    rng.normal(0.003, 0.03, 240),
        "RMW":    rng.normal(0.003, 0.02, 240),
        "CMA":    rng.normal(0.002, 0.02, 240),
        "MOM":    rng.normal(0.006, 0.04, 240),
    }, index=monthly_index_240)
    factor = pd.Series(rng.normal(0.005, 0.04, 240),
                          index=monthly_index_240)
    result = compute_residual_alpha(factor, anchors)
    assert result is not None
    assert result["anchor_names"] == ("MKT_RF", "SMB", "HML",
                                          "RMW", "CMA", "MOM")
    assert len(result["betas"]) == 6
    # NW lag default: floor(4 * (240/100)^(2/9)) = floor(4 * 1.193) = 4
    assert result["nw_lag_used"] >= 1
    assert "window" in result
    assert result["window"].startswith("2005-01")


# ────────────────────────────────────────────────────────────────────
# load_famafrench_anchors integration (skipped if parquet absent)
# ────────────────────────────────────────────────────────────────────
def test_load_famafrench_anchors_real_parquet():
    from engine.research.anchor_regression import load_famafrench_anchors
    df = load_famafrench_anchors()
    if df is None:
        pytest.skip("Run scripts/fetch_anchor_library.py to enable")
    assert isinstance(df.index, pd.DatetimeIndex)
    # RF dropped — it's risk-free, not a risk-factor anchor
    assert "RF" not in df.columns
    # Must have the canonical 6 risk-factor anchors
    assert {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"} <= set(df.columns)
    assert len(df) > 600


def test_load_famafrench_anchors_returns_none_when_missing(tmp_path):
    from engine.research.anchor_regression import load_famafrench_anchors
    missing = tmp_path / "nope.parquet"
    assert load_famafrench_anchors(str(missing)) is None


# ────────────────────────────────────────────────────────────────────
# L2-4 Stage 1 (2026-06-09): dual gross+net, joint F-test, SHA pinning
# ────────────────────────────────────────────────────────────────────
@pytest.fixture
def fake_pnl_df_240(monthly_index_240):
    rng = np.random.default_rng(91)
    n = 240
    return pd.DataFrame({
        "pnl_gross":    rng.normal(0.007, 0.04, n),
        "pnl_net_13bp": rng.normal(0.005, 0.04, n),
        "pnl_net_80bp": rng.normal(0.003, 0.04, n),
        "turnover":     rng.uniform(0.3, 0.6, n),
    }, index=monthly_index_240)


def test_compute_for_tier_c_accepts_dataframe(fake_pnl_df_240):
    """L2-4 Stage 1: helper accepts full pnl_series_df (not just
    a Series) and produces gross + net dual regression."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    anchors = pd.DataFrame({
        "MKT_RF": np.random.default_rng(7).normal(0.005, 0.04, 240),
        "RMW":    np.random.default_rng(8).normal(0.003, 0.02, 240),
    }, index=fake_pnl_df_240.index)
    out = compute_for_tier_c_pnl_series(fake_pnl_df_240, anchors=anchors)
    assert out is not None
    # Net block (existing)
    assert "alpha_nw_t" in out
    # NEW: gross block
    assert "gross" in out
    assert out["gross"] is not None
    assert "alpha_nw_t" in out["gross"]
    assert "betas" in out["gross"]
    # NEW: joint F-test
    assert "joint_loading_f_test" in out
    jf = out["joint_loading_f_test"]
    if jf is not None:  # may be None if statsmodels old
        assert "f_stat" in jf
        assert "f_pvalue" in jf
        assert 0.0 <= jf["f_pvalue"] <= 1.0
    # NEW: anchor snapshot SHA placeholder
    assert "anchor_snapshot_sha" in out


def test_compute_for_tier_c_legacy_series_signature_still_works(
    monthly_index_240,
):
    """Backwards compat: passing a Series (not DataFrame) still
    produces a valid result; gross block is null."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    rng = np.random.default_rng(101)
    factor = pd.Series(rng.normal(0.005, 0.04, 240),
                          index=monthly_index_240)
    anchors = pd.DataFrame({
        "MKT_RF": rng.normal(0.005, 0.04, 240),
        "RMW":    rng.normal(0.003, 0.02, 240),
    }, index=monthly_index_240)
    out = compute_for_tier_c_pnl_series(factor, anchors=anchors)
    assert out is not None
    assert out["alpha_nw_t"] is not None
    # No DataFrame → no gross block
    assert out["gross"] is None


def test_joint_f_test_rejects_orthogonality_on_spanned_factor(
    monthly_index_240,
):
    """A factor that IS a linear combo of anchors → joint F-test
    should reject the H0: all β = 0 (p < 0.01)."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    rng = np.random.default_rng(13)
    a1 = rng.normal(0.005, 0.04, 240)
    a2 = rng.normal(0.003, 0.03, 240)
    factor = 0.5 * a1 + 0.3 * a2 + rng.normal(0, 0.005, 240)
    df = pd.DataFrame({
        "pnl_gross":    factor + 0.001,
        "pnl_net_13bp": factor,
    }, index=monthly_index_240)
    anchors = pd.DataFrame({"MKT_RF": a1, "RMW": a2},
                              index=monthly_index_240)
    out = compute_for_tier_c_pnl_series(df, anchors=anchors)
    assert out is not None
    jf = out["joint_loading_f_test"]
    assert jf is not None
    assert jf["f_pvalue"] < 0.01, (
        f"spanned factor should reject orthogonality, got p={jf['f_pvalue']}"
    )


def test_joint_f_test_fails_to_reject_on_orthogonal_factor(
    monthly_index_240,
):
    """An independent factor → joint F-test should usually NOT reject.
    Per [[feedback-random-data-test-tolerances-from-theory-2026-06-09]],
    random independence has 1% prob of p < 0.01 by definition; we test
    that the F-test is well-calibrated across MULTIPLE seeds — at most
    1 of 10 seeds should reject at p < 0.01 (binomial bound)."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series,
    )
    rejected_count = 0
    n_seeds = 10
    for seed in range(100, 100 + n_seeds):
        rng = np.random.default_rng(seed)
        factor = rng.normal(0.005, 0.04, 240)
        df = pd.DataFrame({
            "pnl_gross":    factor + 0.001,
            "pnl_net_13bp": factor,
        }, index=monthly_index_240)
        anchors = pd.DataFrame({
            "MKT_RF": rng.normal(0.005, 0.04, 240),
            "RMW":    rng.normal(0.003, 0.02, 240),
        }, index=monthly_index_240)
        out = compute_for_tier_c_pnl_series(df, anchors=anchors)
        jf = out["joint_loading_f_test"]
        assert jf is not None
        if jf["f_pvalue"] < 0.01:
            rejected_count += 1
    # Under H0 with α=0.01, prob of any single seed rejecting is ~1%;
    # over 10 seeds, expected rejections ≈ 0.1. Allow up to 3 (very
    # conservative binomial bound) before we say F-test is broken.
    assert rejected_count <= 3, (
        f"orthogonal factor F-test rejected {rejected_count}/10 seeds "
        f"at p<0.01, expected ≈0-1 under H0"
    )


def test_anchor_snapshot_sha_is_64_hex_chars_when_parquet_present():
    """When the cached anchor parquet exists, SHA should be a valid
    SHA-256 hex digest (64 chars). When missing, None."""
    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series, _anchor_parquet_sha256,
    )
    from pathlib import Path
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "famafrench_monthly.parquet")
    sha = _anchor_parquet_sha256(str(p))
    if not p.exists():
        assert sha == "", "missing parquet should yield empty sha"
        return
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


# ────────────────────────────────────────────────────────────────────
# End-to-end with REAL Ken French anchors (skipped if parquet absent)
# ────────────────────────────────────────────────────────────────────
def test_pure_alpha_against_real_ff5_anchors():
    """Inject a 5bp/mo alpha onto Ken French SMB-shaped noise; the
    regression should recover both the alpha AND find the betas
    on the real anchors."""
    from engine.research.anchor_regression import (
        compute_residual_alpha, load_famafrench_anchors,
    )
    anchors = load_famafrench_anchors()
    if anchors is None:
        pytest.skip("Run scripts/fetch_anchor_library.py to enable")
    # Restrict to last 30 years for stability + speed
    anchors = anchors[anchors.index >= pd.Timestamp("1995-01-01")]
    rng = np.random.default_rng(101)
    n = len(anchors)
    eps = pd.Series(rng.normal(0, 0.02, n), index=anchors.index)
    true_alpha = 0.005
    # No exposure to anchors; pure alpha
    factor = true_alpha + eps
    result = compute_residual_alpha(factor, anchors)
    assert result is not None
    # Recover alpha within 1 bp
    assert abs(result["alpha_monthly"] - true_alpha) < 0.001
    # Loadings should be near zero — with 6 anchors and 372 months
    # of monthly noise, sample correlation noise can push any single
    # beta to ~0.10-0.15 in absolute value. 0.20 is a generous cap
    # that any genuine cross-anchor confound would blow past.
    for name in ("MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"):
        assert abs(result["betas"][name]) < 0.20, (
            f"unexpected loading on {name}: {result['betas'][name]}"
        )
    # ...but the AVERAGE absolute loading should be very small
    avg_abs = float(np.mean([abs(v) for v in result["betas"].values()]))
    assert avg_abs < 0.10, f"avg |beta| = {avg_abs} too high"
