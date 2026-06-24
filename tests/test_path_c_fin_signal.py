"""
tests/test_path_c_fin_signal.py — Sprint D-3.5 FIN cross-section composite tests.

Pre-registration: docs/spec_path_d_dhs_behavioral_2factor_v1.md (id=62)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.path_c.fin_signal import (
    DECILE_LONG_THRESHOLD,
    DECILE_SHORT_THRESHOLD,
    compute_fin_composite,
    assign_fin_decile_legs,
    _z_within_quarter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────

def test_decile_thresholds_locked():
    assert DECILE_LONG_THRESHOLD == 0.9
    assert DECILE_SHORT_THRESHOLD == 0.1


# ─────────────────────────────────────────────────────────────────────────────
# z-norm within quarter
# ─────────────────────────────────────────────────────────────────────────────

def test_z_within_quarter_basic():
    df = pd.DataFrame({
        "fiscal_yearq": ["2014Q1"] * 5,
        "nsi":          [-2.0, -1.0, 0.0, 1.0, 2.0],
    })
    z = _z_within_quarter(df, "nsi")
    # symmetric input → z-norm should also be symmetric, mean 0
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=1) - 1.0) < 1e-9


def test_z_within_quarter_separates_quarters():
    df = pd.DataFrame({
        "fiscal_yearq": ["2014Q1", "2014Q1", "2014Q1", "2014Q2", "2014Q2", "2014Q2"],
        "nsi":          [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
    })
    z = _z_within_quarter(df, "nsi")
    q1_z = z.iloc[:3]
    q2_z = z.iloc[3:]
    # Same shape within each quarter despite different scale
    np.testing.assert_allclose(q1_z.values, q2_z.values, rtol=1e-9)


def test_z_within_quarter_nan_passthrough():
    df = pd.DataFrame({
        "fiscal_yearq": ["2014Q1"] * 4,
        "nsi":          [-1.0, 0.0, 1.0, np.nan],
    })
    z = _z_within_quarter(df, "nsi")
    assert pd.isna(z.iloc[3])
    # First 3 should still be normalized
    assert abs(z.iloc[:3].mean()) < 1e-9


def test_z_within_quarter_single_obs_returns_nan():
    df = pd.DataFrame({
        "fiscal_yearq": ["2014Q1"],
        "nsi":          [5.0],
    })
    z = _z_within_quarter(df, "nsi")
    assert pd.isna(z.iloc[0])  # can't compute std from 1 obs


def test_z_within_quarter_zero_std_returns_nan():
    df = pd.DataFrame({
        "fiscal_yearq": ["2014Q1"] * 4,
        "nsi":          [2.0, 2.0, 2.0, 2.0],
    })
    z = _z_within_quarter(df, "nsi")
    assert z.isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# compute_fin_composite
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_fin_composite_basic():
    df = pd.DataFrame({
        "ticker":       ["A", "B", "C", "D"],
        "fiscal_yearq": ["2014Q1"] * 4,
        "nsi":          [-0.2, -0.1, 0.1, 0.2],
        "acc_scaled":   [0.05, -0.05, 0.10, -0.10],
    })
    out = compute_fin_composite(df)
    assert "z_nsi" in out.columns
    assert "z_acc" in out.columns
    assert "fin" in out.columns
    assert "fin_for_long_rank" in out.columns
    # fin should equal mean of components
    np.testing.assert_allclose(
        out["fin"].values,
        ((out["z_nsi"] + out["z_acc"]) / 2.0).values,
        rtol=1e-9,
    )
    # fin_for_long_rank = -fin
    np.testing.assert_allclose(out["fin_for_long_rank"].values, -out["fin"].values, rtol=1e-9)


def test_compute_fin_composite_handles_partial_nan():
    """If one component NaN, fin should equal the other component (z-score)."""
    df = pd.DataFrame({
        "ticker":       ["A", "B", "C", "D"],
        "fiscal_yearq": ["2014Q1"] * 4,
        "nsi":          [-0.2, -0.1, 0.1, 0.2],
        "acc_scaled":   [np.nan, np.nan, np.nan, np.nan],  # all NaN
    })
    out = compute_fin_composite(df)
    # z_acc all NaN → fin should equal z_nsi
    valid_fin = out["fin"].notna()
    np.testing.assert_allclose(
        out.loc[valid_fin, "fin"].values,
        out.loc[valid_fin, "z_nsi"].values,
        rtol=1e-9,
    )


def test_compute_fin_composite_both_nan_yields_nan():
    df = pd.DataFrame({
        "ticker":       ["A", "B"],
        "fiscal_yearq": ["2014Q1", "2014Q1"],
        "nsi":          [np.nan, np.nan],
        "acc_scaled":   [np.nan, np.nan],
    })
    out = compute_fin_composite(df)
    assert out["fin"].isna().all()


def test_compute_fin_composite_empty():
    df = pd.DataFrame({"nsi": [], "acc_scaled": [], "fiscal_yearq": []})
    out = compute_fin_composite(df)
    assert out.empty or out["fin"].isna().all()


def test_compute_fin_composite_missing_columns_raises():
    df = pd.DataFrame({"fiscal_yearq": ["2014Q1"], "nsi": [0.1]})
    with pytest.raises(ValueError, match="missing columns"):
        compute_fin_composite(df)


# ─────────────────────────────────────────────────────────────────────────────
# assign_fin_decile_legs (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def test_assign_fin_decile_legs_produces_long_short_flat():
    """20 firms in 1 quarter; top decile (2 firms) → long, bottom decile (2) → short."""
    rng = np.random.default_rng(42)
    n = 20
    df = pd.DataFrame({
        "ticker":       [f"T{i:02d}" for i in range(n)],
        "fiscal_yearq": ["2014Q1"] * n,
        "nsi":          rng.normal(0, 0.1, n),
        "acc_scaled":   rng.normal(0, 0.05, n),
    })
    out = assign_fin_decile_legs(df)
    assert "leg" in out.columns
    leg_counts = out["leg"].value_counts()
    # Approx decile assignment (some variation acceptable with n=20)
    assert leg_counts.get("long", 0) >= 1
    assert leg_counts.get("short", 0) >= 1
    assert leg_counts.get("flat", 0) >= 10


def test_assign_fin_decile_legs_direction_low_fin_is_long():
    """Spec §2.5 step 2: low FIN composite = LONG (negate convention).
    Construct firms with deliberately different FIN: lowest fin → top decile rank → long."""
    n = 20
    # Make 20 firms with varying nsi (acc_scaled all 0)
    nsi_vals = np.linspace(-0.4, +0.4, n)  # firm 0 has lowest nsi (most buyback)
    df = pd.DataFrame({
        "ticker":       [f"T{i:02d}" for i in range(n)],
        "fiscal_yearq": ["2014Q1"] * n,
        "nsi":          nsi_vals,
        "acc_scaled":   [0.0] * n,
    })
    out = assign_fin_decile_legs(df)
    # Firm 0 (lowest NSI / most buyback) should be LONG
    # Firm 19 (highest NSI / most issuance) should be SHORT
    leg0 = out.loc[out["ticker"] == "T00", "leg"].iloc[0]
    leg19 = out.loc[out["ticker"] == "T19", "leg"].iloc[0]
    assert leg0 == "long"
    assert leg19 == "short"


def test_assign_fin_decile_legs_empty():
    df = pd.DataFrame(columns=["ticker", "fiscal_yearq", "nsi", "acc_scaled"])
    out = assign_fin_decile_legs(df)
    assert "leg" in out.columns
    assert out.empty


def test_assign_fin_decile_legs_separates_quarters():
    """Same firms across 2 quarters; ranks should be quarter-independent."""
    n = 20
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "ticker":       [f"T{i:02d}" for i in range(n)] * 2,
        "fiscal_yearq": ["2014Q1"] * n + ["2014Q2"] * n,
        "nsi":          np.concatenate([rng.normal(0, 0.1, n), rng.normal(0, 0.05, n)]),
        "acc_scaled":   np.concatenate([rng.normal(0, 0.05, n), rng.normal(0, 0.03, n)]),
    })
    out = assign_fin_decile_legs(df)
    # Each quarter should have its own long/short
    q1 = out[out["fiscal_yearq"] == "2014Q1"]
    q2 = out[out["fiscal_yearq"] == "2014Q2"]
    assert q1["leg"].nunique() >= 2
    assert q2["leg"].nunique() >= 2
