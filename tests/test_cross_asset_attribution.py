"""tests/test_cross_asset_attribution.py — JOINT-model FF5+MOM ∪
Industry ∪ Macro pure-function tests.

Per [[feedback-fwl-sequential-residual-trap-2026-06-09]] α is from
JOINT model. Sequential is reporting only.
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
def synth_ff5mom(idx_240):
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
def synth_industry(idx_240):
    rng = np.random.default_rng(13)
    return pd.DataFrame(
        {f"IND{i}": rng.normal(0.005, 0.04, 240) for i in range(6)},
        index=idx_240,
    )


@pytest.fixture
def synth_macro(idx_240):
    rng = np.random.default_rng(23)
    return pd.DataFrame({
        "VIX_change":        rng.normal(0, 5.0, 240),
        "DXY_return":        rng.normal(0, 0.02, 240),
        "BAA_spread_change": rng.normal(0, 0.2, 240),
        "T10Y3M_change":     rng.normal(0, 0.3, 240),
        "T10YIE_change":     rng.normal(0, 0.1, 240),
    }, index=idx_240)


# ────────────────────────────────────────────────────────────────────
# compute_macro_extended_alpha
# ────────────────────────────────────────────────────────────────────
def test_pure_alpha_recovered_with_macro_panel(
    idx_240, synth_ff5mom, synth_industry, synth_macro,
):
    from engine.research.cross_asset_attribution import (
        compute_macro_extended_alpha,
    )
    rng = np.random.default_rng(101)
    true_alpha = 0.006
    factor = pd.Series(true_alpha + rng.normal(0, 0.02, 240),
                          index=idx_240)
    out = compute_macro_extended_alpha(
        factor, synth_ff5mom, synth_industry, synth_macro,
    )
    assert out is not None
    # 3-sigma band per random-data tolerance doctrine
    assert abs(out["alpha_monthly"] - true_alpha) < 0.005
    # Both subset F-tests should NOT reject (independent regressors)
    if out["industry_joint_f_test"]:
        assert out["industry_joint_f_test"]["f_pvalue"] > 0.001
    assert out["macro_joint_f_test"]["f_pvalue"] > 0.001


def test_macro_subset_f_rejects_when_factor_loads_on_macro(
    idx_240, synth_ff5mom, synth_industry, synth_macro,
):
    """factor IS a linear combo of macro shocks → macro subset F
    must reject (p << 0.01)."""
    from engine.research.cross_asset_attribution import (
        compute_macro_extended_alpha,
    )
    rng = np.random.default_rng(31)
    # Heavy negative VIX_change loading + positive DXY (typical
    # carry-crash exposure pattern)
    factor = pd.Series(
        -0.0002 * synth_macro["VIX_change"].values
        + 0.5 * synth_macro["DXY_return"].values
        + rng.normal(0, 0.005, 240),
        index=idx_240,
    )
    out = compute_macro_extended_alpha(
        factor, synth_ff5mom, synth_industry, synth_macro,
    )
    assert out is not None
    assert out["macro_joint_f_test"]["f_pvalue"] < 1e-6
    # Loadings recovered within tolerance
    assert abs(out["macro_betas"]["VIX_change"] - (-0.0002)) < 0.0001
    assert abs(out["macro_betas"]["DXY_return"] - 0.5) < 0.1


def test_industries_optional_for_cross_asset_sleeves(
    idx_240, synth_ff5mom, synth_macro,
):
    """For pure cross-asset sleeves we pass industries=None to skip
    the US-equity-industry panel (mis-specified for carry/TSMOM)."""
    from engine.research.cross_asset_attribution import (
        compute_macro_extended_alpha,
    )
    rng = np.random.default_rng(43)
    factor = pd.Series(0.005 + rng.normal(0, 0.02, 240), index=idx_240)
    out = compute_macro_extended_alpha(
        factor, synth_ff5mom, industries=None, macro=synth_macro,
    )
    assert out is not None
    # Industry section should be empty
    assert out["industry_betas"] == {}
    assert out["industry_joint_f_test"] is None
    # Macro section still computed
    assert out["macro_joint_f_test"] is not None
    assert out["panels_included"]["industry"] == []
    assert len(out["panels_included"]["macro"]) == 5


def test_returns_none_when_overlap_too_short(synth_ff5mom, synth_macro):
    from engine.research.cross_asset_attribution import (
        compute_macro_extended_alpha,
    )
    idx = pd.date_range("2024-01-31", periods=10, freq="ME")
    f = pd.Series(np.zeros(10), index=idx)
    ff5 = pd.DataFrame({"MKT_RF": np.zeros(10)}, index=idx)
    mac = pd.DataFrame({"VIX_change": np.zeros(10)}, index=idx)
    assert compute_macro_extended_alpha(f, ff5, None, mac) is None


def test_returns_none_on_empty_inputs(synth_ff5mom, synth_macro):
    from engine.research.cross_asset_attribution import (
        compute_macro_extended_alpha,
    )
    assert compute_macro_extended_alpha(
        pd.Series(dtype=float), synth_ff5mom, None, synth_macro,
    ) is None


# ────────────────────────────────────────────────────────────────────
# load_macro_anchors + SHA
# ────────────────────────────────────────────────────────────────────
def test_load_macro_anchors_returns_none_when_missing(tmp_path):
    from engine.research.cross_asset_attribution import load_macro_anchors
    assert load_macro_anchors(str(tmp_path / "nope.parquet")) is None


def test_load_macro_anchors_real_parquet():
    from engine.research.cross_asset_attribution import (
        load_macro_anchors, MACRO_COLUMNS,
    )
    df = load_macro_anchors()
    if df is None:
        pytest.skip("macro parquet not cached — run fetcher")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert set(MACRO_COLUMNS) <= set(df.columns)


def test_macro_snapshot_sha_64_hex_when_present():
    from engine.research.cross_asset_attribution import (
        _macro_parquet_sha256,
    )
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "cross_asset_macro_monthly.parquet")
    sha = _macro_parquet_sha256(str(p))
    if not p.exists():
        assert sha == ""
        return
    assert len(sha) == 64


# ────────────────────────────────────────────────────────────────────
# Tier C wiring helper
# ────────────────────────────────────────────────────────────────────
def test_wiring_helper_returns_none_when_macro_missing(monkeypatch):
    from engine.research import cross_asset_attribution as xa
    monkeypatch.setattr(xa, "load_macro_anchors", lambda: None)
    out = xa.compute_for_tier_c_with_macro(
        stage1_result={"alpha_monthly": 0.005, "alpha_nw_t": 2.0,
                          "betas": {}, "anchor_names": []},
        industry_extension=None,
        pnl_series=pd.Series([0.01] * 240),
    )
    assert out is None


def test_wiring_helper_returns_none_when_stage1_missing():
    from engine.research.cross_asset_attribution import (
        compute_for_tier_c_with_macro,
    )
    assert compute_for_tier_c_with_macro(
        stage1_result=None, industry_extension=None,
        pnl_series=pd.Series([0.01] * 240),
    ) is None


def test_wiring_helper_real_data_for_equity_sleeve():
    """End-to-end on real GP/A parquet with industry included."""
    p = (Path(__file__).resolve().parents[1] / "data" / "research_store"
         / "tier_c_pnl" / "dc4cf6beaa247880_GREEN.parquet")
    if not p.exists():
        pytest.skip("GP/A PnL parquet missing")
    mp = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
          / "cross_asset_macro_monthly.parquet")
    if not mp.exists():
        pytest.skip("macro parquet missing")

    from engine.research.anchor_regression import (
        compute_for_tier_c_pnl_series as stage1,
    )
    from engine.research.industry_attribution import (
        compute_for_tier_c_with_stage1_residual as ix_ext,
    )
    from engine.research.cross_asset_attribution import (
        compute_for_tier_c_with_macro as xa_ext,
    )

    gpa = pd.read_parquet(p)
    gpa["date"] = pd.to_datetime(gpa["date"])
    gpa_df = gpa.set_index("date")

    s1 = stage1(gpa_df)
    ix = ix_ext(s1, gpa_df)
    xa = xa_ext(s1, ix, gpa_df, include_industry=True)
    assert xa is not None
    assert math.isfinite(xa["alpha_full_nw_t"])
    # Per Phase 2 Commit 4 (2026-06-09): LRV FX carry anchors auto-loaded
    # when parquet present. model_form reflects which panels were
    # actually included; for equity sleeves running through default
    # path with LRV cached, it gets joined too.
    assert xa["model_form"] in (
        "joint_ff5mom_plus_industry_plus_macro",
        "joint_ff5mom_plus_industry_plus_macro_plus_lrv_fx",
    )
    assert xa["macro_joint_f_test"] is not None
