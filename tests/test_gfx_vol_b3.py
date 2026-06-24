"""tests/test_gfx_vol_b3.py — B.3 MSSS global FX volatility anchor.

Covers:
  1. compute_gfx_vol pure function (the σ_FX math)
  2. msss_gfx_vol AnchorLibrary registration + loader
  3. tuple-form conditional_on in lens_registry.should_execute
     (the fix that lets cross_asset_extension run for FX sleeves)
  4. include_gfx_vol gate: joined for fx, NOT joined for equity
     (sample-truncation protection — GFX_VOL starts 1999, equity
     PnL goes back to 1992)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


GFX_PARQUET = (Path(__file__).resolve().parents[1]
                 / "data" / "anchor_library" / "gfx_vol_monthly.parquet")


# ────────────────────────────────────────────────────────────────────
# 1. compute_gfx_vol — pure σ_FX math
# ────────────────────────────────────────────────────────────────────
def test_gfx_vol_constant_abs_returns():
    """All currencies move |0.5%| every day → σ_FX level = 0.005
    exactly, change = 0 after the first month."""
    from scripts.fetch_gfx_vol_msss import compute_gfx_vol
    days = pd.bdate_range("2020-01-01", "2020-06-30")
    panel = pd.DataFrame(0.005, index=days,
                            columns=list("ABCDEFGHIJ"))
    out = compute_gfx_vol(panel)
    assert (out["GFX_VOL_level"] - 0.005).abs().max() < 1e-12
    assert out["GFX_VOL_change"].iloc[1:].abs().max() < 1e-12


def test_gfx_vol_spike_month_detected():
    """One crisis month with 3x the daily moves → level triples,
    change spikes positive then mean-reverts negative."""
    from scripts.fetch_gfx_vol_msss import compute_gfx_vol
    days = pd.bdate_range("2020-01-01", "2020-04-30")
    panel = pd.DataFrame(0.004, index=days, columns=list("ABCDEF"))
    crisis = (days >= "2020-03-01") & (days <= "2020-03-31")
    panel.loc[crisis, :] = 0.012
    out = compute_gfx_vol(panel)
    mar = out.loc["2020-03-31", "GFX_VOL_level"]
    feb = out.loc["2020-02-29", "GFX_VOL_level"]
    assert abs(mar / feb - 3.0) < 0.01
    assert out.loc["2020-03-31", "GFX_VOL_change"] > 0
    assert out.loc["2020-04-30", "GFX_VOL_change"] < 0


def test_gfx_vol_sparse_days_filtered():
    """Days with < MIN_CCYS_PER_DAY currencies reporting are dropped
    from the daily mean (holiday-calendar mismatch protection)."""
    from scripts.fetch_gfx_vol_msss import (
        compute_gfx_vol, MIN_CCYS_PER_DAY,
    )
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    panel = pd.DataFrame(0.005, index=days, columns=list("ABCDEF"))
    # One day: only 2 currencies report, both with a HUGE move that
    # would distort the monthly mean if not filtered
    sparse_day = days[10]
    panel.loc[sparse_day, :] = np.nan
    panel.loc[sparse_day, ["A", "B"]] = 0.10
    out = compute_gfx_vol(panel)
    # Jan level unaffected by the distorted sparse day
    assert abs(out["GFX_VOL_level"].iloc[0] - 0.005) < 1e-9


def test_gfx_vol_short_months_dropped():
    """Months with < MIN_DAYS_PER_MONTH valid days produce no row."""
    from scripts.fetch_gfx_vol_msss import compute_gfx_vol
    # Only 5 business days of data in the month
    days = pd.bdate_range("2020-01-01", "2020-01-07")
    panel = pd.DataFrame(0.005, index=days, columns=list("ABCDEF"))
    out = compute_gfx_vol(panel)
    assert len(out) == 0


# ────────────────────────────────────────────────────────────────────
# 2. Registration + loader
# ────────────────────────────────────────────────────────────────────
def test_msss_gfx_vol_registered():
    from engine.research.anchor_library_registry import get_library
    lib = get_library("msss_gfx_vol")
    assert lib is not None
    assert lib.units == "decimal"
    # Only the INNOVATION is a regressor — level is non-stationary
    assert lib.anchor_columns == ("GFX_VOL_change",)
    # NOT applicable to equity (sample-truncation + mis-specification)
    assert "equity" not in lib.applicable_asset_classes
    assert "fx" in lib.applicable_asset_classes


@pytest.mark.skipif(not GFX_PARQUET.is_file(),
                     reason="GFX_VOL parquet not cached")
def test_msss_gfx_vol_loads_real_parquet():
    from engine.research.anchor_library_registry import load_library
    df = load_library("msss_gfx_vol")
    assert df is not None
    assert list(df.columns) == ["GFX_VOL_change"]
    # Sanity: daily-|return| innovations are small decimals
    assert df["GFX_VOL_change"].abs().max() < 0.02
    assert len(df) > 200   # ~27 years monthly


# ────────────────────────────────────────────────────────────────────
# 3. Tuple-form conditional_on
# ────────────────────────────────────────────────────────────────────
def _mk_lens(conditional_on):
    from engine.research.lens_registry import LensDeclaration
    return LensDeclaration(
        name="t", version="v", applicable_to={},
        input_protocols=(), output_protocol="CrossAssetExtensionOutput",
        conditional_on=conditional_on, fallback_chain=(),
        output_schema={"primary": "x", "secondary": ()},
        consumed_by=(), runner=lambda *a: None,
    )


def test_tuple_conditional_first_present_wins():
    from engine.research.lens_registry import should_execute
    lens = _mk_lens({
        "lens": ("anchor_regression", "fx_carry_anchor_regression"),
        "condition": lambda out: out.get("alpha_nw_t", 0) >= 1.0,
    })
    # Only the FX lens produced output (the carry path)
    ok, reason = should_execute(
        lens, {"fx_carry_anchor_regression": {"alpha_nw_t": 2.5}},
    )
    assert ok, reason


def test_tuple_conditional_skips_when_neither_present():
    from engine.research.lens_registry import should_execute
    lens = _mk_lens({
        "lens": ("anchor_regression", "fx_carry_anchor_regression"),
        "condition": lambda out: True,
    })
    ok, reason = should_execute(lens, {})
    assert not ok
    assert "produced no output" in reason


def test_tuple_conditional_predicate_still_gates():
    from engine.research.lens_registry import should_execute
    lens = _mk_lens({
        "lens": ("anchor_regression", "fx_carry_anchor_regression"),
        "condition": lambda out: out.get("alpha_nw_t", 0) >= 1.0,
        "skip_reason_if_unmet": "low alpha",
    })
    ok, reason = should_execute(
        lens, {"fx_carry_anchor_regression": {"alpha_nw_t": 0.2}},
    )
    assert not ok
    assert reason == "low alpha"


def test_string_conditional_unchanged():
    """Backward compat: plain-string lens name still works."""
    from engine.research.lens_registry import should_execute
    lens = _mk_lens({
        "lens": "anchor_regression",
        "condition": lambda out: out.get("alpha_nw_t", 0) >= 1.0,
    })
    ok, _ = should_execute(
        lens, {"anchor_regression": {"alpha_nw_t": 3.0}},
    )
    assert ok


# ────────────────────────────────────────────────────────────────────
# 4. include_gfx_vol gate
# ────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not GFX_PARQUET.is_file(),
                     reason="GFX_VOL parquet not cached")
def test_gfx_vol_joined_for_fx_excluded_for_equity():
    """Run compute_for_tier_c_with_macro twice on synthetic data:
    include_gfx_vol=True must put GFX_VOL_change in macro_betas;
    include_gfx_vol=False must not."""
    from engine.research.cross_asset_attribution import (
        compute_for_tier_c_with_macro,
    )
    rng = np.random.default_rng(11)
    idx = pd.date_range("2005-01-31", periods=240, freq="ME")
    pnl = pd.Series(0.004 + rng.normal(0, 0.02, 240), index=idx)
    stage1 = {"alpha_monthly": 0.004, "alpha_nw_t": 2.0,
                "betas": {}, "anchor_names": []}
    df = pd.DataFrame({"pnl_net_13bp": pnl, "pnl_gross": pnl})

    out_fx = compute_for_tier_c_with_macro(
        stage1, None, df, include_industry=False,
        include_lrv_fx=False, include_gfx_vol=True,
    )
    out_eq = compute_for_tier_c_with_macro(
        stage1, None, df, include_industry=False,
        include_lrv_fx=False, include_gfx_vol=False,
    )
    if out_fx is None or out_eq is None:
        pytest.skip("macro parquet not cached")
    assert "GFX_VOL_change" in out_fx["macro_betas"]
    assert "GFX_VOL_change" not in out_eq["macro_betas"]


def test_runner_gate_derived_from_registry(monkeypatch):
    """The cross_asset runner derives include_gfx_vol from the
    registry's applicable_asset_classes — fx gets True, equity
    gets False. Locks the single-source-of-truth contract."""
    from engine.research import cross_asset_attribution as xa
    captured = {}
    monkeypatch.setattr(
        xa, "compute_for_tier_c_with_macro",
        lambda *a, **kw: captured.update(kw) or None,
    )

    class _Spec:
        asset_class = "fx"
    class _TR:
        artifacts = {"pnl_series_df": pd.DataFrame({"pnl_net_13bp": [0.01]})}

    xa.LENS_DECLARATION.runner(
        _Spec(), _TR(),
        {"fx_carry_anchor_regression": {"alpha_nw_t": 2.0}},
    )
    assert captured.get("include_gfx_vol") is True
    assert captured.get("include_industry") is False

    captured.clear()
    class _SpecEq:
        asset_class = "equity"
    xa.LENS_DECLARATION.runner(
        _SpecEq(), _TR(),
        {"anchor_regression": {"alpha_nw_t": 2.0}},
    )
    assert captured.get("include_gfx_vol") is False
    assert captured.get("include_industry") is True


def test_runner_accepts_fx_anchor_stage1():
    """Pre-B.3 regression: runner returned None for FX sleeves
    because it only read prior_outputs['anchor_regression']."""
    from engine.research import cross_asset_attribution as xa

    class _Spec:
        asset_class = "fx"
    class _TR:
        artifacts = {"pnl_series_df": pd.DataFrame()}

    # Empty pnl_df → returns None EARLY, but the point is it must
    # get past the stage1 check. Use a sentinel via monkeypatching:
    # actually simplest — empty pnl_df returns None before stage1.
    # Use non-empty df + verify it reaches compute (returns not-None
    # or None from compute, NOT from the stage1 gate).
    import types
    calls = []
    orig = xa.compute_for_tier_c_with_macro
    xa.compute_for_tier_c_with_macro = (
        lambda *a, **kw: calls.append(1) or None)
    try:
        class _TR2:
            artifacts = {"pnl_series_df":
                            pd.DataFrame({"pnl_net_13bp": [0.01]})}
        xa.LENS_DECLARATION.runner(
            _Spec(), _TR2(),
            {"fx_carry_anchor_regression": {"alpha_nw_t": 3.0}},
        )
    finally:
        xa.compute_for_tier_c_with_macro = orig
    assert calls, ("runner never reached compute — FX stage1 "
                     "not accepted (pre-B.3 bug regressed)")
