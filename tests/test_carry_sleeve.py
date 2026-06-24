"""tests/test_carry_sleeve.py — the deployable carry sleeve return engine.

Pins the pure logic (risk-parity inverse-vol combine + vol-targeting) on synthetic data,
and smoke-tests the real builder on the cached futures curve (skips if cache absent).
The GREEN verdict itself is spec-locked (id=77) and not re-litigated here.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from engine.portfolio.carry_sleeve import (
    build_carry_sleeve_returns,
    risk_parity_combine,
    sleeve_stats,
    vol_target,
)

_CACHE = "data/cache/_cmdty_settle.parquet"


def test_risk_parity_weights_inverse_vol():
    # leg A vol 2x leg B → A should get HALF the weight of B (inverse-vol).
    idx = pd.period_range("2000-01", periods=240, freq="M").to_timestamp()
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 0.04, 240), index=idx)   # vol 4%
    b = pd.Series(rng.normal(0, 0.02, 240), index=idx)   # vol 2%
    comb = risk_parity_combine({"a": a, "b": b})
    # reconstruct implied weights: comb = (wa*a + wb*b); wa/wb ≈ stdB/stdA ≈ 0.5
    wa, wb = 1 / a.std(), 1 / b.std()
    expected = (wa * a + wb * b) / (wa + wb)
    assert np.allclose(comb.values, expected.values)
    assert comb.name == "carry_combined"


def test_risk_parity_handles_nan_and_empty():
    idx = pd.period_range("2000-01", periods=12, freq="M").to_timestamp()
    a = pd.Series([np.nan] * 12, index=idx)              # all-NaN leg dropped
    b = pd.Series(np.linspace(-0.01, 0.01, 12), index=idx)
    comb = risk_parity_combine({"a": a, "b": b})
    assert np.allclose(comb.fillna(0).values, b.values)  # only b survives
    assert risk_parity_combine({}).empty


def test_vol_target_scales_to_target_and_preserves_sharpe():
    idx = pd.period_range("2000-01", periods=300, freq="M").to_timestamp()
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(0.005, 0.03, 300), index=idx)   # ~10.4% ann vol
    out = vol_target(s, target_annual_vol=0.15)
    assert abs(out.std() * np.sqrt(12) - 0.15) < 1e-9       # hit the target vol
    # linear scaling ⇒ Sharpe unchanged
    assert abs((s.mean() / s.std()) - (out.mean() / out.std())) < 1e-9


def test_vol_target_safe_on_degenerate():
    s = pd.Series([0.0, 0.0, 0.0])
    assert vol_target(s).equals(s)   # zero-vol → unchanged, no div-by-zero


def test_sleeve_stats_shape():
    idx = pd.period_range("2000-01", periods=120, freq="M").to_timestamp()
    s = pd.Series(np.random.default_rng(2).normal(0.004, 0.025, 120), index=idx)
    st = sleeve_stats(s)
    assert {"n", "ann_vol", "ann_ret", "sharpe"} <= set(st)
    assert st["n"] == 120 and st["ann_vol"] > 0


@pytest.mark.skipif(not os.path.exists(_CACHE), reason="futures cache absent")
def test_real_builder_smoke():
    # Reuses the validated builders on cached data. Sanity only (GREEN is spec-locked):
    # a non-trivial monthly series, vol-targeted ≈ 10%.
    s = build_carry_sleeve_returns(target_annual_vol=0.10)
    assert isinstance(s, pd.Series) and s.dropna().size > 60
    assert abs(s.std() * np.sqrt(12) - 0.10) < 0.02


@pytest.mark.skipif(not os.path.exists(_CACHE), reason="futures cache absent")
def test_daily_flag_keeps_monthly_path_intact():
    # daily=True is additive: the carry signal is identical, and the monthly path
    # (daily=False) is unchanged. Only the return panel's frequency differs.
    from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
    cw_m, rw_m = commodity_cr(daily=False)
    cw_d, rw_d = commodity_cr(daily=True)
    assert cw_m.equals(cw_d)            # monthly carry signal identical
    assert len(rw_d) > len(rw_m) * 10   # daily panel has many more rows than monthly


@pytest.mark.skipif(not os.path.exists(_CACHE), reason="futures cache absent")
def test_daily_marks_aggregate_to_validated_monthly():
    # THE rigor guard: marking the monthly-rebalanced positions DAILY and aggregating
    # back to monthly must closely track the validated monthly buy-and-hold L/S.
    # (Not identical — daily-rebal vs monthly-hold convention — so we test correlation.)
    from engine.portfolio.carry_sleeve import _daily_xs_ls
    from engine.validation.commodity_carry import build_carry_and_returns as commodity_cr
    from engine.validation.crossasset_carry import build_commodity_carry_ls
    cw, rd = commodity_cr(daily=True)
    daily_ls = _daily_xs_ls(rd, cw, q=0.3)
    monthly_from_daily = (1 + daily_ls).resample("ME").prod() - 1
    val = build_commodity_carry_ls()                     # validated monthly L/S
    j = pd.concat([monthly_from_daily.rename("d"), val.rename("v")], axis=1).dropna()
    assert len(j) > 60, len(j)
    assert j["d"].corr(j["v"]) > 0.85, j["d"].corr(j["v"])               # tracks the validated series
    assert (np.sign(j["d"]) == np.sign(j["v"])).mean() > 0.70            # signs agree most months
