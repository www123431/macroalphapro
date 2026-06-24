"""tests/test_fin_accruals.py — FIN (accruals half) orthogonality test = RED.

Pins the verdict that deploying the DHS FIN factor is NOT worth it: the accruals half has no
residual alpha vs FF5+UMD (+PEAD) and doesn't clear the diversification gate — confirming the
"arbitraged published anomaly" prior empirically. Slow (rebuilds accruals + PEAD); skips if cache
absent.
"""
from __future__ import annotations

import os

import pytest

from engine.validation.fin_accruals import orthogonality_test

_NEEDED = ["data/cache/_compustat_funda.parquet", "data/cache/_pead_ts_panel_2014_2023.parquet",
           "data/cache/crsp_hist_daily_ret.parquet", "data/cache/ff_factors_weekly.parquet"]


@pytest.mark.skipif(not all(os.path.exists(p) for p in _NEEDED), reason="cache absent")
def test_fin_accruals_not_orthogonal():
    r = orthogonality_test()
    assert r["n_months"] > 60, r
    # arbitraged → essentially no residual alpha vs FF5+UMD, sub-significant even with PEAD control
    assert abs(r["alpha_t_vs_ff5umd"]) < 2.0, r
    assert abs(r["alpha_t_vs_ff5umd_pead"]) < 3.0, r
    # fails the diversifier gate (residual-α t≥3 AND |corr|<0.3)
    assert r["orthogonal_diversifier"] is False, r
