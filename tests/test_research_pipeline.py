"""tests/test_research_pipeline.py — the in-app research-automation gate (v1).

Cross-validates that the generalized gate reproduces a verdict we already established by hand
(FIN accruals = RED), and that it refuses underpowered series. log=False so tests never write the
campaign ledger.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from engine.research.pipeline import run_gate

_NEEDED = ["data/cache/_compustat_funda.parquet", "data/cache/_pead_ts_panel_2014_2023.parquet",
           "data/cache/crsp_hist_daily_ret.parquet", "data/cache/ff_factors_weekly.parquet"]


def test_gate_refuses_underpowered():
    idx = pd.period_range("2020-01", periods=10, freq="M").to_timestamp()
    r = run_gate(pd.Series(np.random.default_rng(0).normal(0, 0.01, 10), index=idx),
                 "tiny", pead_control=False, log=False)
    assert r["verdict"] == "UNINTERPRETABLE" and r["available"] is False


@pytest.mark.skipif(not all(os.path.exists(p) for p in _NEEDED), reason="cache absent")
def test_gate_reproduces_accruals_red():
    from engine.validation.fin_accruals import build_accruals_ls
    r = run_gate(build_accruals_ls(), "FIN_accruals_xcheck", mechanism="accruals (Sloan)", log=False)
    assert r["available"]
    # same conclusion as the standalone fin_accruals test: no residual alpha → not GREEN
    assert abs(r["alpha_t_ff5umd"]) < 2.0, r
    assert r["verdict"] in ("RED", "YELLOW"), r           # decisively not GREEN
    assert r["verdict"] == "RED", r
    assert {"deflated_sr", "oos_sharpe", "corr_with_book", "n_trials", "bars"} <= set(r)
