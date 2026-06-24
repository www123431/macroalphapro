"""tests/test_delisting_bias.py — the missing-delisting-return bias is CONSERVATIVE.

Pins the offline finding (no WRDS needed): delisted names skew low-SUE (short side), so the
panel's omission of CRSP delisting returns UNDERSTATES the L/S strategy rather than inflating it.
This downgrades audit residual #1 from "fix needed" to "benign/conservative".
"""
from __future__ import annotations

import os

import pytest

from engine.validation.delisting_bias import quantify_delisting_bias

_PANEL = "data/cache/_pead_ts_panel_2014_2023.parquet"
_RET = "data/cache/crsp_hist_daily_ret.parquet"


@pytest.mark.skipif(not (os.path.exists(_PANEL) and os.path.exists(_RET)), reason="cache absent")
def test_delisting_bias_is_conservative():
    r = quantify_delisting_bias()
    assert r["available"]
    assert r["n_delisted"] > 0 and r["n_delisted_with_sue"] > 50
    # delisted names lower-SUE than the universe ⇒ short-side ⇒ omitted loss = missed short gain
    assert r["mean_sue_delisted"] < r["mean_sue_all"], r
    assert r["delisted_pct_short_side"] > r["delisted_pct_long_side"], r
    assert r["bias_direction"] == "conservative"
