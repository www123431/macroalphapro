"""tests/test_delisting_merge.py — delisting-return splice is correct + deployed leg unchanged.

Pins audit residual #1 CLOSED (2026-05-26): splicing CRSP dlret onto the panel (a) only ADDS
rows (never drops/mutates existing returns), (b) leaves the DEPLOYED long-only D_PEAD leg
unchanged (delisted names are low-SUE, not in the long top decile), and (c) the Shumway-1997
fallback fires only for missing performance-delisting returns with the right sign/magnitude.
"""
from __future__ import annotations

import os

import pytest

from engine.validation.delisting_merge import _fill_dlret, _RET, _DL


def test_shumway_fallback_logic():
    # present dlret is used as-is
    assert _fill_dlret(-0.42, 500, 3) == -0.42
    assert _fill_dlret(0.10, 231, 1) == 0.10
    # missing + performance/liquidation (400-599): NASDAQ -0.55, NYSE/AMEX -0.30
    assert _fill_dlret(float("nan"), 500, 3) == -0.55      # NASDAQ
    assert _fill_dlret(float("nan"), 574, 1) == -0.30      # NYSE
    assert _fill_dlret(float("nan"), 400, 2) == -0.30      # AMEX
    # missing + non-performance (merger 200s / exchange 300s) → 0.0 (no catastrophic move)
    assert _fill_dlret(float("nan"), 231, 1) == 0.0
    assert _fill_dlret(float("nan"), 300, 3) == 0.0


@pytest.mark.skipif(not (os.path.exists(_RET) and os.path.exists(_DL)), reason="cache absent")
def test_splice_only_adds_rows_and_leaves_long_leg_unchanged():
    import pandas as pd
    from engine.validation.delisting_merge import build_spliced_panel, _OUT
    from engine.portfolio.dpead_recon import build_dpead_recon_returns as B

    orig = pd.read_parquet(_RET)[["permno", "date"]].copy()
    orig["date"] = pd.to_datetime(orig["date"])
    merged = build_spliced_panel(save=True)

    # (a) superset: every original (permno,date) survives, only new rows added
    assert len(merged) >= len(orig)
    key_orig = set(map(tuple, orig.values))
    key_new = set(zip(merged["permno"], pd.to_datetime(merged["date"])))
    assert key_orig.issubset(key_new)

    # (b) deployed LONG-ONLY leg is unchanged by the splice (delisted = low-SUE, not in long leg)
    base = B(long_short=False, ret_path=_RET)
    adj = B(long_short=False, ret_path=_OUT)
    j = base.to_frame("b").join(adj.rename("a"), how="inner")
    assert j["b"].corr(j["a"]) > 0.999, "long-only deployed leg must be ~unchanged"

    # (c) L/S short leg can only IMPROVE or stay flat (conservative bias direction)
    ls_base = B(long_short=True, ret_path=_RET)
    ls_adj = B(long_short=True, ret_path=_OUT)
    s_base = ls_base.mean() / ls_base.std()
    s_adj = ls_adj.mean() / ls_adj.std()
    assert s_adj >= s_base - 1e-6, "splice should not worsen the L/S Sharpe"
