"""tests/test_tsmom_crisis.py — self-built TSMOM crisis-sleeve tests.

Validates the universe-agnostic math (signal, sleeve construction, crisis
contribution). The yfinance fetch is network-gated (not unit-tested).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def _synth_px(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    # trending up series + a noisy one
    up = pd.Series(100 * np.cumprod(1 + rng.normal(0.01, 0.03, n)), index=idx)
    noise = pd.Series(100 * np.cumprod(1 + rng.normal(0.0, 0.04, n)), index=idx)
    return pd.DataFrame({"TLT": up, "GLD": noise, "SPY": up, "DBC": noise,
                         "IEF": up, "UUP": noise})


def test_tsmom_signal_sign():
    from engine.validation.tsmom_crisis import tsmom_signal
    px = _synth_px()
    sig = tsmom_signal(px, lookback=12, skip=1)
    # after warmup, an uptrending series should mostly signal +1
    tail = sig["SPY"].dropna().iloc[-20:]
    assert (tail == 1).mean() > 0.6


def test_build_sleeve_vol_targeted():
    from engine.validation.tsmom_crisis import build_tsmom_sleeve
    px = _synth_px(seed=1)
    sleeve = build_tsmom_sleeve(px, target_vol=0.10)
    realized = sleeve.std() * np.sqrt(12)
    # vol-targeting should bring realized vol near the 10% target
    assert 0.05 < realized < 0.20
    assert len(sleeve) > 50


def test_crisis_contribution_patches_hole_logic():
    """patches_hole must be True only when TSMOM>0 AND TLT/GLD<=0."""
    from engine.validation.tsmom_crisis import crisis_contribution, CRISIS_WINDOWS
    # Build a panel where in the 2022 window TLT+GLD fall but a trend
    # sleeve (driven by SPY/DBC) would be positive. We just check the
    # dataclass logic via a constructed sleeve.
    idx = pd.date_range("2021-06-30", periods=18, freq="ME")
    px = pd.DataFrame({
        "TLT": 100 * np.cumprod(np.r_[np.ones(7), np.full(11, 0.97)]),  # falling in 2022
        "GLD": 100 * np.cumprod(np.r_[np.ones(7), np.full(11, 0.99)]),  # flat/down
        "SPY": 100 * np.cumprod(np.r_[np.ones(7), np.full(11, 0.98)]),
        "DBC": 100 * np.ones(18), "IEF": 100*np.ones(18), "UUP": 100*np.ones(18),
    }, index=idx)
    sleeve = pd.Series(np.r_[np.zeros(7), np.full(11, 0.015)], index=idx)  # +ve in 2022
    out = crisis_contribution(px, sleeve)
    c2022 = next(c for c in out if c.window == "2022_RATESHOCK")
    assert c2022.tsmom_ret > 0
    assert c2022.tlt_gld_5050 < 0
    assert c2022.patches_hole is True


def test_multispeed_signal_range_and_blend():
    """Multi-speed signal is the average of component sign signals → in
    [-1, 1], and for a clean uptrend converges to +1 across speeds."""
    from engine.validation.tsmom_crisis import multispeed_signal
    idx = pd.date_range("2010-01-31", periods=40, freq="ME")
    up = pd.Series(100 * np.cumprod(1 + np.full(40, 0.01)), index=idx)
    px = pd.DataFrame({"UP": up})
    s = multispeed_signal(px, (3, 6, 12))
    assert s.min().min() >= -1.0 and s.max().max() <= 1.0
    # after all lookbacks warm up, an uptrend signals +1 on every speed → 1.0
    assert s["UP"].dropna().iloc[-1] == pytest.approx(1.0)


def test_apply_tsmom_cost_reduces_returns():
    from engine.validation.tsmom_crisis import apply_tsmom_cost
    idx = pd.date_range("2010-01-31", periods=24, freq="ME")
    sleeve = pd.Series(0.01, index=idx)
    turnover = pd.Series(0.30, index=idx)   # 30%/mo one-way
    net = apply_tsmom_cost(sleeve, turnover, roundtrip_bps=8.0)
    # drag = 0.30 * 8/10000 = 0.00024/mo
    assert net.iloc[0] == pytest.approx(0.01 - 0.00024, abs=1e-9)
    assert (net < sleeve).all()
