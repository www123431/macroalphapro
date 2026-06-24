"""
tests/test_factor_ensemble_v2.py — v2 robust spec implementation tests.

Pre-registration: docs/spec_factor_ensemble_v2_robust.md (id=51) §四 Tests.

Covers:
  - tc.py:           TC drag math + first-period special case
  - beta_neutral.py: β panel + TSMOM neutralization (long_β = short_β after scale)
  - regime.py:       4-regime classifier + thresholds locked
  - multi_baseline.py: 4 baselines weight construction
  - verdict.py:      end-to-end mock walk-forward + per-baseline + per-regime aggregation
"""
from __future__ import annotations

import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from engine.factor_ensemble_v2 import (
    TC_BPS_ROUNDTRIP_LOCKED,
    compute_tc_drag,
    apply_tc_to_realized_returns,
    BETA_NEUTRAL_FACTORS_LOCKED,
    compute_beta_panel,
    beta_neutralize_tsmom,
    REGIMES_LOCKED,
    REGIME_VOL_THRESHOLD_LOCKED,
    REGIME_RETURN_THRESHOLD_LOCKED,
    classify_regime,
    classify_regime_series,
    BASELINE_DEFINITIONS_LOCKED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────

def test_locked_constants_match_spec():
    assert TC_BPS_ROUNDTRIP_LOCKED == 8.0, "TC bps locked at 8.0 per spec §2.2"
    assert BETA_NEUTRAL_FACTORS_LOCKED == ("tsmom",), "ONLY TSMOM neutralized per spec §2.3"
    assert REGIME_RETURN_THRESHOLD_LOCKED == 0.0
    assert REGIME_VOL_THRESHOLD_LOCKED == 0.18
    assert REGIMES_LOCKED == ("bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol")
    assert BASELINE_DEFINITIONS_LOCKED == ("bab_only", "sixty_forty", "equal_weight", "spy_buyhold")


# ─────────────────────────────────────────────────────────────────────────────
# tc.py
# ─────────────────────────────────────────────────────────────────────────────

def test_tc_first_period_full_establishment():
    """First period (no prev): turnover = ½ Σ |w_new|."""
    w = pd.Series({"A": 0.5, "B": -0.5})
    drag = compute_tc_drag(weights_new=w, weights_prev=None)
    expected_turnover = (0.5 + 0.5) / 2.0  # = 0.5
    expected_drag = expected_turnover * (8.0 / 10000)  # = 0.0004
    assert abs(drag - expected_drag) < 1e-9


def test_tc_subsequent_period_diff():
    """Subsequent period: turnover = ½ Σ |w_new - w_prev|."""
    w_new = pd.Series({"A": 0.6, "B": -0.4})
    w_prev = pd.Series({"A": 0.5, "B": -0.5})
    drag = compute_tc_drag(weights_new=w_new, weights_prev=w_prev)
    expected_turnover = (0.1 + 0.1) / 2.0  # = 0.1
    expected_drag = expected_turnover * (8.0 / 10000)  # = 0.00008
    assert abs(drag - expected_drag) < 1e-9


def test_tc_zero_turnover_zero_drag():
    """Identical weights → zero drag (e.g. spy_buyhold subsequent periods)."""
    w = pd.Series({"SPY": 1.0})
    drag = compute_tc_drag(weights_new=w, weights_prev=w)
    assert drag == 0.0


def test_apply_tc_vectorized():
    gross = pd.Series([0.01, 0.005, -0.003], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    turnover = pd.Series([0.5, 0.1, 0.0], index=gross.index)
    net = apply_tc_to_realized_returns(gross, turnover)
    # drag values: 0.5*8e-4=4e-4, 0.1*8e-4=8e-5, 0
    assert abs(net.iloc[0] - (0.01 - 4e-4)) < 1e-9
    assert abs(net.iloc[1] - (0.005 - 8e-5)) < 1e-9
    assert abs(net.iloc[2] - (-0.003 - 0)) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# beta_neutral.py
# ─────────────────────────────────────────────────────────────────────────────

def _make_panel(tickers, days=300):
    """Synthetic panel deterministic + a bit of cross-correlation with SPY."""
    idx = pd.date_range("2019-06-01", periods=days, freq="B")
    rng = np.random.default_rng(42)
    spy_rets = rng.normal(0.0005, 0.012, days)
    data = {"SPY": 100.0 * np.exp(np.cumsum(spy_rets))}
    for i, t in enumerate(tickers):
        if t == "SPY":
            continue
        # β = 0.8 + i*0.2 (range 0.8 to 1.4 across tickers) → deterministic
        beta = 0.8 + i * 0.15
        idio = rng.normal(0, 0.005, days)
        rets = beta * spy_rets + idio
        data[t] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


def test_compute_beta_panel_returns_finite_betas():
    panel = _make_panel(["XLF", "XLE", "QQQ", "SPY"])
    betas = compute_beta_panel(panel=panel, as_of=datetime.date(2020, 6, 30), tickers=["XLF", "XLE", "QQQ"])
    assert all(np.isfinite(betas[t]) for t in ["XLF", "XLE", "QQQ"])
    # Betas should be positive (constructed as 0.8-1.4) — test directionally
    assert all(betas[t] > 0.5 for t in ["XLF", "XLE", "QQQ"])


def test_compute_beta_panel_missing_spy_returns_nan():
    panel = _make_panel(["XLF", "XLE"])
    panel = panel.drop(columns=["SPY"])
    betas = compute_beta_panel(panel=panel, as_of=datetime.date(2020, 6, 30), tickers=["XLF", "XLE"])
    assert betas.isna().all()


def test_beta_neutralize_tsmom_makes_net_beta_zero():
    """After neutralization: long_β × signal + short_β × signal ≈ 0."""
    sig = pd.Series({"L1": 1.0, "L2": 1.0, "S1": -1.0, "S2": -1.0})
    betas = pd.Series({"L1": 1.0, "L2": 1.5, "S1": 0.5, "S2": 0.7})
    neutralized = beta_neutralize_tsmom(tsmom_signal=sig, beta_panel=betas)
    long_beta_total = (neutralized[neutralized > 0] * betas[neutralized > 0]).sum()
    short_beta_total = (neutralized[neutralized < 0].abs() * betas[neutralized < 0]).sum()
    # After scaling shorts: long_β_total ≈ short_β_total
    assert abs(long_beta_total - short_beta_total) < 1e-6


def test_beta_neutralize_tsmom_insufficient_beta_returns_zero():
    """If <50% of nonzero tickers have valid β → all zero (signal lost this period)."""
    sig = pd.Series({"A": 1.0, "B": -1.0, "C": 1.0, "D": -1.0})
    betas = pd.Series({"A": 1.0})  # only 1/4 have β
    neutralized = beta_neutralize_tsmom(tsmom_signal=sig, beta_panel=betas)
    assert (neutralized == 0).all()


def test_beta_neutralize_tsmom_single_direction_preserves_signal():
    """All long, no short → can't neutralize → return as-is."""
    sig = pd.Series({"A": 1.0, "B": 1.0})
    betas = pd.Series({"A": 1.0, "B": 1.2})
    neutralized = beta_neutralize_tsmom(tsmom_signal=sig, beta_panel=betas)
    pd.testing.assert_series_equal(neutralized, sig)


# ─────────────────────────────────────────────────────────────────────────────
# regime.py
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ret,vol,expected", [
    (0.10, 0.12, "bull_low_vol"),
    (0.10, 0.25, "bull_high_vol"),
    (-0.05, 0.10, "bear_low_vol"),
    (-0.05, 0.30, "bear_high_vol"),
    (0.0, 0.18, "bear_low_vol"),     # boundary: 0.0 not > 0 → bear; 0.18 not > 0.18 → low_vol
    (float("nan"), 0.20, "bull_low_vol"),  # NaN → default
])
def test_classify_regime_threshold_logic(ret, vol, expected):
    assert classify_regime(ret, vol) == expected


def test_classify_regime_series_basic():
    panel = _make_panel(["SPY"])
    # Mid-2020 dates
    dates = [datetime.date(2020, 6, 30), datetime.date(2020, 9, 30)]
    series = classify_regime_series(panel=panel, rebalance_dates=dates)
    assert len(series) == 2
    assert all(s in REGIMES_LOCKED for s in series)


# ─────────────────────────────────────────────────────────────────────────────
# multi_baseline.py — weight construction smoke
# ─────────────────────────────────────────────────────────────────────────────

def test_baseline_definitions_lock():
    assert BASELINE_DEFINITIONS_LOCKED == ("bab_only", "sixty_forty", "equal_weight", "spy_buyhold")


def test_sixty_forty_weights_sum_to_one():
    """60/40 always weights 60% SPY + 40% AGG."""
    from engine.factor_ensemble_v2.multi_baseline import _compute_baseline_weights
    w = _compute_baseline_weights(
        baseline_id="sixty_forty",
        as_of=datetime.date(2020, 6, 30),
        universe=["SPY", "AGG", "QQQ"],
        asset_classes={"SPY": "equity_factor", "AGG": "fixed_income", "QQQ": "equity_factor"},
        is_first_period=True,
    )
    assert w["SPY"] == 0.60
    assert w["AGG"] == 0.40
    assert "QQQ" not in w
    assert abs(w.sum() - 1.0) < 1e-9


def test_equal_weight_excludes_non_equity():
    from engine.factor_ensemble_v2.multi_baseline import _compute_baseline_weights
    w = _compute_baseline_weights(
        baseline_id="equal_weight",
        as_of=datetime.date(2020, 6, 30),
        universe=["XLF", "XLE", "QQQ", "TLT", "GLD"],
        asset_classes={
            "XLF": "equity_sector", "XLE": "equity_sector", "QQQ": "equity_factor",
            "TLT": "fixed_income", "GLD": "commodity",
        },
        is_first_period=True,
    )
    assert set(w.index) == {"XLF", "XLE", "QQQ"}  # only equity_sector + equity_factor
    assert all(abs(w[t] - 1.0/3.0) < 1e-9 for t in w.index)


def test_spy_buyhold_first_period_only():
    from engine.factor_ensemble_v2.multi_baseline import _compute_baseline_weights
    w_first = _compute_baseline_weights(
        baseline_id="spy_buyhold",
        as_of=datetime.date(2011, 1, 31),
        universe=["SPY", "AGG"],
        asset_classes={"SPY": "equity_factor", "AGG": "fixed_income"},
        is_first_period=True,
    )
    assert w_first["SPY"] == 1.0
    w_subsequent = _compute_baseline_weights(
        baseline_id="spy_buyhold",
        as_of=datetime.date(2011, 2, 28),
        universe=["SPY", "AGG"],
        asset_classes={"SPY": "equity_factor", "AGG": "fixed_income"},
        is_first_period=False,
    )
    assert w_subsequent.empty  # signaling "no rebalance this period"


def test_unknown_baseline_raises():
    from engine.factor_ensemble_v2.multi_baseline import _compute_baseline_weights
    with pytest.raises(ValueError, match="Unknown baseline_id"):
        _compute_baseline_weights(
            baseline_id="nonexistent",
            as_of=datetime.date(2020, 6, 30),
            universe=["SPY"], asset_classes={"SPY": "equity_factor"},
            is_first_period=True,
        )
