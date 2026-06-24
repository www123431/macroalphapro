"""tests/test_dpead_short_leg.py — D_PEAD spec-alignment (live≠spec reconciliation, step 1).

Pins the flag-gated spec-compliant L/S construction (spec id=62 Amendment A.1:
combined = long − w·short). The point of step 1 is SAFETY: the default (short_leg_weight=0.0)
must be byte-identical long-only — the currently-deployed behaviour — so adding the L/S path
does NOT destabilize the live book. Flipping it on (re-calibration of book gross/net + RM
gates) is the deliberate next step.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from engine.portfolio.paper_trade_combined import get_d_pead_signal

_CACHE = "data/path_c_dhs/_pead_ts_signal_panel.parquet"
_skip = not os.path.exists(_CACHE)


def _as_of_with_events():
    p = pd.read_parquet(_CACHE, columns=["rdq"])
    return pd.to_datetime(p["rdq"]).max().date()


@pytest.mark.skipif(_skip, reason="PEAD-TS panel absent")
def test_default_is_long_only_unchanged():
    # No flag = deployed behaviour: long-only, weights sum to 1.0, all positive.
    sig = get_d_pead_signal(_as_of_with_events())
    assert sig.status == "OK", sig.notes
    w = sig.weights
    assert len(w) > 0 and (w > 0).all()
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert {a.side for a in sig.trade_attributions} == {"long"}


@pytest.mark.skipif(_skip, reason="PEAD-TS panel absent")
def test_live_strategy_path_stays_long_only():
    # The deployed path (DPeadStrategy.generate_signal → get_d_pead_signal(as_of), no flag)
    # must remain long-only until the book is re-calibrated for shorts.
    from engine.strategies import get_registry
    sig = get_registry().get("D_PEAD").generate_signal(_as_of_with_events())
    if sig.status == "OK":
        assert (sig.weights > 0).all(), "live D_PEAD must stay long-only (flag off by default)"


@pytest.mark.skipif(_skip, reason="PEAD-TS panel absent")
def test_short_leg_makes_market_neutral_ls():
    # short_leg_weight=0.7 (spec A.1) → long − 0.7·short: net 0.3, gross 1.7, both legs.
    sig = get_d_pead_signal(_as_of_with_events(), short_leg_weight=0.7)
    assert sig.status == "OK", sig.notes
    w = sig.weights
    assert (w < 0).any() and (w > 0).any()                 # both legs present
    assert abs(float(w.sum()) - 0.3) < 1e-6                 # net = 1 − 0.7
    assert abs(float(w.abs().sum()) - 1.7) < 1e-6           # gross = 1 + 0.7
    assert {a.side for a in sig.trade_attributions} == {"long", "short"}
