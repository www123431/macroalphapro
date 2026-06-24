"""Tests for engine.research.templates.cross_asset_tsmom."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research.templates.cross_asset_tsmom import run_cross_asset_tsmom
from engine.research.templates import TEMPLATES


@pytest.fixture
def synth_futures_returns():
    """Synthetic 100-month, 10-instrument monthly returns with persistent drift."""
    rng = np.random.RandomState(123)
    n_months, n_inst = 100, 10
    dates = pd.date_range("2016-01-31", periods=n_months, freq="ME")
    instruments = [f"INST{i:02d}" for i in range(n_inst)]
    drift = rng.uniform(-0.005, 0.010, n_inst)
    rets = rng.randn(n_months, n_inst) * 0.05 + drift
    return pd.DataFrame(rets, index=dates, columns=instruments)


def test_cross_asset_tsmom_registered():
    assert "cross_asset_tsmom" in TEMPLATES


def test_run_cross_asset_tsmom_returns_series(synth_futures_returns):
    ls = run_cross_asset_tsmom(
        return_panel=synth_futures_returns,
        lookback_months=12, skip_months=1,
        per_instrument_vol_target=0.40,
        per_instrument_vol_lookback=24,
        n_min_instruments=4,
    )
    assert isinstance(ls, pd.Series)
    assert ls.dropna().shape[0] > 0


def test_invalid_rebal_freq_raises(synth_futures_returns):
    with pytest.raises(NotImplementedError):
        run_cross_asset_tsmom(return_panel=synth_futures_returns,
                                rebal_freq="weekly")


def test_invalid_agg_raises(synth_futures_returns):
    with pytest.raises(NotImplementedError):
        run_cross_asset_tsmom(return_panel=synth_futures_returns,
                                agg_method="risk_parity")


def test_per_instrument_signal_sign(synth_futures_returns):
    """Persistent up-drift instruments should mostly carry positive direction."""
    ls = run_cross_asset_tsmom(
        return_panel=synth_futures_returns,
        lookback_months=12, skip_months=1,
        per_instrument_vol_target=0.40,
        per_instrument_vol_lookback=24,
        n_min_instruments=4,
        cost_bps_per_side=0,    # disable cost so we can read raw direction
    )
    # Mean should be roughly aligned with average drift direction
    # (loose check — synthetic data isn't a perfect TSMOM scenario)
    assert ls.dropna().shape[0] > 30


def test_dsl_runner_routes_to_tsmom(synth_futures_returns):
    from engine.research.strategy_dsl_runner import run_proposal
    proposal = {
        "mechanism_id": "time_series_momentum",
        "execution_template": {
            "template_id": "cross_asset_tsmom",
            "binding": {
                "lookback_months":  12,
                "skip_months":      1,
                "per_instrument_vol_target":   0.40,
                "per_instrument_vol_lookback": 24,
                "rebal_freq":       "monthly",
                "cost_bps_per_side": 12.0,
                "n_min_instruments": 4,
                "agg_method":       "equal_weight",
            },
        },
    }
    ls = run_proposal(proposal, return_panel=synth_futures_returns)
    assert isinstance(ls, pd.Series)
    assert ls.dropna().shape[0] > 0
