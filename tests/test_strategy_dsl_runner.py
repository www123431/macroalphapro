"""Tests for engine.research.strategy_dsl_runner — Layer 3 dispatcher."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research import strategy_dsl_runner as DSL


def test_list_templates_contains_equity_xsmom():
    assert "equity_xsmom" in DSL.list_templates()


def test_run_proposal_missing_execution_template_raises():
    with pytest.raises(ValueError):
        DSL.run_proposal({"mechanism_id": "x"})


def test_run_proposal_unknown_template_raises(synth_prices):
    proposal = {
        "execution_template": {
            "template_id": "fake_template_xyz",
            "binding": {},
        }
    }
    with pytest.raises(KeyError):
        DSL.run_proposal(proposal, price_panel=synth_prices)


def test_run_proposal_equity_xsmom_end_to_end(synth_prices):
    """End-to-end smoke: proposal → equity_xsmom template → returns series."""
    proposal = {
        "mechanism_id": "equity_xsmom_jt",
        "execution_template": {
            "template_id": "equity_xsmom",
            "binding": {
                "lookback_months":  12,
                "skip_months":      1,
                "top_frac":         0.2,
                "bottom_frac":      0.2,
                "weighting":        "equal_weight",
                "rebal_freq":       "monthly",
                "cost_bps_per_side": 12.0,
                "microcap_price_threshold": 5.0,
                "vol_target":       0.10,
                "vol_target_lookback": 12,
            },
        },
    }
    ls = DSL.run_proposal(proposal, price_panel=synth_prices)
    assert isinstance(ls, pd.Series)
    assert ls.index.equals(synth_prices.index)
    # After lookback + vol-target warmup, must have non-NaN values
    assert ls.dropna().shape[0] > 0


@pytest.fixture
def synth_prices():
    """Synthetic 50-month, 30-ticker price panel."""
    rng = np.random.RandomState(42)
    n_months = 50
    n_tickers = 30
    dates = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    # Random walk with drift
    drift = rng.uniform(-0.005, 0.01, n_tickers)
    vol = rng.uniform(0.04, 0.10, n_tickers)
    rets = rng.randn(n_months, n_tickers) * vol + drift
    prices = pd.DataFrame(
        np.cumprod(1 + rets, axis=0) * 100.0,
        index=dates, columns=tickers,
    )
    return prices
