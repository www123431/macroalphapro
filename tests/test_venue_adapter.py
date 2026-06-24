"""tests/test_venue_adapter.py — VenueAdapter Protocol + resolver tests.

Confirms:
  1. BacktestReplayAdapter satisfies VenueAdapter Protocol + returns
     non-empty series for PIT SN
  2. AlpacaPaperAdapter + WrdsForwardSimAdapter are correctly stubbed
     (raise NotImplementedError, not silent broken)
  3. resolve_venue_adapter_for_sleeve picks the right concrete class
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

import engine.research.sleeves  # noqa: F401  (sleeve registration)

from engine.research.venue_adapter import (
    AlpacaPaperAdapter, BacktestReplayAdapter, VenueAdapter, VenueType,
    WrdsForwardSimAdapter, list_venue_assignments,
    resolve_venue_adapter_for_sleeve,
)


class TestBacktestReplayAdapter:
    def test_satisfies_protocol(self):
        a = BacktestReplayAdapter("post_earnings_drift_pit_sn")
        assert isinstance(a, VenueAdapter)
        assert a.venue_type == VenueType.BACKTEST_REPLAY
        assert a.supports_real_orders is False

    def test_get_forward_returns_filters_by_start_date(self):
        a = BacktestReplayAdapter("post_earnings_drift_pit_sn")
        # PIT SN data goes 2014-01-31 → 2024-03-31; ask for 2023+
        cutoff = _dt.datetime(2023, 1, 1)
        s = a.get_forward_monthly_returns(cutoff)
        assert isinstance(s, pd.Series)
        assert s.index.min() >= pd.Timestamp("2023-01-01")

    def test_submit_target_weights_is_noop(self):
        a = BacktestReplayAdapter("post_earnings_drift_pit_sn")
        result = a.submit_target_weights({"AAPL": 0.5, "MSFT": 0.5})
        assert result.status == "no_op"
        assert result.error is None


class TestAlpacaPaperAdapter:
    def test_satisfies_protocol_shape(self):
        a = AlpacaPaperAdapter("post_earnings_drift_pit_sn")
        assert isinstance(a, VenueAdapter)
        assert a.venue_type == VenueType.ALPACA_PAPER
        assert a.supports_real_orders is True

    def test_get_forward_returns_raises_not_implemented(self):
        a = AlpacaPaperAdapter("post_earnings_drift_pit_sn")
        with pytest.raises(NotImplementedError, match="deferred"):
            a.get_forward_monthly_returns(_dt.datetime(2026, 1, 1))

    def test_submit_target_weights_raises_not_implemented(self):
        a = AlpacaPaperAdapter("post_earnings_drift_pit_sn")
        with pytest.raises(NotImplementedError, match="deferred"):
            a.submit_target_weights({"AAPL": 0.5})


class TestWrdsForwardSimAdapter:
    def test_satisfies_protocol_shape(self):
        a = WrdsForwardSimAdapter("cross_asset_carry")
        assert isinstance(a, VenueAdapter)
        assert a.venue_type == VenueType.WRDS_FORWARD_SIM
        assert a.supports_real_orders is False

    def test_get_forward_returns_raises_not_implemented(self):
        a = WrdsForwardSimAdapter("cross_asset_carry")
        with pytest.raises(NotImplementedError, match="deferred"):
            a.get_forward_monthly_returns(_dt.datetime(2026, 1, 1))

    def test_submit_target_weights_is_noop_not_raise(self):
        # WRDS sim is by definition no-op for orders, not deferred
        a = WrdsForwardSimAdapter("cross_asset_carry")
        result = a.submit_target_weights({"@ES": 0.5})
        assert result.status == "no_op"


class TestResolver:
    def test_defaults_to_backtest_replay_for_unknown_sleeve(self):
        a = resolve_venue_adapter_for_sleeve("nonexistent_sleeve")
        assert isinstance(a, BacktestReplayAdapter)

    def test_pit_sn_resolves_to_backtest_replay_currently(self):
        a = resolve_venue_adapter_for_sleeve("post_earnings_drift_pit_sn")
        # Currently default; will flip to ALPACA_PAPER when adapter shipped
        assert a.venue_type == VenueType.BACKTEST_REPLAY

    def test_list_venue_assignments_returns_dict(self):
        d = list_venue_assignments()
        assert isinstance(d, dict)
        assert "post_earnings_drift_pit_sn" in d
