"""engine/research/venue_adapter.py — pluggable forward-return source
for SLM PAPER_TRADE / SHADOW states.

DESIGN INTENT (post-2026-05-31 user critique on Alpaca integration):
  "我们的策略有一部分能走 alpaca,有一部分要通过别的方式进行 simulation"

The 7 deployed sleeves split across 3 venue capabilities:
  - US equity / ETF        → Alpaca paper (when forward sim is wanted)
  - Cross-asset futures    → IB futures paper OR WRDS-based forward sim
  - Default (audit-style)  → Backtest replay using cached parquet

This module provides the Protocol all venue adapters implement, plus:
  - BacktestReplayAdapter (default; current SLM behavior, ZERO behavior
    change — reads sleeve.returns() from parquet)
  - AlpacaPaperAdapter (STUB — raises NotImplementedError until
    sleeve.target_weights() interface is implemented + AlpacaAdapter
    wired through)
  - WrdsForwardSimAdapter (STUB — uses Almgren-Chriss cost model on
    cached settle data for futures sleeves; placeholder)

This is an ARCHITECTURE-ONLY commit. Concrete Alpaca / WRDS adapters
are deferred until PIT SN graduates to SHADOW (~12mo from now per the
PAPER_TRADE → SHADOW gate). The Protocol + default + stubs let future
integration plug in WITHOUT refactoring SLM core.

Caller pattern:

    adapter = resolve_venue_adapter_for_sleeve(strategy_id)
    forward_returns = adapter.get_forward_monthly_returns(
        start_date=record.paper_trade_started,
    )
    # Pass forward_returns to paper_trade_monitor instead of
    # sleeve.returns() (which is the same in the BacktestReplay case
    # but DIFFERENT in Alpaca / WRDS-sim cases).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd

from engine.research.sleeve_registry import get_sleeve


class VenueType(str, Enum):
    """The 3 supported venue modes per the user's heterogeneous-book
    observation. Add new modes here when wiring real adapters."""

    BACKTEST_REPLAY = "backtest_replay"   # default; cached parquet
    ALPACA_PAPER = "alpaca_paper"         # equity/ETF; real paper fills
    WRDS_FORWARD_SIM = "wrds_forward_sim" # futures; Almgren-Chriss sim
    IB_PAPER_FUTURES = "ib_paper_futures" # placeholder for IB integration


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of submit_target_weights() call.

    For BacktestReplayAdapter this is always (status='no_op', ...) since
    no order actually goes out. For real adapters it captures the
    order-submission outcome.
    """

    status: str                          # "no_op" | "submitted" | "rejected"
    submitted_at: Optional[_dt.datetime] = None
    venue: str = ""
    orders_submitted: int = 0
    diagnostic: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@runtime_checkable
class VenueAdapter(Protocol):
    """Pluggable forward-return source. All adapters MUST implement
    get_forward_monthly_returns() and submit_target_weights().

    Protocols (not ABC) so adapters don't need inheritance — they just
    need the methods. Enables third-party adapters with zero coupling.
    """

    @property
    def venue_type(self) -> VenueType: ...

    @property
    def supports_real_orders(self) -> bool: ...

    def get_forward_monthly_returns(
        self, start_date: _dt.datetime,
    ) -> pd.Series:
        """Monthly returns observed from this venue since start_date.
        For BacktestReplay this is just sleeve.returns()[start:];
        for real venues this reads PnL from the venue API."""
        ...

    def submit_target_weights(
        self, weights: dict[str, float],
    ) -> SubmitResult:
        """Submit a target-weight portfolio to the venue. For backtest
        replay this is a no-op; for real venues this routes to the
        execution stack."""
        ...


# ── Default: BacktestReplay (zero behavior change vs pre-Phase-5) ──────


class BacktestReplayAdapter:
    """Default venue adapter — reads sleeve.returns() from parquet.

    This is what SLM currently uses for PAPER_TRADE evaluation. Keeps
    the existing "audit-style" paper trade workflow as the default,
    with zero behavior change until a sleeve explicitly opts into
    Alpaca / WRDS-sim.
    """

    venue_type = VenueType.BACKTEST_REPLAY
    supports_real_orders = False

    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id

    def get_forward_monthly_returns(
        self, start_date: _dt.datetime,
    ) -> pd.Series:
        sleeve = get_sleeve(self.strategy_id)
        full = sleeve.returns()
        if start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        return full[full.index >= start_date]

    def submit_target_weights(
        self, weights: dict[str, float],
    ) -> SubmitResult:
        return SubmitResult(
            status="no_op",
            venue=self.venue_type.value,
            diagnostic={"reason": "BacktestReplayAdapter does not submit orders"},
        )


# ── STUB: Alpaca paper (equity / ETF) ──────────────────────────────────


class AlpacaPaperAdapter:
    """STUB. Wires sleeve.target_weights() through
    engine.execution.alpaca_adapter.AlpacaAdapter + rebalancer.

    Deferred until PIT SN approaches SHADOW. The shape of the API is
    fixed here so future implementation is mechanical.
    """

    venue_type = VenueType.ALPACA_PAPER
    supports_real_orders = True

    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id

    def get_forward_monthly_returns(
        self, start_date: _dt.datetime,
    ) -> pd.Series:
        raise NotImplementedError(
            "AlpacaPaperAdapter.get_forward_monthly_returns: deferred to "
            "Phase 5+ when PIT SN approaches SHADOW. Will read Alpaca "
            "account history via AlpacaAdapter.get_account() + "
            "monthly NAV diff. See engine/execution/alpaca_adapter.py."
        )

    def submit_target_weights(
        self, weights: dict[str, float],
    ) -> SubmitResult:
        raise NotImplementedError(
            "AlpacaPaperAdapter.submit_target_weights: deferred to "
            "Phase 5+. Will use engine.execution.rebalancer to translate "
            "weights → orders + AlpacaAdapter.submit_order() per name. "
            "Requires sleeve to implement target_weights() interface."
        )


# ── STUB: WRDS-based forward sim (futures sleeves) ─────────────────────


class WrdsForwardSimAdapter:
    """STUB. For carry / TSMOM sleeves where Alpaca paper is not an
    option (no futures support). Uses cached WRDS settle data + the
    existing Almgren-Chriss cost model to simulate forward fills.

    Forward returns = next-month settle returns - Almgren impact.
    Conservative bias by design.
    """

    venue_type = VenueType.WRDS_FORWARD_SIM
    supports_real_orders = False

    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id

    def get_forward_monthly_returns(
        self, start_date: _dt.datetime,
    ) -> pd.Series:
        raise NotImplementedError(
            "WrdsForwardSimAdapter: deferred to Phase 5+. Will pull next-"
            "month settle from data/cache/_cmdty_settle / _fx_settle / "
            "_rates_settle, apply Almgren-Chriss impact (matching the "
            "cost_model audit blocks), and emit synthetic monthly returns."
        )

    def submit_target_weights(
        self, weights: dict[str, float],
    ) -> SubmitResult:
        return SubmitResult(
            status="no_op",
            venue=self.venue_type.value,
            diagnostic={"reason": "WrdsForwardSimAdapter is a sim, no real orders"},
        )


# ── Resolver ───────────────────────────────────────────────────────────


_DEFAULT_VENUE_ASSIGNMENTS: dict[str, VenueType] = {
    # Until each sleeve YAML explicitly declares a venue_adapter, the
    # resolver falls back to this static map. Equity-only sleeves default
    # to BACKTEST_REPLAY (audit-style); when ready they can flip to
    # ALPACA_PAPER. Futures sleeves CANNOT use Alpaca and default to
    # BACKTEST_REPLAY for now (would flip to WRDS_FORWARD_SIM when ready).
    "post_earnings_drift": VenueType.BACKTEST_REPLAY,
    "post_earnings_drift_pit_sn": VenueType.BACKTEST_REPLAY,
    # Future entries when adapters land:
    # "post_earnings_drift_pit_sn": VenueType.ALPACA_PAPER,
    # "cross_asset_carry":          VenueType.WRDS_FORWARD_SIM,
    # "time_series_momentum":       VenueType.WRDS_FORWARD_SIM,
}


def resolve_venue_adapter_for_sleeve(strategy_id: str) -> VenueAdapter:
    """Pick the right VenueAdapter for a sleeve.

    Resolution order:
      1. (Future) sleeve YAML venue_adapter.type field — when added
      2. _DEFAULT_VENUE_ASSIGNMENTS map (current implementation)
      3. Fallback to BacktestReplayAdapter (safe default)
    """
    venue = _DEFAULT_VENUE_ASSIGNMENTS.get(
        strategy_id, VenueType.BACKTEST_REPLAY,
    )
    if venue == VenueType.BACKTEST_REPLAY:
        return BacktestReplayAdapter(strategy_id)
    if venue == VenueType.ALPACA_PAPER:
        return AlpacaPaperAdapter(strategy_id)
    if venue == VenueType.WRDS_FORWARD_SIM:
        return WrdsForwardSimAdapter(strategy_id)
    raise ValueError(f"no adapter implementation for venue {venue.value!r}")


def list_venue_assignments() -> dict[str, str]:
    """For UI / debugging: current venue assignments by strategy_id."""
    return {sid: vt.value for sid, vt in _DEFAULT_VENUE_ASSIGNMENTS.items()}
