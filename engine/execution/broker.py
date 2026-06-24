"""engine/execution/broker.py — broker-agnostic execution interface (Phase B, post-paper-trade).

The systematic engine produces DETERMINISTIC target weights (engine.portfolio.*). This layer is
strictly DOWNSTREAM of that decision: it translates target weights into orders and routes them to
a broker's PAPER/sandbox account. It NEVER decides what to hold — it executes what the math already
decided (0-LLM-in-DECISION preserved; this is plumbing, not judgment).

Adapters (sim / Alpaca / IB / OANDA) implement ExecutionAdapter so the rebalancer is broker-agnostic
(aligns with the source-agnostic fetch-layer plan). PAPER-ONLY is enforced per-adapter: a live
endpoint must be refused, never silently traded — there is no real capital in this project.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Position:
    ticker: str
    qty: float                 # signed shares (+ long, - short)
    avg_price: float = 0.0
    market_value: float = 0.0


@dataclass(frozen=True)
class Account:
    equity: float              # total account value (cash + positions MV)
    cash: float = 0.0
    buying_power: float = 0.0


@dataclass(frozen=True)
class Order:
    ticker: str
    qty: float                 # signed: + buy, - sell
    type: str = "market"
    note: str = ""

    @property
    def side(self) -> str:
        return "buy" if self.qty > 0 else "sell"


@dataclass(frozen=True)
class Fill:
    ticker: str
    qty: float                 # signed shares actually filled
    price: float
    order_id: str = ""


@dataclass
class RebalanceReport:
    broker: str
    paper: bool
    equity_before: float
    target_weights: dict[str, float]
    orders: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    skipped_below_min: dict[str, float] = field(default_factory=dict)  # ticker -> delta_usd skipped
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "broker": self.broker,
            "paper": self.paper,
            "equity_before": round(self.equity_before, 2),
            "n_orders": len(self.orders),
            "n_fills": len(self.fills),
            "orders": [{"ticker": o.ticker, "qty": round(o.qty, 4), "side": o.side, "note": o.note}
                       for o in self.orders],
            "fills": [{"ticker": f.ticker, "qty": round(f.qty, 4), "price": round(f.price, 4)}
                      for f in self.fills],
            "skipped_below_min": {k: round(v, 2) for k, v in self.skipped_below_min.items()},
            "warnings": self.warnings,
        }


class ExecutionAdapter(abc.ABC):
    """Broker-agnostic execution surface. Implementations MUST be paper/sandbox only."""

    name: str = "abstract"

    @property
    @abc.abstractmethod
    def is_paper(self) -> bool:
        """True iff this adapter targets a PAPER/sandbox account. The rebalancer refuses to
        submit orders if this is False."""

    @abc.abstractmethod
    def get_account(self) -> Account:
        ...

    @abc.abstractmethod
    def get_positions(self) -> dict[str, Position]:
        ...

    @abc.abstractmethod
    def get_prices(self, tickers: list[str]) -> dict[str, float]:
        ...

    @abc.abstractmethod
    def submit_order(self, order: Order) -> Fill:
        """Submit a single order. Returns the resulting Fill (paper fill for sim/sandbox)."""
