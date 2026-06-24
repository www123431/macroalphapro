"""engine/execution/sim_adapter.py — in-house paper-fill simulator (no broker, no keys).

A fully offline ExecutionAdapter: deterministic fills at the price you feed it (optional slippage),
file-backed state so a daily loop accumulates real positions + a NAV track. This lets us validate
the ENTIRE execution plumbing (target weights → orders → fills → positions → NAV) before wiring any
broker key — and serves as the always-available fallback. is_paper is True by construction.

Prices: SimAdapter does not fetch market data itself; callers pass a price source via set_prices()
(e.g. yfinance close, or the engine's cached panel) before rebalancing. This keeps the sim free of
network/data dependencies and deterministic in tests.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from engine.execution.broker import Account, ExecutionAdapter, Fill, Order, Position

_DEFAULT_STATE = "data/execution/sim_state.json"


@dataclass
class _SimState:
    cash: float
    qty: dict[str, float] = field(default_factory=dict)        # ticker -> shares
    avg_px: dict[str, float] = field(default_factory=dict)     # ticker -> avg cost
    nav_history: list[dict] = field(default_factory=list)      # [{date, nav}]


class SimAdapter(ExecutionAdapter):
    name = "sim"

    def __init__(self, starting_cash: float = 1_000_000.0,
                 slippage_bps: float = 0.0,
                 state_path: str | None = _DEFAULT_STATE,
                 reset: bool = False):
        self.slippage_bps = float(slippage_bps)
        self.state_path = state_path
        self._prices: dict[str, float] = {}
        if state_path and os.path.exists(state_path) and not reset:
            self._state = self._load(state_path)
        else:
            self._state = _SimState(cash=float(starting_cash))
            if state_path and reset:
                self._save()

    # ---- price source (caller-provided; sim does no network) ----
    def set_prices(self, prices: dict[str, float]) -> None:
        self._prices.update({k: float(v) for k, v in prices.items() if v and v > 0})

    def get_prices(self, tickers: list[str]) -> dict[str, float]:
        return {t: self._prices[t] for t in tickers if t in self._prices}

    # ---- adapter surface ----
    @property
    def is_paper(self) -> bool:
        return True

    def _mtm(self) -> float:
        return self._state.cash + sum(
            q * self._prices.get(t, self._state.avg_px.get(t, 0.0))
            for t, q in self._state.qty.items())

    def get_account(self) -> Account:
        eq = self._mtm()
        return Account(equity=eq, cash=self._state.cash, buying_power=max(0.0, self._state.cash))

    def get_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for t, q in self._state.qty.items():
            if abs(q) < 1e-9:
                continue
            px = self._prices.get(t, self._state.avg_px.get(t, 0.0))
            out[t] = Position(ticker=t, qty=q, avg_price=self._state.avg_px.get(t, 0.0),
                              market_value=q * px)
        return out

    def submit_order(self, order: Order) -> Fill:
        px = self._prices.get(order.ticker)
        if not px or px <= 0:
            raise ValueError(f"no price for {order.ticker}")
        # slippage works against us: pay more to buy, receive less to sell
        slip = self.slippage_bps / 10_000.0
        fill_px = px * (1 + slip) if order.qty > 0 else px * (1 - slip)
        cost = order.qty * fill_px
        self._state.cash -= cost
        prev_q = self._state.qty.get(order.ticker, 0.0)
        new_q = prev_q + order.qty
        # update avg cost only when increasing a same-sign position
        if prev_q == 0 or (prev_q > 0) == (order.qty > 0):
            prev_cost = abs(prev_q) * self._state.avg_px.get(order.ticker, fill_px)
            self._state.avg_px[order.ticker] = (
                (prev_cost + abs(order.qty) * fill_px) / max(abs(new_q), 1e-9)) if new_q != 0 else 0.0
        if abs(new_q) < 1e-9:
            self._state.qty.pop(order.ticker, None)
            self._state.avg_px.pop(order.ticker, None)
        else:
            self._state.qty[order.ticker] = new_q
        return Fill(ticker=order.ticker, qty=order.qty, price=fill_px, order_id="sim")

    # ---- NAV track persistence ----
    def mark_nav(self, date: str) -> float:
        nav = self._mtm()
        self._state.nav_history.append({"date": date, "nav": round(nav, 2)})
        self._save()
        return nav

    def nav_history(self) -> list[dict]:
        return list(self._state.nav_history)

    def save(self) -> None:
        self._save()

    # ---- (de)serialize ----
    def _save(self) -> None:
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"cash": self._state.cash, "qty": self._state.qty,
                       "avg_px": self._state.avg_px, "nav_history": self._state.nav_history},
                      f, indent=2)

    @staticmethod
    def _load(path: str) -> _SimState:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return _SimState(cash=float(d.get("cash", 0.0)), qty=dict(d.get("qty", {})),
                         avg_px=dict(d.get("avg_px", {})), nav_history=list(d.get("nav_history", [])))
