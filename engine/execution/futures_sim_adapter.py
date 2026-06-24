"""engine/execution/futures_sim_adapter.py — realistic internal futures paper sim.

The futures legs (carry + CTA sleeve B) run here — our code, durable, 0 KYC, fed by free yfinance
continuous-futures prices. To be FAITHFUL to real futures trading (the whole point), it models the
frictions a naive "fill at settle" sim hides:
  - WHOLE CONTRACTS at real per-contract notionals (futures_specs) — you cannot hold 0.3 of a CL
    contract; positions are integer contracts (micro contracts where available for finer granularity).
  - FUTURES ACCOUNTING — buying a future costs no notional (margin only); P&L is marked daily as
    contracts × notional × return. Equity = initial + cumulative P&L − costs (NOT cash − notional).
  - SLIPPAGE on traded notional + ROLL cost charged on contract turnover.

Plugs into the same ExecutionAdapter ABC (rebalancer/reconcile/multi_venue unchanged). get_prices
returns the PER-CONTRACT NOTIONAL so the rebalancer sizes target contracts = weight×equity/notional;
submit_order rounds to whole contracts. mark(returns) advances NAV by the period's front-contract
returns (the same series B/carry are defined on).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from engine.execution.broker import Account, ExecutionAdapter, Fill, Order, Position
from engine.execution.futures_specs import contract_notional

_DEFAULT_STATE = "data/execution/futures_sim_state.json"


@dataclass
class _FState:
    equity: float
    contracts: dict[str, float] = field(default_factory=dict)   # sym -> integer contracts
    notional: dict[str, float] = field(default_factory=dict)    # sym -> current per-contract notional
    nav_history: list[dict] = field(default_factory=list)


class FuturesSimAdapter(ExecutionAdapter):
    name = "futures_sim"

    def __init__(self, starting_equity: float = 10_000_000.0, use_micro: bool = True,
                 slippage_bps: float = 1.0, roll_bps: float = 1.0,
                 state_path: str | None = _DEFAULT_STATE, reset: bool = False):
        self.use_micro = use_micro
        self.slippage_bps = float(slippage_bps)
        self.roll_bps = float(roll_bps)
        self.state_path = state_path
        if state_path and os.path.exists(state_path) and not reset:
            self._s = self._load(state_path)
        else:
            self._s = _FState(equity=float(starting_equity))
            if state_path and reset:
                self._save()

    # ---- marks: set current per-contract notional per sym (caller: multiplier×live price, or specs) ----
    def set_notionals(self, notionals: dict[str, float]) -> None:
        self._s.notional.update({k: float(v) for k, v in notionals.items() if v and v > 0})

    def seed_notionals_from_specs(self, syms) -> None:
        for s in syms:
            cn = contract_notional(s, use_micro=self.use_micro)
            if cn and s not in self._s.notional:
                self._s.notional[s] = cn

    def get_prices(self, tickers: list[str]) -> dict[str, float]:
        """Per-contract NOTIONAL (so rebalancer sizes target contracts = weight×equity/notional)."""
        out = {}
        for t in tickers:
            cn = self._s.notional.get(t) or contract_notional(t, use_micro=self.use_micro)
            if cn:
                out[t] = cn
        return out

    @property
    def is_paper(self) -> bool:
        return True

    def get_account(self) -> Account:
        # futures tie up margin, not notional → "cash" ≈ equity (free collateral); report equity.
        return Account(equity=self._s.equity, cash=self._s.equity, buying_power=self._s.equity)

    def get_positions(self) -> dict[str, Position]:
        out = {}
        for s, c in self._s.contracts.items():
            if abs(c) < 1e-9:
                continue
            cn = self._s.notional.get(s, contract_notional(s, use_micro=self.use_micro) or 0.0)
            out[s] = Position(ticker=s, qty=c, avg_price=cn, market_value=c * cn)
        return out

    def submit_order(self, order: Order) -> Fill:
        cn = self._s.notional.get(order.ticker) or contract_notional(order.ticker, use_micro=self.use_micro)
        if not cn or cn <= 0:
            raise ValueError(f"no contract spec/notional for {order.ticker}")
        whole = float(round(order.qty))                    # WHOLE contracts only
        if abs(whole) < 1:
            return Fill(ticker=order.ticker, qty=0.0, price=cn, order_id="fsim-skip")
        # slippage on traded notional (futures pay no notional outlay, but slippage is a real cost)
        self._s.equity -= abs(whole) * cn * (self.slippage_bps / 1e4)
        self._s.contracts[order.ticker] = self._s.contracts.get(order.ticker, 0.0) + whole
        if abs(self._s.contracts[order.ticker]) < 1e-9:
            self._s.contracts.pop(order.ticker, None)
        return Fill(ticker=order.ticker, qty=whole, price=cn, order_id="fsim")

    def mark(self, returns: dict[str, float], date: str | None = None,
             rolled: set[str] | None = None) -> float:
        """Advance NAV by the period's front-contract returns: P&L = Σ contracts×notional×ret.
        Notionals compound by return. `rolled` syms pay a roll cost on their notional."""
        rolled = rolled or set()
        for s, c in list(self._s.contracts.items()):
            r = returns.get(s)
            if r is None:
                continue
            cn = self._s.notional.get(s, 0.0)
            self._s.equity += c * cn * float(r)            # mark-to-market P&L
            self._s.notional[s] = cn * (1 + float(r))      # notional drifts with price
            if s in rolled:
                self._s.equity -= abs(c) * cn * (self.roll_bps / 1e4)
        if date:
            self._s.nav_history.append({"date": date, "nav": round(self._s.equity, 2)})
        self._save()
        return self._s.equity

    def nav_history(self) -> list[dict]:
        return list(self._s.nav_history)

    def save(self) -> None:
        """Persist state (submit_order updates positions in memory only; call after rebalancing
        on a run with no mark, else the first run's contracts wouldn't hit disk)."""
        self._save()

    def _save(self) -> None:
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"equity": self._s.equity, "contracts": self._s.contracts,
                       "notional": self._s.notional, "nav_history": self._s.nav_history}, f, indent=2)

    @staticmethod
    def _load(path: str) -> _FState:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return _FState(equity=float(d.get("equity", 0.0)), contracts=dict(d.get("contracts", {})),
                       notional=dict(d.get("notional", {})), nav_history=list(d.get("nav_history", [])))
