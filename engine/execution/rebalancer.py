"""engine/execution/rebalancer.py — deterministic target-weight → orders translation (0-LLM core).

PURE math: given target weights (from the systematic engine), the broker's current account equity,
current positions, and prices, compute the orders that move the book toward the targets. No LLM, no
judgment — this is the execution arithmetic. `rebalance()` then routes those orders through any
ExecutionAdapter (sim / Alpaca paper / ...), enforcing PAPER-ONLY before submitting anything.
"""
from __future__ import annotations

import logging

from engine.execution.broker import (Account, ExecutionAdapter, Fill, Order,
                                      Position, RebalanceReport)

logger = logging.getLogger(__name__)

DEFAULT_MIN_ORDER_USD = 50.0     # don't churn the book for sub-$50 drift
DEFAULT_GROSS_CAP = 1.0          # paper book targets ≤ 100% gross by default (safety backstop)


def compute_orders(target_weights: dict[str, float],
                   account: Account,
                   positions: dict[str, Position],
                   prices: dict[str, float],
                   *,
                   min_order_usd: float = DEFAULT_MIN_ORDER_USD,
                   allow_fractional: bool = True,
                   gross_cap: float = DEFAULT_GROSS_CAP) -> tuple[list[Order], dict[str, float], list[str]]:
    """Return (orders, skipped_below_min, warnings).

    target $ per ticker = weight × equity; target shares = target$/price; order = target − current.
    Tickers held but absent from target_weights are liquidated (target weight 0). Orders whose
    |delta notional| < min_order_usd are skipped (anti-churn). gross_cap is a defensive backstop:
    if the targets' gross exceeds it, a warning is emitted (the systematic engine is supposed to
    vol-target already; this just catches a malformed target set)."""
    equity = float(account.equity)
    warnings: list[str] = []
    if equity <= 0:
        return [], {}, [f"non-positive equity {equity}; no orders"]

    gross = sum(abs(float(w)) for w in target_weights.values())
    if gross > gross_cap + 1e-9:
        warnings.append(f"target gross {gross:.3f} > cap {gross_cap:.2f} - submitting anyway (paper)")

    # union of tickers to consider: targets + currently held (held-not-in-target → liquidate)
    tickers = set(target_weights) | set(positions)
    orders: list[Order] = []
    skipped: dict[str, float] = {}

    for tk in sorted(tickers):
        w = float(target_weights.get(tk, 0.0))
        px = float(prices.get(tk, 0.0) or 0.0)
        cur_qty = float(positions[tk].qty) if tk in positions else 0.0
        if px <= 0:
            if cur_qty != 0 or w != 0:
                warnings.append(f"{tk}: missing/zero price - skipped")
            continue
        target_qty = (w * equity) / px
        if not allow_fractional:
            target_qty = float(int(target_qty))   # whole shares toward zero
        delta_qty = target_qty - cur_qty
        delta_usd = delta_qty * px
        if abs(delta_usd) < min_order_usd:
            if abs(delta_usd) > 1e-9:
                skipped[tk] = delta_usd
            continue
        orders.append(Order(ticker=tk, qty=delta_qty,
                            note=f"w={w:+.4f} tgt={target_qty:.2f} cur={cur_qty:.2f}"))
    return orders, skipped, warnings


def rebalance(adapter: ExecutionAdapter,
              target_weights: dict[str, float],
              *,
              min_order_usd: float = DEFAULT_MIN_ORDER_USD,
              allow_fractional: bool = True,
              dry_run: bool = False) -> RebalanceReport:
    """Pull live account/positions/prices, compute orders, and (unless dry_run) submit them through
    the adapter. PAPER-ONLY: refuses to submit if the adapter is not a paper/sandbox account."""
    if not adapter.is_paper and not dry_run:
        raise RuntimeError(
            f"adapter '{adapter.name}' is NOT a paper account — refusing to submit "
            "(this project trades no real capital). Use dry_run=True to inspect orders.")

    account = adapter.get_account()
    positions = adapter.get_positions()
    tickers = sorted(set(target_weights) | set(positions))
    prices = adapter.get_prices(tickers)

    orders, skipped, warnings = compute_orders(
        target_weights, account, positions, prices,
        min_order_usd=min_order_usd, allow_fractional=allow_fractional)

    report = RebalanceReport(
        broker=adapter.name, paper=adapter.is_paper, equity_before=account.equity,
        target_weights=dict(target_weights), orders=orders,
        skipped_below_min=skipped, warnings=warnings)

    if dry_run:
        report.warnings.append("dry_run: orders computed but NOT submitted")
        return report

    for o in orders:
        try:
            fill = adapter.submit_order(o)
            report.fills.append(fill)
        except Exception as exc:                      # one bad order must not abort the rebalance
            report.warnings.append(f"{o.ticker} order failed: {exc}")
            logger.warning("submit_order failed for %s: %s", o.ticker, exc)
    return report
