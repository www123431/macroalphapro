"""engine/execution/multi_venue.py — consolidate a book across multiple paper venues.

The full book runs on >1 venue: equity legs on a real broker paper account (Alpaca), the futures
legs (carry + CTA sleeve B) on the internal SimAdapter (our code, durable, no KYC, fed by free
yfinance continuous-futures prices). This module sums their accounts/positions into ONE book NAV
and reconciles vs the full model — and flags the residual (targets no venue can hold, e.g. the
single-name shorts Alpaca can't borrow). 0-LLM, read-only.

Each venue is an ExecutionAdapter tagged with the kind of exposure it carries. Cross-venue cash is
NOT fungible (each sim/broker has its own cash) — the consolidated equity is the SUM, which is the
honest paper-book view (we measure strategy performance, not unified-margin efficiency).
"""
from __future__ import annotations

import math

from engine.execution.broker import ExecutionAdapter

DRIFT_EPS = 0.002


def book_risk_weights() -> dict:
    """The DEPLOYED equity/carry risk split — SINGLE SOURCE OF TRUTH, read from
    engine.portfolio.combined_book.DEFAULT_CARRY_RISK_WEIGHT (NOT hardcoded here). Verified config:
    carry risk weight 0.30 → equity 0.70 / carry 0.30 (risk-weighted, each side vol-targeted first).
    The trend sleeve (spec 75) is NOT in this deployed blend (a candidate, separate)."""
    from engine.portfolio.combined_book import DEFAULT_CARRY_RISK_WEIGHT as w
    return {"equity": round(1.0 - float(w), 4), "carry": round(float(w), 4),
            "source": "combined_book.DEFAULT_CARRY_RISK_WEIGHT"}


def risk_weighted_book_return(equity_ret, carry_ret):
    """Consolidated book return = (1-w)·equity + w·carry, w from the deployed config. Operates on
    RETURN series → SCALE-INVARIANT: the venues' dollar-scale mismatch (e.g. $100k equity vs $10M
    futures sim) does NOT distort the book (each venue's return is scale-free). NOTE the 70/30 is a
    RISK split — callers must pass returns each already vol-targeted to the same vol (combined_book
    does this); a live blend therefore only becomes valid once enough NAV has accrued to vol-target."""
    import pandas as pd
    rw = book_risk_weights()
    j = pd.concat([equity_ret.rename("e"), carry_ret.rename("c")], axis=1).dropna()
    return (rw["equity"] * j["e"] + rw["carry"] * j["c"]).rename("book")


def consolidate(adapters: dict[str, ExecutionAdapter]) -> dict:
    """Sum equity/cash across venues; merge positions, tagging each ticker with its venue(s)."""
    total_equity = 0.0
    total_cash = 0.0
    per_venue: dict[str, dict] = {}
    position_mv: dict[str, float] = {}
    position_venue: dict[str, str] = {}
    for name, ad in adapters.items():
        acct = ad.get_account()
        pos = ad.get_positions()
        total_equity += float(acct.equity)
        total_cash += float(acct.cash)
        per_venue[name] = {"equity": round(float(acct.equity), 2), "paper": ad.is_paper,
                           "n_positions": len(pos)}
        for tk, p in pos.items():
            position_mv[tk] = position_mv.get(tk, 0.0) + float(p.market_value)
            position_venue[tk] = name if tk not in position_venue else f"{position_venue[tk]}+{name}"
    return {"total_equity": round(total_equity, 2), "total_cash": round(total_cash, 2),
            "per_venue": per_venue, "position_mv": position_mv, "position_venue": position_venue}


def reconcile_multi(adapters: dict[str, ExecutionAdapter], target_weights: dict[str, float]) -> dict:
    """Consolidated target-vs-actual across venues. actual_weight = Σ_venue position MV / total equity.
    Flags targets that NO venue holds (queued / unborrowable shorts / untradeable) as breaks."""
    con = consolidate(adapters)
    eq = con["total_equity"] or 0.0
    pmv = con["position_mv"]

    tickers = sorted(set(target_weights) | set(pmv))
    rows = []
    sse = 0.0
    gross_target = 0.0
    gross_actual = 0.0
    for tk in tickers:
        tgt = float(target_weights.get(tk, 0.0))
        act = (pmv.get(tk, 0.0) / eq) if eq else 0.0
        drift = act - tgt
        sse += drift * drift
        gross_target += abs(tgt)
        gross_actual += abs(act)
        rows.append({"ticker": tk, "target_weight": round(tgt, 5), "actual_weight": round(act, 5),
                     "drift": round(drift, 5), "venue": con["position_venue"].get(tk, ""),
                     "on_target": abs(drift) <= DRIFT_EPS})

    targeted_not_held = sorted(tk for tk, w in target_weights.items()
                               if abs(w) > DRIFT_EPS and tk not in pmv)
    # honest short-borrow residual: SHORT targets (w<0) that no venue holds
    short_residual = sorted(tk for tk in targeted_not_held if target_weights.get(tk, 0.0) < 0)
    held_not_targeted = sorted(tk for tk in pmv if tk not in target_weights)

    rows.sort(key=lambda r: abs(r["drift"]), reverse=True)
    short_residual_wt = round(sum(abs(target_weights[t]) for t in short_residual), 4)
    return {
        "venues": list(adapters), "total_equity": round(eq, 2), "cash": con["total_cash"],
        "per_venue": con["per_venue"],
        "n_targets": int(sum(1 for w in target_weights.values() if abs(w) > DRIFT_EPS)),
        "n_held": len(pmv),
        "gross_target": round(gross_target, 4), "gross_actual": round(gross_actual, 4),
        "tracking_error": round(math.sqrt(sse), 5),
        "n_on_target": int(sum(1 for r in rows if r["on_target"])),
        "breaks": {"targeted_not_held": targeted_not_held, "held_not_targeted": held_not_targeted,
                   "short_borrow_residual": short_residual,
                   "short_borrow_residual_weight": short_residual_wt},
        "rows": rows,
    }
