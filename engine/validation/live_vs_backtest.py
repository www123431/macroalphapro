"""engine/validation/live_vs_backtest.py — accumulating "does live deliver backtest" tracker.

Audit item F (docs/live_delivers_backtest_audit_2026-05-25.md): the #1 way to know whether the
live book actually earns the backtest is to TRACK realized-vs-expected from day one. With only
~7 live NAV days now this is statistically meaningless — the POINT is to stand up the comparison
so it accrues into a real verdict over ~6-12 months. No new storage: the live NAV history (which
grows daily) IS the accumulating record; this just computes the comparison on demand.

Honest framing baked in:
  - the backtest reference is the LIVE 5-sleeve replay (long-biased book), NOT the 1.04
    market-neutral alpha+carry research construct (see the D_PEAD reconciliation decision);
  - a `significant` flag + min-days gate make clear when (not) to read anything into it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REPLAY = "data/portfolio_replay/v1_combined_replay_verdict.json"
_TRADING_DAYS = 252
MIN_DAYS_FOR_SIGNIFICANCE = 126   # ~6 months of trading days before a daily-diff t-stat means anything


def _backtest_expectation() -> dict:
    p = Path(_REPLAY)
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    cm = d.get("combined_metrics", d)
    if not isinstance(cm, dict):
        return {}
    return {k: cm.get(k) for k in ("ann_ret", "ann_vol", "sharpe", "max_dd", "n_weeks")}


def build_tracking(days_back: int = 750) -> dict:
    """Compare the live paper book's realized returns to the backtest expectation.
    Returns a dict with live stats, backtest-expected stats, the tracking gap, and an
    honest `significant` flag (n_live_days vs MIN_DAYS_FOR_SIGNIFICANCE)."""
    from engine.agents.persona.tools import read_nav_history

    bt = _backtest_expectation()
    nav = json.loads(read_nav_history(days_back))
    days = [x for x in (nav.get("days") or []) if x.get("daily_dietz") is not None]
    if not days:
        return {"available": False, "reason": nav.get("message") or "no live NAV returns yet",
                "backtest_expected": bt}

    rets = np.array([float(x["daily_dietz"]) for x in days], dtype=float)
    n = int(len(rets))
    live_ann_ret = float(rets.mean() * _TRADING_DAYS)
    live_ann_vol = float(rets.std() * np.sqrt(_TRADING_DAYS))
    live_sharpe = float(live_ann_ret / live_ann_vol) if live_ann_vol > 0 else float("nan")
    live_cum = float(np.prod(1.0 + rets) - 1.0)

    exp_ann = float(bt.get("ann_ret") or 0.0)
    exp_daily = exp_ann / _TRADING_DAYS
    exp_cum = float((1.0 + exp_daily) ** n - 1.0)

    diff = rets - exp_daily
    t_stat = (float(diff.mean() / diff.std() * np.sqrt(n))
              if (n > 1 and diff.std() > 0) else None)
    enough = n >= MIN_DAYS_FOR_SIGNIFICANCE

    return {
        "available": True,
        "n_live_days": n,
        "live_window": {"start": days[0].get("date"), "end": days[-1].get("date")},
        "live": {"ann_ret": round(live_ann_ret, 4), "ann_vol": round(live_ann_vol, 4),
                 "sharpe": round(live_sharpe, 3) if live_sharpe == live_sharpe else None,
                 "cum_return": round(live_cum, 5)},
        "backtest_expected": {k: bt.get(k) for k in ("ann_ret", "ann_vol", "sharpe", "max_dd")},
        "tracking": {
            "ann_ret_diff": round(live_ann_ret - exp_ann, 4),
            "live_cum": round(live_cum, 5),
            "expected_cum": round(exp_cum, 5),
            "t_stat_vs_expected": round(t_stat, 3) if t_stat is not None else None,
        },
        "significant": bool(enough),
        "min_days_for_significance": MIN_DAYS_FOR_SIGNIFICANCE,
        "note": (
            f"{n} live trading days — {'sufficient' if enough else 'FAR below'} the "
            f"~{MIN_DAYS_FOR_SIGNIFICANCE} (~6mo) needed before realized-vs-expected means anything. "
            f"Accumulating tracker, NOT yet a verdict. Backtest reference = the live 5-sleeve replay "
            f"(long-biased book), NOT the 1.04 market-neutral research construct."
        ),
    }
