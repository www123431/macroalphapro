"""engine/execution/reconcile.py — honest target-vs-actual reconciliation (0-LLM, read-only).

The submit response is NOT the truth (a queued market order reports no fill yet; some symbols get
rejected). Truth = read the broker's ACTUAL account + positions and compare to the target weights
the systematic engine produced. This powers the /api/execution surface and the UI reconciliation
card — the thing that exposes execution slippage / tracking error / breaks that a marked view hides.
"""
from __future__ import annotations

import math

from engine.execution.broker import ExecutionAdapter

DRIFT_EPS = 0.002      # |actual − target| weight below this is "on target"


def reconcile(adapter: ExecutionAdapter, target_weights: dict[str, float]) -> dict:
    """Compare target weights to the broker's actual positions. Pure read; submits nothing."""
    acct = adapter.get_account()
    positions = adapter.get_positions()
    eq = float(acct.equity) or 0.0

    tickers = sorted(set(target_weights) | set(positions))
    rows = []
    sse = 0.0
    gross_target = 0.0
    gross_actual = 0.0
    for tk in tickers:
        tgt_w = float(target_weights.get(tk, 0.0))
        held = positions.get(tk)
        act_mv = float(held.market_value) if held else 0.0
        act_w = (act_mv / eq) if eq else 0.0
        drift = act_w - tgt_w
        sse += drift * drift
        gross_target += abs(tgt_w)
        gross_actual += abs(act_w)
        rows.append({
            "ticker": tk,
            "target_weight": round(tgt_w, 5),
            "actual_weight": round(act_w, 5),
            "drift": round(drift, 5),
            "held_qty": round(float(held.qty), 4) if held else 0.0,
            "on_target": abs(drift) <= DRIFT_EPS,
        })

    targeted_not_held = sorted(
        tk for tk, w in target_weights.items()
        if abs(w) > DRIFT_EPS and (tk not in positions or abs(positions[tk].qty) < 1e-9))
    held_not_targeted = sorted(tk for tk in positions if tk not in target_weights)

    rows.sort(key=lambda r: abs(r["drift"]), reverse=True)
    return {
        "broker": adapter.name,
        "paper": adapter.is_paper,
        "equity": round(eq, 2),
        "cash": round(float(acct.cash), 2),
        "n_targets": int(sum(1 for w in target_weights.values() if abs(w) > DRIFT_EPS)),
        "n_positions": len(positions),
        "gross_target": round(gross_target, 4),
        "gross_actual": round(gross_actual, 4),
        "tracking_error": round(math.sqrt(sse), 5),      # L2 weight distance target↔actual
        "max_abs_drift": round(max((abs(r["drift"]) for r in rows), default=0.0), 5),
        "n_on_target": int(sum(1 for r in rows if r["on_target"])),
        "breaks": {
            "targeted_not_held": targeted_not_held,       # want it, broker has none (rejected/queued/untradeable)
            "held_not_targeted": held_not_targeted,        # broker holds it, no longer a target
        },
        "rows": rows,
    }
