"""scripts/run_execution.py — paper-execution cron entry (Phase B, 2026-05-27).

Runs the consolidated paper book's execution, ALIGNED TO US MARKET HOURS (not the 06:30 SGT
book-computation cron — that just needs the prior close; execution wants RTH for clean fills):

  EQUITY legs → Alpaca paper (real broker). Rebalance to the current ui_artifact target weights.
    Anti-churn (min_order_usd) means hold days place ~no orders, so a DAILY run is safe and respects
    each sleeve's own cadence implicitly (target only moves when signals do).
  CARRY legs → internal futures sim (durable, $10M institutional scale, whole contracts). MONTHLY:
    only acts on the FIRST trading day of the month (gate inside).
  (Trend sleeve deploys on Alpaca ETFs per spec 75 — a separate candidate track, not wired here yet.)

PAPER-ONLY (0 real capital): the rebalancer/adapters refuse non-paper accounts.

  python scripts/run_execution.py                  # equity (daily) + carry (monthly gate)
  python scripts/run_execution.py --no-carry       # equity only
  python scripts/run_execution.py --dry-run        # compute, don't submit

Schedule (US RTH covers SGT/China 21:30-04:00 summer / 22:30-05:00 winter → 23:00 is RTH both):
  Windows Task Scheduler, DAILY 23:00 local:
    schtasks /Create /TN "PaperExecution" /TR "<py> <root>\\scripts\\run_execution.py" /SC DAILY /ST 23:00
  cron (server):  30 15 * * 1-5  cd /path/intern && python scripts/run_execution.py >> logs/exec.log 2>&1
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _is_first_trading_day_of_month(d: datetime.date) -> bool:
    try:
        from engine.daily_batch import _is_first_trading_day_of_month as gate
        return gate(d)
    except Exception:
        # fallback: first weekday of the month
        first = d.replace(day=1)
        while first.weekday() >= 5:
            first += datetime.timedelta(days=1)
        return d == first


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-equity", action="store_true", help="skip the Alpaca equity rebalance")
    ap.add_argument("--no-carry", action="store_true", help="skip the carry futures forward step")
    ap.add_argument("--dry-run", action="store_true", help="compute orders but do not submit")
    ap.add_argument("--force-carry", action="store_true", help="run carry even if not month-start")
    args = ap.parse_args()

    today = datetime.date.today()
    out: dict = {"date": today.isoformat(), "dry_run": args.dry_run}

    # ── EQUITY → Alpaca (daily; anti-churn handles hold days) ──
    if not args.no_equity:
        try:
            from engine.execution.run_paper_execution import run as run_equity
            r = run_equity(use_alpaca=True, submit=not args.dry_run)
            rep = r.get("report", {})
            out["equity"] = {"as_of": r.get("as_of"), "n_orders": rep.get("n_orders"),
                             "n_fills": rep.get("n_fills"),
                             "dropped_untradable": list((r.get("dropped_untradable") or {}).keys())}
        except Exception as exc:
            out["equity_error"] = str(exc)[:200]

    # ── CARRY → futures sim (monthly: first trading day of month) ──
    if not args.no_carry:
        if args.force_carry or _is_first_trading_day_of_month(today):
            try:
                from engine.execution.futures_book import run_futures_forward
                fr = run_futures_forward(submit=not args.dry_run)
                out["carry"] = {k: fr.get(k) for k in
                                ("target_month", "n_target", "n_contracts_held", "equity", "n_marked")}
            except Exception as exc:
                out["carry_error"] = str(exc)[:200]
        else:
            out["carry"] = "skipped (not first trading day of month)"

    print(json.dumps(out, ensure_ascii=False))
    return 0 if ("equity_error" not in out and "carry_error" not in out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
