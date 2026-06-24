"""scripts/run_book_monitor.py — Phase 2.0 step 9c CLI entry.

  python scripts/run_book_monitor.py [--dry-run]
                                      [--events-window-days 30]
                                      [--dedup-window-days 7]
                                      [--force-emit]
                                      [--json]

Loads recent research events, runs Employee D's pattern rules,
dedups against recently-emitted doctrine_signal_detected events,
emits fresh signals via emit.doctrine_signal_detected.

No LLM call — D is pure rules. Cost: filesystem reads only.

Default daily cron:  python scripts/run_book_monitor.py
On-demand audit:     python scripts/run_book_monitor.py --dry-run --json
Operator re-fire:    python scripts/run_book_monitor.py --force-emit

Subscribers wake up automatically — Employee A's gatherer reads
doctrine_signal_detected events on each synthesis call.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="Run rules + dedup but skip emit.")
    ap.add_argument("--events-window-days", type=int, default=30,
                     help="Recency window for events passed to rules (default 30).")
    ap.add_argument("--dedup-window-days", type=int, default=7,
                     help="Window for dedup vs prior signals (default 7).")
    ap.add_argument("--force-emit", action="store_true",
                     help="Bypass dedup — re-fire even matching prior signal. "
                          "Use ONLY when intentionally re-promoting a signal.")
    ap.add_argument("--json", action="store_true",
                     help="Machine-readable JSON output.")
    ap.add_argument("--verbose", action="store_true",
                     help="DEBUG-level logging.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from engine.agents.book_monitor.runner import run_book_monitor
    result = run_book_monitor(
        events_window_days = args.events_window_days,
        dedup_window_days  = args.dedup_window_days,
        dry_run            = args.dry_run,
        force_emit         = args.force_emit,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[book_monitor] run_ts            : {result['run_ts']}")
        print(f"[book_monitor] dry_run           : {result['dry_run']}")
        print(f"[book_monitor] events_scanned   : {result['n_events_scanned']}")
        print(f"[book_monitor] rules_run        : {', '.join(result['rules_run'])}")
        print(f"[book_monitor] hits_total       : {result['n_hits_total']}")
        print(f"[book_monitor] hits_fresh       : {result['n_hits_fresh']}")
        print(f"[book_monitor] emitted          : {result['n_emitted']}")
        for h in result["hits"]:
            status = "FRESH" if h["is_fresh"] else "dedup"
            print(f"")
            print(f"  [{status}] {h['rule_name']} · severity={h['severity']}")
            print(f"    family       : {h.get('family')}")
            print(f"    subject_id   : {h['subject_id']}")
            print(f"    summary      : {h['summary']}")
            m = h.get("metrics") or {}
            if "red_count" in m:
                print(f"    red_count    : {m['red_count']} "
                      f"(threshold {m.get('threshold')}, window "
                      f"{m.get('window_days')}d)")
        if result["errors"]:
            print(f"")
            print(f"[book_monitor] errors           : {result['errors']}")

    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
