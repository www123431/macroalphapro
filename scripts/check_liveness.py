"""scripts/check_liveness.py — P0e of liveness layer (2026-06-02).

External watcher cron. Recommended schedule: every 15 min.
Reads data/research/liveness_heartbeat.jsonl, classifies state,
exits with code matching severity so the scheduler can chain alerts.

Exit codes (so Task Scheduler / Watchdog can route on the integer):
  0   OK or INFO_OFF_HOURS / INFO_WEEKEND
  1   WARN_STATUS    — yesterday halted or partial
  2   ALERT_NO_SHOW  — cron failed to fire

Usage:
  python scripts/check_liveness.py            # current UTC now
  python scripts/check_liveness.py --json     # machine-readable
  python scripts/check_liveness.py --expected-hour-utc 22

Wire into Watchdog: a cron rule that runs this every 15 min and
escalates exit code 2 to a HIGH alert.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_VERDICT_EXIT = {
    "OK":              0,
    "INFO_OFF_HOURS":  0,
    "INFO_WEEKEND":    0,
    "WARN_STATUS":     1,
    "ALERT_NO_SHOW":   2,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Liveness watcher — read heartbeat ledger, raise alarm if cron missed"
    )
    parser.add_argument("--json", action="store_true",
                         help="emit JSON verdict on stdout")
    parser.add_argument("--expected-hour-utc", type=int, default=None,
                         help="override DEFAULT_EXPECTED_HOUR_UTC (22 = 06:00 SGT)")
    parser.add_argument("--grace-min", type=int, default=None,
                         help="override DEFAULT_NO_SHOW_GRACE_MIN (90)")
    parser.add_argument("--trading-days-only", action="store_true", default=True,
                         help="ignore weekends (default True)")
    args = parser.parse_args()

    from engine.research.liveness_heartbeat import (
        assess_liveness, DEFAULT_EXPECTED_HOUR_UTC, DEFAULT_NO_SHOW_GRACE_MIN,
    )
    now = _dt.datetime.utcnow()
    verdict = assess_liveness(
        now_utc=now,
        expected_hour_utc=args.expected_hour_utc if args.expected_hour_utc is not None
                          else DEFAULT_EXPECTED_HOUR_UTC,
        no_show_grace_min=args.grace_min if args.grace_min is not None
                           else DEFAULT_NO_SHOW_GRACE_MIN,
        trading_days_only=args.trading_days_only,
    )

    if args.json:
        print(json.dumps(verdict, indent=2, default=str))
    else:
        v = verdict.get("verdict", "UNKNOWN")
        print(f"[liveness] {v}: {verdict.get('explanation','')}")
        latest = verdict.get("latest")
        if isinstance(latest, dict):
            print(f"  latest hb: as_of={latest.get('as_of')} status={latest.get('status')} "
                  f"n_orders={latest.get('n_orders')} n_fills={latest.get('n_fills')}")

    return _VERDICT_EXIT.get(verdict.get("verdict", ""), 1)


if __name__ == "__main__":
    sys.exit(main())
