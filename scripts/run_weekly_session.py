"""scripts/run_weekly_session.py — Phase 2.0 step 14a CLI.

  python scripts/run_weekly_session.py [--dry-run]
                                        [--session-id cos-2026-06-06]
                                        [--json]

Runs one chief_of_staff weekly session: D → A → B in order, emits a
chief_of_staff_session_run event aggregating the results.

Designed for weekly cron — Monday 03:00 UTC default. SETUP HINTS:

  ── Linux / macOS (crontab) ──
  Add ONE line to `crontab -e`:
    0 3 * * 1  cd /path/to/intern && /usr/bin/python scripts/run_weekly_session.py >> data/cos_log.txt 2>&1
  Verify with `crontab -l` + `tail -f data/cos_log.txt` after the
  first Monday.

  ── Windows (Task Scheduler) ──
  1. taskschd.msc → Create Basic Task → Name: "MacroAlphaPro Weekly"
  2. Trigger: Weekly · Monday · 03:00
  3. Action: Start a program
       Program:  python.exe (full path)
       Args:     ${REPO_ROOT}\Desktop\intern\scripts\run_weekly_session.py
       Start in: ${REPO_ROOT}\Desktop\intern
  4. Settings → "Run whether user is logged on or not" + store creds
  5. After saving, right-click → Run to test once

  ── On-demand from the UI (no cron needed) ──
  /lab/today now has a "Run now" button (Phase 2.0 step 15) that
  triggers POST /api/chief_of_staff/run with the same orchestrator.
  Useful for ad-hoc / mid-week runs.

Same-day re-runs are SAFE:
  - D's runner has its own dedup window (default 7 days) so identical
    cluster signals don't fire twice
  - A's runner emits one event per call; same substrate same week
    will likely produce identical empty result
  - B's runner is idempotent per (hypothesis_id) → already-reviewed
    rows are skipped

Cost ceiling per session: ≤ $0.10 (A's Sonnet call) + ≤ $0.05 × N (B's
Sonnet calls, capped at max_hypotheses=10 default) + ≤ $0.05 (memo).
D is free (pure rules). Realistic month-of-Mondays cost: ≤ $1-2.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="Run substeps with dry_run=True; no event emits, "
                          "no hypotheses persisted, no verdicts persisted.")
    ap.add_argument("--session-id", type=str, default=None,
                     help="Deterministic correlation id. Defaults to "
                          "'cos-<YYYY-MM-DD>'.")
    ap.add_argument("--json", action="store_true",
                     help="Machine-readable JSON output.")
    ap.add_argument("--verbose", action="store_true",
                     help="DEBUG logging.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from engine.agents.chief_of_staff.runner import run_weekly_session
    result = run_weekly_session(
        session_id = args.session_id,
        dry_run    = args.dry_run,
    )

    if args.json:
        d = dataclasses.asdict(result)
        print(json.dumps(d, indent=2, ensure_ascii=False))
    else:
        print(f"[cos] session_id          : {result.session_id}")
        print(f"[cos] run_ts              : {result.run_ts}")
        print(f"[cos] dry_run             : {result.dry_run}")
        print(f"[cos] session_event_id    : {result.session_event_id}")
        print(f"")
        print(f"[D] events_scanned       : {result.d_result.get('n_events_scanned', 0)}")
        print(f"[D] hits_total           : {result.d_result.get('n_hits_total', 0)}")
        print(f"[D] hits_fresh           : {result.d_result.get('n_hits_fresh', 0)}")
        print(f"[D] emitted              : {result.d_emitted}")
        print(f"")
        print(f"[A] candidates           : {result.a_n_candidates}")
        print(f"[A] written              : {result.a_n_written}")
        a_snap = result.a_result.get("snapshot") or {}
        print(f"[A] snapshot             : "
              f"{a_snap.get('recent_summaries', 0)} papers, "
              f"{a_snap.get('deployed_sleeves', 0)} sleeves, "
              f"{a_snap.get('recent_events', 0)} events")
        print(f"")
        print(f"[B] candidates queued    : {result.b_result.get('n_candidates', 0)}")
        print(f"[B] reviewed             : {result.b_n_reviewed}")
        print(f"[B] verdicts persisted   : {result.b_result.get('n_persisted', 0)}")
        print(f"[B] queue pending appr.  : {result.b_n_pending_approval}")
        print(f"")
        if result.memo:
            print("=" * 70)
            print("WEEKLY MEMO")
            print("=" * 70)
            print(f"  {result.memo.get('headline', '')}")
            print()
            for i, b in enumerate(result.memo.get("bullets") or [], 1):
                print(f"  {i}. {b}")
            print()
            print(f"  Next focus: {result.memo.get('whats_next', '')}")
            print()
        if result.errors:
            print(f"[cos] errors             : {result.errors}")

    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
