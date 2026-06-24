"""scripts/autopilot_live_run.py — cron entry for F14b live runs (2026-06-05).

Wraps engine.agents.autopilot_live.run_top1() with logging + exit code:

  exit 0  on GREEN / MARGINAL / RED — verdict emitted, run logged
  exit 1  on hard failure (compose error, no candidates, etc.)

Cron usage (Windows Task Scheduler / dev.bat startup hook):
  python scripts/autopilot_live_run.py [--force]

Manual usage (smoke test):
  python scripts/autopilot_live_run.py --force
    (force=bypass compose cache)

Output:
  - data/autopilot/_live/<date>.json           run log (LiveRunResult)
  - docs/capability_evidence/autopilot/*.md    human-auditable evidence
  - data/research_store/events.jsonl           appended events
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                     help="Bypass compose cache (re-materialize returns)")
    ap.add_argument("--force-da", action="store_true",
                     help="Fire Devil's Advocate even on RED verdicts (smoke test only)")
    args = ap.parse_args()

    from engine.agents.autopilot_live import run_top1

    try:
        result = run_top1(force_compose=args.force, force_da=args.force_da)
    except Exception as exc:
        logger.exception("F14b live run failed: %s", exc)
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1

    print()
    print("=" * 70)
    print(f"F14b live run COMPLETE: {result.verdict} (score {result.score}/4)")
    if result.raw_verdict and result.raw_verdict != result.verdict:
        print(f"   (raw verdict {result.raw_verdict}/{result.raw_score} -> "
               f"DA-downgraded to {result.verdict}/{result.score})")
    print("=" * 70)
    print(f"  subject_id:         {result.subject_id}")
    print(f"  source hyp:         {result.source_hypothesis_id[:8]}")
    print(f"  family/signal:      {result.family}/{result.signal_type}")
    print(f"  n_obs:              {result.n_obs}")
    print(f"  IS Sharpe:          {result.is_sharpe:+.2f}")
    print(f"  OOS Sharpe:         {result.oos_sharpe:+.2f}")
    print(f"  t-stat:             {result.t_stat:+.2f}")
    print(f"  Deflated SR:        {result.deflated_sr:+.3f}")
    print(f"  Max DD:             {result.max_dd*100:+.1f}%")
    print()
    print(f"  DA tag:             {result.da_tag}")
    if result.da_fired:
        print(f"  DA severity:        {result.da_severity}  (confidence {result.da_confidence:.2f})")
        print(f"  DA attack:          {result.da_attack_vector}")
    else:
        print(f"  DA skipped:         verdict was RED (DA does not challenge kills)")
    print()
    print(f"  evidence event:     {result.evidence_event_id}")
    print(f"  verdict event:      {result.verdict_event_id}")
    if result.da_event_id:
        print(f"  DA event:           {result.da_event_id}")
    print(f"  evidence file:      {result.capability_evidence_path}")
    print(f"  elapsed:            {result.elapsed_s}s")
    print()
    print("NOTE: F14b only emits research verdicts. Any PROMOTE_TO_PAPER_TRADE "
           "or library.yaml deploy decision remains a manual human action.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
