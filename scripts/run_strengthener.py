"""scripts/run_strengthener.py — Phase 2.0 step 11b CLI.

  python scripts/run_strengthener.py [--dry-run]
                                      [--max-hypotheses 10]
                                      [--json]

Loops PROPOSED + LLM_SYNTHESIS hypothesis rows in hypotheses.jsonl,
calls Employee B's review per row, persists verdicts to
data/strengthener/verdicts.jsonl.

Idempotent: B skips hypotheses already in verdicts.jsonl. Re-runs are
safe — they only pick up NEW PROPOSED rows since last invocation.

Cost ceiling: ~$0.05 per review (Sonnet 4.6, single tool-use call) ×
N hypotheses, capped at max_hypotheses (default 10 = ~$0.50/run).

Designed to run AFTER each A synthesis pass — daily cron, or
on-demand from the chief_of_staff weekly session (step 14).
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
                     help="Call B but skip verdict persistence (preview).")
    ap.add_argument("--max-hypotheses", type=int, default=10,
                     help="Cap per-run cost gate (default 10).")
    ap.add_argument("--json", action="store_true",
                     help="Machine-readable JSON.")
    ap.add_argument("--verbose", action="store_true",
                     help="DEBUG logging.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from engine.agents.strengthener.runner import run_strengthener_pipeline
    result = run_strengthener_pipeline(
        dry_run        = args.dry_run,
        max_hypotheses = args.max_hypotheses,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[strengthener] run_ts        : {result['run_ts']}")
        print(f"[strengthener] dry_run       : {result['dry_run']}")
        print(f"[strengthener] candidates    : {result['n_candidates']}")
        print(f"[strengthener] reviewed      : {result['n_reviewed']}")
        print(f"[strengthener] persisted     : {result['n_persisted']}")
        for v in result["verdicts"]:
            vt = v["verdict_type"]
            print(f"")
            print(f"  [{vt}] {v['hypothesis_id'][:8]} · "
                  f"conf={v['confidence']:.2f}")
            print(f"    {v['one_line_summary']}")
            if v.get("similar_to_deployed"):
                print(f"    similar_to     : {v['similar_to_deployed']}")
            if v.get("replaces_decaying"):
                print(f"    replaces_decay : {v['replaces_decaying']}")
            if v.get("blocking_doctrine_id"):
                print(f"    blocking       : {v['blocking_doctrine_id']}")
                print(f"    amendment      : {v.get('proposed_amendment_summary') or ''}")
            if v.get("recommended_pipeline_action"):
                print(f"    next_action    : {v['recommended_pipeline_action']}")
        if result["errors"]:
            print(f"")
            print(f"[strengthener] errors        : {result['errors']}")

    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
