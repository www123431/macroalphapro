"""scripts/run_papers_curator_synthesis.py — Phase 2.0 step 5a CLI entry.

  python scripts/run_papers_curator_synthesis.py [--dry-run]
                                                 [--summaries-days 14]
                                                 [--events-days 30]
                                                 [--tag session:cos-2026-06-06]
                                                 [--json]

Reads the current substrate state (papers_curator summaries +
deployed sleeves + recent events + doctrine snippets), calls Sonnet 4.6
for 0-3 cross-source candidates, persists to hypotheses.jsonl with
extraction_method=LLM_SYNTHESIS.

Cost: ≤ $0.10/call (Sonnet 4.6 single-shot, no retry).

Default daily cron call:  python scripts/run_papers_curator_synthesis.py
Weekly chief_of_staff:    + --tag session:cos-<date>
UI button preview:        + --dry-run --json

Exits non-zero only on internal errors (gather/write). LLM returning
0 candidates is NOT an error — it's the prompt-encoded "prefer empty
over weak" path.
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
                     help="Run LLM call but skip persistence (preview mode).")
    ap.add_argument("--summaries-days", type=int, default=14,
                     help="Recency window for paper summaries (default 14).")
    ap.add_argument("--events-days", type=int, default=30,
                     help="Recency window for events (default 30).")
    ap.add_argument("--tag", action="append", default=[],
                     help="Extra tag(s) added to written rows.")
    ap.add_argument("--json", action="store_true",
                     help="Emit machine-readable JSON instead of human summary.")
    ap.add_argument("--verbose", action="store_true",
                     help="DEBUG-level logging.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline

    result = run_synthesis_pipeline(
        dry_run        = args.dry_run,
        summaries_days = args.summaries_days,
        events_days    = args.events_days,
        extra_tags     = tuple(args.tag),
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        snap = result["snapshot"]
        print(f"[synthesis] run_ts            : {result['run_ts']}")
        print(f"[synthesis] dry_run           : {result['dry_run']}")
        print(f"[synthesis] snapshot          : "
              f"{snap.get('recent_summaries', 0)} papers, "
              f"{snap.get('deployed_sleeves', 0)} sleeves, "
              f"{snap.get('recent_events', 0)} events, "
              f"{snap.get('doctrine_snippets', 0)} doctrine snippets")
        print(f"[synthesis] candidates        : {result['n_candidates']}")
        print(f"[synthesis] written           : {result['n_written']}")
        for i, c in enumerate(result["candidates"]):
            print(f"")
            print(f"  candidate {i+1}: {c.get('claim', '')[:100]}")
            print(f"    family           : {c.get('mechanism_family')} / "
                  f"{c.get('mechanism_subtype')}")
            print(f"    direction/magn   : {c.get('predicted_direction')} / "
                  f"{c.get('predicted_magnitude')}")
            print(f"    cochrane         : {c.get('cochrane_frame')}")
            print(f"    expected_prior   : {c.get('expected_outcome_prior')}")
            paps = c.get("synthesizes_paper_ids", [])
            evs  = c.get("synthesizes_event_ids", [])
            print(f"    provenance       : {len(paps)} papers + {len(evs)} events")
            gc = c.get("graveyard_conflicts") or []
            dc = c.get("doctrine_conflicts") or []
            if gc:
                print(f"    graveyard_conf   : {gc}")
            if dc:
                print(f"    doctrine_conf    : {dc}")
        if result["errors"]:
            print(f"")
            print(f"[synthesis] errors            : {result['errors']}")

    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
