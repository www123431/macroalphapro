"""scripts/papers_curator_summarize.py — eager-summary on YES-filtered papers.

Reads judgments.jsonl, picks YES-filtered candidates not yet in
summaries.jsonl, runs Deepseek 5-field summary, appends.

  python scripts/papers_curator_summarize.py [--limit 20] [--also-no] [--print]

  --limit:    cap per-run (default 20). Cost gate so a backlog doesn't
              burn $$$ on a single launch.
  --also-no:  ALSO summarize NO-filtered candidates (lazy / user-requested
              mode). Default OFF — daily kickoff only summarizes YES.
  --print:    print each summary's recommended_action + 1-liner.

Cost: ~$0.01/summary. With ~5-10 YES/day ≈ $0.10/day; --also-no can
~10× that on a big backlog.
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
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--also-no", action="store_true",
                     help="Also summarize NO-filtered (lazy / user-request mode)")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    from engine.agents.papers_curator import (
        load_cache, latest_by_paper, latest_summary_by_paper,
        summarize_paper, append_summary,
    )

    cache_by_key = {(c.source, c.source_id): c for c in load_cache()}
    judgments = latest_by_paper()
    summaries = latest_summary_by_paper()

    # Pick candidates to summarize. Order: YES first (always),
    # NO only if --also-no.
    to_do = []
    for key, j in judgments.items():
        if key in summaries:
            continue   # already summarized
        if key not in cache_by_key:
            continue   # judgment without cache row (shouldn't happen)
        if not j.is_tradable_factor and not args.also_no:
            continue
        to_do.append((cache_by_key[key], j))

    # Stable order: highest confidence YES first; NO papers tail
    to_do.sort(key=lambda p: (not p[1].is_tradable_factor,
                               -p[1].confidence,
                               p[0].source_id))
    to_do = to_do[:args.limit]

    if not to_do:
        print(f"Nothing to summarize. cache={len(cache_by_key)} "
               f"judgments={len(judgments)} summaries={len(summaries)}")
        return 0

    print(f"Summarizing {len(to_do)} candidates "
           f"(YES + NO={'yes' if args.also_no else 'no'})")

    n_ingest = 0
    n_read = 0
    n_skip = 0
    n_fail = 0
    for i, (c, j) in enumerate(to_do, start=1):
        triggered = "auto_yes" if j.is_tradable_factor else "user_request_no"
        s = summarize_paper(c, j, triggered_by=triggered)
        if s is None:
            n_fail += 1
            if args.print:
                print(f"  [{i}/{len(to_do)}] FAIL  {c.source_id}  {c.title[:60]}")
            continue
        append_summary(s)
        if s.recommended_action == "INGEST":
            n_ingest += 1
        elif s.recommended_action == "READ_AND_DISCARD":
            n_read += 1
        else:
            n_skip += 1
        if args.print:
            print(f"  [{i}/{len(to_do)}] {s.recommended_action:<17}  "
                   f"{c.source_id}  {c.title[:55]}")
            print(f"      thesis: {s.thesis[:120]}")
            if s.risk_flags:
                print(f"      risks:  {', '.join(s.risk_flags[:4])}")

    print()
    print(f"Done: INGEST={n_ingest}  READ_AND_DISCARD={n_read}  "
           f"SKIP={n_skip}  FAIL={n_fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
