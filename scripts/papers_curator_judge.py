"""scripts/papers_curator_judge.py — run filter on un-judged candidates.

Reads cache.jsonl, picks candidates not yet in judgments.jsonl (or
present but with an older judgment than today), runs Deepseek 1-line
filter, appends to judgments.jsonl. Idempotent.

  python scripts/papers_curator_judge.py [--limit 50] [--print]

Wall: ~2-4s per paper (Deepseek). 30 papers ≈ 1-2 min.
Cost: ~$0.001/paper × N. Cheap.

Wired into daily kickoff in run_app.py (runs after crawler).
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
    ap.add_argument("--limit", type=int, default=50,
                     help="Max candidates to judge this run (default 50)")
    ap.add_argument("--print", action="store_true",
                     help="Print each judgment as it's written")
    args = ap.parse_args()

    from engine.agents.papers_curator import (
        load_cache, latest_by_paper, judge_paper, append_judgment,
    )

    cache = load_cache()
    judged = latest_by_paper()
    unjudged = [c for c in cache if (c.source, c.source_id) not in judged]

    if not unjudged:
        print(f"All {len(cache)} cached candidates already judged. No-op.")
        return 0

    to_judge = unjudged[:args.limit]
    print(f"Judging {len(to_judge)} of {len(unjudged)} unjudged candidates"
           f" (cache total {len(cache)}, already judged {len(judged)})")

    n_yes = 0
    n_no = 0
    n_fail = 0
    for i, c in enumerate(to_judge, start=1):
        j = judge_paper(c)
        if j is None:
            n_fail += 1
            if args.print:
                print(f"  [{i}/{len(to_judge)}] FAIL  {c.source}/{c.source_id}  {c.title[:60]}")
            continue
        append_judgment(j)
        if j.is_tradable_factor:
            n_yes += 1
        else:
            n_no += 1
        if args.print:
            tag = "YES" if j.is_tradable_factor else "no "
            print(f"  [{i}/{len(to_judge)}] {tag}  conf={j.confidence:.2f}  "
                   f"{j.category_guess:<11}  {c.title[:60]}")
            print(f"      reason: {j.one_line_reason[:140]}")

    print()
    print(f"Done: YES={n_yes}, no={n_no}, FAIL={n_fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
