"""scripts/papers_curator_crawl.py — Employee A daily crawl (Phase 1).

Fetches latest papers from all enabled sources (currently: arxiv q-fin),
dedups against the local cache, appends new entries. Idempotent —
re-running within a day is safe (a no-op for already-cached IDs).

  python scripts/papers_curator_crawl.py [--arxiv-max 50] [--print]

Output:
  data/papers_curator/cache.jsonl   (append-only)
  stdout: summary "fetched N, new K, cache total M"

Wire into run_app.py daily kickoff later (next phase).

Cost: $0 (no LLM). Wall: ~2-5s for arxiv.
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
    ap.add_argument("--arxiv-max", type=int, default=50,
                     help="Max arxiv results per run (default 50)")
    ap.add_argument("--print", action="store_true",
                     help="Print new candidate titles to stdout")
    args = ap.parse_args()

    from engine.agents.papers_curator import (
        crawl_all, save_new_candidates, CACHE_PATH, load_cache,
    )

    cands = crawl_all(arxiv_max=args.arxiv_max)
    print(f"Fetched: {len(cands)} candidates")

    n_before = len(load_cache())
    n_new = save_new_candidates(cands)
    print(f"New (dedup against {n_before}): {n_new}")
    print(f"Cache total now: {n_before + n_new}")
    print(f"Cache file: {CACHE_PATH}")

    if args.print and n_new > 0:
        print()
        print("New candidates:")
        # Re-load to get the newly appended ones at the tail
        all_cands = load_cache()
        for c in all_cands[-n_new:]:
            cats = ",".join(c.categories[:3]) if c.categories else "(no cat)"
            print(f"  [{c.source}/{c.source_id}] {cats}  {c.title[:80]}")

    # Write today's sentinel so daily-kickoff doesn't re-fire. Sentinel
    # written even if 0 new candidates (the crawl itself succeeded;
    # absence of new papers ≠ failure to crawl). Skipped entirely if
    # cands is empty (which means crawl failed — let it retry on next
    # launch).
    if cands:
        import datetime as _dt
        sentinel_dir = REPO_ROOT / "data" / "papers_curator" / "_runs"
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
        (sentinel_dir / f"{today}.ok").write_text(
            f"fetched={len(cands)} new={n_new}", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
