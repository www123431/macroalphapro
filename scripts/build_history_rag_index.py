"""
Build (or rebuild) the project history RAG index.

Usage
-----
  # Full rebuild (idempotent — upsert by doc_id):
  python scripts/build_history_rag_index.py

  # Incremental — only rows created/modified in last N days:
  python scripts/build_history_rag_index.py --since-days 7

  # Only one source:
  python scripts/build_history_rag_index.py --sources decision_log

  # Reset + full rebuild (drops the collection first):
  python scripts/build_history_rag_index.py --reset

Operationally
-------------
First run downloads ~1.1GB sentence-transformers model (paraphrase-
multilingual-mpnet-base-v2) into the HuggingFace cache; subsequent runs
load from disk in ~5 seconds.

Output is JSON to stdout: {source_type: n_indexed} counters.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.agents.history_rag import (  # noqa: E402
    build_index, collection_stats, reset_store, SourceType,
)

_VALID_SOURCES = [s.value for s in SourceType]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reset", action="store_true",
                   help="Drop the collection before indexing (full rebuild).")
    p.add_argument("--since-days", type=int, default=None,
                   help="Only re-index rows modified in the last N days.")
    p.add_argument("--sources", nargs="*", default=None,
                   choices=_VALID_SOURCES,
                   help="Restrict to a subset of sources.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress logs (counters JSON still goes to stdout).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.reset:
        if not args.quiet:
            print("Resetting collection...", file=sys.stderr)
        reset_store()

    sources = None
    if args.sources:
        sources = [SourceType(s) for s in args.sources]

    modified_since = None
    if args.since_days is not None:
        modified_since = (
            datetime.datetime.utcnow() - datetime.timedelta(days=args.since_days)
        )

    t0 = time.time()
    counters = build_index(sources=sources, modified_since=modified_since)
    elapsed = time.time() - t0

    stats = collection_stats()
    output = {
        "indexed":         counters,
        "elapsed_seconds": round(elapsed, 2),
        "store_total":     stats["n_total"],
        "store_by_source": stats["by_source"],
    }
    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
