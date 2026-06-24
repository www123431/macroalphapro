"""engine.agents.papers_curator.store — dedup cache (jsonl append-only).

Schema-stable persistence for crawled paper candidates. Dedup key =
(source, source_id). Re-crawling the same arxiv id is a no-op.

File: data/papers_curator/cache.jsonl
Schema: one PaperCandidate per line as JSON.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from engine.agents.papers_curator.crawler import PaperCandidate

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CACHE_PATH = _REPO_ROOT / "data" / "papers_curator" / "cache.jsonl"


def load_cache() -> list[PaperCandidate]:
    """Return all cached candidates. Order preserved (chronological by
    fetched_ts, but caller should sort if needed).

    Defensively drops rows missing `source` OR `source_id` — these
    would corrupt dedup (every empty-key row would collide on the
    ('', '') tuple) and produce empty downstream candidates. Caught
    in 2026-06-07 failure-surface walk: a malformed `{"x": 1}` line
    parses as JSON but lacks the required keys.
    """
    if not CACHE_PATH.is_file():
        return []
    out: list[PaperCandidate] = []
    n_dropped = 0
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                pc = PaperCandidate.from_dict(json.loads(line))
            except Exception as exc:
                logger.warning("cache.jsonl line %d malformed; skipping: %s",
                               ln_no, exc)
                continue
            if not pc.source or not pc.source_id:
                logger.warning("cache.jsonl line %d missing source/source_id; "
                               "dropping", ln_no)
                n_dropped += 1
                continue
            out.append(pc)
    if n_dropped:
        logger.info("cache.jsonl: dropped %d rows missing required keys",
                     n_dropped)
    return out


def _cache_keys() -> set[tuple[str, str]]:
    """Set of (source, source_id) tuples currently in cache."""
    return {(c.source, c.source_id) for c in load_cache()}


def save_new_candidates(candidates: list[PaperCandidate]) -> int:
    """Append candidates whose (source, source_id) is not yet in cache.
    Returns the count of new candidates actually written.
    """
    if not candidates:
        return 0
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen = _cache_keys()
    new_count = 0
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        for c in candidates:
            key = (c.source, c.source_id)
            if key in seen:
                continue
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            seen.add(key)
            new_count += 1
    logger.info("cache: appended %d new of %d candidates (cache size now %d)",
                 new_count, len(candidates), len(seen))
    return new_count
