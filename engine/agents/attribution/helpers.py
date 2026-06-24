"""engine.agents.attribution.helpers — Layer 4 piece 3a join helpers.

Read-time JOIN primitives over the existing stores. These are the
building blocks the piece-3b query layer composes into lifecycle
records and rollup aggregates.

Design principle: NO new storage. We don't denormalize. Every helper
reads from the existing source-of-truth stores (events / hypotheses /
verdicts / resolutions / lessons / cache).

Why this matters: when we add piece 3c (watchlist + doctrine
reweighting based on outcomes), the rollups it consumes are computed
freshly from current state, not from stale denormalized rows.
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CACHE_PATH = _REPO_ROOT / "data" / "papers_curator" / "cache.jsonl"
_WATCHLIST_PATH = _REPO_ROOT / "data" / "papers_curator" / "watchlist.yaml"


# ────────────────────────────────────────────────────────────────────
# Paper-source lookup — bridges A's synthesizes_paper_ids to which
# source (arxiv vs semantic_scholar via watchlist) brought it in.
# ────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _build_paper_source_map() -> dict[str, str]:
    """Read cache.jsonl once + build {(source,source_id) projected
    key -> source string}. Cached for the process lifetime; the cache
    can be cleared with clear_caches() if substrate changed mid-run.

    Key formats covered (since A's synthesizes_paper_ids can use any):
      - 'arxiv/2606.12345'     (source/source_id composite)
      - '2606.12345'           (bare source_id)
      - 'semantic_scholar/ss_pid'
      - 'ss_pid'
    All are indexed pointing back to source string.
    """
    out: dict[str, str] = {}
    if not _CACHE_PATH.is_file():
        return out
    with _CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            src = str(row.get("source") or "")
            sid = str(row.get("source_id") or "")
            if not src or not sid:
                continue
            # Index both the composite key and the bare id
            composite = f"{src}/{sid}"
            out[composite] = src
            out[sid] = src
    return out


def get_paper_source(paper_id: str) -> Optional[str]:
    """Look up the source ('arxiv' / 'semantic_scholar' / etc.) for
    a paper_id, by joining against cache.jsonl.

    paper_id formats handled: 'arxiv/2606.x', 'semantic_scholar/x',
    'x' (bare id). Returns None if not in cache.

    Used by attribution rollups: 'this hypothesis cited 3 papers; how
    many were from arxiv RSS vs the adversarial author watchlist?'"""
    if not paper_id:
        return None
    table = _build_paper_source_map()
    return table.get(paper_id)


# ────────────────────────────────────────────────────────────────────
# Paper → watchlist author resolution
# ────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _build_paper_author_map() -> dict[str, tuple[str, ...]]:
    """{paper_id (composite or bare) → tuple of author NAMES}.
    Used to attribute a candidate's citations back to specific authors
    on the watchlist."""
    out: dict[str, tuple[str, ...]] = {}
    if not _CACHE_PATH.is_file():
        return out
    with _CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            src = str(row.get("source") or "")
            sid = str(row.get("source_id") or "")
            authors = tuple(
                str(a) for a in (row.get("authors") or [])
                if a
            )
            if not authors:
                continue
            composite = f"{src}/{sid}"
            out[composite] = authors
            out[sid] = authors
    return out


def get_paper_authors(paper_id: str) -> tuple[str, ...]:
    """Return the author NAMES (not SS ids) attached to a paper_id by
    joining against cache.jsonl. Empty tuple if paper unknown."""
    if not paper_id:
        return ()
    table = _build_paper_author_map()
    return table.get(paper_id, ())


@functools.lru_cache(maxsize=1)
def _watchlist_name_set() -> frozenset[str]:
    """Lowercased name set of authors on the watchlist. Used to test
    whether a paper's authors include a watchlisted name."""
    try:
        from engine.agents.papers_curator.watchlist import load_watchlist
        return frozenset(a.name.strip().lower()
                          for a in load_watchlist())
    except Exception as exc:
        logger.warning("watchlist load failed: %s", exc)
        return frozenset()


def paper_from_watchlist_authors(paper_id: str) -> tuple[str, ...]:
    """Return the SUBSET of paper authors that are on the watchlist.
    Empty tuple means 'no watchlist author cited this paper'.

    Used for attribution: 'this candidate cited X papers, of which
    Y were authored by watchlist members — and those Y had GREEN
    rate Z'."""
    authors = get_paper_authors(paper_id)
    if not authors:
        return ()
    wl = _watchlist_name_set()
    if not wl:
        return ()
    hits = tuple(a for a in authors if a.strip().lower() in wl)
    return hits


# ────────────────────────────────────────────────────────────────────
# Cache invalidation — called by long-running processes if substrate
# changed mid-run (e.g. tests, chief_of_staff between phases)
# ────────────────────────────────────────────────────────────────────
def clear_caches() -> None:
    """Clear all lru_cache entries in this module. Necessary if
    cache.jsonl or watchlist.yaml mutated mid-process."""
    _build_paper_source_map.cache_clear()
    _build_paper_author_map.cache_clear()
    _watchlist_name_set.cache_clear()
