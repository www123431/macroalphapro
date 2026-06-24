"""engine.agents.papers_curator.watchlist — Stage A piece 2.

Manages the adversarial author watchlist for anti-mental-rut paper
acquisition.

Per project_anti_rut_doctrine_2026-06-07.md:
  The watchlist is INTENTIONALLY curated for cognitive diversity, not
  topical alignment. These are authors whose work the principal is
  *less likely* to encounter through normal arxiv RSS or organic
  reading. They span:

    - Mean-reversion / momentum-crash specialists (anti-momentum)
    - Post-publication decay researchers (anti-published-factor)
    - Multiple-testing / overfitting critics (anti-naive-discovery)
    - Crisis / regime-shift specialists (anti-stationarity-assumption)
    - Cross-asset / macro views (anti-single-asset-narrative)

Storage: data/papers_curator/watchlist.yaml. Version-controlled in
git so the curation history has an audit trail (changes to who's on
the list = changes to principal's anti-rut intent).

Schema (frozen v1):
  schema_version:  1
  authors:
    - author_id:        Semantic Scholar id (resolved on first crawl)
      name:             principal-readable name
      rationale:        why this author is on the watchlist
      added_ts:         iso UTC when added
      last_crawled_ts:  iso UTC of last successful watchlist crawl
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_REPO_ROOT     = Path(__file__).resolve().parent.parent.parent.parent
WATCHLIST_PATH = _REPO_ROOT / "data" / "papers_curator" / "watchlist.yaml"

WATCHLIST_SCHEMA_VERSION = 1


@_dc.dataclass(frozen=True)
class WatchlistAuthor:
    """One curated author on the adversarial watchlist."""
    author_id:        str           # Semantic Scholar id ("" until resolved)
    name:             str           # human-readable, used for SS lookup
    rationale:        str           # why this author breaks principal's bias
    added_ts:         str
    last_crawled_ts:  str = ""      # empty = never crawled

    def to_dict(self) -> dict:
        return _dc.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WatchlistAuthor":
        return cls(
            author_id       = str(d.get("author_id", "")),
            name            = str(d.get("name", "")),
            rationale       = str(d.get("rationale", "")),
            added_ts        = str(d.get("added_ts", "")),
            last_crawled_ts = str(d.get("last_crawled_ts", "")),
        )


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ────────────────────────────────────────────────────────────────────
# Load / save
# ────────────────────────────────────────────────────────────────────
def load_watchlist(path: Optional[Path] = None) -> list[WatchlistAuthor]:
    """Read the watchlist. Empty file or missing file → []."""
    p = path or WATCHLIST_PATH
    if not p.is_file():
        return []
    try:
        import yaml
    except ImportError:
        logger.warning("watchlist: PyYAML missing — load disabled")
        return []
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("watchlist: %s parse failed: %s", p, exc)
        return []
    authors = raw.get("authors") or []
    return [WatchlistAuthor.from_dict(a) for a in authors if isinstance(a, dict)]


def save_watchlist(authors: list[WatchlistAuthor],
                     *, path: Optional[Path] = None) -> None:
    """Write the watchlist atomically. Sorted by name for stable diffs."""
    p = path or WATCHLIST_PATH
    try:
        import yaml
    except ImportError:
        logger.warning("watchlist: PyYAML missing — save disabled")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    sorted_authors = sorted(authors, key=lambda a: a.name.lower())
    payload = {
        "schema_version": WATCHLIST_SCHEMA_VERSION,
        "authors":        [a.to_dict() for a in sorted_authors],
    }
    # Atomic: write to temp + rename
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    tmp.replace(p)


# ────────────────────────────────────────────────────────────────────
# Mutations — used by CLI / future UI
# ────────────────────────────────────────────────────────────────────
def add_author(name: str, *, rationale: str,
                 author_id: str = "",
                 path: Optional[Path] = None) -> WatchlistAuthor:
    """Add a new author to the watchlist. Idempotent on name match
    (case-insensitive)."""
    current = load_watchlist(path=path)
    lower = name.strip().lower()
    for a in current:
        if a.name.strip().lower() == lower:
            return a   # already present
    new = WatchlistAuthor(
        author_id       = author_id,
        name            = name.strip(),
        rationale       = rationale.strip(),
        added_ts        = _utc_iso(),
        last_crawled_ts = "",
    )
    current.append(new)
    save_watchlist(current, path=path)
    return new


def update_after_crawl(author_id: str, *,
                         path: Optional[Path] = None) -> None:
    """Mark the author as just crawled — used by the watchlist crawler
    so the next run can prioritize stale authors."""
    current = load_watchlist(path=path)
    updated = []
    for a in current:
        if a.author_id == author_id:
            updated.append(_dc.replace(a, last_crawled_ts=_utc_iso()))
        else:
            updated.append(a)
    save_watchlist(updated, path=path)


def resolve_author_id(name: str, *,
                        path: Optional[Path] = None) -> Optional[str]:
    """Resolve an unresolved (no author_id) entry by name via Semantic
    Scholar search. Persists the discovered id back to the watchlist.

    Returns the resolved id, or None on lookup failure (the entry
    stays unresolved; next call retries)."""
    try:
        from engine.agents.papers_curator.semantic_scholar import (
            search_author_by_name,
        )
        matches = search_author_by_name(name, limit=3)
    except Exception as exc:
        logger.warning("watchlist: SS resolve failed for %s: %s", name, exc)
        return None
    if not matches:
        return None
    # First match is the most likely (SS sorts by relevance)
    resolved_id = matches[0].author_id

    current = load_watchlist(path=path)
    updated = []
    for a in current:
        if a.name.strip().lower() == name.strip().lower():
            updated.append(_dc.replace(a, author_id=resolved_id))
        else:
            updated.append(a)
    save_watchlist(updated, path=path)
    return resolved_id
