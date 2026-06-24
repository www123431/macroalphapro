"""engine/data/cache_manager.py — Layer 1: versioned cache + provenance.

Wraps the data/cache_v2/ directory. Each entry:
  data/cache_v2/{token}/{hash}.parquet
  data/cache_v2/_meta/{token}/{hash}.json

The metadata file records:
  - fetched_ts:      ISO timestamp
  - source:          which fetcher in the chain served this data
  - source_tier:     paid | free | scraped
  - schema_version:  fetcher's declared schema version
  - source_query:    parameters used (start, end, kwargs)
  - row_count:       quick integrity check
  - wallclock_seconds: cost transparency

Cache key = hash(token + start + end + sorted(kw.items())).

Doctrine:
- Cache MISS → call orchestrator → fetch → write cache
- Cache HIT but stale (TTL exceeded) → re-fetch + overwrite
- Cache HIT with mismatched schema_version → invalidate + re-fetch
- Forced refresh: orchestrator passes force=True

Flexibility ↔ Rigor balance:
- FLEX: cache transparently accelerates repeated queries; supports any fetcher
- RIGOR: schema_version + provenance metadata catch silent schema drift;
   row_count check catches corrupted writes
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache_v2"
META_DIR = CACHE_DIR / "_meta"


def _cache_key(token: str, start: str, end: str, kw: dict) -> str:
    payload = json.dumps(
        {"token": token, "start": start, "end": end, "kw": dict(sorted(kw.items()))},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _data_path(token: str, key: str) -> Path:
    return CACHE_DIR / token / f"{key}.parquet"


def _meta_path(token: str, key: str) -> Path:
    return META_DIR / token / f"{key}.json"


def get(token: str, start: str, end: str,
         *, max_age_days: float | None = None,
         expected_schema_version: int | None = None,
         **kw) -> tuple[pd.DataFrame | None, dict | None]:
    """Read from cache. Returns (df, meta) or (None, None) on miss/stale.

    Args:
      token:                 inventory token
      start, end:            YYYY-MM-DD range
      max_age_days:          cache invalidated if older than this; None = no TTL
      expected_schema_version: cache invalidated if mismatched; None = ignore
      **kw:                  additional query params (e.g. universe)
    """
    key = _cache_key(token, start, end, kw)
    data_p = _data_path(token, key)
    meta_p = _meta_path(token, key)
    if not data_p.exists() or not meta_p.exists():
        return None, None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("cache meta unreadable for %s/%s: %s", token, key, exc)
        return None, None

    if max_age_days is not None:
        fetched = datetime.datetime.fromisoformat(
            meta.get("fetched_ts", "").rstrip("Z")
        )
        age_days = (datetime.datetime.utcnow() - fetched).total_seconds() / 86400
        if age_days > max_age_days:
            return None, None

    if (expected_schema_version is not None
        and meta.get("schema_version") != expected_schema_version):
        return None, None

    try:
        df = pd.read_parquet(data_p)
    except Exception as exc:
        logger.warning("cache data unreadable for %s/%s: %s", token, key, exc)
        return None, None
    return df, meta


def put(token: str, start: str, end: str, df: pd.DataFrame,
         *, source: str, source_tier: str,
         schema_version: int = 1, wallclock_seconds: float = 0.0,
         extra_meta: dict | None = None,
         **kw) -> str:
    """Write to cache with provenance metadata. Returns the cache key."""
    key = _cache_key(token, start, end, kw)
    data_p = _data_path(token, key)
    meta_p = _meta_path(token, key)
    data_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(data_p, index=False)
    meta = {
        "token":              token,
        "cache_key":          key,
        "fetched_ts":         datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source":             source,
        "source_tier":        source_tier,
        "schema_version":     schema_version,
        "source_query":       {"start": start, "end": end, "kw": dict(kw)},
        "row_count":          len(df),
        "wallclock_seconds":  round(wallclock_seconds, 3),
    }
    if extra_meta:
        meta["extra"] = extra_meta
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return key


def invalidate(token: str, *, start: str | None = None, end: str | None = None,
                **kw) -> int:
    """Invalidate cache entries. If start/end/kw provided, removes only the
    specific entry. If only token, removes all entries for that token.

    Returns number of entries removed."""
    if start and end:
        key = _cache_key(token, start, end, kw)
        removed = 0
        for p in (_data_path(token, key), _meta_path(token, key)):
            if p.exists():
                p.unlink()
                removed += 1
        return removed
    # remove all for token
    removed = 0
    for sub in (CACHE_DIR / token, META_DIR / token):
        if sub.exists():
            for p in sub.glob("*"):
                p.unlink()
                removed += 1
    return removed


def list_cached(token: str | None = None) -> list[dict]:
    """List metadata for all cached entries (optionally filter by token)."""
    if not META_DIR.exists():
        return []
    out = []
    pattern = f"{token}/*.json" if token else "**/*.json"
    for meta_p in META_DIR.glob(pattern):
        try:
            out.append(json.loads(meta_p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out
