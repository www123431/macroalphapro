"""engine.agents.history_rag.bm25_retrieve — sparse BM25 retrieval.

Complements dense (vector) retrieval. Term-specific queries like
"Pastor Stambaugh" or "KMPV" have rare lexical tokens that pure
sentence-embedding models compress away. BM25's per-token IDF
weighting catches them.

Per the 2026-06-03 eval v2 diagnostic, spec_registry queries with
proper-noun terms remained at R@5 = 0.50 even after stratified
retrieval. This module + the RRF merge in retrieve.py raises that
toward the production bar.

Implementation:
  - One-time BM25 index build over the entire chromadb corpus
    (text + title concatenated). Cached at
    .streamlit/rag_store/bm25_index.pkl.
  - Build is invalidated when the chromadb collection count changes
    (cheap consistency check; not a true hash but suffices for
    drift detection).
  - Tokenization is whitespace + lowercase + simple regex. Mixed
    Chinese/English: Chinese chars become individual tokens (works
    because BM25 cares about co-occurrence not bigram order).
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from engine.agents.history_rag.config import PERSIST_DIR

logger = logging.getLogger(__name__)


_BM25_CACHE_PATH = PERSIST_DIR / "bm25_index.pkl"


@dataclass
class _BM25Cache:
    n_docs:        int                       # for cache invalidation
    doc_ids:       list[str]                  # parallel to bm25 corpus order
    titles:        list[str]
    source_types:  list[str]
    bm25:          object                     # rank_bm25.BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Whitespace + non-word boundary split. Chinese chars become
    1-char tokens. Lowercase. Strip empty."""
    if not text:
        return []
    # Split on whitespace and non-alphanumeric, preserving Chinese chars
    tokens = re.findall(r"[a-zA-Z0-9_]+|[一-鿿]", text.lower())
    return [t for t in tokens if t]


def _build_index() -> _BM25Cache:
    """Pull all docs from chromadb, build BM25Okapi index, cache to disk."""
    from rank_bm25 import BM25Okapi
    from engine.agents.history_rag.store import get_store

    coll = get_store()
    n = coll.count()

    # Pull all docs (chromadb get() returns all when no where filter)
    res = coll.get(limit=n, include=["documents", "metadatas"])
    ids        = res.get("ids") or []
    docs       = res.get("documents") or []
    metas_list = res.get("metadatas") or []

    titles       = [(m or {}).get("title", "") for m in metas_list]
    source_types = [(m or {}).get("source_type", "") for m in metas_list]

    # Concatenate title + text — title-only is too sparse; text-only
    # misses the bare filename specs (whose title IS the keyword).
    corpus_tokens = [
        _tokenize(((titles[i] or "") + " " + (docs[i] or "")))
        for i in range(len(ids))
    ]
    bm25 = BM25Okapi(corpus_tokens)

    cache = _BM25Cache(
        n_docs=n, doc_ids=list(ids),
        titles=titles, source_types=source_types, bm25=bm25,
    )

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    with _BM25_CACHE_PATH.open("wb") as fh:
        pickle.dump(cache, fh)
    logger.info("bm25 index built: %d docs", n)
    return cache


def _load_index() -> _BM25Cache | None:
    if not _BM25_CACHE_PATH.is_file():
        return None
    try:
        with _BM25_CACHE_PATH.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        logger.exception("bm25 cache load failed; will rebuild")
        return None


def _get_index() -> _BM25Cache:
    """Return cached BM25 index, rebuilding if corpus size changed."""
    from engine.agents.history_rag.store import get_store

    cache = _load_index()
    if cache is None:
        return _build_index()

    # Cheap consistency check: rebuild if doc count changed
    coll = get_store()
    current_n = coll.count()
    if current_n != cache.n_docs:
        logger.info("bm25 cache stale (%d vs %d); rebuilding", cache.n_docs, current_n)
        return _build_index()
    return cache


# ── Public API ─────────────────────────────────────────────────────


@dataclass
class BM25Hit:
    doc_id:       str
    title:        str
    source_type:  str
    score:        float
    rank:         int       # 1-based


def bm25_search(query: str, top_k: int = 20) -> list[BM25Hit]:
    """Run BM25 over the cached corpus, return top_k by score."""
    if not query or not query.strip():
        return []
    cache = _get_index()
    qtoks = _tokenize(query)
    if not qtoks:
        return []
    scores = cache.bm25.get_scores(qtoks)
    ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        BM25Hit(
            doc_id=cache.doc_ids[i],
            title=cache.titles[i],
            source_type=cache.source_types[i],
            score=float(s),
            rank=rank + 1,
        )
        for rank, (i, s) in enumerate(ranked)
        if s > 0.0   # drop zero-scored (no term overlap)
    ]
