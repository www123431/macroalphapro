"""
ChromaDB persistent client + collection manager + embedding wrapper.

Single source of truth for everything that touches the on-disk vector
store. Other modules in engine.agents.history_rag import ``get_store``
to obtain the ready-to-use Collection object — they never instantiate
chromadb.PersistentClient directly. This guarantees:

  - one process holds at most one client (chromadb's SQLite-backed lock)
  - the embedding function is consistently applied
  - the persist directory is initialized lazily

Resource model
--------------
The sentence-transformers model is loaded once on first call, then kept
warm in module state. ~1GB RAM hit; acceptable for a research console.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Module-level singletons — populated lazily.
_client_lock     = threading.Lock()
_client          = None  # chromadb.PersistentClient
_collection      = None  # chromadb.api.models.Collection.Collection
_embed_fn        = None  # SentenceTransformerEmbeddingFunction


def _build_embedder():
    """Lazily build the chromadb-compatible embedder backed by sentence-transformers.

    Returns a chromadb.utils.embedding_functions object so chroma calls
    .__call__(texts) directly during upsert / query. Loading the model
    triggers a ~1.1GB HuggingFace download on first run.
    """
    from chromadb.utils import embedding_functions  # local import: chromadb optional at module-load time
    from engine.agents.history_rag.config import EMBED_MODEL_NAME

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME,
        # device left at default ('cpu' for our build); GPU path untested.
    )


def _build_client():
    """Build (or reopen) the on-disk persistent client."""
    import chromadb
    from chromadb.config import Settings
    from engine.agents.history_rag.config import ensure_persist_dir

    persist_dir = ensure_persist_dir()
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(
            anonymized_telemetry=False,   # do not phone home
            allow_reset=True,             # for tests / reset_store()
        ),
    )


def get_store():
    """Return the project_history_v1 Collection, building it on first call.

    Idempotent + thread-safe. Subsequent calls reuse the cached collection
    handle for the lifetime of the Python process.
    """
    global _client, _collection, _embed_fn
    if _collection is not None:
        return _collection
    with _client_lock:
        if _collection is not None:
            return _collection
        from engine.agents.history_rag.config import COLLECTION_NAME
        _client     = _build_client()
        _embed_fn   = _build_embedder()
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embed_fn,
            metadata={"hnsw:space": "cosine"},  # cosine sim is the right default for sentence embeddings
        )
        logger.info(
            "history_rag store opened: collection=%s n_docs=%d",
            COLLECTION_NAME, _collection.count(),
        )
        return _collection


def reset_store() -> None:
    """Drop the collection + recreate empty. Use only in tests / re-index."""
    global _client, _collection, _embed_fn
    from engine.agents.history_rag.config import COLLECTION_NAME

    with _client_lock:
        if _client is None:
            _client = _build_client()
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        if _embed_fn is None:
            _embed_fn = _build_embedder()
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("history_rag store reset: collection=%s", COLLECTION_NAME)


def collection_stats() -> dict[str, Any]:
    """Return basic counters for the active collection.

    Cheap call (chromadb keeps the count cached). Safe to call from UI
    on every render.
    """
    coll = get_store()
    out: dict[str, Any] = {
        "n_total": coll.count(),
    }
    # Cheap source-type breakdown when corpus is small enough.
    if out["n_total"] <= 5000:
        try:
            page = coll.get(limit=out["n_total"], include=["metadatas"])
            metas = page.get("metadatas") or []
            from collections import Counter
            c = Counter(
                (m or {}).get("source_type", "unknown") for m in metas
            )
            out["by_source"] = dict(c)
        except Exception as exc:
            logger.warning("collection_stats by_source failed: %s", exc)
            out["by_source"] = {}
    else:
        out["by_source"] = {"_skipped": "corpus too large for cheap counting"}
    return out
