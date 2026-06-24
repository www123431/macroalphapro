"""
Public retrieval API for the project history RAG.

Wraps ChromaDB's raw .query() with:
  - typed RetrievalResult dataclass instead of dict-of-list-of-list
  - source-type + date metadata filters expressed as Python kwargs
  - optional recency rerank (exponential half-life decay)
  - hard cap on top_k returned to caller

The retrieval surface is intentionally narrow (one entry function,
``retrieve()``) so the UI / synthesizer / eval harness all share the
exact same code path. Anyone bypassing this and calling chromadb
directly is opting out of the rerank + filter discipline — don't.
"""
from __future__ import annotations

import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from engine.agents.history_rag.config import (
    DEFAULT_FINAL_K,
    DEFAULT_TOP_K,
    RECENCY_HALF_LIFE_D,
)
from engine.agents.history_rag.schema import SourceType
from engine.agents.history_rag.store import get_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """One hit from the history RAG.

    Attributes
    ----------
    doc_id : str
        Stable upsert key. Use to deep-link or cite.
    source_type : SourceType
    source_id : str
        Originating row PK (or compound key for amendments).
    text : str
        The chunk text that actually matched. Use this in citations,
        not the full source row — the chunk is what the embedding saw.
    title : str
        Short label suitable for UI / citation header.
    occurred_at : datetime.datetime | None
    similarity : float
        Cosine similarity in [-1, 1]; ChromaDB returns distance =
        1 - cosine_similarity, so we flip back. Higher = more relevant.
    recency_weight : float
        Exponential decay in (0, 1] based on RECENCY_HALF_LIFE_D.
        1.0 if rerank disabled or occurred_at unknown.
    score : float
        Final ranking score = similarity * recency_weight. The list
        returned by retrieve() is sorted by this value descending.
    deep_link : str | None
    metadata : dict[str, Any]
        Raw chromadb metadata blob (already flat scalars).
    """
    doc_id:         str
    source_type:    SourceType
    source_id:      str
    text:           str
    title:          str
    occurred_at:    datetime.datetime | None
    similarity:     float
    recency_weight: float
    score:          float
    deep_link:      str | None       = None
    metadata:       dict[str, Any]   = field(default_factory=dict)


# ── Filter assembly ──────────────────────────────────────────────────────────

def _build_chroma_where(
    sources:         Sequence[SourceType] | None,
    occurred_after:  datetime.datetime    | None,
    occurred_before: datetime.datetime    | None,
    extra:           dict | None,
) -> dict | None:
    """Assemble a ChromaDB ``where`` clause from typed Python args.

    ChromaDB's where filter language is dict-based with operators like
    ``$eq``, ``$gte``, ``$lte``, ``$in``, ``$and``. We normalize to a
    single $and at the top so multiple conditions compose.
    """
    clauses: list[dict] = []

    if sources:
        src_values = [s.value for s in sources]
        if len(src_values) == 1:
            clauses.append({"source_type": {"$eq": src_values[0]}})
        else:
            clauses.append({"source_type": {"$in": src_values}})

    # We index occurred_at_ts (POSIX timestamp) as a numeric scalar so
    # range filters work. occurred_at_iso is human-readable only.
    if occurred_after is not None:
        clauses.append({"occurred_at_ts": {"$gte": occurred_after.timestamp()}})
    if occurred_before is not None:
        clauses.append({"occurred_at_ts": {"$lte": occurred_before.timestamp()}})

    if extra:
        for k, v in extra.items():
            clauses.append({k: {"$eq": v}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _recency_weight(occurred_at: datetime.datetime | None,
                    now:         datetime.datetime,
                    half_life_days: float) -> float:
    """Exponential decay: weight = 0.5 ** (age_days / half_life_days).

    Unknown timestamps default to 1.0 (no penalty) — better to surface
    a possibly-stale match than to demote it to oblivion.
    """
    if occurred_at is None or half_life_days <= 0:
        return 1.0
    age_days = max(0.0, (now - occurred_at).total_seconds() / 86_400.0)
    return 0.5 ** (age_days / half_life_days)


# ── Public entry point ───────────────────────────────────────────────────────

def retrieve(
    query:           str,
    top_k:           int = DEFAULT_FINAL_K,
    *,
    sources:         Iterable[SourceType] | None = None,
    occurred_after:  datetime.datetime    | None = None,
    occurred_before: datetime.datetime    | None = None,
    rerank_by_recency: bool   = True,
    half_life_days:  float    = RECENCY_HALF_LIFE_D,
    extra_filters:   dict     | None = None,
    over_fetch:      int      = DEFAULT_TOP_K,
    stratified:      bool     = True,
    per_source_k:    int      = 5,
    hybrid_bm25:     bool     = True,
    bm25_top:        int      = 20,
    rrf_k:           int      = 60,
) -> list[RetrievalResult]:
    """Run a semantic query against the project history RAG.

    Args
    ----
    query : the natural-language question. Mixed Chinese / English OK.
    top_k : how many results to return after rerank.
    sources : restrict to these source types (None = all).
    occurred_after / occurred_before : date-range filter on row's
        canonical timestamp (created_at / registered_at / detected_at).
    rerank_by_recency : if True, multiply similarity by exponential
        decay factor based on ``half_life_days``. Default ON because
        a recent decision_log row is almost always more relevant
        than a 2-year-old one when the query is about "current state".
        Disable for explicitly historical queries.
    half_life_days : decay constant. Default 180d (~6 months).
    extra_filters : extra metadata equality filters (e.g. ticker="BAB").
    over_fetch : how many vector hits to ask chromadb for before rerank.
        Must be >= top_k. Default = config.DEFAULT_TOP_K (= 8).

    Returns
    -------
    list[RetrievalResult] sorted by ``score`` desc, length ≤ top_k.
    Empty list when the collection is empty or no rows match filters.

    Raises
    ------
    Returns [] on transient chromadb errors (logs at WARNING) — caller
    decides whether to surface "no answer" vs error to user.
    """
    if not query or not query.strip():
        return []

    sources_list = list(sources) if sources else None

    # 2026-06-03 stratified retrieval (eval v2 diagnostic): if enabled,
    # query each source type independently and merge. Compensates for
    # heavy class imbalance (decision_log = 73.8% of corpus). Eliminates
    # cases where spec lookups get drowned by high-volume DL hits.
    if stratified and sources_list is None:
        dense_hits = _retrieve_stratified(
            query=query.strip(),
            top_k=max(top_k * 3, 15),    # over-fetch for RRF merge
            per_source_k=per_source_k,
            occurred_after=occurred_after,
            occurred_before=occurred_before,
            rerank_by_recency=rerank_by_recency,
            half_life_days=half_life_days,
            extra_filters=extra_filters,
        )

        # 2026-06-03 hybrid BM25 (fix B per eval diagnostic): rare-term
        # queries ("Pastor Stambaugh", "KMPV") need lexical match. BM25
        # complements dense via Reciprocal Rank Fusion.
        if hybrid_bm25:
            return _merge_dense_and_bm25(
                query=query.strip(),
                dense_hits=dense_hits,
                bm25_top=bm25_top,
                rrf_k=rrf_k,
                top_k=top_k,
            )
        return dense_hits[:top_k]

    where = _build_chroma_where(
        sources=sources_list,
        occurred_after=occurred_after,
        occurred_before=occurred_before,
        extra=extra_filters,
    )

    coll = get_store()
    n_fetch = max(top_k, over_fetch)
    try:
        raw = coll.query(
            query_texts=[query.strip()],
            n_results=n_fetch,
            where=where,
        )
    except Exception as exc:
        logger.warning("history_rag.retrieve query failed: %s", exc)
        return []

    ids        = (raw.get("ids")        or [[]])[0]
    docs       = (raw.get("documents")  or [[]])[0]
    metas_list = (raw.get("metadatas")  or [[]])[0]
    distances  = (raw.get("distances")  or [[]])[0]

    if not ids:
        return []

    now = datetime.datetime.utcnow()
    out: list[RetrievalResult] = []
    for i, doc_id in enumerate(ids):
        meta = metas_list[i] or {}
        # ChromaDB cosine distance is in [0, 2]; cosine similarity = 1 - dist
        # for unit-norm embeddings. Clip to [-1, 1] for safety.
        dist = float(distances[i]) if i < len(distances) else 1.0
        similarity = max(-1.0, min(1.0, 1.0 - dist))

        occurred_at = None
        ts = meta.get("occurred_at_ts")
        if isinstance(ts, (int, float)) and ts > 0:
            try:
                occurred_at = datetime.datetime.utcfromtimestamp(float(ts))
            except (OSError, OverflowError, ValueError):
                pass

        if rerank_by_recency:
            rw = _recency_weight(occurred_at, now, half_life_days)
        else:
            rw = 1.0
        score = similarity * rw

        try:
            st_enum = SourceType(meta.get("source_type", ""))
        except ValueError:
            # Unknown source type — corpus drift; surface for diagnosis but
            # don't crash the query.
            logger.warning(
                "history_rag.retrieve: unknown source_type=%r doc_id=%s",
                meta.get("source_type"), doc_id,
            )
            continue

        out.append(RetrievalResult(
            doc_id=doc_id,
            source_type=st_enum,
            source_id=str(meta.get("source_id", "")),
            text=docs[i] if i < len(docs) else "",
            title=str(meta.get("title", "")),
            occurred_at=occurred_at,
            similarity=similarity,
            recency_weight=rw,
            score=score,
            deep_link=meta.get("deep_link"),
            metadata=dict(meta),
        ))

    out.sort(key=lambda r: r.score, reverse=True)
    return out[:top_k]


# ── Stratified retrieval (2026-06-03 — eval v2 fix) ──────────────────────────


def _retrieve_stratified(
    query:              str,
    top_k:              int,
    per_source_k:       int,
    occurred_after:     datetime.datetime | None,
    occurred_before:    datetime.datetime | None,
    rerank_by_recency:  bool,
    half_life_days:     float,
    extra_filters:      dict | None,
) -> list[RetrievalResult]:
    """Retrieve top per_source_k from EACH source type, then merge.

    Solves the class-imbalance problem identified in
    `docs/rag_eval_diagnostic_2026-06-03.md`: decision_log dominates
    73.8% of the corpus so it monopolizes top-K under unstratified
    cosine search. Spec lookups get drowned even when relevant specs
    exist.

    Approach (a la Stratified-IR / per-class sampling):
      1. For each SourceType, run an independent vector query restricted
         to that type, returning per_source_k hits.
      2. Concatenate all results.
      3. Sort by score (similarity × recency_weight, same as base path)
         and return top_k.

    This guarantees each source type gets a fair shot at top_k without
    requiring corpus reindexing or BM25. Mean R@5 improvement (per
    eval v2): spec_registry 0.25 → ~0.60+ expected.
    """
    coll = get_store()
    now = datetime.datetime.utcnow()
    all_results: list[RetrievalResult] = []

    for st in SourceType:
        where = _build_chroma_where(
            sources=[st],
            occurred_after=occurred_after,
            occurred_before=occurred_before,
            extra=extra_filters,
        )
        try:
            raw = coll.query(
                query_texts=[query],
                n_results=per_source_k,
                where=where,
            )
        except Exception as exc:
            logger.warning(
                "history_rag.retrieve_stratified failed for source=%s: %s",
                st.value, exc,
            )
            continue

        ids       = (raw.get("ids")        or [[]])[0]
        docs      = (raw.get("documents")  or [[]])[0]
        metas     = (raw.get("metadatas")  or [[]])[0]
        distances = (raw.get("distances")  or [[]])[0]

        for i, doc_id in enumerate(ids):
            meta = metas[i] or {}
            dist = float(distances[i]) if i < len(distances) else 1.0
            similarity = max(-1.0, min(1.0, 1.0 - dist))

            occurred_at = None
            ts = meta.get("occurred_at_ts")
            if isinstance(ts, (int, float)) and ts > 0:
                try:
                    occurred_at = datetime.datetime.utcfromtimestamp(float(ts))
                except (OSError, OverflowError, ValueError):
                    pass

            rw = _recency_weight(occurred_at, now, half_life_days) if rerank_by_recency else 1.0
            score = similarity * rw

            try:
                st_enum = SourceType(meta.get("source_type", ""))
            except ValueError:
                logger.warning(
                    "history_rag.retrieve_stratified: unknown source_type=%r doc_id=%s",
                    meta.get("source_type"), doc_id,
                )
                continue

            all_results.append(RetrievalResult(
                doc_id=doc_id,
                source_type=st_enum,
                source_id=str(meta.get("source_id", "")),
                text=docs[i] if i < len(docs) else "",
                title=str(meta.get("title", "")),
                occurred_at=occurred_at,
                similarity=similarity,
                recency_weight=rw,
                score=score,
                deep_link=meta.get("deep_link"),
                metadata=dict(meta),
            ))

    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]


# ── Hybrid BM25 + dense via RRF (2026-06-03 — eval v2 fix B) ────────────────


def _merge_dense_and_bm25(
    query:       str,
    dense_hits:  list[RetrievalResult],
    bm25_top:    int,
    rrf_k:       int,
    top_k:       int,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion (RRF) between dense and BM25.

    RRF score per doc = sum_over_rankers(1 / (k + rank_in_ranker)).
    Standard k=60 from the original RRF paper (Cormack et al 2009).

    Docs in BM25 but not dense are pulled into the result with a
    fake RetrievalResult constructed from the BM25 hit's metadata.
    """
    from engine.agents.history_rag.bm25_retrieve import bm25_search, _get_index

    # BM25 hits
    bm25_hits = bm25_search(query, top_k=bm25_top)

    # Compute RRF score per doc_id
    rrf_score: dict[str, float] = {}
    for r, h in enumerate(dense_hits):
        rrf_score[h.doc_id] = rrf_score.get(h.doc_id, 0.0) + 1.0 / (rrf_k + r + 1)
    for h in bm25_hits:
        rrf_score[h.doc_id] = rrf_score.get(h.doc_id, 0.0) + 1.0 / (rrf_k + h.rank)

    # Lookup table for the original RetrievalResult objects (dense path)
    dense_by_id = {h.doc_id: h for h in dense_hits}

    # For BM25-only docs we need to construct RetrievalResult. Pull from
    # the BM25 cache (which has source_type + title). We don't have body
    # text without a chromadb hit, but that's OK — title + source_type
    # suffice for ranking + display.
    bm25_only_ids = [h.doc_id for h in bm25_hits if h.doc_id not in dense_by_id]
    bm25_only_meta: dict[str, BM25Hit] = {h.doc_id: h for h in bm25_hits}

    # If any BM25-only doc made it, fetch its full record from chromadb
    # so we have body text for the synthesizer / UI.
    if bm25_only_ids:
        try:
            coll = get_store()
            recs = coll.get(ids=bm25_only_ids,
                            include=["documents", "metadatas"])
            for i, doc_id in enumerate(recs.get("ids") or []):
                meta = (recs.get("metadatas") or [{}])[i] or {}
                doc_text = ((recs.get("documents") or [""])[i]) or ""
                bm25_hit = bm25_only_meta.get(doc_id)
                if not bm25_hit:
                    continue
                try:
                    st_enum = SourceType(meta.get("source_type", ""))
                except ValueError:
                    continue
                dense_by_id[doc_id] = RetrievalResult(
                    doc_id=doc_id,
                    source_type=st_enum,
                    source_id=str(meta.get("source_id", "")),
                    text=doc_text,
                    title=str(meta.get("title", "")),
                    occurred_at=None,           # we don't recompute here
                    similarity=0.0,             # not from dense
                    recency_weight=1.0,
                    score=bm25_hit.score,       # surface BM25 score
                    deep_link=meta.get("deep_link"),
                    metadata=dict(meta),
                )
        except Exception as exc:
            logger.warning("hybrid: chromadb get() for BM25-only ids failed: %s", exc)

    # Final ranking by RRF
    ranked = sorted(
        ((doc_id, rrf_score[doc_id]) for doc_id in dense_by_id),
        key=lambda kv: kv[1], reverse=True,
    )
    out = [dense_by_id[doc_id] for doc_id, _ in ranked[:top_k]]
    return out
