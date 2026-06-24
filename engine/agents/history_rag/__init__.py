"""
Project History RAG (P2 deliverable, 2026-05-07).

Supervisor / interviewer-facing research agent. Natural-language query →
vector retrieval over project history (DecisionLog / SpecRegistry +
amendments / PendingApproval / AgentReflection / AuditFinding) → optional
LLM synthesis with strict citation back to source rows.

Design red-lines:
  - 0-LLM-in-evaluation invariant preserved. LLM only synthesizes; never
    scores, never decides. Retrieval-only mode is the default.
  - Local-first: ChromaDB persists to disk; embedding model is a local
    sentence-transformers checkpoint (paraphrase-multilingual-mpnet-base-v2).
  - Citation-mandatory: every synthesized answer cites source row IDs;
    supervisor can deep-link back to the originating page.

Public API:
  from engine.agents.history_rag import (
      get_store, build_index, retrieve, synthesize_answer,
  )

Module map:
  config.py     - constants (model name, persist dir, collection name)
  schema.py     - IndexedDoc dataclass + source-type enum
  store.py      - ChromaDB client + collection management
  index.py      - row-to-chunk indexers for 5 sources (P2.2)
  retrieve.py   - semantic search + filters + recency rerank (P2.3)
  synthesize.py - Gemini 2.5 Flash synthesizer with citation enforcement (P2.4)
"""
from __future__ import annotations

from engine.agents.history_rag.config import (
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    PERSIST_DIR,
)
from engine.agents.history_rag.schema import IndexedDoc, SourceType
from engine.agents.history_rag.store import (
    get_store,
    reset_store,
    collection_stats,
)
from engine.agents.history_rag.index import build_index
from engine.agents.history_rag.retrieve import retrieve, RetrievalResult
from engine.agents.history_rag.synthesize import (
    synthesize_answer,
    SynthesizedAnswer,
    Citation,
    get_synthesis_cost_status,
)

__all__ = [
    "COLLECTION_NAME",
    "EMBED_MODEL_NAME",
    "PERSIST_DIR",
    "IndexedDoc",
    "SourceType",
    "get_store",
    "reset_store",
    "collection_stats",
    "build_index",
    "retrieve",
    "RetrievalResult",
    "synthesize_answer",
    "SynthesizedAnswer",
    "Citation",
    "get_synthesis_cost_status",
]
