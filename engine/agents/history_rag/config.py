"""
History RAG — constants & configuration.

These values are project-wide invariants. Changing any of them is a schema
break (all indexed embeddings must be regenerated).
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Embedding model ───────────────────────────────────────────────────────────
# paraphrase-multilingual-mpnet-base-v2 is the production-grade choice for
# mixed Chinese / English content. ~1.1GB on first download; cached locally
# under ~/.cache/huggingface/hub/. Embedding dim = 768.
#
# DO NOT switch models without rebuilding the entire index — embedding
# vectors from different models live in different geometric spaces.
EMBED_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBED_DIM:        int = 768

# ── Persistence ───────────────────────────────────────────────────────────────
# ChromaDB persistent client writes to disk. Path is gitignored so the index
# does not bloat the repo (~10-50MB per 1000 docs depending on metadata).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PERSIST_DIR:     Path = _PROJECT_ROOT / ".streamlit" / "rag_store"
COLLECTION_NAME: str  = "project_history_v1"

# ── Chunking ──────────────────────────────────────────────────────────────────
# Documents longer than CHUNK_TARGET_CHARS get split into overlapping chunks
# at sentence boundaries. Overlap preserves cross-chunk context for queries
# that span paragraph boundaries.
CHUNK_TARGET_CHARS:  int = 800   # ~ 200 tokens for Chinese, 300 for English
CHUNK_OVERLAP_CHARS: int = 120

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K:       int   = 8     # vector hits before rerank
DEFAULT_FINAL_K:     int   = 5     # rows shown to user after rerank
RECENCY_HALF_LIFE_D: float = 180.0 # days; older docs get downweighted

# ── LLM synthesis ─────────────────────────────────────────────────────────────
# 0-LLM-in-evaluation invariant: LLM is in the *generation* loop only.
# Retrieval is deterministic; final scoring of relevance is deterministic.
# LLM may only summarize already-retrieved + cited evidence.
SYNTHESIS_MODEL:        str   = "gemini-2.5-flash"
SYNTHESIS_MAX_TOKENS:   int   = 1024
SYNTHESIS_TEMPERATURE:  float = 0.2  # low; we want grounded summary not creative writing
SYNTHESIS_DAILY_BUDGET: float = 0.05 # USD; ~50 calls/day at $0.001 each


def ensure_persist_dir() -> Path:
    """Create the persist directory if missing. Idempotent."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return PERSIST_DIR
