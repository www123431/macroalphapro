"""
Indexed-document schema for the history RAG store.

Every row added to ChromaDB gets a stable doc_id of the form
    {source_type}:{source_id}[#chunk_{i}]
so re-indexing the same row is idempotent (upsert by doc_id) and chunked
documents stay groupable for citation rendering.
"""
from __future__ import annotations

import datetime
import enum
from dataclasses import dataclass, field
from typing import Any


class SourceType(str, enum.Enum):
    """Canonical source-type identifiers used in doc_id and metadata.

    Adding a new source: add the enum value here, then implement an
    indexer in engine.agents.history_rag.index returning IndexedDoc
    objects with this source_type.
    """
    DECISION_LOG       = "decision_log"
    SPEC_REGISTRY      = "spec_registry"
    SPEC_AMENDMENT     = "spec_amendment"   # amendment_log entries within a spec
    PENDING_APPROVAL   = "pending_approval"
    AGENT_REFLECTION   = "agent_reflection"
    AUDIT_FINDING      = "audit_finding"
    # Tier 1 polish (2026-05-07): hard-coded self-description docs that
    # describe what each agentic capability / page can answer. Indexed
    # into the same vector store so meta-queries ("what can you do") +
    # navigational queries ("how do I X") retrieve a real, on-topic doc
    # instead of forcing LLM to synthesize nonsense from decision_log
    # content. NOT user data — content comes from self_help_docs.py.
    SYSTEM_HELP        = "system_help"


@dataclass
class IndexedDoc:
    """A single document ready for embedding + chromadb upsert.

    Attributes
    ----------
    doc_id : str
        Stable, unique identifier. Format:
            "{source_type}:{source_id}"           — single-chunk docs
            "{source_type}:{source_id}#chunk_{i}" — multi-chunk docs
    source_type : SourceType
        Origin table / synthesis category.
    source_id : str
        Primary-key (or compound key) of the originating row, stringified.
    text : str
        The text chunk that will be embedded.
    title : str
        Short human-readable label rendered in citations.
    occurred_at : datetime.datetime | None
        Wall-clock timestamp of the underlying event (e.g. decision_date,
        registered_at, detected_at). Used by recency reranker; None means
        "treat as old".
    metadata : dict[str, Any]
        Free-form extra fields (severity, ticker, regime, etc) used for
        UI filtering. Must be JSON-serializable; ChromaDB rejects nested
        objects, so flatten to scalars at construction time.
    deep_link : str | None
        Optional UI route the supervisor clicks to land on the source row,
        e.g. "pages/decisions.py?id=42".
    """
    doc_id:      str
    source_type: SourceType
    source_id:   str
    text:        str
    title:       str
    occurred_at: datetime.datetime | None = None
    metadata:    dict[str, Any]           = field(default_factory=dict)
    deep_link:   str | None               = None

    def chroma_metadata(self) -> dict[str, Any]:
        """Return the metadata blob to pass to ChromaDB.

        ChromaDB only accepts (str|int|float|bool) scalars in metadata.
        We embed the structured fields into a flat dict + add the
        source_type, occurred_at-iso, title, deep_link. The free-form
        ``metadata`` dict is filtered to scalars only; non-scalar values
        are dropped (caller's responsibility to flatten upstream).
        """
        out: dict[str, Any] = {
            "source_type": self.source_type.value,
            "source_id":   self.source_id,
            "title":       self.title,
        }
        if self.occurred_at is not None:
            out["occurred_at_iso"] = self.occurred_at.isoformat()
            out["occurred_at_ts"]  = self.occurred_at.timestamp()
        if self.deep_link:
            out["deep_link"] = self.deep_link
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
        return out
