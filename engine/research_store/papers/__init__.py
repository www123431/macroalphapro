"""engine.research_store.papers — canonical paper registry.

Why this exists separately from `engine.research_store.red_lessons`:

  - Lessons reference papers, but papers OUTLIVE individual lessons. A
    single paper (e.g. Frazzini-Pedersen 2014) can simultaneously be
    the motivation for a deployed GREEN sleeve (K1_BAB) AND be cited
    by a RED Lesson's critique chain AND be on the doctrine shelf for
    its methodology.
  - Status (GREEN / YELLOW / RED) is a property of the candidate, NOT
    the paper. The same paper does not change its semantic embedding
    when a candidate's status flips. Therefore: ONE paper registry,
    ONE ChromaDB collection, partitioned LOGICALLY by `shelves`
    multi-label metadata, not physically by status.

Public API:

    from engine.research_store.papers import (
        PaperRegistryEntry, Shelf, SHELF_DOCS, FulltextStatus,
        load_registry, save_entry, find_by_doi, latest_per_doi,
    )
"""
from engine.research_store.papers.shelves import (
    Shelf,
    SHELF_DOCS,
)
from engine.research_store.papers.schema import (
    PaperRegistryEntry,
    FulltextStatus,
    REGISTRY_SCHEMA_VERSION,
    IngestionReason,
    IngestionReasonSource,
    IntentCategory,
)
from engine.research_store.papers.store import (
    load_registry,
    save_entry,
    find_by_doi,
    latest_per_doi,
    REGISTRY_PATH,
)
from engine.research_store.papers.amend import (
    amend_entry,
)

__all__ = [
    "Shelf", "SHELF_DOCS",
    "PaperRegistryEntry", "FulltextStatus", "REGISTRY_SCHEMA_VERSION",
    "IngestionReason", "IngestionReasonSource", "IntentCategory",
    "load_registry", "save_entry", "find_by_doi", "latest_per_doi",
    "REGISTRY_PATH",
    "amend_entry",
]
