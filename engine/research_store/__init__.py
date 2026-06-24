"""engine.research_store — typed event store for Claude↔project handoff.

The canonical record of "what happened in research". Append-only JSONL +
controlled subject vocabulary + typed emit/query helpers. Replaces the
previous ad-hoc scattering across capability_evidence/, memory/,
factory_ledger.jsonl, gate_runs.jsonl.

Public API:
    emit.*               — write events (see emit.py for the 8 helpers)
    registry.*           — manage controlled subject vocabulary
    store.filter_events  — typed queries
    schema.*             — types (EventType, SubjectType, Verdict, ResearchEvent)

See CLAUDE.md "Research Event Emission Doctrine" for when to emit.
"""
from engine.research_store import emit, registry, store, schema
from engine.research_store.exceptions import (
    ResearchStoreError,
    SubjectNotRegisteredError,
    ArtifactMissingError,
    InvalidEventError,
    DuplicateEventError,
)
from engine.research_store.schema import EventType, SubjectType, Verdict, ResearchEvent

__all__ = [
    "emit", "registry", "store", "schema",
    "EventType", "SubjectType", "Verdict", "ResearchEvent",
    "ResearchStoreError", "SubjectNotRegisteredError",
    "ArtifactMissingError", "InvalidEventError", "DuplicateEventError",
]
