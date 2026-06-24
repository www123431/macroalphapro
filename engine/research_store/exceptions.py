"""engine.research_store.exceptions — typed errors for emit-time validation.

All errors carry enough context for the user to fix without re-checking
docs. Friendly-but-strict: error messages include "did you mean" hints
and the exact remediation command.
"""
from __future__ import annotations


class ResearchStoreError(Exception):
    """Base for all research_store errors. Catch this to handle any
    emit-time validation failure uniformly."""


class SubjectNotRegisteredError(ResearchStoreError):
    """The subject_id passed to emit() is not in the registry. Either it's
    a typo, or it's a genuinely new subject that needs explicit registration."""

    def __init__(self, subject_id: str, suggestions: list[str] | None = None):
        self.subject_id = subject_id
        self.suggestions = suggestions or []
        lines = [f"subject_id {subject_id!r} not in registry."]
        if self.suggestions:
            lines.append("did you mean one of:")
            for s in self.suggestions:
                lines.append(f"  - {s!r}")
        lines.append("")
        lines.append("to add as new subject:")
        lines.append(f"  registry.register_subject({subject_id!r}, subject_type=..., family=...)")
        if self.suggestions:
            lines.append("to register as alias of an existing subject:")
            lines.append(f"  registry.register_alias(canonical={self.suggestions[0]!r}, alias={subject_id!r})")
        super().__init__("\n".join(lines))


class ArtifactMissingError(ResearchStoreError):
    """One or more artifact paths passed to emit() do not exist on disk.
    Artifacts MUST exist before emit; the event is a pointer, the file is
    the content. Emit AFTER you've written the artifacts."""

    def __init__(self, missing: dict[str, str]):
        self.missing = missing
        lines = [f"{len(missing)} artifact path(s) do not exist on disk:"]
        for role, path in missing.items():
            lines.append(f"  - {role}: {path}")
        lines.append("")
        lines.append("artifacts must exist BEFORE emit() — the event references "
                     "them as evidence, it does not create them.")
        super().__init__("\n".join(lines))


class InvalidEventError(ResearchStoreError):
    """The event payload violates a schema constraint (bad enum value,
    summary too long, etc.). Message details the violation."""


class DuplicateEventError(ResearchStoreError):
    """An event with this event_id already exists. Events are immutable —
    if you need to amend, emit a new event with parent_event_ids pointing
    to the prior."""

    def __init__(self, event_id: str):
        self.event_id = event_id
        super().__init__(
            f"event_id {event_id!r} already in store. Events are immutable. "
            f"To correct, emit a new event with parent_event_ids=({event_id!r},)."
        )
