"""engine.research_store.schema — typed contract for the research event store.

Single source of truth for what counts as a research event. All emit() helpers
construct ResearchEvent instances and the store rejects anything that doesn't
conform. Schema-version-bumped on every breaking change; consumers MUST
handle the legacy versions they care about.

Doctrine (2026-06-02): events are immutable. To "correct" an event, emit a
new event with the corrected payload and parent_event_ids pointing to the
prior. NEVER mutate events.jsonl in place.
"""
from __future__ import annotations

import dataclasses as _dc
from enum import Enum
from typing import Any


SCHEMA_VERSION = 1


class EventType(str, Enum):
    """Canonical event taxonomy. New types require a schema bump + downstream
    consumer audit. 8 types intentionally covers ~90% of research operations;
    if a 9th is genuinely needed, add it deliberately, don't widen by reflex.

    F4 (2026-06-05): added candidate_pipeline_started. Was previously
    untracked — user picks a forward_vector and runs the pipeline, but no
    research event records the decision. With F4 the lifecycle is
    candidate_pipeline_started -> factor_verdict_filed, fully queryable.

    Phase 2.0 step 4c (2026-06-06): added papers_curator_synthesis_run.
    Each time Employee A's cross-source synthesis fires, record:
      - which papers / events were read (provenance)
      - what Sonnet proposed (claim count + which families)
      - what was DROPPED (LLM returned [] vs N; conflicts honestly listed)
      - generation metadata the writer drops from hypotheses.jsonl
        (cochrane_frame / novelty / expected_outcome_prior)
    Makes the synthesis behavior auditable + lets the chief_of_staff
    orchestrator (step 14, later) query "how many synthesis runs this
    week, how many candidates, how many persisted, how many graveyard-
    blocked" without re-running anything.

    Phase 2.0 step 9a (2026-06-06): added doctrine_signal_detected.
    Employee D (book monitor) fires this when a deterministic rule
    sees a pattern in events.jsonl worth doctrine-amending attention
    (e.g. ≥3 RED verdicts in same family within 30 days; deployed
    sleeve EWMA Sharpe falls below threshold; gate rejection rate
    spikes). Subscribers: Employee A's gatherer reads these into the
    next synthesis snapshot — first real coupling between the 4
    employees. Subject points to the thing being flagged (the
    representative factor / sleeve), with the cluster context in
    metrics.

    Phase 2.1a (2026-06-06): added forward_vector_created. Bridges
    the gap that surfaced when the principal corrected the mental
    model — hypothesis = brainstorm (LLM_SYNTHESIS), not paper-stated
    extraction. Fired when the principal approves a B verdict in
    /approvals; signals the corresponding hypothesis_id is now
    eligible for the forward_vector queue. generate_forward_vectors
    (Phase 2.1b) reads these events to gate which LLM_SYNTHESIS rows
    appear in /research/forward.

    Phase 2.0 step 13 (2026-06-06): added memory_amendment_proposed.
    Fired when the principal approves a B verdict whose verdict_type
    is DOCTRINE_AMENDMENT_NEEDED in /approvals. Distinct from
    memory_doctrine_locked because the memory file is NOT yet edited
    — only a draft amendment markdown is written for the principal
    to review + apply manually. The eventual memory_doctrine_locked
    will reference this event via parent_event_ids when the actual
    file is updated.

    Phase 2.0 step 8 (2026-06-06): added candidate_skipped_pre_compute.
    Fired by autopilot_live.run_top1 when pre-compute DA (step 6+7)
    returns worth_running=False on a top-1 candidate. Closes the
    lineage gap — without this event, a skipped candidate would
    disappear from the audit trail (the existing candidate_pipeline_
    started → factor_verdict_filed chain skips the case where the
    pipeline never STARTED). Carries spec_hash + attack_vector + DA
    reasoning so downstream queries ("which candidates did
    pre-compute kill last week and why?") work.

    Phase 2.0 step 14a (2026-06-06): added chief_of_staff_session_run.
    One event per weekly session that orchestrates D → A → B. The
    event aggregates the substeps' counts (D signals emitted, A
    candidates produced, B verdicts persisted) with parent_event_ids
    pointing to the substep emits. NO LLM in the orchestrator itself —
    it's a deterministic Python sequencer; each substep's LLM call
    happens in its own typed module (which keeps the chief_of_staff
    surface Pattern 5-safe, no multi-agent debate).
    """
    factor_verdict_filed         = "factor_verdict_filed"
    memory_doctrine_locked       = "memory_doctrine_locked"
    spec_amended                 = "spec_amended"
    deploy_changed               = "deploy_changed"
    decay_alert                  = "decay_alert"
    dq_breach                    = "dq_breach"
    council_critique             = "council_critique"
    capability_evidence_filed    = "capability_evidence_filed"
    candidate_pipeline_started   = "candidate_pipeline_started"
    papers_curator_synthesis_run = "papers_curator_synthesis_run"
    doctrine_signal_detected     = "doctrine_signal_detected"   # step 9a
    chief_of_staff_session_run   = "chief_of_staff_session_run" # step 14a
    forward_vector_created       = "forward_vector_created"     # Phase 2.1a
    candidate_skipped_pre_compute = "candidate_skipped_pre_compute"  # step 8
    memory_amendment_proposed    = "memory_amendment_proposed"  # step 13
    post_green_rigor_run         = "post_green_rigor_run"       # Phase 4.1 (2026-06-13)


class SubjectType(str, Enum):
    """What kind of thing the event is about. Used by queries to filter.

    Phase 2.0 step 4c (2026-06-06): added agent_run for events that are
    ABOUT an agent's execution rather than a research artifact. The
    subject_id is a stable agent identity (e.g. 'papers_curator') so
    queries can ask 'how often did Employee A fire' / 'what was its
    last 10 outcomes' without going through the existing factor/sleeve
    taxonomies (which don't fit — an A synthesis run isn't 'about' a
    single factor, it produces 0-N candidate factors).
    """
    factor          = "factor"
    sleeve          = "sleeve"
    spec            = "spec"
    memory_doctrine = "memory_doctrine"
    data_quality    = "data_quality"
    capacity        = "capacity"
    book            = "book"
    agent_run       = "agent_run"


class Verdict(str, Enum):
    """Verdict taxonomy. GREEN/MARGINAL/RED come from the strict gate;
    NEUTRAL means 'event has no verdict semantic' (e.g. memory amendment)."""
    GREEN    = "GREEN"
    MARGINAL = "MARGINAL"
    RED      = "RED"
    NEUTRAL  = "NEUTRAL"


@_dc.dataclass(frozen=True)
class ResearchEvent:
    """An immutable record of one significant research action.

    Identity:
        event_id        — UUID4 string, unique per event
        ts              — ISO-8601 UTC, e.g. '2026-06-02T14:23:00Z'
        session_id      — opaque session id (Claude session UUID or cron run id)
        actor           — who emitted ('claude-opus-4-7' / 'engine.daily_batch'
                          / 'user' — descriptive, not a security boundary)

    Subject:
        subject_type    — taxonomy of WHAT this is about
        subject_id      — registered subject identifier (must be in registry)
        family          — optional family label for grouping (e.g. 'pead',
                          'carry', 'position_weighting')

    Outcome:
        event_type      — what KIND of event happened
        verdict         — verdict if applicable; NEUTRAL otherwise
        metrics         — typed numeric metrics dict; consumer-specific shape
                          but keys should be self-descriptive
        summary         — 1-2 sentence human summary (max ~280 chars)

    Provenance:
        artifacts       — {role: path} map of files that ARE the event content.
                          All paths MUST exist on disk at emit time.
        parent_event_ids — tuple of event_ids this event depends on (DAG).
                          Used for lineage queries.
        git_sha         — current git HEAD at emit time
        tags            — free-form labels for ad-hoc query filtering

    Versioning:
        schema_version  — bumped on incompatible schema change. Consumers
                          should accept the versions they understand and
                          warn (not crash) on newer versions.
    """
    event_id:           str
    event_type:         EventType
    ts:                 str
    session_id:         str
    actor:              str
    subject_type:       SubjectType
    subject_id:         str
    verdict:            Verdict
    metrics:            dict[str, Any]
    artifacts:          dict[str, str]
    parent_event_ids:   tuple[str, ...]
    family:             str | None
    tags:               tuple[str, ...]
    summary:            str
    git_sha:            str
    schema_version:     int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize for jsonl storage. Enums → string values; tuples → lists."""
        d = _dc.asdict(self)
        # Enums serialize as their string value (StrEnum-style, works on 3.10)
        d["event_type"]   = self.event_type.value
        d["subject_type"] = self.subject_type.value
        d["verdict"]      = self.verdict.value
        d["parent_event_ids"] = list(self.parent_event_ids)
        d["tags"]             = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResearchEvent":
        """Deserialize from jsonl row. Enums reconstructed; lists → tuples."""
        return cls(
            event_id         = d["event_id"],
            event_type       = EventType(d["event_type"]),
            ts               = d["ts"],
            session_id       = d["session_id"],
            actor            = d["actor"],
            subject_type     = SubjectType(d["subject_type"]),
            subject_id       = d["subject_id"],
            verdict          = Verdict(d["verdict"]),
            metrics          = dict(d.get("metrics") or {}),
            artifacts        = dict(d.get("artifacts") or {}),
            parent_event_ids = tuple(d.get("parent_event_ids") or ()),
            family           = d.get("family"),
            tags             = tuple(d.get("tags") or ()),
            summary          = d.get("summary", ""),
            git_sha          = d.get("git_sha", ""),
            schema_version   = int(d.get("schema_version", 1)),
        )
