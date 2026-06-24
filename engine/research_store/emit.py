"""engine.research_store.emit — high-level emit helpers (the public API).

This is the ONLY surface Claude / cron jobs / users should use to record
research events. Direct writes to events.jsonl are forbidden by doctrine
and bypass validation.

Each helper:
  1. Validates inputs (subject registered, artifacts on disk)
  2. Constructs a typed ResearchEvent
  3. Appends to store atomically

Returns the event_id so the caller can chain (parent_event_ids in a
later emit).
"""
from __future__ import annotations

import os
import uuid
import datetime as _dt
import logging
from pathlib import Path
from typing import Optional

from engine.research_store import registry, store
from engine.research_store.exceptions import (
    ArtifactMissingError, InvalidEventError,
)
from engine.research_store.manifest import current_git_sha
from engine.research_store.schema import EventType, SubjectType, Verdict, ResearchEvent

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Sentinel actor used when caller doesn't override. Override per emit if
# you're a cron job ('engine.daily_batch') or user ('user') etc.
DEFAULT_ACTOR = "claude-opus-4-7"

_SUMMARY_MAX_CHARS = 400


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_artifacts(artifacts: dict[str, str]) -> None:
    """Every artifact path must exist on disk. Paths are interpreted
    relative to the repo root unless absolute."""
    missing: dict[str, str] = {}
    for role, path in (artifacts or {}).items():
        if not path:
            missing[role] = "(empty path)"
            continue
        p = Path(path)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        if not p.exists():
            missing[role] = str(path)
    if missing:
        raise ArtifactMissingError(missing)


def _validate_summary(summary: str) -> None:
    if not summary or not summary.strip():
        raise InvalidEventError("summary is required (1-2 sentences)")
    if len(summary) > _SUMMARY_MAX_CHARS:
        raise InvalidEventError(
            f"summary too long ({len(summary)} > {_SUMMARY_MAX_CHARS} chars). "
            f"keep it 1-2 sentences; detail belongs in the evidence doc."
        )


def _read_active_session() -> Optional[dict]:
    """Read the active session pointer (engine.sessions.store.get_active).
    Returns None if no active session or sessions module unavailable.
    Imported lazily to avoid circular import at module load."""
    try:
        from engine.sessions import store as session_store
        return session_store.get_active()
    except Exception:
        return None


def _emit(
    event_type: EventType,
    subject_id: str,
    verdict: Verdict,
    metrics: dict,
    artifacts: dict[str, str],
    summary: str,
    parent_event_ids: tuple = (),
    family: Optional[str] = None,
    tags: tuple = (),
    session_id: Optional[str] = None,
    actor: str = DEFAULT_ACTOR,
) -> str:
    """Inner constructor. The public helpers (factor_verdict, etc.) call
    this with their event_type fixed.

    Active-session integration (P2 2026-06-02): if no session_id passed,
    read engine.sessions.store.get_active() and auto-tag with the active
    session_id + session_type. Claude / cron callers don't need to know
    about sessions — the doctrine takes care of itself.
    """
    subj = registry.require(subject_id)        # raises with helpful suggestion
    _validate_artifacts(artifacts)
    _validate_summary(summary)

    # Resolve session context — explicit param wins; else read singleton.
    effective_session_id = session_id
    extra_tags: list[str] = []
    if not effective_session_id:
        active = _read_active_session()
        if active:
            effective_session_id = active.get("session_id", "unknown")
            stype = active.get("session_type")
            if stype:
                extra_tags.append(f"session:{effective_session_id}")
                extra_tags.append(f"session_type:{stype}")
    if not effective_session_id:
        effective_session_id = os.environ.get("CLAUDE_SESSION_ID", "unknown")

    final_tags = tuple(tuple(tags or ()) + tuple(extra_tags))

    event = ResearchEvent(
        event_id         = str(uuid.uuid4()),
        event_type       = event_type,
        ts               = _utc_iso(),
        session_id       = effective_session_id,
        actor            = actor,
        subject_type     = SubjectType(subj.subject_type),
        subject_id       = subj.subject_id,    # canonical, not the alias
        verdict          = verdict,
        metrics          = dict(metrics or {}),
        artifacts        = dict(artifacts or {}),
        parent_event_ids = tuple(parent_event_ids or ()),
        family           = family if family is not None else subj.family,
        tags             = final_tags,
        summary          = summary.strip(),
        git_sha          = current_git_sha(),
    )
    store.append(event)
    logger.info("emitted %s for subject=%s verdict=%s event_id=%s",
                event_type.value, subj.subject_id, verdict.value, event.event_id)

    # 2026-06-04 AI-native Step 1: publish to the in-process EventBus so
    # subscribers (audit_verifier, future personas) react synchronously.
    # Fire-and-log; bus failure must NOT corrupt the jsonl write above.
    _publish_to_bus(event)

    return event.event_id


def _publish_to_bus(event: ResearchEvent) -> None:
    """Mirror a ResearchEvent onto the cross-agent EventBus as an
    AgentEvent payload. Failure logs and returns; never raises into the
    emit caller because the jsonl write is the canonical record."""
    try:
        from engine.agents.event_bus import get_event_bus
        bus = get_event_bus()
        bus.publish(
            event_type   = event.event_type.value,
            payload      = {
                "research_event_id": event.event_id,
                "subject_id":        event.subject_id,
                "subject_type":      event.subject_type.value,
                "verdict":           event.verdict.value,
                "family":            event.family,
                "session_id":        event.session_id,
                "ts":                event.ts,
                "summary":           event.summary,
                # Lineage hooks — subscribers need these without a store reload
                "parent_event_ids":  list(event.parent_event_ids),
                "artifacts":         dict(event.artifacts),
                "metrics":           dict(event.metrics),
            },
            source_agent = event.actor,
        )
    except Exception as exc:
        logger.warning("_publish_to_bus failed for %s: %s",
                       event.event_id, exc, exc_info=True)


# ── Public helpers (8 canonical event types) ────────────────────────


def factor_verdict(
    subject_id: str,
    verdict: str,                          # "GREEN" / "MARGINAL" / "RED"
    metrics: dict,
    artifacts: dict[str, str],
    summary: str,
    parent_event_ids: tuple = (),
    family: Optional[str] = None,
    tags: tuple = (),
    actor: str = DEFAULT_ACTOR,
) -> str:
    """A factor study completes with a strict-gate verdict.

    `artifacts` should include at minimum 'evidence_doc' (path to the
    capability_evidence markdown). 'data_dir' (path to the run outputs)
    is strongly encouraged. Both must exist on disk before this call.
    """
    return _emit(
        EventType.factor_verdict_filed,
        subject_id=subject_id,
        verdict=Verdict(verdict),
        metrics=metrics, artifacts=artifacts, summary=summary,
        parent_event_ids=parent_event_ids, family=family, tags=tags, actor=actor,
    )


def candidate_pipeline_started(
    subject_id: str,
    spec_hash: str,
    source_hypothesis_id: str,
    family: Optional[str] = None,
    metrics: Optional[dict] = None,
    artifacts: Optional[dict[str, str]] = None,
    parent_event_ids: tuple = (),
    tags: tuple = (),
    actor: str = DEFAULT_ACTOR,
) -> str:
    """F4 (2026-06-05): a candidate pipeline run has been triggered.

    Emitted at /api/pipeline/prepare-from-fv entry — BEFORE the
    composer build + stream begin. Closes the lineage gap from a
    forward_vector decision to its eventual factor_verdict_filed
    event.

    Pre-F4, "show me every forward_vector that's been tested" was an
    unanswerable query — you had to grep gate_runs.jsonl for matching
    proposal_name strings and hope the naming was stable. With F4,
    one filter_events call returns the typed history.

    subject_id should be the canonical factor identity (proposal_name
    or fv-derived) so the eventual factor_verdict_filed event links
    by parent_event_ids. spec_hash and source_hypothesis_id go into
    metrics for traceability.
    """
    m = dict(metrics or {})
    m["spec_hash"] = spec_hash
    m["source_hypothesis_id"] = source_hypothesis_id
    return _emit(
        EventType.candidate_pipeline_started,
        subject_id=subject_id,
        verdict=Verdict.NEUTRAL,    # decision, not a verdict
        metrics=m,
        artifacts=artifacts or {},
        summary=(
            f"Candidate pipeline started for {subject_id} "
            f"(spec_hash={spec_hash[:8]}, hyp_id={source_hypothesis_id[:8]})"
        ),
        parent_event_ids=parent_event_ids, family=family,
        tags=tags, actor=actor,
    )


def memory_locked(
    subject_id: str,
    artifacts: dict[str, str],
    summary: str,
    parent_event_ids: tuple = (),
    family: Optional[str] = None,
    tags: tuple = (),
    actor: str = DEFAULT_ACTOR,
) -> str:
    """A memory file (doctrine) was written or amended.

    `artifacts['memory_doc']` should point to the memory/*.md file.
    """
    return _emit(
        EventType.memory_doctrine_locked,
        subject_id=subject_id,
        verdict=Verdict.NEUTRAL,
        metrics={}, artifacts=artifacts, summary=summary,
        parent_event_ids=parent_event_ids, family=family, tags=tags, actor=actor,
    )


def spec_amended(
    subject_id: str,
    artifacts: dict[str, str],
    summary: str,
    metrics: Optional[dict] = None,
    parent_event_ids: tuple = (),
    family: Optional[str] = None,
    tags: tuple = (),
    actor: str = DEFAULT_ACTOR,
) -> str:
    """A spec document (docs/spec_*.md) was amended.

    `artifacts['spec_doc']` should point to the new revision. If a hash
    is meaningful, include it in metrics as `metrics['spec_hash']`.
    """
    return _emit(
        EventType.spec_amended,
        subject_id=subject_id,
        verdict=Verdict.NEUTRAL,
        metrics=metrics or {}, artifacts=artifacts, summary=summary,
        parent_event_ids=parent_event_ids, family=family, tags=tags, actor=actor,
    )


def deploy_changed(
    subject_id: str,
    metrics: dict,
    artifacts: dict[str, str],
    summary: str,
    parent_event_ids: tuple = (),
    tags: tuple = (),
    actor: str = DEFAULT_ACTOR,
) -> str:
    """The active deployment config changed (sleeve added / dropped /
    weight changed). subject_id should be the sleeve / book affected."""
    return _emit(
        EventType.deploy_changed,
        subject_id=subject_id,
        verdict=Verdict.NEUTRAL,
        metrics=metrics, artifacts=artifacts, summary=summary,
        parent_event_ids=parent_event_ids, tags=tags, actor=actor,
    )


def decay_alert(
    subject_id: str,
    verdict: str,                          # "RED" (HARD) / "MARGINAL" (SOFT) / "NEUTRAL" (INFO)
    metrics: dict,
    artifacts: dict[str, str],
    summary: str,
    parent_event_ids: tuple = (),
    tags: tuple = (),
    actor: str = "engine.decay_sentinel",
) -> str:
    """A decay sentinel fired against a deployed sleeve.
    subject_id should be the sleeve."""
    return _emit(
        EventType.decay_alert,
        subject_id=subject_id,
        verdict=Verdict(verdict),
        metrics=metrics, artifacts=artifacts, summary=summary,
        parent_event_ids=parent_event_ids, tags=tags, actor=actor,
    )


def dq_breach(
    subject_id: str,
    verdict: str,                          # "RED" (HARD_HALT) / "MARGINAL" (SOFT_WARN)
    metrics: dict,
    artifacts: dict[str, str],
    summary: str,
    parent_event_ids: tuple = (),
    tags: tuple = (),
    actor: str = "engine.dq_inspector",
) -> str:
    """A data-quality breach. subject_id is the data feed / panel affected."""
    return _emit(
        EventType.dq_breach,
        subject_id=subject_id,
        verdict=Verdict(verdict),
        metrics=metrics, artifacts=artifacts, summary=summary,
        parent_event_ids=parent_event_ids, tags=tags, actor=actor,
    )


def council_critique(
    subject_id: str,
    verdict: str,                          # mirrors council consensus
    metrics: dict,
    artifacts: dict[str, str],
    summary: str,
    parent_event_ids: tuple = (),
    family: Optional[str] = None,
    tags: tuple = (),
    actor: str = "engine.council",
) -> str:
    """A council critique run produced a verdict on a candidate."""
    return _emit(
        EventType.council_critique,
        subject_id=subject_id,
        verdict=Verdict(verdict),
        metrics=metrics, artifacts=artifacts, summary=summary,
        parent_event_ids=parent_event_ids, family=family, tags=tags, actor=actor,
    )


def candidate_skipped_pre_compute(
    *,
    subject_id:           str,
    spec_hash:             str,
    source_hypothesis_id:  str,
    attack_vector:         str,
    reasoning:             str,
    confidence:            float,
    family:                Optional[str] = None,
    parent_event_ids:      tuple = (),
    actor:                 str = "engine.agents.autopilot_pre_compute_da",
) -> str:
    """Phase 2.0 step 8 (2026-06-06): autopilot pre-compute DA killed
    this candidate before strict-gate spend.

    Subject is the auto_<hash> factor subject autopilot_live registers
    via _ensure_subject. The skip event closes the lineage gap so
    queries like "which candidates did pre-compute DA kill last week
    and why?" work — without this, the candidate never appeared in
    events.jsonl because candidate_pipeline_started is emitted AFTER
    compose succeeds, and factor_verdict_filed is emitted AFTER
    strict-gate math. Skipped candidates would have been
    indistinguishable from "never picked" without this event type.

    Verdict: NEUTRAL (skip is neither GREEN nor RED — it's a "did
    not test, here's why").
    """
    metrics = {
        "spec_hash":            spec_hash,
        "source_hypothesis_id": source_hypothesis_id,
        "attack_vector":        attack_vector,
        "confidence":           float(confidence),
        "reasoning":            reasoning[:600],
    }

    summary = (
        f"Pre-compute DA skipped candidate {source_hypothesis_id[:8]} "
        f"({family or '?'}): {attack_vector[:120]}"
    )

    return _emit(
        EventType.candidate_skipped_pre_compute,
        subject_id        = subject_id,
        verdict           = Verdict.NEUTRAL,
        metrics           = metrics,
        artifacts         = {},
        summary           = summary,
        parent_event_ids  = parent_event_ids,
        family            = family,
        tags              = ("pre_compute_da", "skipped"),
        actor             = actor,
    )


def post_green_rigor_run(
    *,
    subject_id:            str,
    rigor_id:              str,
    verdict_event_id:      Optional[str],
    original_verdict:      str,           # "GREEN" / "MARGINAL"
    oos_status:            str,           # SURVIVED / DEGRADED / DEAD / SKIPPED
    spanning_status:       str,           # SPANNING_PASSED / INDETERMINATE / SUBSUMED / SKIPPED
    borrow_status:         Optional[str], # SURVIVED / MARGINAL / DEAD / SKIPPED
    flags:                 list,
    oos_nw_t:              Optional[float] = None,
    spanning_alpha_t:      Optional[float] = None,
    borrow_adj_nw_t:       Optional[float] = None,
    rigor_ledger_path:     Optional[str] = None,
    family:                Optional[str] = None,
    parent_event_ids:      tuple = (),
    actor:                 str = "engine.research.post_green_rigor",
) -> str:
    """Phase 4.1 (2026-06-13): post-GREEN rigor pipeline finished a 3-check
    pass on a verdict. Records:
      - which checks ran (post-pub OOS / FF5+MOM spanning / borrow cost)
      - what each returned
      - critical flags (DEAD_POST_PUB, SUBSUMED_BY_FF5_MOM,
        DEAD_UNDER_BORROW_COST, etc)

    Why event-level (not just ledger jsonl): downstream consumers
    (/approvals UI, daily digest, belief layer second-pass) need a typed
    queryable signal; ledger scraping doesn't compose with the rest of
    research_store's filter_events API. parent_event_ids links to the
    originating verdict event so lineage queries work:
        "show me every GREEN whose post-pub OOS DIED" =
        filter_events(type=post_green_rigor_run, tag=DEAD_POST_PUB)

    Verdict semantics:
      RED      = at least one CRITICAL flag (DEAD_POST_PUB / DEAD_UNDER_BORROW_COST / SUBSUMED)
      MARGINAL = at least one non-critical concern (DEGRADED / MARGINAL_UNDER_BORROW_COST)
      GREEN    = all checks SURVIVED / SPANNING_PASSED / SURVIVED_BORROW
      NEUTRAL  = no checks were able to run (all SKIPPED)
    """
    critical_flags = {"DEAD_POST_PUB", "SUBSUMED_BY_FF5_MOM", "DEAD_UNDER_BORROW_COST"}
    concern_flags  = {"DEGRADED_POST_PUB", "MARGINAL_UNDER_BORROW_COST"}
    if any(f in critical_flags for f in (flags or [])):
        verdict = Verdict.RED
    elif any(f in concern_flags for f in (flags or [])):
        verdict = Verdict.MARGINAL
    elif oos_status == "SURVIVED" or spanning_status == "SPANNING_PASSED" or borrow_status == "SURVIVED":
        verdict = Verdict.GREEN
    else:
        verdict = Verdict.NEUTRAL

    metrics = {
        "rigor_id":          rigor_id,
        "original_verdict":  original_verdict,
        "oos_status":        oos_status,
        "oos_nw_t":          oos_nw_t,
        "spanning_status":   spanning_status,
        "spanning_alpha_t":  spanning_alpha_t,
        "borrow_status":     borrow_status,
        "borrow_adj_nw_t":   borrow_adj_nw_t,
        "flags":             list(flags or []),
    }

    artifacts = {}
    if rigor_ledger_path:
        artifacts["rigor_ledger"] = rigor_ledger_path

    summary = (
        f"rigor on {original_verdict} verdict: OOS={oos_status} "
        f"spanning={spanning_status} borrow={borrow_status or 'n/a'}"
        + (f" [flags: {','.join(flags)}]" if flags else "")
    )[:400]

    return _emit(
        EventType.post_green_rigor_run,
        subject_id        = subject_id,
        verdict           = verdict,
        metrics           = metrics,
        artifacts         = artifacts,
        summary           = summary,
        parent_event_ids  = tuple(parent_event_ids or ((verdict_event_id,) if verdict_event_id else ())),
        family            = family,
        tags              = tuple(["post_green_rigor"] + list(flags or [])),
        actor             = actor,
    )


def papers_curator_synthesis_run(
    *,
    n_candidates:        int,
    n_written:           int,
    snapshot:            dict,
    candidates:          list[dict],
    errors:              list[str] = (),
    dry_run:             bool = False,
    doctrine_snippet_ids: list[str] = (),
    parent_event_ids:    tuple = (),
    actor:               str = "engine.agents.papers_curator.synthesis_runner",
) -> str:
    """Phase 2.0 step 4c (2026-06-06): one event per synthesis run.

    Records what Sonnet 4.6 was shown + what it proposed + what
    was DROPPED. The hypotheses.jsonl writer stores the persisted
    rows; this event stores the generation-time metadata that the
    writer drops (cochrane_frame / novelty / expected_prior /
    graveyard_conflicts / doctrine_conflicts) + the snapshot the
    LLM read + the cost/latency facts.

    Why both stores: hypotheses.jsonl is the testable artifact (B
    will read it); events.jsonl is the run history (chief_of_staff
    + UI will read it). They have different consumers + different
    retention semantics.

    subject_id is hard-coded 'papers_curator' (subject_type=agent_run)
    so queries like "last 30 days of Employee A activity" return
    every synthesis run regardless of what families were proposed.

    Verdict:
      n_candidates == 0           → NEUTRAL (honest-empty path)
      n_written  > 0 errors empty → GREEN
      errors non-empty            → MARGINAL  (partial success)

    No artifacts — synthesis is pure-LLM, no doc files to reference.
    The candidates list is embedded in `metrics["candidates_summary"]`
    so the event is self-contained.
    """
    if errors:
        verdict = Verdict.MARGINAL
    elif n_candidates == 0:
        verdict = Verdict.NEUTRAL
    else:
        verdict = Verdict.GREEN

    # Lean down candidate dicts for the event — keep what audit/UI needs,
    # drop redundant long fields (claim already short, methodology lives
    # in hypothesis row).
    candidates_summary = [{
        "claim":                  (c.get("claim") or "")[:200],
        "mechanism_family":       c.get("mechanism_family"),
        "mechanism_subtype":      c.get("mechanism_subtype"),
        "predicted_direction":    c.get("predicted_direction"),
        "predicted_magnitude":    c.get("predicted_magnitude"),
        "cochrane_frame":         c.get("cochrane_frame"),
        "novelty_vs_known":       c.get("novelty_vs_known"),
        "expected_outcome_prior": c.get("expected_outcome_prior"),
        "addresses_decay_in":     c.get("addresses_decay_in"),
        "n_papers":               len(c.get("synthesizes_paper_ids") or []),
        "n_events":               len(c.get("synthesizes_event_ids") or []),
        "graveyard_conflicts":    list(c.get("graveyard_conflicts") or []),
        "doctrine_conflicts":     list(c.get("doctrine_conflicts") or []),
        # Phase 2.2b: citation quality (None until verifier wires up;
        # consumers should treat missing as "not verified this run")
        "citation_quality":       c.get("citation_quality"),
    } for c in (candidates or [])]

    metrics = {
        "n_candidates":            n_candidates,
        "n_written":                n_written,
        "n_dropped_by_writer":      max(0, n_candidates - n_written),
        "dry_run":                  bool(dry_run),
        "snapshot_papers":          int(snapshot.get("recent_summaries", 0)),
        "snapshot_sleeves":         int(snapshot.get("deployed_sleeves", 0)),
        "snapshot_events":          int(snapshot.get("recent_events", 0)),
        "snapshot_doctrine":        int(snapshot.get("doctrine_snippets", 0)),
        "snapshot_ts":              snapshot.get("snapshot_ts", ""),
        # Phase 2.0 + Layer 4 attribution (2026-06-07):
        # the exact memory_file_ids A retrieved + saw during this synthesis.
        # Lets 6-month rollups answer "which doctrine entries are
        # associated with GREEN-verdict candidates" — the foundation
        # for doctrine retrieval reweighting (piece 3c).
        "doctrine_snippet_ids":     list(doctrine_snippet_ids or ()),
        "candidates_summary":       candidates_summary,
        "errors_count":             len(errors or ()),
    }

    if errors:
        # Keep top-3 error strings in metrics for audit forensics
        metrics["errors_sample"] = list(errors[:3])

    summary = (
        f"papers_curator synthesis: {n_candidates} candidate(s), "
        f"{n_written} written"
        + (f" [dry-run]" if dry_run else "")
        + (f", {len(errors)} error(s)" if errors else "")
    )

    return _emit(
        EventType.papers_curator_synthesis_run,
        subject_id        = "papers_curator",
        verdict           = verdict,
        metrics           = metrics,
        artifacts         = {},
        summary           = summary,
        parent_event_ids  = parent_event_ids,
        family            = None,
        tags              = ("synthesis",) + (("dry_run",) if dry_run else ()),
        actor             = actor,
    )


def memory_amendment_proposed(
    *,
    hypothesis_id:               str,
    blocking_doctrine_id:        str,
    proposed_amendment_summary:  str,
    b_reasoning:                 str,
    draft_doc_path:              str,
    b_confidence:                float = 0.5,
    parent_event_ids:            tuple = (),
    actor:                       str = "engine.agents.strengthener.approval_view",
) -> str:
    """Phase 2.0 step 13 (2026-06-06): the principal approved B's
    DOCTRINE_AMENDMENT_NEEDED verdict in /approvals.

    DISTINCT from memory_doctrine_locked — that event fires when the
    memory file is actually LOCKED (written/amended). This event is
    upstream: the principal said "yes, draft this amendment", but the
    actual memory file hasn't been edited yet. The draft is parked
    in `data/strengthener/amendment_drafts/<hyp_id>.md` for the
    principal to review + apply manually (memory file edits go
    through Claude's Write tool, not autonomous file mutation).

    Subject is the blocking_doctrine_id (the memory file slug) —
    queries like "all amendment proposals targeting memory file X"
    are one-liners. Auto-registers if missing (per the existing
    auto-register convention for emit helpers).

    parent_event_ids should include B's session/synthesis chain if
    available. The eventual memory_doctrine_locked event will
    reference THIS event in its parent_event_ids when the actual
    memory file is updated, closing the proposal→lock lineage.
    """
    metrics = {
        "hypothesis_id":              hypothesis_id,
        "proposed_amendment_summary": proposed_amendment_summary[:400],
        "b_confidence":               float(b_confidence),
        "b_reasoning":                b_reasoning[:600],
    }

    summary = (
        f"Amendment proposed for {blocking_doctrine_id}: "
        f"{proposed_amendment_summary[:200]}"
    )

    # Auto-register the doctrine subject if missing
    try:
        registry.register_subject(
            subject_id   = blocking_doctrine_id,
            subject_type = SubjectType.memory_doctrine,
            description  = "Auto-registered via memory_amendment_proposed "
                           "(B-proposed doctrine amendment awaiting principal "
                           "manual edit of the memory file)",
            created_by   = "emit.memory_amendment_proposed",
        )
    except Exception:
        pass

    return _emit(
        EventType.memory_amendment_proposed,
        subject_id        = blocking_doctrine_id,
        verdict           = Verdict.NEUTRAL,
        metrics           = metrics,
        artifacts         = {"amendment_draft": draft_doc_path},
        summary           = summary,
        parent_event_ids  = parent_event_ids,
        family            = None,
        tags              = ("memory_amendment", "proposed"),
        actor             = actor,
    )


def forward_vector_created(
    *,
    hypothesis_id:         str,
    verdict_type:          str,                    # B's verdict type: APPROVE_FOR_PIPELINE / DOCTRINE_AMENDMENT_NEEDED
    b_confidence:          float,                  # B's confidence on the underlying review
    extraction_method:     str,                    # llm_synthesis / human_authored
    mechanism_family:      Optional[str] = None,
    summary_override:      Optional[str] = None,
    parent_event_ids:      tuple = (),
    actor:                 str = "engine.agents.strengthener.approval_view",
) -> str:
    """Phase 2.1a (2026-06-06): the principal approved a B verdict in
    /approvals → the corresponding hypothesis is now eligible for
    forward_vector queue.

    Why this event exists: before Phase 2.1, B's APPROVE_FOR_PIPELINE
    verdicts had no downstream consumer. The principal could click
    'approved' in /approvals but the hypothesis sat in PROPOSED state
    forever; generate_forward_vectors required source_paper_id which
    LLM_SYNTHESIS doesn't have. This event closes that gap.

    parent_event_ids SHOULD include the B-verdict's
    papers_curator_synthesis_run event (lineage to the synthesis run
    that produced the candidate). The B verdict itself isn't yet
    emitted as a research event (deferred — would need its own
    event type), so the chain is:
      papers_curator_synthesis_run → forward_vector_created → (later)
      candidate_pipeline_started → factor_verdict_filed.

    subject_id is the hypothesis_id; subject_type is factor (the
    existing convention for hypothesis-rooted events).

    Verdict mapping:
      verdict_type == "APPROVE_FOR_PIPELINE"       → NEUTRAL  (intent
                                                      registered; no
                                                      strict-gate verdict yet)
      verdict_type == "DOCTRINE_AMENDMENT_NEEDED"  → NEUTRAL  (different
                                                      downstream path —
                                                      memory_amendment
                                                      handler, not fv queue;
                                                      event still recorded
                                                      for audit)
    """
    metrics = {
        "verdict_type":      verdict_type,
        "b_confidence":      float(b_confidence),
        "extraction_method": extraction_method,
    }

    summary = summary_override or (
        f"forward_vector created for {hypothesis_id[:8]} "
        f"(B verdict {verdict_type}, conf={b_confidence:.2f}, "
        f"extraction={extraction_method})"
    )

    # Auto-register the hypothesis_id as a factor subject (idempotent).
    # Hypothesis UUIDs aren't pre-registered in subjects.yaml — they
    # become "factors" the moment they're flagged for the pipeline.
    # Matches the existing convention (autopilot_live.py / shadow_emit
    # register `auto_<hash>` factor subjects on the fly).
    try:
        registry.register_subject(
            subject_id   = hypothesis_id,
            subject_type = SubjectType.factor,
            family       = mechanism_family,
            description  = f"Auto-registered for forward_vector_created "
                           f"(B-approved hypothesis, extraction="
                           f"{extraction_method})",
            created_by   = "emit.forward_vector_created",
        )
    except Exception:
        # If registration fails (e.g. type mismatch with existing
        # registration), let _emit's registry.require fail loudly so
        # the bug surfaces.
        pass

    return _emit(
        EventType.forward_vector_created,
        subject_id        = hypothesis_id,
        verdict           = Verdict.NEUTRAL,
        metrics           = metrics,
        artifacts         = {},
        summary           = summary,
        parent_event_ids  = parent_event_ids,
        family            = mechanism_family,
        tags              = ("forward_vector", "b_approved",
                              extraction_method),
        actor             = actor,
    )


_DOCTRINE_SIGNAL_SEVERITY_TO_VERDICT = {
    "INFO":     Verdict.NEUTRAL,
    "WARN":     Verdict.MARGINAL,
    "CRITICAL": Verdict.RED,
}


def chief_of_staff_session_run(
    *,
    session_id:           str,
    d_emitted:            int,
    a_n_candidates:       int,
    a_n_written:          int,
    b_n_reviewed:         int,
    b_n_pending_approval: int,
    parent_event_ids:     tuple = (),
    errors:               list[str] = (),
    summary_override:     Optional[str] = None,
    memo_headline:        Optional[str] = None,   # step 14b
    actor:                str = "engine.agents.chief_of_staff.runner",
) -> str:
    """Phase 2.0 step 14a (2026-06-06): one event per weekly chief_of_staff
    session that orchestrates D → A → B.

    `session_id` is the deterministic correlation id for this run
    (e.g. 'cos-2026-06-06'); it's also surfaced as a tag so
    downstream queries can find every event in a given session.

    Verdict mapping:
      errors empty AND any substep produced output → GREEN
      errors empty AND everything was 0           → NEUTRAL
      errors present                              → MARGINAL

    `parent_event_ids` should include every substep emit (D's
    doctrine_signal_detected events + A's papers_curator_synthesis_run
    event). B doesn't yet emit its own event (Step 14b will add one),
    so it's not in parent_event_ids — its work is summarized in
    b_n_reviewed / b_n_pending_approval metrics.
    """
    if errors:
        verdict = Verdict.MARGINAL
    elif (d_emitted + a_n_candidates + a_n_written + b_n_reviewed) > 0:
        verdict = Verdict.GREEN
    else:
        verdict = Verdict.NEUTRAL

    metrics = {
        "session_id":            session_id,
        "d_emitted":             int(d_emitted),
        "a_n_candidates":        int(a_n_candidates),
        "a_n_written":           int(a_n_written),
        "b_n_reviewed":          int(b_n_reviewed),
        "b_n_pending_approval":  int(b_n_pending_approval),
        "errors_count":          len(errors or ()),
    }
    if errors:
        metrics["errors_sample"] = list(errors[:5])
    if memo_headline:
        metrics["memo_headline"] = memo_headline[:200]

    # Step 14b: the memo lives in data/chief_of_staff/weekly_memos.jsonl
    # but we surface the path on the event so audit queries find it
    # via artifacts['memo_doc']. Existence check is at emit-time via
    # _validate_artifacts — skip pointing at a non-existent file by
    # only adding artifact when memo was actually written.
    artifacts: dict[str, str] = {}
    if memo_headline:
        from pathlib import Path as _P
        _memo_path = _REPO_ROOT / "data" / "chief_of_staff" / "weekly_memos.jsonl"
        if _memo_path.is_file():
            artifacts["memo_doc"] = str(_memo_path.relative_to(_REPO_ROOT))

    summary = summary_override or (
        f"weekly session {session_id}: "
        f"D emitted {d_emitted}, "
        f"A {a_n_candidates} cand / {a_n_written} written, "
        f"B {b_n_reviewed} reviewed / {b_n_pending_approval} pending"
        + (f", {len(errors)} error(s)" if errors else "")
    )

    return _emit(
        EventType.chief_of_staff_session_run,
        subject_id        = "chief_of_staff",
        verdict           = verdict,
        metrics           = metrics,
        artifacts         = artifacts,
        summary           = summary,
        parent_event_ids  = parent_event_ids,
        family            = None,
        tags              = ("chief_of_staff", f"session:{session_id}"),
        actor             = actor,
    )


def doctrine_signal_detected(
    *,
    subject_id:          str,
    pattern_name:        str,
    metrics:             dict,
    summary:             str,
    severity:            str = "WARN",          # INFO / WARN / CRITICAL
    parent_event_ids:    tuple = (),
    family:              Optional[str] = None,
    tags:                tuple = (),
    actor:               str = "engine.agents.book_monitor",
) -> str:
    """Phase 2.0 step 9a (2026-06-06): Employee D pattern-detection emit.

    Fired when a deterministic rule (NOT an LLM) sees a pattern in the
    research store worth doctrine-amending attention. The first three
    canonical rules (built in step 9b):

      - family_red_cluster:   ≥N RED verdicts in same family / window
      - sleeve_sharpe_decay:  DEPLOYED sleeve EWMA Sharpe below floor
      - gate_rejection_spike: strict gate reject rate above threshold

    The subject_id is the thing being flagged — typically a
    representative factor (most-recent RED in the cluster) or the
    affected sleeve. The cluster context (family, window, member ids)
    lives in metrics.

    Subscribers (read these events as input):
      - Employee A's synthesis gatherer (already wired —
        doctrine_signal_detected is in the
        _SYNTHESIS_RELEVANT_EVENT_TYPES set).
      - chief_of_staff orchestrator (step 14).
      - /approvals when severity=CRITICAL (step 13).

    Verdict mapping:
      INFO     → NEUTRAL  (notable but not actionable)
      WARN     → MARGINAL (default — actionable doctrine signal)
      CRITICAL → RED      (drop everything, this hits live capital)

    `pattern_name` MUST be a registered rule name (the runner enforces
    this; here it's just stored as metrics["pattern_name"] for query).
    `parent_event_ids` SHOULD link to the source events that triggered
    the pattern (e.g. the 3 factor_verdict_filed RED events that
    cluster) so lineage queries can walk back to root causes.
    """
    verdict = _DOCTRINE_SIGNAL_SEVERITY_TO_VERDICT.get(severity.upper())
    if verdict is None:
        raise InvalidEventError(
            f"unknown severity {severity!r}; choose from "
            f"{sorted(_DOCTRINE_SIGNAL_SEVERITY_TO_VERDICT)}"
        )

    enriched_metrics = dict(metrics or {})
    enriched_metrics["pattern_name"] = pattern_name
    enriched_metrics["severity"]     = severity.upper()

    return _emit(
        EventType.doctrine_signal_detected,
        subject_id        = subject_id,
        verdict           = verdict,
        metrics           = enriched_metrics,
        artifacts         = {},
        summary           = summary,
        parent_event_ids  = parent_event_ids,
        family            = family,
        tags              = ("doctrine_signal", pattern_name) + tuple(tags or ()),
        actor             = actor,
    )


def capability_evidence_filed(
    subject_id: str,
    verdict: str,
    artifacts: dict[str, str],
    summary: str,
    metrics: Optional[dict] = None,
    parent_event_ids: tuple = (),
    family: Optional[str] = None,
    tags: tuple = (),
    actor: str = DEFAULT_ACTOR,
) -> str:
    """A capability evidence doc was filed (docs/capability_evidence/*.md).

    Usually emitted ALONGSIDE a factor_verdict (parent_event_ids includes the
    verdict event). This one carries the doc-filing fact specifically — query
    by event_type=capability_evidence_filed to find all docs.
    """
    return _emit(
        EventType.capability_evidence_filed,
        subject_id=subject_id,
        verdict=Verdict(verdict),
        metrics=metrics or {}, artifacts=artifacts, summary=summary,
        parent_event_ids=parent_event_ids, family=family, tags=tags, actor=actor,
    )
