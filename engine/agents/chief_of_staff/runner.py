"""engine.agents.chief_of_staff.runner — Phase 2.0 step 14a +
Stage A piece 7b.

Deterministic Python sequencer that orchestrates one weekly session:

  0. (piece 7b) refresh substrate → fetch arxiv + nber + ssrn +
                              watchlist + forward-citations into
                              cache.jsonl BEFORE A reads it
  1. run D (book_monitor)  → emit fresh doctrine_signal_detected events
                              that A's gatherer will read on step 2
  2. run A (synthesis)     → produce candidates if substrate warrants
  3. run B (strengthener)  → review whatever A persisted (or anything
                              previously persisted but not yet reviewed)
  4. emit chief_of_staff_session_run aggregating the substep results

Decisions are deterministic:
  - Skip A step never (D's signals matter for the audit trail even
    if A returns empty).
  - Skip B step ONLY when run_strengthener_pipeline finds 0 candidates
    (idempotency built into the strengthener runner already covers
    this case; we still call it to keep the audit trail uniform).

NO LLM in this module. Each substep already does its own typed
LLM call internally. Adding an LLM at the orchestrator level would
approach Pattern 5 (banned); it lives in step 14b as a separate
post-session memo writer if/when the principal wants one.

Same fail-safe + structured-result + dry-run contract as the
individual runners. An exception in any substep is caught,
appended to errors[], and the next substep still runs.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_session_id() -> str:
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    return f"cos-{today}"


# ────────────────────────────────────────────────────────────────────
# Result shape
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class SessionResult:
    """Typed result of one orchestrator session.

    The dict form (asdict) is what the CLI / future API surface
    consumes. Kept as a dataclass for editor introspection."""
    session_id:           str
    run_ts:               str
    dry_run:              bool
    d_result:             dict
    a_result:             dict
    b_result:             dict
    session_event_id:     Optional[str]
    errors:               list
    # Roll-up convenience fields
    d_emitted:            int
    a_n_candidates:       int
    a_n_written:          int
    b_n_reviewed:         int
    b_n_pending_approval: int
    # Step 14b: weekly memo from Sonnet (None when dry_run or memo
    # generation failed)
    memo:                 Optional[dict] = None
    # Piece 7b (2026-06-07): substrate refresh result (step 0).
    # None when refresh_substrate=False; otherwise the to_dict() form
    # of SubstrateRunResult.
    substrate_result:     Optional[dict] = None
    # Stage B P2 piece 2 (2026-06-07): sleeve_fix proposer result
    # (step 1.5). None when propose_sleeve_fixes=False; otherwise the
    # dict returned by propose_sleeve_fixes().
    sleeve_fix_result:    Optional[dict] = None
    # Stage B P3c (2026-06-07): active-B sleeve strengthen scan result
    # (step 1.7). None when run_strengthen_scan=False; otherwise the
    # dict returned by run_sleeve_strengthen_scan().
    strengthen_scan_result: Optional[dict] = None


def _count_pending_b_approvals() -> int:
    """How many B verdicts are sitting in /approvals waiting on the
    principal? Surfaced in the session event so the principal sees
    'B queue has N items' without opening the UI."""
    try:
        from engine.agents.strengthener.approval_view import list_pending_approvals
        return list_pending_approvals().get("n_pending", 0)
    except Exception as exc:
        logger.warning("chief_of_staff: pending B approvals lookup failed: %s", exc)
        return 0


# ────────────────────────────────────────────────────────────────────
# Public entry
# ────────────────────────────────────────────────────────────────────
def run_weekly_session(
    *,
    session_id:           Optional[str] = None,
    dry_run:              bool = False,
    a_summaries_days:     int = 14,
    a_events_days:        int = 30,
    a_extra_tags:         tuple[str, ...] = (),
    b_max_hypotheses:     int = 10,
    d_dedup_window_days:  int = 7,
    d_events_window_days: int = 30,
    hypotheses_path:      Optional[Path] = None,
    verdicts_path:        Optional[Path] = None,
    # Piece 7b: substrate refresh (step 0)
    refresh_substrate:    bool = True,
    substrate_sources:    Optional[tuple[str, ...]] = None,
    # Stage B P2 piece 2: D->B fix-proposal coupling (step 1.5)
    propose_sleeve_fixes:   bool = True,
    sleeve_fix_max_signals: int = 10,
    sleeve_fix_days:        int = 30,
    # Stage B P3c: active-B per-sleeve strengthen scan (step 1.7)
    run_strengthen_scan:    bool = True,
    strengthen_max_sleeves: int = 3,
    strengthen_force:       bool = False,
) -> SessionResult:
    """Run one weekly session. Returns SessionResult.

    Args:
      session_id:    correlation id ('cos-2026-06-06' by default).
                     Surfaced as tag on every substep emit + on the
                     final chief_of_staff_session_run event so all
                     events from one session can be queried together.
      dry_run:       propagated to A + B; D's emits ARE still skipped
                     in dry_run mode (D writes to events.jsonl which
                     A reads — dry-run must not pollute that).

    Substep ordering MATTERS:
      D first so its emits land BEFORE A reads events.jsonl.
      A second so its candidates land BEFORE B's runner picks them up.
      B last so it can review the freshest A output.
    """
    if session_id is None:
        session_id = _default_session_id()
    run_ts = _utc_iso()
    session_tag = f"session:{session_id}"
    # Tag A's persisted hypotheses with the session id so the audit
    # trail correlates ("which hypotheses came from this session?")
    a_tags = tuple(a_extra_tags) + (session_tag,)

    errors: list[str] = []

    # ── STEP 0: substrate refresh (piece 7b) ────────────────
    # Runs BEFORE D so the cache.jsonl that A reads on step 2 is
    # already fresh. Failures here are isolated — D/A/B still run
    # against whatever cache state existed before, so a transient
    # NBER outage doesn't kill the weekly session.
    substrate_dict: Optional[dict] = None
    if refresh_substrate:
        try:
            from engine.agents.chief_of_staff.substrate import (
                ALL_SOURCES, run_weekly_substrate,
            )
            sources = (substrate_sources
                        if substrate_sources is not None
                        else ALL_SOURCES)
            substrate_res = run_weekly_substrate(
                dry_run         = dry_run,
                enabled_sources = sources,
            )
            substrate_dict = substrate_res.to_dict()
            for e in substrate_res.errors:
                errors.append(f"substrate: {e}")
        except Exception as exc:
            logger.exception("chief_of_staff: substrate step raised")
            errors.append(f"substrate_step: {exc}")

    # ── STEP 1: D (book_monitor) ─────────────────────────────
    d_result: dict = {
        "n_events_scanned": 0, "n_hits_total": 0, "n_hits_fresh": 0,
        "n_emitted": 0, "event_ids": [], "errors": [],
    }
    try:
        from engine.agents.book_monitor.runner import run_book_monitor
        d_result = run_book_monitor(
            events_window_days = d_events_window_days,
            dedup_window_days  = d_dedup_window_days,
            dry_run            = dry_run,
        )
        for e in d_result.get("errors", []):
            errors.append(f"D: {e}")
    except Exception as exc:
        logger.exception("chief_of_staff: D step raised")
        errors.append(f"D_step: {exc}")

    # ── STEP 1.5: D→B coupling (piece 7b-2 / Stage B P2) ─────
    # Turn freshly-emitted doctrine_signal_detected events into
    # B-reviewable fix-proposal Hypotheses. Runs AFTER D so it can
    # see the just-emitted signals; runs BEFORE B so B's review
    # picks them up the same run. Isolated from D/A/B failures.
    sleeve_fix_dict: Optional[dict] = None
    if propose_sleeve_fixes:
        try:
            from engine.agents.strengthener.sleeve_fix_proposer import (
                propose_sleeve_fixes as _propose_sleeve_fixes,
            )
            sleeve_fix_dict = _propose_sleeve_fixes(
                days            = sleeve_fix_days,
                max_signals     = sleeve_fix_max_signals,
                dry_run         = dry_run,
                hypotheses_path = hypotheses_path,
            )
            for e in sleeve_fix_dict.get("errors", []):
                errors.append(f"sleeve_fix: {e}")
        except Exception as exc:
            logger.exception("chief_of_staff: sleeve_fix step raised")
            errors.append(f"sleeve_fix_step: {exc}")

    # ── STEP 1.7: active-B sleeve strengthen scan (P3c) ──────
    # Runs AFTER sleeve_fix_proposer (1.5) so the D->B fix queue is
    # already populated; runs BEFORE A so A's synthesis sees the
    # active-B strengthen proposals as part of the substrate snapshot.
    # max_sleeves default 3 → rotates through 9 deployed sleeves over
    # ~3 weeks. Isolated from D/A/B failures.
    strengthen_scan_dict: Optional[dict] = None
    if run_strengthen_scan:
        try:
            from engine.agents.strengthener.sleeve_strengthen_scan import (
                run_sleeve_strengthen_scan,
            )
            strengthen_scan_dict = run_sleeve_strengthen_scan(
                max_sleeves     = strengthen_max_sleeves,
                force           = strengthen_force,
                dry_run         = dry_run,
                hypotheses_path = hypotheses_path,
            )
            for e in strengthen_scan_dict.get("errors", []):
                errors.append(f"strengthen_scan: {e}")
        except Exception as exc:
            logger.exception("chief_of_staff: strengthen_scan step raised")
            errors.append(f"strengthen_scan_step: {exc}")

    # ── STEP 2: A (papers_curator synthesis) ─────────────────
    a_result: dict = {
        "n_candidates": 0, "n_written": 0,
        "written_hypothesis_ids": [], "event_id": None, "errors": [],
    }
    try:
        from engine.agents.papers_curator.synthesis_runner import run_synthesis_pipeline
        a_result = run_synthesis_pipeline(
            dry_run         = dry_run,
            summaries_days  = a_summaries_days,
            events_days     = a_events_days,
            created_by      = "chief_of_staff_weekly",
            extra_tags      = a_tags,
            hypotheses_path = hypotheses_path,
        )
        for e in a_result.get("errors", []):
            errors.append(f"A: {e}")
    except Exception as exc:
        logger.exception("chief_of_staff: A step raised")
        errors.append(f"A_step: {exc}")

    # ── STEP 3: B (strengthener) ─────────────────────────────
    b_result: dict = {
        "n_candidates": 0, "n_reviewed": 0, "n_persisted": 0,
        "verdicts": [], "errors": [],
    }
    try:
        from engine.agents.strengthener.runner import run_strengthener_pipeline
        b_result = run_strengthener_pipeline(
            dry_run         = dry_run,
            max_hypotheses  = b_max_hypotheses,
            hypotheses_path = hypotheses_path,
            verdicts_path   = verdicts_path,
        )
        for e in b_result.get("errors", []):
            errors.append(f"B: {e}")
    except Exception as exc:
        logger.exception("chief_of_staff: B step raised")
        errors.append(f"B_step: {exc}")

    # ── STEP 4: emit session-summary event ────────────────────
    session_event_id: Optional[str] = None
    parent_event_ids: list[str] = []
    parent_event_ids.extend(d_result.get("event_ids") or [])
    if a_result.get("event_id"):
        parent_event_ids.append(a_result["event_id"])

    b_pending = _count_pending_b_approvals()

    # ── STEP 4: weekly memo (step 14b) ──────────────────────
    # Generate BEFORE emit so the memo file path can be carried as
    # an artifact on the session event. Skipped on dry_run so
    # weekly_memos.jsonl isn't polluted with preview runs.
    memo_dict: Optional[dict] = None
    if not dry_run:
        try:
            from engine.agents.chief_of_staff.memo import generate_memo
            memo = generate_memo(
                session_id     = session_id,
                session_result = {
                    "d_result": d_result,
                    "a_result": a_result,
                    "b_result": b_result,
                },
                pending_b      = b_pending,
            )
            if memo is not None:
                memo_dict = memo.to_dict()
        except Exception as exc:
            logger.exception("chief_of_staff: memo step raised")
            errors.append(f"memo: {exc}")

    # ── STEP 5: emit session-summary event ──────────────────
    if not dry_run:
        try:
            from engine.research_store import emit
            session_event_id = emit.chief_of_staff_session_run(
                session_id           = session_id,
                d_emitted            = int(d_result.get("n_emitted", 0)),
                a_n_candidates       = int(a_result.get("n_candidates", 0)),
                a_n_written          = int(a_result.get("n_written", 0)),
                b_n_reviewed         = int(b_result.get("n_reviewed", 0)),
                b_n_pending_approval = b_pending,
                parent_event_ids     = tuple(parent_event_ids),
                errors               = errors,
                memo_headline        = (memo_dict or {}).get("headline"),
            )
        except Exception as exc:
            logger.exception("chief_of_staff: session emit raised")
            errors.append(f"session_emit: {exc}")

    return SessionResult(
        session_id           = session_id,
        run_ts               = run_ts,
        dry_run              = dry_run,
        d_result             = d_result,
        a_result             = a_result,
        b_result             = b_result,
        session_event_id     = session_event_id,
        errors               = errors,
        d_emitted            = int(d_result.get("n_emitted", 0)),
        a_n_candidates       = int(a_result.get("n_candidates", 0)),
        a_n_written          = int(a_result.get("n_written", 0)),
        b_n_reviewed         = int(b_result.get("n_reviewed", 0)),
        b_n_pending_approval = b_pending,
        memo                 = memo_dict,
        substrate_result       = substrate_dict,
        sleeve_fix_result      = sleeve_fix_dict,
        strengthen_scan_result = strengthen_scan_dict,
    )
