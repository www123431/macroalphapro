"""api/routes_daily_directive.py — Chief of Staff "Daily Directive" endpoint.

The "employee-feeling" agent the user kept asking for: every morning,
aggregate state across all surfaces and tell them what to do.

Output shape:
  {
    "generated_ts": "...",
    "blockers":    [{"kind", "title", "detail", "where"}],
    "pending":     [{"kind", "title", "count", "where"}],
    "today":       [{"rank", "title", "rationale", "where"}],
    "stats":       {dq, decay, queue, sessions, audit},
  }

Pure aggregation. NO LLM call in v1 — heuristic + thresholds. An LLM
narration layer (Chief of Staff persona) can layer on later as a
`summary` field. The structured fields above are the contract.

Why pure deterministic first:
  - Cheap & reproducible — no API spend
  - Easy to debug — every section maps to a discrete query
  - LLM can later READ this and add prose; that's a strict superset
"""
from __future__ import annotations

import json
import datetime as _dt
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents/daily_directive", tags=["agents"])

_REPO_ROOT       = Path(__file__).resolve().parent.parent
_LINEAGE_RESULTS = _REPO_ROOT / "data" / "audit_verifier" / "lineage_results.jsonl"


# ── Output schema ────────────────────────────────────────────


class DirectiveItem(BaseModel):
    kind:      str            # "dq_halt" | "decay_action" | "queue_approved" | ...
    title:     str            # human-facing one-liner
    detail:    Optional[str] = None
    where:     Optional[str] = None   # route or page to go to
    count:     Optional[int] = None   # for pending-section items
    rank:      Optional[int] = None   # for today-section items (1=highest)
    rationale: Optional[str] = None   # why it's the right next thing
    severity:  Optional[str] = None   # "high" | "medium" | "low"


class DirectionItem(BaseModel):
    rank:                  int
    source_paper_id:       str
    paper_title:           str
    source_hypothesis_id:  str
    claim:                 str
    family:                str
    mechanism_subtype:     Optional[str] = None
    predicted_direction:   Optional[str] = None
    data_status:           str
    priority:              str
    pm_status:             str
    scores:                dict
    graveyard_verdict:     str
    graveyard_n_scanned:   int
    rationale:             str


class DirectiveStats(BaseModel):
    dq_verdict:           str
    decay_overall:        str
    forward_approved:     int
    pending_intents:      int
    active_session_type:  Optional[str] = None
    lessons_last_72h:     int
    lineage_warn_or_fail_24h: int


class DailyDirective(BaseModel):
    generated_ts: str
    blockers:     list[DirectiveItem]
    pending:      list[DirectiveItem]
    today:        list[DirectiveItem]
    directions:   list[DirectionItem]
    stats:        DirectiveStats


# ── Aggregation helpers ──────────────────────────────────────


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_dq() -> dict:
    try:
        from engine.dq.inspector import current_dq_state
        return current_dq_state() or {}
    except Exception:
        # Fall back to reading the same path the /api/dq endpoint reads
        p = _REPO_ROOT / "data" / "dq" / "current.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}


def _get_decay() -> dict:
    p = _REPO_ROOT / "data" / "decay" / "report.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _count_pending_intents() -> tuple[int, list[dict]]:
    try:
        from engine.intents import store as intents_store
        rows = intents_store.list_by_status("pending", limit=50)
        return len(rows), rows
    except Exception:
        # Direct jsonl read fallback
        p = _REPO_ROOT / "data" / "intents" / "intents.jsonl"
        if not p.exists():
            return 0, []
        try:
            rows = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if r.get("status") == "pending":
                            rows.append(r)
                    except Exception:
                        continue
            return len(rows), rows[:50]
        except Exception:
            return 0, []


def _get_active_session() -> Optional[dict]:
    try:
        from engine.sessions import store as session_store
        return session_store.get_active()
    except Exception:
        return None


def _count_forward_approved() -> tuple[int, list[dict]]:
    """Top approved forward vectors with data=have, by priority."""
    try:
        from engine.research_store.forward_vectors import generate_forward_vectors
        from engine.research_store.forward_vectors.review import load_latest_reviews
        vecs = generate_forward_vectors()
        reviews = load_latest_reviews()
        approved = []
        for v in vecs:
            r = reviews.get(v.source_hypothesis_id)
            if r and r.status.value == "approved":
                approved.append({
                    "hypothesis_id":  v.source_hypothesis_id,
                    "family":         v.mechanism_family.value,
                    "subtype":        v.mechanism_subtype,
                    "claim":          v.claim,
                    "priority":       v.priority.value,
                })
        return len(approved), approved[:10]
    except Exception as exc:
        logger.warning("daily_directive: forward count failed: %s", exc)
        return 0, []


def _count_recent_lessons(hours: int = 72) -> int:
    try:
        from engine.research_store import store
        cutoff = (_dt.datetime.utcnow() - _dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        evs = store.filter_events(event_type="factor_verdict_filed", since=cutoff, limit=50)
        return len(evs)
    except Exception:
        return 0


def _count_recent_lineage_warns(hours: int = 24) -> tuple[int, list[dict]]:
    """Count WARN+FAIL rows from audit_verifier in last N hours."""
    if not _LINEAGE_RESULTS.exists():
        return 0, []
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(hours=hours)
    flagged = []
    try:
        with _LINEAGE_RESULTS.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("verdict") not in ("WARN", "FAIL"):
                    continue
                ts = row.get("verified_ts", "")
                try:
                    t = _dt.datetime.fromisoformat(ts.rstrip("Z"))
                except Exception:
                    continue
                if t >= cutoff:
                    flagged.append(row)
    except Exception:
        pass
    return len(flagged), flagged[:10]


# ── Directive composition ────────────────────────────────────


def _build_blockers(dq: dict, decay: dict, active: Optional[dict]) -> list[DirectiveItem]:
    out: list[DirectiveItem] = []
    if dq.get("verdict") == "HALT":
        out.append(DirectiveItem(
            kind     = "dq_halt",
            severity = "high",
            title    = f"DQ HALT — {dq.get('n_breaches', '?')} breach(es)",
            detail   = "Pipeline-test intents are refused server-side until DQ clears.",
            where    = "/lab/cockpit",
        ))
    if decay.get("overall") == "ACTION":
        out.append(DirectiveItem(
            kind     = "decay_action",
            severity = "high",
            title    = f"Decay ACTION — {decay.get('n_mechanisms', '?')} sleeve(s) flagged",
            detail   = "At least one deployed sleeve requires immediate review (re-test or de-allocate).",
            where    = "/lab/decay",
        ))
    if active and active.get("session_type"):
        # Optional: warn if session has been open > 6h (might be stuck)
        opened_ts = active.get("opened_ts")
        if opened_ts:
            try:
                t = _dt.datetime.fromisoformat(opened_ts.rstrip("Z"))
                age_h = (_dt.datetime.utcnow() - t).total_seconds() / 3600
                if age_h > 6:
                    out.append(DirectiveItem(
                        kind     = "session_stale",
                        severity = "medium",
                        title    = f"Session open {int(age_h)}h — verify exit conditions",
                        detail   = f"Active {active.get('session_type')} session "
                                   f"{active.get('session_id', '')[:8]}. "
                                   f"Close if done OR abandon with reason.",
                        where    = "/lab/sessions",
                    ))
            except Exception:
                pass
    return out


def _build_pending(
    n_intents: int,
    n_approved: int,
    n_lineage_warn: int,
) -> list[DirectiveItem]:
    out: list[DirectiveItem] = []
    if n_intents > 0:
        out.append(DirectiveItem(
            kind  = "pending_intents",
            count = n_intents,
            title = f"{n_intents} intent(s) waiting for Claude",
            detail= "Start Claude Code with the poll hook OR ack/fulfill manually.",
            where = "/lab/today",
        ))
    if n_approved > 0:
        out.append(DirectiveItem(
            kind  = "approved_forward",
            count = n_approved,
            title = f"{n_approved} approved forward vector(s) in queue",
            detail= "Ready to test. Pick the highest-priority + data=have first.",
            where = "/research/enhance",
        ))
    if n_lineage_warn > 0:
        out.append(DirectiveItem(
            kind  = "lineage_warn",
            count = n_lineage_warn,
            title = f"{n_lineage_warn} recent lineage WARN/FAIL from audit_verifier",
            detail= "Some recent verdicts lack paired evidence or have invalid parents.",
            where = "/research/lessons",
        ))
    return out


def _build_today(
    blockers:    list[DirectiveItem],
    pending:     list[DirectiveItem],
    approved_top: list[dict],
    active:      Optional[dict],
) -> list[DirectiveItem]:
    """Build the prioritized 'do this next' list. 1-3 items, ROI-ranked.

    Heuristic rules (ordered, first-match-wins per slot):

      A. Any blocker exists → top item is "resolve blocker X"
      B. Active session in_flight → top item is "satisfy exit conditions"
      C. Approved forward vector available → top item is "enhance flow"
      D. Recent verdict → top item is "review lessons + plan next test"
      E. Nothing pending → "scan inbox / read 1 paper"

    Then 1-2 secondary items for variety. Max 3 total.
    """
    out: list[DirectiveItem] = []
    rank = 1

    # Slot 1 — primary action
    if blockers:
        b = blockers[0]
        out.append(DirectiveItem(
            rank      = rank,
            kind      = f"resolve_{b.kind}",
            title     = f"Resolve: {b.title}",
            rationale = "Blockers gate every downstream action. Nothing else compiles until this clears.",
            where     = b.where,
            severity  = "high",
        )); rank += 1
    elif active and active.get("session_type") == "research_new":
        out.append(DirectiveItem(
            rank      = rank,
            kind      = "satisfy_session_exit",
            title     = f"Finish active research_new session ({active.get('session_id', '')[:8]})",
            rationale = "Session exit gate needs ≥1 factor_verdict_filed + ≥1 capability_evidence_filed. Close cleanly OR abandon.",
            where     = "/research/enhance",
            severity  = "medium",
        )); rank += 1
    elif approved_top:
        top = approved_top[0]
        out.append(DirectiveItem(
            rank      = rank,
            kind      = "enhance_top_candidate",
            title     = f"Test top approved: {top['family']} / {top['subtype']}",
            rationale = f"Priority={top['priority']}. Paper-grounded, PM-approved, data on disk. Straight into pipeline.",
            where     = "/research/enhance",
            severity  = "medium",
        )); rank += 1
    else:
        out.append(DirectiveItem(
            rank      = rank,
            kind      = "scan_inbox",
            title     = "Scan Inbox + Forward queue for new opportunities",
            rationale = "No active blockers, no in-flight session, queue empty. Surface new direction.",
            where     = "/inbox",
            severity  = "low",
        )); rank += 1

    # Slot 2 — secondary action (pending decisions worth addressing)
    if pending and rank <= 3:
        # Pick the FIRST pending that's not already covered by slot 1
        for p in pending:
            if "resolve" in (out[0].kind or ""):
                # If primary was a resolve, secondary is "address pending stack"
                out.append(DirectiveItem(
                    rank      = rank,
                    kind      = f"address_{p.kind}",
                    title     = p.title,
                    rationale = "After blocker clears, drain the pending queue. ROI scales with how long it's sat.",
                    where     = p.where,
                    severity  = "medium",
                )); rank += 1
                break
            elif p.kind != "approved_forward" or (out[0].kind != "enhance_top_candidate"):
                out.append(DirectiveItem(
                    rank      = rank,
                    kind      = f"address_{p.kind}",
                    title     = p.title,
                    rationale = p.detail or "",
                    where     = p.where,
                    severity  = "low",
                )); rank += 1
                break

    # Slot 3 — strategic / preventive
    if rank <= 3:
        out.append(DirectiveItem(
            rank      = rank,
            kind      = "graveyard_review",
            title     = "Spend 10min reading 1 recent RED lesson",
            rationale = "Best return on time: every RED you internalize prevents a future repeat candidate from making it past PM approval.",
            where     = "/research/lessons?verdict=red",
            severity  = "low",
        ))

    return out


# ── Endpoint ────────────────────────────────────────────────


@router.get("", response_model=DailyDirective)
def get_daily_directive() -> DailyDirective:
    """Returns the current 'what should I do?' directive. Synchronous,
    pure aggregation, no LLM. Safe to poll (cheap, ~10ms on a warm
    process)."""
    dq             = _get_dq()
    decay          = _get_decay()
    active         = _get_active_session()
    n_intents, _   = _count_pending_intents()
    n_approved, approved_top = _count_forward_approved()
    n_lessons_72  = _count_recent_lessons(hours=72)
    n_lineage_w24, _ = _count_recent_lineage_warns(hours=24)

    blockers = _build_blockers(dq, decay, active)
    pending  = _build_pending(n_intents, n_approved, n_lineage_w24)
    today    = _build_today(blockers, pending, approved_top, active)

    # AI-native Commit D — paper-corpus directions. Pure ROI-ranked
    # untested hypotheses with paper trace + graveyard score + book
    # orthogonality. Trimmed to top-3 for the daily tile; full list
    # via /api/agents/directions.
    directions: list[DirectionItem] = []
    try:
        from engine.agents.direction_proposer import propose_directions
        d_out = propose_directions(top=3)
        directions = [DirectionItem(**d) for d in d_out.get("directions", [])]
    except Exception as exc:
        logger.warning("daily_directive: directions build failed: %s", exc)

    return DailyDirective(
        generated_ts = _utc_iso(),
        blockers     = blockers,
        pending      = pending,
        today        = today,
        directions   = directions,
        stats        = DirectiveStats(
            dq_verdict           = dq.get("verdict") or "—",
            decay_overall        = decay.get("overall") or "—",
            forward_approved     = n_approved,
            pending_intents      = n_intents,
            active_session_type  = (active or {}).get("session_type"),
            lessons_last_72h     = n_lessons_72,
            lineage_warn_or_fail_24h = n_lineage_w24,
        ),
    )


# ── Dedicated directions endpoint ───────────────────────────


class DirectionsResponse(BaseModel):
    generated_ts:         str
    deployed_families:    list[str]
    n_candidates_scanned: int
    directions:           list[DirectionItem]


@router.get("/directions", response_model=DirectionsResponse)
def get_directions(
    top: int = 10,
    family: Optional[str] = None,
) -> DirectionsResponse:
    """Full ranked direction list (Daily Directive shows top 3 inline).
    Supports family filter (e.g. ?family=CARRY)."""
    try:
        from engine.agents.direction_proposer import propose_directions
        d_out = propose_directions(top=top, family=family)
        return DirectionsResponse(
            generated_ts         = d_out["generated_ts"],
            deployed_families    = list(d_out.get("deployed_families") or []),
            n_candidates_scanned = int(d_out.get("n_candidates_scanned") or 0),
            directions           = [DirectionItem(**d) for d in d_out.get("directions", [])],
        )
    except Exception as exc:
        logger.exception("get_directions failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300])
