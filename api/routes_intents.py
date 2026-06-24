"""api/routes_intents.py — typed-intent persistence layer.

Closes Collab-断点 1+2 (R3.1 audit): until today the page-level CTAs
("Audit session", "Pipeline test", "Open session →") were nothing
more than URL jumps. Claude doesn't read browser URLs, so when the
user clicked one the intent evaporated. The user then had to
re-articulate it in Claude Code chat ("audit crisis_hedge for me").

This module persists intents as a typed jsonl event log that Claude
can poll on a hook. Each intent has a stable kind + subject + payload;
the lifecycle is pending → acknowledged → fulfilled. Fire-and-forget
from the UI side, structured pickup on the Claude side.

Intent shape (data/research_store/intents.jsonl):
  {
    intent_id:     str    UUID4
    kind:          str    audit_subject | pipeline_test | research_test
                          | ingest_paper | re_audit_decay | etc
    subject_type:  str    mechanism | hypothesis | paper | lesson | sleeve
    subject_id:    str    canonical id of the subject
    filed_ts:      str    ISO-8601 UTC
    filed_by:      str    user | claude | cron
    source_page:   str    /lab/library/detail | /research/forward | …
    payload:       dict   kind-specific extra (family, proposal_name,
                          paper_id, notes…)
    status:        str    pending | acknowledged | fulfilled
    ack_ts:        str?
    ack_by:        str?
    fulfill_ts:    str?
    fulfill_by:    str?
    fulfill_event_id: str?     research_store event_id when known
  }

Endpoints
  POST   /api/intents/file              file a new intent
  GET    /api/intents/pending           list pending intents
                                        (default last 24h, status=pending)
  POST   /api/intents/{intent_id}/ack   mark acknowledged (Claude sees it)
  POST   /api/intents/{intent_id}/fulfill  mark done
  GET    /api/intents                   browse history (debugging)

Append-only — to amend a pending intent, file a new one referencing
the previous via payload.parent_intent_id. Same doctrine as the
research_store event log.
"""
from __future__ import annotations

import datetime as _dt
import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/api/intents", tags=["intents"])


_REPO_ROOT = Path(__file__).resolve().parent.parent
_INTENTS   = _REPO_ROOT / "data" / "research_store" / "intents.jsonl"


# ─── lifecycle helpers ────────────────────────────────────────────


VALID_KINDS = {
    "audit_subject",
    "pipeline_test",
    "research_test",
    "ingest_paper",
    "re_audit_decay",
    "review_lesson",
    "explore_hypothesis",
    "annotate_doctrine",
}

VALID_SUBJECT_TYPES = {
    "mechanism", "hypothesis", "paper", "lesson", "sleeve", "session", "axis",
}

VALID_STATUSES = {"pending", "acknowledged", "fulfilled", "rejected"}


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir() -> None:
    _INTENTS.parent.mkdir(parents=True, exist_ok=True)


def _append(record: dict) -> None:
    _ensure_dir()
    with _INTENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_all() -> list[dict]:
    if not _INTENTS.is_file():
        return []
    rows: list[dict] = []
    for line in _INTENTS.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s: continue
        try: rows.append(json.loads(s))
        except json.JSONDecodeError: continue
    return rows


def _latest_per_intent(rows: list[dict]) -> dict[str, dict]:
    """Latest version per intent_id (newest wins). Status transitions
    are recorded as new rows; the latest row defines the current state.

    We treat the row with the most recent of (fulfill_ts, ack_ts, filed_ts)
    as the head — fulfill > ack > file in the lifecycle."""
    def head_ts(r: dict) -> str:
        return r.get("fulfill_ts") or r.get("ack_ts") or r.get("filed_ts") or ""
    latest: dict[str, dict] = {}
    for r in rows:
        iid = r.get("intent_id")
        if not iid: continue
        cur = latest.get(iid)
        if cur is None or head_ts(r) >= head_ts(cur):
            latest[iid] = r
    return latest


# ─── request models ──────────────────────────────────────────────


class FileIntentRequest(BaseModel):
    kind:          str
    subject_type:  str
    subject_id:    str
    source_page:   str  = ""
    filed_by:      str  = "user"
    payload:       dict = {}


class FileIntentResponse(BaseModel):
    intent_id:  str
    filed_ts:   str
    status:     str


class AckRequest(BaseModel):
    ack_by: str = "claude"


class FulfillRequest(BaseModel):
    fulfill_by:       str           = "claude"
    fulfill_event_id: Optional[str] = None
    note:             str           = ""


class IntentRow(BaseModel):
    intent_id:        str
    kind:             str
    subject_type:     str
    subject_id:       str
    filed_ts:         str
    filed_by:         str
    source_page:      str
    payload:          dict
    status:           str
    ack_ts:           Optional[str] = None
    ack_by:           Optional[str] = None
    fulfill_ts:       Optional[str] = None
    fulfill_by:       Optional[str] = None
    fulfill_event_id: Optional[str] = None


# ─── endpoints ───────────────────────────────────────────────────


def _dq_is_halt() -> tuple[bool, str]:
    """Cheap DQ status probe. Used by /file to guard execution-class
    intents from being queued when the pipeline can't actually run.

    Failures (DQ subsystem broken / unavailable) return (False, "")
    so we DON'T block intents on telemetry-class errors."""
    try:
        from api.main import dq_report
        d = dq_report() or {}
        if (d.get("verdict") or "").upper() == "HALT":
            return True, str(d.get("rationale") or "DQ verdict HALT")
    except Exception:
        pass
    return False, ""


# Kinds that REQUIRE the daily paper-trade pipeline to be runnable.
# Filing one of these while DQ is HALT just creates a pending intent
# that can't be fulfilled — Claude would pick it up, try to run, and
# bail. Better to refuse upfront with a clear explanation.
_EXECUTION_KINDS = {"pipeline_test", "research_test", "re_audit_decay"}


@router.post("/file", response_model=FileIntentResponse)
def file_intent(req: FileIntentRequest):
    """File a new intent. Validates kind + subject_type against the
    enums; everything else is free-form payload. P1-C 2026-06-04:
    execution-class intents are refused when DQ is HALT, since the
    paper-trade pipeline they target can't run anyway."""
    if req.kind not in VALID_KINDS:
        raise HTTPException(status_code=400,
            detail=f"unknown kind {req.kind!r}; allowed: {sorted(VALID_KINDS)}")
    if req.subject_type not in VALID_SUBJECT_TYPES:
        raise HTTPException(status_code=400,
            detail=f"unknown subject_type {req.subject_type!r}; "
                   f"allowed: {sorted(VALID_SUBJECT_TYPES)}")
    if not req.subject_id.strip():
        raise HTTPException(status_code=400, detail="subject_id required")

    if req.kind in _EXECUTION_KINDS:
        halt, reason = _dq_is_halt()
        if halt:
            raise HTTPException(status_code=409, detail={
                "error":     "dq_halt",
                "kind":      req.kind,
                "rationale": reason[:400],
                "hint":      "Resolve DQ breaches first (open /lab/cockpit), "
                             "then re-file the intent.",
            })

    intent_id = uuid.uuid4().hex
    now = _utc_iso()
    record = {
        "intent_id":        intent_id,
        "kind":             req.kind,
        "subject_type":     req.subject_type,
        "subject_id":       req.subject_id.strip(),
        "filed_ts":         now,
        "filed_by":         req.filed_by or "user",
        "source_page":      req.source_page or "",
        "payload":          dict(req.payload or {}),
        "status":           "pending",
        "ack_ts":           None,
        "ack_by":           None,
        "fulfill_ts":       None,
        "fulfill_by":       None,
        "fulfill_event_id": None,
    }
    _append(record)

    # AI-native Step 4 (2026-06-04) — publish to EventBus so reactive
    # subscribers (graveyard_collision agent) can run synchronously.
    # Best-effort: bus failure must NOT corrupt the intent file write.
    try:
        from engine.agents.event_bus import get_event_bus
        get_event_bus().publish(
            event_type   = "intent_filed",
            payload      = {
                "intent_id":    intent_id,
                "kind":         req.kind,
                "subject_type": req.subject_type,
                "subject_id":   record["subject_id"],
                "payload":      record["payload"],
            },
            source_agent = record["filed_by"],
        )
    except Exception as _exc:
        import logging as _l
        _l.getLogger(__name__).warning("intent_filed bus publish failed: %s", _exc)

    return FileIntentResponse(intent_id=intent_id, filed_ts=now, status="pending")


@router.get("/pending", response_model=list[IntentRow])
def list_pending(
    since_minutes: int = 1440,
    kind:          Optional[str] = None,
    subject_type:  Optional[str] = None,
):
    """For Claude / cron hooks: list pending intents, newest first.

    Defaults to last 24h. Pass kind=audit_subject to filter to one
    intent class. Returns [] when nothing's queued — quiet steady-state
    means nothing for Claude to pick up."""
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(minutes=since_minutes)
    rows = _load_all()
    latest = _latest_per_intent(rows)
    out: list[IntentRow] = []
    for r in latest.values():
        if r.get("status") != "pending":
            continue
        ts_str = r.get("filed_ts", "")
        try:
            ts = _dt.datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        if ts < cutoff:
            continue
        if kind and r.get("kind") != kind:
            continue
        if subject_type and r.get("subject_type") != subject_type:
            continue
        out.append(IntentRow(**r))
    out.sort(key=lambda x: x.filed_ts, reverse=True)
    return out


@router.post("/{intent_id}/ack", response_model=IntentRow)
def ack_intent(intent_id: str, req: AckRequest):
    """Mark intent acknowledged (Claude saw it; not yet done)."""
    latest = _latest_per_intent(_load_all())
    if intent_id not in latest:
        raise HTTPException(status_code=404, detail=f"intent {intent_id} not found")
    cur = latest[intent_id]
    if cur.get("status") == "fulfilled":
        raise HTTPException(status_code=409, detail="intent already fulfilled")
    new = {**cur,
           "status": "acknowledged",
           "ack_ts": _utc_iso(),
           "ack_by": req.ack_by or "claude"}
    _append(new)
    return IntentRow(**new)


@router.post("/{intent_id}/fulfill", response_model=IntentRow)
def fulfill_intent(intent_id: str, req: FulfillRequest):
    """Mark intent fulfilled. Optionally link the research_store event
    that captures the actual work output (factor_verdict_filed,
    capability_evidence_filed, etc.)."""
    latest = _latest_per_intent(_load_all())
    if intent_id not in latest:
        raise HTTPException(status_code=404, detail=f"intent {intent_id} not found")
    cur = latest[intent_id]
    new = {**cur,
           "status":           "fulfilled",
           "fulfill_ts":       _utc_iso(),
           "fulfill_by":       req.fulfill_by or "claude",
           "fulfill_event_id": req.fulfill_event_id or None}
    if req.note:
        # Append the note to payload under a fulfillment_note key.
        new["payload"] = {**new.get("payload", {}), "fulfillment_note": req.note}
    _append(new)
    return IntentRow(**new)


@router.get("", response_model=list[IntentRow])
def list_all(
    limit: int = 200,
    status: Optional[str] = None,
):
    """Browse intent history (debugging / audit). Newest first."""
    latest = _latest_per_intent(_load_all())
    rows = list(latest.values())
    if status:
        rows = [r for r in rows if r.get("status") == status]
    rows.sort(key=lambda x: x.get("filed_ts", ""), reverse=True)
    return [IntentRow(**r) for r in rows[:limit]]
