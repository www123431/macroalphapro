"""engine.governance.approval_ledger — append-only governance approval ledger.

Each row in data/governance/approval_ledger.jsonl is one EVENT on an
approval request: create / approve / reject / expire. To get the current
state of request `ar_xxx`, fold over its events (latest decision wins).

Why append-only + jsonl (not DB):
  1. Audit trail is the WHOLE POINT — never overwrite, never delete.
  2. Diff-friendly + grep-friendly for ops investigation.
  3. No schema migration needed; doctrine can evolve without locking
     us into a SQL model.
  4. Survives DB resets / replays (data/governance/ is sacred).

Doctrine (the things this enforces):
  - Every promote-to-paper-trade, promote-to-live, weight-change,
    or manifest-edit MUST first create_request() and then await
    approve() before execution.
  - cooling_off_seconds default = 86400 (24h). Approve before cooling_off
    elapsed = "FAST_APPROVE", recorded explicitly; institutional best
    practice is 24h cooling-off even when you're a one-person book.
  - expires_at default = 7 days from create. Expired requests cannot
    be approved; must be re-created.
  - Rejection requires a non-empty rejection_reason (≥10 chars).

See: project_approval_gateway_2026-06-02 memo.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
_LEDGER_DIR     = _REPO_ROOT / "data" / "governance"
_LEDGER_PATH    = _LEDGER_DIR / "approval_ledger.jsonl"


# Closed enumerations (no typo-induced data sharding)
ApprovalType = Literal[
    "deploy_config_promote",   # change active_config_id in active_deployment.yaml
    "weight_method_change",    # within-sleeve weighting scheme change (Phase A winner)
    "sleeve_weight_change",    # change base_weight of an existing sleeve
    "sleeve_add",              # add a new sleeve to active config
    "sleeve_remove",           # remove a sleeve
    "manifest_edit",           # any other change to active_deployment.yaml
]

ApprovalStatus = Literal["pending", "approved", "rejected", "expired"]
EventKind      = Literal["create", "approve", "reject", "expire"]


@dataclasses.dataclass(frozen=True)
class ApprovalEvent:
    """One row in the ledger. Folding all rows by id gives current state."""
    id:               str
    ts:               str
    event:            EventKind
    request_type:     str
    title:            str
    summary:          str
    proposed_payload: dict[str, Any]
    current_state:    dict[str, Any]
    evidence_pack:    dict[str, Any]
    cooling_off_seconds: int
    created_at:       str
    expires_at:       str
    decided_by:       Optional[str] = None
    decision_reason:  Optional[str] = None
    fast_approve:     bool = False    # True if approved before cooling_off elapsed
    execution_log:    Optional[str] = None


def _utc_iso(d: _dt.datetime | None = None) -> str:
    d = d or _dt.datetime.utcnow()
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_hash(payload: str) -> str:
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=4).hexdigest()


def _ensure_dir():
    _LEDGER_DIR.mkdir(parents=True, exist_ok=True)


def _new_id(request_type: str) -> str:
    """Stable but unique id: ar_<utc>_<type-prefix>_<rand>."""
    ts = _utc_iso().replace(":", "").replace("-", "")
    prefix = (request_type or "x").split("_")[0][:4]
    rand = _short_hash(f"{ts}{os.urandom(4).hex()}")
    return f"ar_{ts}_{prefix}_{rand}"


def _append_event(event: dict[str, Any]) -> None:
    _ensure_dir()
    with _LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


# ── Public API ────────────────────────────────────────────────────


def create_request(
    *,
    request_type:     ApprovalType,
    title:            str,
    summary:          str,
    proposed_payload: dict[str, Any],
    current_state:    dict[str, Any],
    evidence_pack:    Optional[dict[str, Any]] = None,
    cooling_off_seconds: int = 86400,
    expires_in_seconds:  int = 7 * 86400,
    created_by:       Optional[str] = None,
) -> str:
    """Create an approval request. Returns the request id.

    The created event has status "pending". After cooling_off_seconds the
    request becomes APPROVABLE (in the normal track); approving before
    cooling_off is allowed but flagged fast_approve=True for the audit
    trail.

    The request expires after expires_in_seconds; expired requests cannot
    be approved without re-creating.
    """
    if request_type not in (
        "deploy_config_promote", "weight_method_change",
        "sleeve_weight_change", "sleeve_add", "sleeve_remove",
        "manifest_edit",
    ):
        raise ValueError(f"unknown request_type: {request_type!r}")
    if not title or not isinstance(title, str):
        raise ValueError("title required")
    if not summary or not isinstance(summary, str):
        raise ValueError("summary required")

    now = _dt.datetime.utcnow()
    rid = _new_id(request_type)
    event = {
        "id":               rid,
        "ts":               _utc_iso(now),
        "event":            "create",
        "request_type":     request_type,
        "title":            title,
        "summary":          summary,
        "proposed_payload": dict(proposed_payload),
        "current_state":    dict(current_state),
        "evidence_pack":    dict(evidence_pack or {}),
        "cooling_off_seconds": int(cooling_off_seconds),
        "created_at":       _utc_iso(now),
        "expires_at":       _utc_iso(now + _dt.timedelta(seconds=expires_in_seconds)),
        "decided_by":       None,
        "decision_reason":  created_by or None,   # who created (note in this field)
        "fast_approve":     False,
        "execution_log":    None,
    }
    _append_event(event)
    logger.info("approval.create id=%s type=%s title=%r", rid, request_type, title)
    return rid


def approve_request(
    request_id: str,
    *,
    approved_by: str,
    note:        Optional[str] = None,
    force_pre_cooling: bool = False,
) -> dict[str, Any]:
    """Approve a pending request.

    Args:
        approved_by: human identifier — "supervisor:xizhe", "auto:test", etc.
        note:        optional reason / context for the approval
        force_pre_cooling: if True, allow approval before cooling_off elapsed
                           (will be flagged fast_approve=True in the ledger)
    """
    state = get_request(request_id)
    if state is None:
        raise ValueError(f"unknown request id: {request_id}")
    if state["status"] != "pending":
        raise ValueError(f"request {request_id} is {state['status']}, not pending")

    now = _dt.datetime.utcnow()
    expires = _dt.datetime.strptime(state["expires_at"], "%Y-%m-%dT%H:%M:%SZ")
    if now > expires:
        raise ValueError(f"request {request_id} expired at {state['expires_at']}")

    created = _dt.datetime.strptime(state["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    cooling_until = created + _dt.timedelta(seconds=int(state["cooling_off_seconds"]))
    fast = now < cooling_until
    if fast and not force_pre_cooling:
        raise ValueError(
            f"request {request_id} still in cooling-off until "
            f"{cooling_until.strftime('%Y-%m-%dT%H:%M:%SZ')} "
            f"({int((cooling_until - now).total_seconds())}s remaining); "
            f"pass force_pre_cooling=True to override"
        )

    event = {
        "id":               request_id,
        "ts":               _utc_iso(now),
        "event":            "approve",
        "request_type":     state["request_type"],
        "title":            state["title"],
        "summary":          state["summary"],
        "proposed_payload": state["proposed_payload"],
        "current_state":    state["current_state"],
        "evidence_pack":    state["evidence_pack"],
        "cooling_off_seconds": state["cooling_off_seconds"],
        "created_at":       state["created_at"],
        "expires_at":       state["expires_at"],
        "decided_by":       approved_by,
        "decision_reason":  note,
        "fast_approve":     fast,
        "execution_log":    None,
    }
    _append_event(event)
    logger.info("approval.approve id=%s by=%s fast=%s", request_id, approved_by, fast)
    return get_request(request_id)


def reject_request(
    request_id: str,
    *,
    rejected_by: str,
    reason:      str,
) -> dict[str, Any]:
    """Reject a pending request. Reason is mandatory and must be ≥ 10 chars."""
    state = get_request(request_id)
    if state is None:
        raise ValueError(f"unknown request id: {request_id}")
    if state["status"] != "pending":
        raise ValueError(f"request {request_id} is {state['status']}, not pending")
    if not isinstance(reason, str) or len(reason.strip()) < 10:
        raise ValueError("rejection reason must be ≥ 10 chars")

    event = {
        "id":               request_id,
        "ts":               _utc_iso(),
        "event":            "reject",
        "request_type":     state["request_type"],
        "title":            state["title"],
        "summary":          state["summary"],
        "proposed_payload": state["proposed_payload"],
        "current_state":    state["current_state"],
        "evidence_pack":    state["evidence_pack"],
        "cooling_off_seconds": state["cooling_off_seconds"],
        "created_at":       state["created_at"],
        "expires_at":       state["expires_at"],
        "decided_by":       rejected_by,
        "decision_reason":  reason.strip(),
        "fast_approve":     False,
        "execution_log":    None,
    }
    _append_event(event)
    logger.info("approval.reject id=%s by=%s reason=%r",
                request_id, rejected_by, reason[:60])
    return get_request(request_id)


# ── Read-side helpers ─────────────────────────────────────────────


def _iter_events() -> Iterable[dict[str, Any]]:
    if not _LEDGER_PATH.is_file():
        return iter([])
    def gen():
        with _LEDGER_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    logger.exception("approval_ledger: skipping malformed row")
    return gen()


def _fold(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold a list of events for the same id into the current state."""
    if not events:
        return {}
    events = sorted(events, key=lambda e: e["ts"])
    state = dict(events[0])  # the create event
    state["status"] = "pending"
    for ev in events[1:]:
        if ev["event"] == "approve":
            state["status"] = "approved"
            state["decided_by"]      = ev["decided_by"]
            state["decision_reason"] = ev["decision_reason"]
            state["fast_approve"]    = ev.get("fast_approve", False)
            state["decided_ts"]      = ev["ts"]
        elif ev["event"] == "reject":
            state["status"] = "rejected"
            state["decided_by"]      = ev["decided_by"]
            state["decision_reason"] = ev["decision_reason"]
            state["decided_ts"]      = ev["ts"]
        elif ev["event"] == "expire":
            state["status"] = "expired"
            state["decided_ts"]      = ev["ts"]
    # Auto-derive expired if past expires_at and still pending
    if state.get("status") == "pending":
        try:
            now = _dt.datetime.utcnow()
            expires = _dt.datetime.strptime(state["expires_at"], "%Y-%m-%dT%H:%M:%SZ")
            if now > expires:
                state["status"] = "expired"
        except Exception:
            pass
    return state


def get_request(request_id: str) -> Optional[dict[str, Any]]:
    """Return the folded current state of a single request, or None."""
    events = [e for e in _iter_events() if e.get("id") == request_id]
    if not events:
        return None
    return _fold(events)


def list_requests(
    *,
    status: Optional[str] = None,
    request_type: Optional[str] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List requests (newest first) optionally filtered by status / type.

    Folds all events once, then filters + sorts. Cheap up to ~10k rows.
    """
    by_id: dict[str, list[dict[str, Any]]] = {}
    for ev in _iter_events():
        by_id.setdefault(ev["id"], []).append(ev)
    folded = [_fold(evs) for evs in by_id.values()]
    if status is not None:
        folded = [r for r in folded if r.get("status") == status]
    if request_type is not None:
        folded = [r for r in folded if r.get("request_type") == request_type]
    folded.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return folded[:max(1, int(limit))]


def count_pending() -> int:
    return len(list_requests(status="pending", limit=10000))
