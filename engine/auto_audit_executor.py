"""
engine/auto_audit_executor.py — Execute approved auto_audit proposals (R-1.E, 2026-05-06).

Called from `engine.memory.resolve_pending_approval` when:
  pa.approval_type == 'auto_audit_proposal' AND approved=True

Executor scope is intentionally narrow:
  • amendment_kind ∈ {clarification, scope_narrow, threshold_tweak, hypothesis_amend,
                      endpoint_swap, superseded}  → call engine.preregistration.amend_spec
  • amendment_kind == 'no_action'                  → mark AuditFinding status='IGNORED' + notes
  • Anything else → no-op + warning log

Critical invariant (project-wide):
  THE EXECUTOR DOES NOT EDIT CODE. amend_spec records hashes of *current* file
  state — supervisor must edit the file MANUALLY before approving the
  proposal. If file edit hasn't happened yet, amend_spec just records the
  unchanged hash with the supervisor's rationale; this is acceptable but
  callers should be aware.

Failure modes are surfaced via the return dict, not via raise — caller
(resolve_pending_approval) writes them into PendingApproval.notes.
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, Optional

from engine.auto_audit_models import AuditFinding, AuditProposal
from engine.memory import PendingApproval, SessionFactory

logger = logging.getLogger(__name__)


def _find_proposal_for_pa(pa_id: int) -> Optional[AuditProposal]:
    with SessionFactory() as s:
        return (
            s.query(AuditProposal)
             .filter(AuditProposal.pending_approval_id == pa_id)
             .first()
        )


def execute_approved_proposal(pa_id: int, supervisor_id: str = "human") -> Dict[str, Any]:
    """
    Look up the AuditProposal for this PendingApproval row and execute the
    recommended option per amendment_kind.

    Returns {ok, action_taken, exec_detail}. Does NOT modify pa.status —
    that's the resolve_pending_approval caller's job.
    """
    prop = _find_proposal_for_pa(pa_id)
    if prop is None:
        return {"ok": False, "action_taken": "none",
                "exec_detail": {"error": f"no AuditProposal links to PendingApproval #{pa_id}"}}

    if prop.gate_status != "pass":
        return {"ok": False, "action_taken": "none",
                "exec_detail": {"error": f"gate_status={prop.gate_status!r}, won't execute"}}

    try:
        payload = json.loads(prop.parsed_payload_json) if prop.parsed_payload_json else {}
    except Exception as exc:
        return {"ok": False, "action_taken": "none",
                "exec_detail": {"error": f"payload unparseable: {exc}"}}

    kind = payload.get("amendment_kind")
    rationale = (payload.get("rationale_short") or "").strip()
    rec_idx = payload.get("recommendation_index", 0)
    options = payload.get("options") or []
    rec = options[rec_idx] if 0 <= rec_idx < len(options) else None

    AMENDABLE_KINDS = {
        "clarification", "scope_narrow", "threshold_tweak",
        "hypothesis_amend", "endpoint_swap", "superseded",
    }

    if kind == "no_action":
        return _mark_finding_ignored(prop.finding_id, rationale, supervisor_id)

    if kind in AMENDABLE_KINDS:
        return _execute_amendment(prop, payload, rec, supervisor_id)

    return {"ok": False, "action_taken": "none",
            "exec_detail": {"error": f"unhandled amendment_kind={kind!r}"}}


def _mark_finding_ignored(finding_id: int, rationale: str, supervisor_id: str) -> Dict[str, Any]:
    with SessionFactory() as s:
        f = s.get(AuditFinding, finding_id)
        if f is None:
            return {"ok": False, "action_taken": "none",
                    "exec_detail": {"error": f"finding {finding_id} not found"}}
        f.status = "IGNORED"
        # Tag rationale with timestamp + supervisor for audit trail
        ts = datetime.datetime.utcnow().isoformat(timespec="seconds")
        existing = (f.notes or "").strip()
        new_note = f"[{ts} approved-as-no_action by {supervisor_id}] {rationale}"
        f.notes = (existing + "\n\n" + new_note).strip() if existing else new_note
        s.commit()
        return {
            "ok":            True,
            "action_taken":  "marked_ignored",
            "exec_detail":   {"finding_id": finding_id, "rationale": rationale[:200]},
        }


def _execute_amendment(prop: AuditProposal,
                       payload: Dict[str, Any],
                       rec: Optional[Dict[str, Any]],
                       supervisor_id: str) -> Dict[str, Any]:
    """
    Call amend_spec for each spec_path appearing in the recommended option's
    files_to_touch. spec_paths must be currently registered (otherwise the
    proposal would have failed the V2 gate or the spec was unregistered
    after gate passed — corner case worth surfacing).
    """
    from engine.preregistration import amend_spec
    from engine.memory import SpecRegistry

    if rec is None:
        return {"ok": False, "action_taken": "none",
                "exec_detail": {"error": "recommended option missing"}}

    kind = payload["amendment_kind"]
    rationale = (payload.get("rationale_short") or "").strip()
    files = rec.get("files_to_touch") or []

    if not rationale or len(rationale) < 20:
        # amend_spec enforces this too, but fail early for clearer error
        return {"ok": False, "action_taken": "none",
                "exec_detail": {"error": "rationale_short < 20 chars; cannot amend_spec"}}

    if not files:
        return {"ok": False, "action_taken": "none",
                "exec_detail": {"error": "recommended option has no files_to_touch"}}

    results: list[Dict[str, Any]] = []
    n_amended = 0
    n_skipped = 0
    n_failed  = 0
    augmented_rationale = (
        f"[auto_audit proposal #{prop.id}, supervisor={supervisor_id}] {rationale}"
    )

    with SessionFactory() as s:
        registered_paths = {
            r.spec_path for r in s.query(SpecRegistry).filter(SpecRegistry.status == "active").all()
        }

    for path in files:
        path_norm = (path or "").strip().replace("\\", "/")
        if not path_norm:
            continue
        if path_norm not in registered_paths:
            results.append({
                "path":   path_norm,
                "status": "skipped_unregistered",
                "note":   "spec not registered; supervisor must register first",
            })
            n_skipped += 1
            continue
        try:
            sid = amend_spec(path_norm, kind=kind, reason=augmented_rationale[:500])
            results.append({"path": path_norm, "status": "amended", "spec_id": sid})
            n_amended += 1
        except Exception as exc:
            logger.exception("auto_audit_executor: amend_spec failed for %s", path_norm)
            results.append({"path": path_norm, "status": "amend_failed", "error": str(exc)[:200]})
            n_failed += 1

    # Mark finding RESOLVED on at least one successful amend
    if n_amended > 0:
        with SessionFactory() as s:
            f = s.get(AuditFinding, prop.finding_id)
            if f is not None:
                f.status = "RESOLVED"
                s.commit()

    overall_ok = (n_failed == 0 and n_amended > 0)
    return {
        "ok":            overall_ok,
        "action_taken":  f"amended {n_amended} / failed {n_failed} / skipped {n_skipped}",
        "exec_detail":   {
            "amendment_kind": kind,
            "results":        results,
            "supervisor_id":  supervisor_id,
        },
    }
