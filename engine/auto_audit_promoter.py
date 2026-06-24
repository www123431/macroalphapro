"""
engine/auto_audit_promoter.py — Promote gate-passed AuditProposal to
PendingApproval queue (R-1.E, 2026-05-06).

Bridge between Layer 2 (R-1.D auto_audit_gate) and the existing supervisor
governance queue. Runs synchronously at the end of `gate_proposal()` when
gate_status='pass'. The supervisor sees the proposal in
pages/operations.py (the canonical approval surface — per
feedback_no_duplicate_ui_for_same_action, all approve/reject buttons live
in one place).

Hash chain integrity: existing PendingApproval lifecycle handles the
review_narrative_snapshot + prev_narrative_hash linkage at resolve time,
so this module only needs to:
  • write a PendingApproval row mapped from AuditProposal payload fields
  • back-link via AuditProposal.pending_approval_id + AuditFinding.status='PROMOTED'

Idempotent: if AuditProposal already has pending_approval_id set, returns
existing.
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict

from engine.auto_audit_models import AuditFinding, AuditProposal
from engine.memory import PendingApproval, SessionFactory

logger = logging.getLogger(__name__)


def promote_to_pending_approval(proposal_id: int) -> Dict[str, Any]:
    """
    Promote a gate_status='pass' AuditProposal to a PendingApproval row.

    Returns {ok, proposal_id, pending_approval_id, priority, skipped?}.
    """
    with SessionFactory() as s:
        prop = s.get(AuditProposal, proposal_id)
        if prop is None:
            return {"ok": False, "proposal_id": proposal_id, "error": "proposal_not_found"}
        if prop.gate_status != "pass":
            return {
                "ok":          False,
                "proposal_id": proposal_id,
                "error":       f"cannot promote: gate_status={prop.gate_status!r}",
            }

        # Idempotency: already promoted?
        if prop.pending_approval_id is not None:
            return {
                "ok":                  True,
                "proposal_id":         proposal_id,
                "pending_approval_id": int(prop.pending_approval_id),
                "skipped":             "already_promoted",
            }

        try:
            payload = json.loads(prop.parsed_payload_json) if prop.parsed_payload_json else {}
        except Exception:
            return {"ok": False, "proposal_id": proposal_id, "error": "payload_unparseable"}

        finding = s.get(AuditFinding, prop.finding_id)
        if finding is None:
            return {"ok": False, "proposal_id": proposal_id, "error": "finding_missing"}

        priority = "high" if prop.governance_required else "normal"
        summary = (payload.get("summary") or "")[:500]
        # Use rationale_short to populate notes — supervisor sees it as
        # context when reviewing in the Operations queue.
        rationale = payload.get("rationale_short") or ""

        # Build the review_narrative_snapshot from the proposal payload —
        # gives the supervisor a one-glance understanding without diving
        # into the AuditProposal table.
        rec_idx = payload.get("recommendation_index", 0)
        opts = payload.get("options") or []
        rec_action = opts[rec_idx]["action"] if 0 <= rec_idx < len(opts) else "(no recommendation)"
        snapshot_text = (
            f"AUTO-AUDIT PROPOSAL #{proposal_id}\n"
            f"Source rule:        {finding.rule_name}\n"
            f"Finding severity:   {finding.severity}\n"
            f"Diagnosis:          {payload.get('diagnosis', '')[:600]}\n"
            f"Recommended action: {rec_action[:300]}\n"
            f"Amendment kind:     {payload.get('amendment_kind', '')}\n"
            f"Governance required: {prop.governance_required}\n"
            f"Rationale (verbatim): {rationale[:500]}"
        )

        # PendingApproval.sector is NOT NULL — sentinel "_audit_" for non-sector
        # approval types. Mirrors anomaly_screener_promote convention.
        pa = PendingApproval(
            approval_type       = "auto_audit_proposal",
            approval_class      = "llm_output",
            priority            = priority,
            sector              = "_audit_",
            ticker              = "_AUDIT",
            triggered_condition = summary,
            triggered_date      = datetime.date.today(),
            triggered_price     = None,
            suggested_weight    = None,
            position_rank       = None,
            contradicts_quant   = False,
            llm_confidence      = None,
            status              = "pending",
            review_narrative_snapshot = snapshot_text,
        )
        s.add(pa)
        s.flush()
        prop.pending_approval_id = pa.id
        finding.status = "PROMOTED"
        finding.pending_approval_id = pa.id
        s.commit()
        logger.info(
            "auto_audit: promoted proposal #%d → PendingApproval #%d (priority=%s, gov=%s)",
            proposal_id, pa.id, priority, prop.governance_required,
        )
        return {
            "ok":                  True,
            "proposal_id":         proposal_id,
            "pending_approval_id": pa.id,
            "priority":            priority,
            "governance_required": bool(prop.governance_required),
        }
