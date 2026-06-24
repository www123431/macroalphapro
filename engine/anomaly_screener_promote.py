"""
engine/anomaly_screener_promote.py — D4.6 of S6 anomaly_screener (2026-05-05)

Pre-registration: docs/decisions/s6_anomaly_screener_spec_2026-05-05.md
Sibling: D2 HITL slim (docs/decisions/hitl_architecture_audit_2026-05-05.md)

Promotes high-confidence LLM flags to PendingApproval queue with
approval_class='llm_output' + approval_type='anomaly_screener'. They appear
in Operations Governance Queue (D3 tab) for supervisor review.

Promotion rules (avoid swamping supervisor):
  • LLM detector only (rule_baseline_a/b stay as audit data, no queue entry)
  • Confidence Likert ≥ 3 (i.e. ≥ "two converging signals")
  • One promotion per (scan_date, ticker) — no duplicates
  • Max 5 promotions per day to bound supervisor load (10-15 min/day cap)

Isolation invariants enforced:
  • S6 anomaly_screener cannot trigger paper_trading E arm changes
    (only strategy_arm_toggle approval_type can; promotion writes
    approval_type='anomaly_screener' which has no executor in
    resolve_pending_approval).
  • promotion_id back-link stored in AnomalyFlag.pending_approval_id for audit.
"""
from __future__ import annotations

import datetime
import logging

from engine.memory import (
    AnomalyFlag,
    PendingApproval,
    SessionFactory,
)

logger = logging.getLogger(__name__)

# Pre-registered promotion thresholds (centralized in engine/config.py 2026-05-06)
from engine.config import PROMOTE_MIN_LIKERT, PROMOTE_DAILY_CAP


def promote_llm_flags_to_queue(scan_date: datetime.date) -> dict:
    """
    Promote eligible LLM flags from `scan_date` to PendingApproval queue.

    Returns {scan_date, n_eligible, n_promoted, promoted_ids}.
    """
    promoted_ids: list[int] = []
    with SessionFactory() as session:
        candidates = (
            session.query(AnomalyFlag)
            .filter(AnomalyFlag.detector == "llm")
            .filter(AnomalyFlag.scan_date == scan_date)
            .filter(AnomalyFlag.confidence_likert >= PROMOTE_MIN_LIKERT)
            .filter(AnomalyFlag.pending_approval_id.is_(None))
            .order_by(AnomalyFlag.confidence_likert.desc(), AnomalyFlag.id.asc())
            .all()
        )
        n_eligible = len(candidates)

        for flag in candidates[:PROMOTE_DAILY_CAP]:
            # Defensive: dedupe at the queue level too — one PendingApproval per
            # (anomaly_screener, scan_date, ticker)
            existing = (
                session.query(PendingApproval)
                .filter(PendingApproval.approval_type == "anomaly_screener")
                .filter(PendingApproval.triggered_date == scan_date)
                .filter(PendingApproval.ticker == flag.ticker)
                .first()
            )
            if existing:
                # Back-link in case the flag wasn't linked yet
                if not flag.pending_approval_id:
                    flag.pending_approval_id = existing.id
                continue

            triggered_text = (
                f"[{flag.event_class}] {flag.evidence_summary or '(no evidence)'} "
                f"| Likert {flag.confidence_likert}/5"
            )
            pa = PendingApproval(
                approval_type       = "anomaly_screener",
                approval_class      = "llm_output",
                priority            = "high" if flag.confidence_likert >= 4 else "normal",
                sector              = flag.sector,
                ticker              = flag.ticker,
                triggered_condition = triggered_text[:500],
                triggered_date      = scan_date,
                triggered_price     = None,
                suggested_weight    = None,
                position_rank       = None,
                contradicts_quant   = False,
                llm_confidence      = int(flag.confidence_likert * 20),
                # Map Likert 1-5 to 0-100 for back-compat with llm_confidence column
                status              = "pending",
            )
            session.add(pa)
            session.flush()
            flag.pending_approval_id = pa.id
            promoted_ids.append(pa.id)

        session.commit()

    return {
        "scan_date":    str(scan_date),
        "n_eligible":   n_eligible,
        "n_promoted":   len(promoted_ids),
        "promoted_ids": promoted_ids,
        "daily_cap":    PROMOTE_DAILY_CAP,
        "min_likert":   PROMOTE_MIN_LIKERT,
    }


def attach_supervisor_label_to_flag(approval_id: int, useful: bool) -> dict:
    """
    Mirror supervisor's PendingApproval decision back to the AnomalyFlag for
    M2 metric. Called by Operations UI when supervisor resolves an
    anomaly_screener case.
    """
    with SessionFactory() as session:
        pa = session.get(PendingApproval, approval_id)
        if pa is None or pa.approval_type != "anomaly_screener":
            return {"ok": False, "message": "not an anomaly_screener approval"}
        flag = (
            session.query(AnomalyFlag)
            .filter(AnomalyFlag.pending_approval_id == approval_id)
            .first()
        )
        if flag is None:
            return {"ok": False, "message": "no anomaly_flag linked"}
        flag.supervisor_useful   = bool(useful)
        flag.supervisor_label_at = datetime.datetime.utcnow()
        session.commit()
        return {"ok": True, "flag_id": flag.id, "useful": bool(useful)}
