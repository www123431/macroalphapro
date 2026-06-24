"""
P-AUDIT v1 M3 trader-workstation backend (clarification amendment 2026-05-04).

Three deterministic functions for multi-alert workflow:

    get_throughput_today(today=None) -> dict
        Today's already-resolved + pending counts + nearest-deadline.

    check_alert_can_batch(approval_id) -> tuple[bool, str]
        Red-line check: does this alert allow batch approval, or must it be
        resolved with an independent rationale?
        Returns (can_batch, reason). reason is empty when can_batch=True.

    bulk_resolve_pending_approvals(
        approval_ids, approved, resolved_by, review_rationale, review_category,
    ) -> dict
        Loop resolve_pending_approval per id with shared rationale + category.
        Each row gets independent DB write per CFA GIPS §III.A.18 (shared input
        for UX, not DB merge). Skips ids that fail check_alert_can_batch.

Cross-references:
- Spec § F-pre M2 red-line list (HARKing / contradicts_quant=True / priority∈{critical,urgent})
- engine.memory.resolve_pending_approval (single-row resolver)
- feedback_no_llm_as_judge.md (rationale must be supervisor-typed)
"""
from __future__ import annotations

import datetime
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Throughput counter
# ─────────────────────────────────────────────────────────────────────────────

def get_throughput_today(today: datetime.date | None = None) -> dict:
    """
    Today's resolved + pending counts. Returns:
        n_approved_today / n_rejected_today / n_pending /
        nearest_deadline_days / nearest_deadline_id
    """
    from engine.memory import PendingApproval, SessionFactory
    from sqlalchemy import func

    if today is None:
        today = datetime.date.today()

    with SessionFactory() as s:
        n_approved = (
            s.query(func.count(PendingApproval.id))
             .filter(PendingApproval.status == "approved")
             .filter(func.date(PendingApproval.resolved_at) == today)
             .scalar()
        ) or 0
        n_rejected = (
            s.query(func.count(PendingApproval.id))
             .filter(PendingApproval.status == "rejected")
             .filter(func.date(PendingApproval.resolved_at) == today)
             .scalar()
        ) or 0
        n_pending = (
            s.query(func.count(PendingApproval.id))
             .filter(PendingApproval.status == "pending")
             .scalar()
        ) or 0

        nearest = (
            s.query(PendingApproval.id, PendingApproval.approval_deadline)
             .filter(PendingApproval.status == "pending")
             .filter(PendingApproval.approval_deadline.isnot(None))
             .order_by(PendingApproval.approval_deadline.asc())
             .first()
        )
        if nearest:
            nid, ndeadline = int(nearest[0]), nearest[1]
            try:
                days = (ndeadline - today).days if ndeadline else None
            except Exception:
                days = None
        else:
            nid, days = None, None

    return {
        "as_of_date":              today.isoformat(),
        "n_approved_today":        int(n_approved),
        "n_rejected_today":        int(n_rejected),
        "n_pending":               int(n_pending),
        "nearest_deadline_id":     nid,
        "nearest_deadline_days":   days,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch eligibility check (red-line guard)
# ─────────────────────────────────────────────────────────────────────────────

# Red lines: these conditions force independent rationale (cannot batch).
_HIGH_STAKE_PRIORITIES = ("critical", "urgent")


def check_alert_can_batch(approval_id: int) -> tuple[bool, str]:
    """
    Returns (can_batch, reason).
    can_batch=False when:
      - priority is critical/urgent (high-stake, GIPS §III.A.18 individually documented)
      - contradicts_quant=True (LLM/quant disagreement, must be reasoned individually)
      - active HARKing flag on the governing spec (R1-R4 detected)
    """
    from engine.memory import (
        PendingApproval, HARKingFlag, SessionFactory,
    )

    with SessionFactory() as s:
        pa = s.get(PendingApproval, int(approval_id))
        if pa is None:
            return (False, f"approval {approval_id} not found")

        if (pa.priority or "").lower() in _HIGH_STAKE_PRIORITIES:
            return (False, f"priority={pa.priority} requires independent rationale")

        if bool(pa.contradicts_quant):
            return (False, "contradicts_quant=True requires independent rationale")

        # Best-effort HARKing check via decision linkage
        if pa.watchlist_entry_id:
            try:
                from engine.memory import WatchlistEntry, DecisionLog
                wl = s.get(WatchlistEntry, int(pa.watchlist_entry_id))
                if wl and wl.decision_log_id:
                    dl = s.get(DecisionLog, int(wl.decision_log_id))
                    if dl and dl.spec_hash:
                        from engine.memory import SpecRegistry
                        sr = (
                            s.query(SpecRegistry)
                             .filter(SpecRegistry.current_hash == dl.spec_hash)
                             .first()
                        )
                        if sr:
                            n_active = (
                                s.query(HARKingFlag)
                                 .filter(HARKingFlag.spec_path == sr.spec_path)
                                 .filter(HARKingFlag.resolved_at.is_(None))
                                 .count()
                            )
                            if n_active > 0:
                                return (False, f"governing spec has {n_active} active HARKing flag(s)")
            except Exception:
                pass

    return (True, "")


# ─────────────────────────────────────────────────────────────────────────────
# Bulk resolve
# ─────────────────────────────────────────────────────────────────────────────

def bulk_resolve_pending_approvals(
    approval_ids:     list[int],
    approved:         bool,
    resolved_by:      str,
    review_rationale: str,
    review_category:  str,
) -> dict:
    """
    Resolve multiple PendingApproval rows with a shared rationale + category.

    Per row: still calls engine.memory.resolve_pending_approval() so each row
    fires its own downstream side-effects (WatchlistEntry transitions, position
    zeroing for risk_control, etc.) and audit trail (resolved_at timestamp,
    review_rationale + review_category persisted INDEPENDENTLY per CFA GIPS
    §III.A.18). The "shared" inputs are convenience for the UI; the database
    writes one rationale per row.

    Red-line guard: rows that fail check_alert_can_batch() are SKIPPED with a
    reason; caller must resolve them individually via the per-row UI.

    Returns:
      {
        "submitted":  int,                     # ids the caller passed in
        "resolved":   list[{id, ok, message}], # actually resolved
        "skipped":    list[{id, reason}],      # red-line blocks
      }
    """
    from engine.memory import resolve_pending_approval

    out_resolved: list[dict] = []
    out_skipped:  list[dict] = []

    for aid in approval_ids:
        try:
            ok_batch, reason = check_alert_can_batch(int(aid))
        except Exception as e:
            ok_batch, reason = False, f"check failed: {e}"

        if not ok_batch:
            out_skipped.append({"id": int(aid), "reason": reason})
            continue

        try:
            result = resolve_pending_approval(
                approval_id      = int(aid),
                approved         = bool(approved),
                resolved_by      = resolved_by,
                review_rationale = review_rationale,
                review_category  = review_category,
            )
            out_resolved.append({
                "id":      int(aid),
                "ok":      bool(result.get("ok")),
                "message": str(result.get("message") or ""),
            })
        except Exception as e:
            out_resolved.append({
                "id":      int(aid),
                "ok":      False,
                "message": f"exception: {e}",
            })

    return {
        "submitted": len(approval_ids),
        "resolved":  out_resolved,
        "skipped":   out_skipped,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Daily review markdown export (Tetlock 2015 aggregated retrospective)
# ─────────────────────────────────────────────────────────────────────────────

def compose_daily_review_markdown(today: datetime.date | None = None) -> str:
    """
    Compose a markdown 'analyst daily review' note: aggregates all PendingApproval
    rows resolved today (approved or rejected) with their per-row narrative
    paragraphs + verdict + rationale + category. Plus pending items still open.

    For each row, reuses engine.decision_context.compose_supervisor_narrative
    (deterministic templating, 0 LLM). Output is markdown-formatted, ready for
    paste into a daily review log or supervisor diary.

    No LLM, no API calls — pure SQL aggregation + narrative templating.
    """
    from engine.memory import PendingApproval, SessionFactory
    from engine.approval_context import get_approval_context
    from engine.decision_context import compose_supervisor_narrative
    from sqlalchemy import or_, func

    if today is None:
        today = datetime.date.today()

    with SessionFactory() as s:
        rows_today = (
            s.query(PendingApproval)
             .filter(or_(
                 func.date(PendingApproval.resolved_at) == today,
                 PendingApproval.status == "pending",
             ))
             .order_by(
                 PendingApproval.status.asc(),
                 PendingApproval.resolved_at.asc().nullslast(),
                 PendingApproval.id.asc(),
             )
             .all()
        )
        ids = [int(r.id) for r in rows_today]

    lines: list[str] = []
    lines.append(f"# Daily Approval Review · {today.isoformat()}")
    lines.append("")
    lines.append(
        f"_Generated by Macro Alpha Pro — deterministic narrative templating, "
        f"0 LLM. Per Tetlock 2015 aggregated retrospective principle._"
    )
    lines.append("")

    if not ids:
        lines.append("_No approval activity today._")
        return "\n".join(lines)

    n_approved = 0; n_rejected = 0; n_pending = 0

    for aid in ids:
        try:
            ctx = get_approval_context(aid)
        except Exception:
            continue
        if not ctx.get("found"):
            continue
        base = ctx.get("base", {}) or {}
        dc   = ctx.get("decision_context", {}) or {}
        status = base.get("status") or "pending"
        if status == "approved": n_approved += 1
        elif status == "rejected": n_rejected += 1
        else: n_pending += 1

        try:
            narrative = compose_supervisor_narrative(base, dc)
        except Exception as e:
            narrative = f"_(narrative composer failed: {e})_"

        # Header
        sec = base.get("sector") or "—"
        tk = base.get("ticker") or "—"
        atype = base.get("approval_type") or "—"
        verdict_label = {
            "approved": "APPROVED",
            "rejected": "REJECTED",
            "pending":  "PENDING",
            "expired":  "EXPIRED",
        }.get(status, status.upper())
        lines.append(f"## #{aid} · {atype} · {sec} {tk} · **{verdict_label}**")
        lines.append("")

        # Resolved metadata
        with SessionFactory() as s2:
            row = s2.get(PendingApproval, aid)
            if row is not None:
                if row.resolved_at:
                    lines.append(f"- Resolved at: `{row.resolved_at.isoformat(timespec='minutes')}` by `{row.resolved_by or 'system'}`")
                if row.review_category:
                    lines.append(f"- Category: `{row.review_category}`")
                if row.review_rationale:
                    lines.append(f"- Rationale: {row.review_rationale}")
                if row.rejection_reason and status == "rejected":
                    lines.append(f"- Rejection reason: {row.rejection_reason}")
        lines.append("")
        lines.append(narrative)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary header inserted at top
    summary = (
        f"**Summary**: approved={n_approved} · rejected={n_rejected} · pending={n_pending}\n\n"
    )
    lines.insert(3, summary)
    return "\n".join(lines)
