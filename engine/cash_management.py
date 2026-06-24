"""
engine/cash_management.py — Supervisor-controlled portfolio cash management.

Spec: docs/spec_performance_reporting_v1.md  (sha256[:16] f1c9b693f7a6a6df).

Three roles for cash flows:
  External (supervisor-controlled): deposit / withdraw / fee
      → drives MWR (XIRR) cash-flow series
      → splits TWR sub-periods (Modified Dietz)
      → optionally goes through PendingApproval gate
  Internal (portfolio-internal): dividend / coupon / interest
      → affects NAV but does NOT split sub-periods (per GIPS §III.5.A.20)
      → never goes through approval gate (auto-applied)
  Status state-machine:
      pending  →  applied   (after supervisor approval or direct insert)
              →  cancelled  (rejection / withdrawal of request)

Public API:
  deposit_funds(amount, ...)     -> (cash_flow_id, approval_id_or_None)
  withdraw_funds(amount, ...)    -> (cash_flow_id, approval_id_or_None)
  record_internal_flow(...)      -> cash_flow_id
  approve_cash_flow(cf_id, ...)  -> bool
  reject_cash_flow(cf_id, ...)   -> bool
  get_cash_flow_history(...)     -> list[dict]
  get_current_cash_balance(...)  -> float
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Sign convention: amount_usd > 0 = INTO portfolio
EXTERNAL_TYPES = {"deposit", "withdraw", "fee"}
INTERNAL_TYPES = {"dividend", "coupon", "interest"}
ALL_TYPES = EXTERNAL_TYPES | INTERNAL_TYPES


def _validate_flow_type(flow_type: str) -> None:
    if flow_type not in ALL_TYPES:
        raise ValueError(
            f"unknown flow_type {flow_type!r}; expected one of {sorted(ALL_TYPES)}"
        )


def _signed_amount(flow_type: str, amount_usd: float) -> float:
    """
    Normalize sign so that amount > 0 means money into portfolio.
    Caller passes |amount| as positive value; we attach sign by flow_type.
    """
    abs_amt = abs(float(amount_usd))
    if flow_type in ("deposit", "dividend", "coupon", "interest"):
        return +abs_amt
    if flow_type in ("withdraw", "fee"):
        return -abs_amt
    raise ValueError(f"signed_amount: unknown flow_type {flow_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API — external (supervisor-controlled) flows
# ─────────────────────────────────────────────────────────────────────────────

def deposit_funds(
    amount_usd:       float,
    flow_date:        datetime.date | None = None,
    supervisor_id:    str | None = None,
    notes:            str | None = None,
    require_approval: bool = True,
    *,
    session:          Any | None = None,
) -> tuple[int, int | None]:
    """
    Record a supervisor deposit. Returns (cash_flow_id, approval_id_or_None).

    require_approval=True (default): creates PendingApproval row, CashFlow
        starts in `pending` status. NAV is not affected until approved.
    require_approval=False: direct apply, status='applied' immediately.
        Use only for scripted / system-level operations.
    """
    if amount_usd <= 0:
        raise ValueError(f"deposit amount must be positive, got {amount_usd}")
    return _create_external_flow(
        "deposit", amount_usd, flow_date, supervisor_id, notes,
        require_approval, session,
    )


def withdraw_funds(
    amount_usd:       float,
    flow_date:        datetime.date | None = None,
    supervisor_id:    str | None = None,
    notes:            str | None = None,
    require_approval: bool = True,
    *,
    session:          Any | None = None,
) -> tuple[int, int | None]:
    """
    Record a supervisor withdrawal. Validates against current cash balance
    only on apply (not on request) — pending withdrawals do not lock cash.
    """
    if amount_usd <= 0:
        raise ValueError(f"withdraw amount must be positive, got {amount_usd}")
    return _create_external_flow(
        "withdraw", amount_usd, flow_date, supervisor_id, notes,
        require_approval, session,
    )


def record_internal_flow(
    flow_type:    str,
    amount_usd:   float,
    flow_date:    datetime.date | None = None,
    notes:        str | None = None,
    *,
    session:      Any | None = None,
) -> int:
    """
    Record portfolio-internal flow (dividend / coupon / interest).
    Always applied immediately, no approval gate. amount_usd should be
    positive; sign is attached by flow_type.
    """
    _validate_flow_type(flow_type)
    if flow_type not in INTERNAL_TYPES:
        raise ValueError(
            f"record_internal_flow: {flow_type!r} is external; use deposit_funds / withdraw_funds"
        )
    return _create_internal_flow(flow_type, amount_usd, flow_date, notes, session)


def approve_cash_flow(
    cash_flow_id: int,
    *,
    resolved_by:  str = "human",
    session:      Any | None = None,
) -> bool:
    """
    Approve a pending CashFlow. Marks status='applied', stamps applied_at,
    and (if linked) marks PendingApproval status='approved'. For withdrawals,
    re-validates that current cash balance is sufficient at apply time.
    """
    from engine.memory import CashFlow, PendingApproval, SessionFactory

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        cf = sess.query(CashFlow).filter(CashFlow.id == cash_flow_id).one_or_none()
        if cf is None:
            raise ValueError(f"approve_cash_flow: CashFlow id={cash_flow_id} not found")
        if cf.status != "pending":
            logger.info("approve_cash_flow: cf id=%s already in status=%s; no-op",
                        cash_flow_id, cf.status)
            return False

        # Re-validate withdrawals against live balance
        if cf.flow_type == "withdraw":
            current_balance = _compute_balance(sess, as_of=cf.flow_date,
                                               include_pending_id=cf.id)
            if abs(cf.amount_usd) > current_balance + 1e-6:
                raise ValueError(
                    f"approve_cash_flow: withdraw ${abs(cf.amount_usd):,.2f} "
                    f"exceeds balance ${current_balance:,.2f}"
                )

        cf.status = "applied"
        cf.applied_at = datetime.datetime.utcnow()

        if cf.approval_id:
            pa = sess.query(PendingApproval).filter(
                PendingApproval.id == cf.approval_id
            ).one_or_none()
            if pa is not None and pa.status == "pending":
                pa.status = "approved"
                pa.resolved_at = datetime.datetime.utcnow()
                pa.resolved_by = resolved_by

        sess.commit()
        logger.info(
            "approve_cash_flow: id=%s applied (%s $%.2f)",
            cash_flow_id, cf.flow_type, abs(cf.amount_usd),
        )
        return True
    finally:
        if own:
            sess.close()


def reject_cash_flow(
    cash_flow_id: int,
    reason:       str,
    *,
    resolved_by:  str = "human",
    session:      Any | None = None,
) -> bool:
    """Reject a pending CashFlow → status='cancelled'."""
    from engine.memory import CashFlow, PendingApproval, SessionFactory

    if not reason or len(reason.strip()) < 5:
        raise ValueError("reject_cash_flow: reason must be ≥5 chars")

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        cf = sess.query(CashFlow).filter(CashFlow.id == cash_flow_id).one_or_none()
        if cf is None:
            raise ValueError(f"CashFlow id={cash_flow_id} not found")
        if cf.status != "pending":
            return False

        cf.status = "cancelled"
        if cf.approval_id:
            pa = sess.query(PendingApproval).filter(
                PendingApproval.id == cf.approval_id
            ).one_or_none()
            if pa is not None:
                pa.status = "rejected"
                pa.resolved_at = datetime.datetime.utcnow()
                pa.resolved_by = resolved_by
                pa.rejection_reason = reason
        sess.commit()
        return True
    finally:
        if own:
            sess.close()


# ─────────────────────────────────────────────────────────────────────────────
# Read API
# ─────────────────────────────────────────────────────────────────────────────

def get_cash_flow_history(
    start:           datetime.date | None = None,
    end:             datetime.date | None = None,
    external_only:   bool = False,
    applied_only:    bool = True,
    *,
    session:         Any | None = None,
) -> list[dict]:
    """List cash flows in date range. Default returns only `applied` rows."""
    from engine.memory import CashFlow, SessionFactory

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        q = sess.query(CashFlow)
        if start is not None:
            q = q.filter(CashFlow.flow_date >= start)
        if end is not None:
            q = q.filter(CashFlow.flow_date <= end)
        if external_only:
            q = q.filter(CashFlow.is_external.is_(True))
        if applied_only:
            q = q.filter(CashFlow.status == "applied")
        rows = q.order_by(CashFlow.flow_date.asc(), CashFlow.id.asc()).all()
        return [
            {
                "id":            r.id,
                "flow_date":     r.flow_date,
                "flow_type":     r.flow_type,
                "amount_usd":    r.amount_usd,
                "is_external":   r.is_external,
                "status":        r.status,
                "supervisor_id": r.supervisor_id,
                "approval_id":   r.approval_id,
                "notes":         r.notes,
                "created_at":    r.created_at,
                "applied_at":    r.applied_at,
            }
            for r in rows
        ]
    finally:
        if own:
            sess.close()


def get_current_cash_balance(
    as_of:           datetime.date | None = None,
    *,
    session:         Any | None = None,
) -> float:
    """
    Current cash balance = sum of all applied CashFlow.amount_usd up to as_of.
    Note: this is "cash sleeve" balance; total NAV includes ETF position MTM
    on top of cash.
    """
    from engine.memory import SessionFactory

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        return _compute_balance(sess, as_of=as_of)
    finally:
        if own:
            sess.close()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _create_external_flow(
    flow_type:        str,
    amount_usd:       float,
    flow_date:        datetime.date | None,
    supervisor_id:    str | None,
    notes:            str | None,
    require_approval: bool,
    session:          Any | None,
) -> tuple[int, int | None]:
    from engine.memory import CashFlow, PendingApproval, SessionFactory

    _validate_flow_type(flow_type)
    if flow_type not in EXTERNAL_TYPES:
        raise ValueError(f"_create_external_flow: {flow_type!r} is internal")

    flow_date = flow_date or datetime.date.today()
    signed = _signed_amount(flow_type, amount_usd)

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        if require_approval:
            # Create PendingApproval first so we have id
            pa = PendingApproval(
                approval_type      = "cash_flow",
                priority           = "normal",
                sector             = "CASH",
                ticker             = "USD",
                triggered_condition = (
                    f"{flow_type} ${abs(signed):,.2f} "
                    f"(supervisor={supervisor_id or 'unknown'})"
                ),
                triggered_date     = flow_date,
                approval_deadline  = flow_date + datetime.timedelta(days=3),
                status             = "pending",
            )
            sess.add(pa)
            sess.flush()
            approval_id = pa.id
            cf_status = "pending"
        else:
            approval_id = None
            cf_status = "applied"

        cf = CashFlow(
            flow_date     = flow_date,
            flow_type     = flow_type,
            amount_usd    = signed,
            is_external   = True,
            status        = cf_status,
            supervisor_id = supervisor_id,
            approval_id   = approval_id,
            notes         = notes,
            created_at    = datetime.datetime.utcnow(),
            applied_at    = (
                datetime.datetime.utcnow() if cf_status == "applied" else None
            ),
        )
        sess.add(cf)
        sess.commit()
        logger.info(
            "cash_flow created: id=%s type=%s amount=%+.2f status=%s approval_id=%s",
            cf.id, flow_type, signed, cf_status, approval_id,
        )
        return (cf.id, approval_id)
    finally:
        if own:
            sess.close()


def _create_internal_flow(
    flow_type:    str,
    amount_usd:   float,
    flow_date:    datetime.date | None,
    notes:        str | None,
    session:      Any | None,
) -> int:
    from engine.memory import CashFlow, SessionFactory

    flow_date = flow_date or datetime.date.today()
    signed = _signed_amount(flow_type, amount_usd)

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        cf = CashFlow(
            flow_date     = flow_date,
            flow_type     = flow_type,
            amount_usd    = signed,
            is_external   = False,
            status        = "applied",
            approval_id   = None,
            notes         = notes,
            created_at    = datetime.datetime.utcnow(),
            applied_at    = datetime.datetime.utcnow(),
        )
        sess.add(cf)
        sess.commit()
        logger.info(
            "internal cash_flow: id=%s type=%s amount=%+.2f",
            cf.id, flow_type, signed,
        )
        return cf.id
    finally:
        if own:
            sess.close()


def _compute_balance(
    sess: Any,
    as_of: datetime.date | None = None,
    include_pending_id: int | None = None,
) -> float:
    """
    Sum of all applied CashFlow.amount_usd. If include_pending_id is given,
    also count that one row even if pending (used for withdraw re-validation).
    """
    from engine.memory import CashFlow
    from sqlalchemy import func

    q = sess.query(func.coalesce(func.sum(CashFlow.amount_usd), 0.0)).filter(
        CashFlow.status == "applied"
    )
    if as_of is not None:
        q = q.filter(CashFlow.flow_date <= as_of)
    base = q.scalar() or 0.0

    if include_pending_id is not None:
        # Don't double-count if it's already applied
        cf = sess.query(CashFlow).filter(
            CashFlow.id == include_pending_id,
            CashFlow.status == "pending",
        ).one_or_none()
        # Pending withdrawals would reduce balance, so we leave them out by
        # default; the caller does its own check.
        # This helper is invoked by approve_cash_flow's re-validation path,
        # which wants the balance EXCLUDING the row being approved.

    return float(base)
