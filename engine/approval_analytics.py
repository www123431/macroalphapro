"""
P-AUDIT v1 — Cross-time supervisor approval analytics (3d).

Spec: docs/spec_supervisor_approval_panel_v1.md (forward-registered 2026-05-04).

Three deterministic analytics functions:

    get_approval_rate_by_period(group="month")     -> pd.DataFrame
    get_category_outcome_correlation(min_n=5)      -> pd.DataFrame
    get_supervisor_override_pattern()              -> dict

All values are SQL aggregations over PendingApproval × DecisionLog.
Insufficient-data guard: any cell with n < min_n surfaces NaN/null;
caller / UI is expected to render "insufficient data" placeholder.

0 LLM. No RAG. Pure groupby on persisted columns.
"""
from __future__ import annotations

import datetime
from typing import Any

import pandas as pd


_VALID_GROUPS = ("day", "week", "month", "quarter")


# ─────────────────────────────────────────────────────────────────────────────
# 3d.1  Approval rate over time
# ─────────────────────────────────────────────────────────────────────────────

def get_approval_rate_by_period(
    group: str = "month",
    *,
    session: Any | None = None,
) -> pd.DataFrame:
    """
    Per-period rollup. Columns:
        period   pd.Period            indexed by group
        n_total  int                  rows in the period
        n_approved int
        n_rejected int
        n_expired  int
        n_pending  int
        approval_rate  float          n_approved / max(1, n_resolved)
        rejection_rate float          n_rejected / max(1, n_resolved)
    """
    from engine.memory import PendingApproval, SessionFactory

    if group not in _VALID_GROUPS:
        raise ValueError(f"group must be one of {_VALID_GROUPS}; got {group!r}")
    period_alias = {"day": "D", "week": "W", "month": "M", "quarter": "Q"}[group]

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        rows = (
            sess.query(
                PendingApproval.triggered_date,
                PendingApproval.status,
            ).all()
        )
    finally:
        if own:
            sess.close()

    if not rows:
        return pd.DataFrame(
            columns=["period", "n_total", "n_approved", "n_rejected",
                     "n_expired", "n_pending", "approval_rate", "rejection_rate"],
        )

    df = pd.DataFrame(rows, columns=["triggered_date", "status"])
    df["triggered_date"] = pd.to_datetime(df["triggered_date"])
    df["period"] = df["triggered_date"].dt.to_period(period_alias)

    pivot = (
        df.groupby("period")["status"]
          .value_counts()
          .unstack(fill_value=0)
    )
    for col in ("approved", "rejected", "expired", "pending"):
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot.rename(
        columns={
            "approved": "n_approved",
            "rejected": "n_rejected",
            "expired":  "n_expired",
            "pending":  "n_pending",
        }
    )

    pivot["n_total"]    = (
        pivot["n_approved"] + pivot["n_rejected"]
        + pivot["n_expired"] + pivot["n_pending"]
    )
    n_resolved = pivot["n_approved"] + pivot["n_rejected"] + pivot["n_expired"]
    pivot["approval_rate"]  = pivot["n_approved"] / n_resolved.replace({0: pd.NA})
    pivot["rejection_rate"] = pivot["n_rejected"] / n_resolved.replace({0: pd.NA})

    pivot = pivot.reset_index()
    pivot["period"] = pivot["period"].astype(str)
    return pivot[
        ["period", "n_total", "n_approved", "n_rejected",
         "n_expired", "n_pending", "approval_rate", "rejection_rate"]
    ].sort_values("period")


# ─────────────────────────────────────────────────────────────────────────────
# 3d.2  review_category × outcome correlation
# ─────────────────────────────────────────────────────────────────────────────

def get_category_outcome_correlation(
    min_n: int = 5,
    *,
    session: Any | None = None,
) -> pd.DataFrame:
    """
    For each review_category × verdict bucket with n ≥ min_n:
        review_category       str
        verdict               approved/rejected
        n                     int
        hit_rate              float  (active_return > 0 share)
        mean_active_return    float  (active_return mean across linked decisions)
        median_dd             float  (median MAE)

    Cells with n < min_n drop to NaN to surface "insufficient data".
    """
    from engine.memory import PendingApproval, WatchlistEntry, DecisionLog, SessionFactory

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        joined = (
            sess.query(
                PendingApproval.review_category,
                PendingApproval.status,
                DecisionLog.active_return,
                DecisionLog.mae,
                DecisionLog.accuracy_score,
            )
            .join(
                WatchlistEntry,
                WatchlistEntry.id == PendingApproval.watchlist_entry_id,
                isouter=True,
            )
            .join(
                DecisionLog,
                DecisionLog.id == WatchlistEntry.decision_log_id,
                isouter=True,
            )
            .filter(PendingApproval.status.in_(("approved", "rejected")))
            .all()
        )
    finally:
        if own:
            sess.close()

    if not joined:
        return pd.DataFrame(
            columns=["review_category", "verdict", "n",
                     "hit_rate", "mean_active_return", "median_dd"]
        )

    df = pd.DataFrame(
        joined,
        columns=["review_category", "verdict",
                 "active_return", "mae", "accuracy"],
    )
    df["review_category"] = df["review_category"].fillna("(unspecified)")

    grouped = df.groupby(["review_category", "verdict"])
    out_rows: list[dict] = []
    for (cat, verdict), g in grouped:
        n = len(g)
        if n < min_n:
            out_rows.append({
                "review_category":  cat,
                "verdict":          verdict,
                "n":                n,
                "hit_rate":         None,
                "mean_active_return": None,
                "median_dd":        None,
            })
            continue

        non_null_ret = g["active_return"].dropna()
        non_null_mae = g["mae"].dropna()
        hit_rate = (
            float((non_null_ret > 0).mean()) if not non_null_ret.empty else None
        )
        mean_ret = (
            float(non_null_ret.mean()) if not non_null_ret.empty else None
        )
        median_dd = (
            float(non_null_mae.median()) if not non_null_mae.empty else None
        )
        out_rows.append({
            "review_category":  cat,
            "verdict":          verdict,
            "n":                n,
            "hit_rate":         hit_rate,
            "mean_active_return": mean_ret,
            "median_dd":        median_dd,
        })

    return pd.DataFrame(out_rows).sort_values(["review_category", "verdict"])


# ─────────────────────────────────────────────────────────────────────────────
# 3d.3  Supervisor override pattern
# ─────────────────────────────────────────────────────────────────────────────

def get_supervisor_override_pattern(
    *,
    session: Any | None = None,
) -> dict:
    """
    Surface "where supervisor approves contrarian LLM suggestions".

    Filter to approvals where contradicts_quant=True, then count:
        n_total                  total contradicts_quant rows
        n_approved               of those, status=approved
        n_rejected               of those, status=rejected
        approval_rate            n_approved / (n_approved + n_rejected)
        hit_rate_when_approved   among approved rows, share with linked
                                 DecisionLog.accuracy_score >= 0.5
    """
    from engine.memory import PendingApproval, WatchlistEntry, DecisionLog, SessionFactory

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        rows = (
            sess.query(
                PendingApproval.id,
                PendingApproval.status,
                DecisionLog.accuracy_score,
            )
            .join(
                WatchlistEntry,
                WatchlistEntry.id == PendingApproval.watchlist_entry_id,
                isouter=True,
            )
            .join(
                DecisionLog,
                DecisionLog.id == WatchlistEntry.decision_log_id,
                isouter=True,
            )
            .filter(PendingApproval.contradicts_quant.is_(True))
            .all()
        )
    finally:
        if own:
            sess.close()

    n_total = len(rows)
    n_approved = sum(1 for _, st, _ in rows if st == "approved")
    n_rejected = sum(1 for _, st, _ in rows if st == "rejected")

    approved_with_acc = [acc for _, st, acc in rows if st == "approved" and acc is not None]
    hit_rate_when_approved = (
        float(sum(1 for a in approved_with_acc if a >= 0.5) / len(approved_with_acc))
        if approved_with_acc else None
    )

    n_resolved = n_approved + n_rejected
    return {
        "n_total":                n_total,
        "n_approved":             n_approved,
        "n_rejected":             n_rejected,
        "approval_rate":          (n_approved / n_resolved) if n_resolved else None,
        "hit_rate_when_approved": hit_rate_when_approved,
        "n_with_outcome":         len(approved_with_acc),
    }
