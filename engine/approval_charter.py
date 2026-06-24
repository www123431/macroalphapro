"""engine/approval_charter.py — the HITL governance-inbox charter (2026-05-24).

Single source of truth for ONE question: what is the human approval inbox *for*?

Doctrine (human-ON-the-loop, not human-IN-the-loop):
    A systematic book's edge comes from removing discretionary human gates from
    the trade path. The LIVE 5-strategy book never routes here — it auto-executes
    on schedule and its risk control is the automatic Risk-Manager HARD-HALT
    (skips the day's persist + writes _HALT.json), NOT an inbox item.

    The inbox is therefore reserved for the two things a real systematic shop
    *does* put in front of a human:
      1. actions the human ORIGINATED (a typed CoS directive → propose_action), and
      2. genuine GOVERNANCE / EXCEPTION decisions that are out-of-model-support
         (OOD) and where a wrong automated action is convex/catastrophic —
         model-deployment gates (factor-candidate promotion), universe changes,
         cash flow, audit-promotion review.

    Anything in-distribution and systematic does NOT belong here. The legacy
    sector-overlay's per-stop approvals (`risk_control`) are the one discretionary
    organ that violated this; they are RETIRED → recorded as audit traces, never
    gated to the inbox again (the sector book is dormant — see MEMORY.md).

This module is pure (no DB, no I/O) so it can be imported by the engine writers,
the API serializer, and the tests without side effects.
"""
from __future__ import annotations

import datetime as _dt

# ── Routing predicate ────────────────────────────────────────────────────────

# Types that legitimately belong in the human governance inbox: human-originated
# (advisory = a CoS propose_action) + OOD/convex-loss governance decisions.
GOVERNANCE_INBOX_TYPES: frozenset[str] = frozenset({
    "advisory",             # CoS propose_action — human ORIGINATED the directive
    "overlay",              # CoS propose_action (position) — human-originated discretionary overlay
    "factor_candidate",     # factor passed BH multiple-testing → model-deployment gate
    "universe_change",      # quarterly universe exit/add — capital/universe governance
    "cash_flow",            # external cash in/out — capital-flow governance
    "auto_audit_proposal",  # LLM-proposed audit promotion → human review
    "track_b",              # Track-B research decision → human review
    "anomaly_screener",     # record-only monitor acknowledgment (no trade)
})

# Discretionary-overlay signal types that must NEVER create a blocking inbox item.
# The sector pipeline (LLM-driven sector rotation) is the decommissioned organ;
# its stop signals are recorded as audit traces instead of gated to a human.
RETIRED_DISCRETIONARY_TYPES: frozenset[str] = frozenset({
    "risk_control",         # sector-overlay stop suggestions — retired 2026-05-24
})

# Of all types, only these three are wired to auto-execute on approval in
# resolve_pending_approval(); every other type is record-only (status change +
# audit trail, no book mutation). Pinned here so the UI/API never overstate what
# "Approve" does. `risk_control` stays listed for any *legacy* pending row, but
# no new ones are created (see RETIRED_DISCRETIONARY_TYPES).
EXECUTING_TYPES: frozenset[str] = frozenset({"entry", "risk_control", "rebalance", "overlay"})

# Record-only resolution marks for retired sector stops (kept in History/audit).
SECTOR_RETIRED_RESOLVED_BY: str = "auto_retired_sector"
SECTOR_RETIRED_NOTE: str = (
    "sector overlay retired 2026-05-24 — stop signal recorded as audit trace, "
    "not gated to the inbox (charter: discretionary organ decommissioned)"
)


def is_governance_inbox_item(approval_type: str | None) -> bool:
    """True iff this approval_type legitimately belongs in the human inbox."""
    return (approval_type or "") in GOVERNANCE_INBOX_TYPES


def is_retired_discretionary(approval_type: str | None) -> bool:
    """True iff this type is a retired discretionary-overlay signal (record-only)."""
    return (approval_type or "") in RETIRED_DISCRETIONARY_TYPES


def retired_trace_fields(now: _dt.datetime | None = None) -> dict:
    """Field overrides that turn a would-be `pending` row into a record-only,
    pre-resolved audit trace (routine_review). Spread into PendingApproval(...)."""
    return {
        "status":         "approved",
        "resolved_at":    now or _dt.datetime.utcnow(),
        "resolved_by":    SECTOR_RETIRED_RESOLVED_BY,
        "approval_class": "routine_review",
        "review_rationale": SECTOR_RETIRED_NOTE,
    }


# ── Per-type "what does Approve actually do" effect lines ─────────────────────
# Faithful to resolve_pending_approval: only EXECUTING_TYPES move the book.
_EFFECT: dict[str, dict] = {
    "entry": {
        "executes": True,
        "en": "Approve → open position at the suggested weight.",
        "zh": "批准 → 按建议权重建仓（设为活跃持仓）。",
    },
    "risk_control": {
        "executes": True,
        "en": "Approve → stop-out: zero this position (sell to 0). [legacy — retired path]",
        "zh": "批准 → 止损清仓：该持仓权重清零并记一笔卖出。（遗留项 — 通路已退役）",
    },
    "rebalance": {
        "executes": True,
        "en": "Approve → run a full-portfolio rebalance.",
        "zh": "批准 → 跑一次全组合再平衡。",
    },
    "overlay": {
        "executes": True,
        "en": "Approve → set this position in your discretionary OVERLAY sleeve (separate "
              "from the systematic book; RM-cap validated). Reject → nothing changes.",
        "zh": "批准 → 在你的自由裁量【叠加层 overlay】里建/调该仓位（独立于系统化主账本，"
              "经 RM 限额校验）。拒绝 → 不变。",
    },
    "advisory": {
        "executes": False,
        "en": "Record-only → CoS proposal; approving logs your decision, no trade.",
        "zh": "仅记录 → CoS 提案；批准只留痕，不动仓位。",
    },
    "anomaly_screener": {
        "executes": False,
        "en": "Record-only → acknowledge the anomaly-monitor flag; no trade.",
        "zh": "仅记录 → 确认异常监控标记；不动仓位。",
    },
    "auto_audit_proposal": {
        "executes": False,
        "en": "Governance record → mark the audit proposal reviewed; no trade.",
        "zh": "治理记录 → 标记审计提案已审阅；不动仓位。",
    },
    "factor_candidate": {
        "executes": False,
        "en": "Governance → record your verdict on this factor candidate (model-deployment gate; no auto-deploy).",
        "zh": "治理决策 → 记录对该因子候选的裁决（模型部署 gate；不自动入册）。",
    },
    "universe_change": {
        "executes": False,
        "en": "Governance → record your verdict on this universe change; no auto-edit.",
        "zh": "治理决策 → 记录对 universe 调整的裁决；不自动改动。",
    },
    "cash_flow": {
        "executes": False,
        "en": "Governance record → log the cash-flow verdict; no trade.",
        "zh": "治理记录 → 现金流裁决留痕；不动仓位。",
    },
    "track_b": {
        "executes": False,
        "en": "Governance record → log the Track-B decision; no trade.",
        "zh": "治理记录 → Track B 决策留痕；不动仓位。",
    },
}
_EFFECT_DEFAULT = {
    "executes": False,
    "en": "Record-only → approving logs your decision; no automatic execution.",
    "zh": "仅记录 → 批准只留痕；不自动执行。",
}


def approval_effect(approval_type: str | None) -> dict:
    """Return {'executes': bool, 'en': str, 'zh': str} — what Approve actually does.

    The single honest answer to "减仓还是干什么": only entry/risk_control/rebalance
    move the book; everything else is record-only governance.
    """
    return dict(_EFFECT.get(approval_type or "", _EFFECT_DEFAULT))


# ── Charter prose (inbox banner) ──────────────────────────────────────────────
CHARTER = {
    "en": (
        "Governance & exception queue. The live 5-strategy systematic book does "
        "NOT route here — it auto-executes on schedule and its risk control is the "
        "automatic Risk-Manager HARD-HALT, not an inbox item. This queue is only "
        "for actions you originated (CoS proposals) and out-of-model governance "
        "decisions (model-deployment gates, universe/cash/audit review). You decide; "
        "the engine executes. The LLM never auto-approves."
    ),
    "zh": (
        "治理与例外队列。实跑的 5 策略系统化主账本不走这里 —— 它按计划自动执行，"
        "风控是自动的 Risk-Manager HARD-HALT，不是信箱项。本队列只接两类：你主动"
        "发起的动作（CoS 提案）+ 模型够不到的治理决策（模型部署 gate、universe/现金/"
        "审计审阅）。你决定，引擎执行；LLM 绝不自动审批。"
    ),
}
