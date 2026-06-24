"""
engine/auto_audit_models.py — ORM tables for the Auto-Audit Loop (R-1.A 2026-05-06)

Persistence layer so weekly drift comparisons (R-1.B), LLM proposals (R-1.C),
and PendingApproval back-links (R-1.E) all share one history.

Schema choices:
  • Reuses the project's single Base from engine/db_models so init_db()'s
    create_all picks these tables up automatically — no separate migration.
  • proposal_id / pending_approval_id are nullable Integer (no FK constraint)
    because the target tables (AuditProposal R-1.C; PendingApproval) live in
    other modules and we don't want a circular import. Application code
    enforces the link.
  • snapshot_json holds a JSON-serialised dict describing the contradiction
    in enough detail for an LLM to reason over it without re-running the
    rule.
"""
from __future__ import annotations

import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index, Integer, String, Text,
)

from engine.db_models import Base


class AuditRun(Base):
    """One execution of the audit orchestrator (one cron tick)."""
    __tablename__ = "auto_audit_runs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_at       = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    scope        = Column(String(20), nullable=False)   # 'critical' | 'weekly'
    n_rules_run   = Column(Integer, default=0)
    n_findings    = Column(Integer, default=0)
    n_errors      = Column(Integer, default=0)
    n_suppressed  = Column(Integer, default=0)          # silenceable mechanism (R-1.B.3)
    duration_sec  = Column(Float, default=0.0)
    exit_status   = Column(String(20), default="ok")    # 'ok' | 'partial' | 'error' | 'no_rules'


class AuditFinding(Base):
    """One contradiction detected by one rule in one run."""
    __tablename__ = "auto_audit_findings"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    run_id              = Column(Integer, nullable=False)         # FK -> auto_audit_runs.id
    rule_name           = Column(String(100), nullable=False)
    severity            = Column(String(10),  nullable=False)     # 'LOW' | 'MID' | 'HIGH'
    detected_at         = Column(DateTime, default=datetime.datetime.utcnow)
    snapshot_json       = Column(Text, nullable=False)            # JSON contradiction detail
    proposal_id         = Column(Integer, nullable=True)          # FK -> auto_audit_proposals.id (R-1.C)
    pending_approval_id = Column(Integer, nullable=True)          # FK -> pending_approvals.id (R-1.E)
    status              = Column(String(20), default="OPEN")      # OPEN|PROPOSED|PROMOTED|RESOLVED|IGNORED
    notes               = Column(Text, nullable=True)             # IGNORE rationale (R-1.E ≥20 chars enforced in UI)


class AuditProposal(Base):
    """
    LLM-generated remediation proposal for one AuditFinding (R-1.C, 2026-05-06).

    Lifecycle:
      generation_status: 'pending' → 'success' | 'generation_failed' | 'deferred_quota'
      gate_status:       NULL      → 'pending' (R-1.D set) → 'pass' | 'fail'

    One proposal per finding (unique constraint). Re-running the proposer on
    a finding that already has a row is a no-op — see scripts/run_auto_audit_proposals.

    raw_response_text + parsed_payload_json + prompt_hash form the
    reproducibility triple: given the same inputs + model_version, the
    proposer should produce a cache-equivalent output (Gemini temperature=0).
    """
    __tablename__ = "auto_audit_proposals"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    finding_id              = Column(Integer, nullable=False, unique=True)  # FK -> auto_audit_findings.id
    generated_at            = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    model_version           = Column(String(50), nullable=False)
    prompt_hash             = Column(String(64), nullable=False)
    input_tokens            = Column(Integer, default=0)
    output_tokens           = Column(Integer, default=0)
    cost_usd                = Column(Float,   default=0.0)

    raw_response_text       = Column(Text, nullable=True)
    parsed_payload_json     = Column(Text, nullable=True)

    generation_status       = Column(String(20), default="pending")
    failure_reason          = Column(Text, nullable=True)

    # Layer 2 (R-1.D) populates these:
    gate_status             = Column(String(20), nullable=True)   # 'pending' | 'pass' | 'fail'
    gate_failure_reasons_json = Column(Text, nullable=True)
    governance_required     = Column(Boolean, default=False, nullable=False)

    # Layer 2/3 (R-1.E) populates this when promoter writes the PA row:
    pending_approval_id     = Column(Integer, nullable=True)      # FK -> pending_approvals.id


# Indexes for the UI side (R-1.E pages/auto_audit.py): browse latest runs by
# scope + drill into a single run's findings without table scan.
Index("ix_auto_audit_runs_scope_run_at", AuditRun.scope, AuditRun.run_at.desc())
Index("ix_auto_audit_findings_run_id",   AuditFinding.run_id)
Index("ix_auto_audit_findings_status",   AuditFinding.status)
Index("ix_auto_audit_proposals_finding_id", AuditProposal.finding_id, unique=True)
Index("ix_auto_audit_proposals_generation_status", AuditProposal.generation_status)
Index("ix_auto_audit_proposals_gate_status", AuditProposal.gate_status)
