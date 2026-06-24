"""api/routes_strengthener.py — Phase 2.0 step 12 backend.

Read-only + resolve endpoints over B's verdicts.jsonl. Parallel to
the legacy /api/approvals (ticker-level) and /api/governance/approvals
(deploy decisions) — each surface has its own typed shape because
the underlying decisions have different fields and lifecycles.

GET  /api/strengthener/approvals?include_resolved=false
     → list of pending APPROVE_FOR_PIPELINE + DOCTRINE_AMENDMENT_NEEDED
       verdicts the principal hasn't decided on yet.

POST /api/strengthener/approvals/resolve
     {hypothesis_id, decision, rationale}
     → append a resolution row. decision ∈ {approved, rejected, deferred}.

Resolution is the only state change here — the runner is OFFLINE
(scripts/run_strengthener.py). This API does NOT trigger downstream
pipelines on `approved`; the principal still chooses whether to actually
run F14b strict-gate on the candidate.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/strengthener", tags=["strengthener"])


# ──────────────────────────────────────────────────────────────────────
# Response shape
# ──────────────────────────────────────────────────────────────────────
class StrengthenerApprovalRow(BaseModel):
    hypothesis_id:               str
    verdict_type:                str
    one_line_summary:            str
    confidence:                  float
    reasoning:                   str
    similar_to_deployed:         Optional[str] = None
    replaces_decaying:           Optional[str] = None
    blocking_doctrine_id:        Optional[str] = None
    proposed_amendment_summary:  Optional[str] = None
    recommended_pipeline_action: Optional[str] = None
    risk_flags:                  list[str]
    review_ts:                   str
    model:                       str
    resolved:                    bool
    resolution:                  Optional[dict] = None


class StrengthenerApprovalsDigest(BaseModel):
    n_pending:  int
    n_resolved: int
    rows:       list[StrengthenerApprovalRow]


@router.get("/approvals", response_model=StrengthenerApprovalsDigest)
def list_approvals(include_resolved: bool = Query(False)):
    """Return pending strengthener verdicts. Pending = APPROVE_FOR_PIPELINE
    or DOCTRINE_AMENDMENT_NEEDED that the principal hasn't decided on.

    REJECT verdicts are NOT surfaced — B already decided, no human
    action needed.
    """
    try:
        from engine.agents.strengthener.approval_view import list_pending_approvals
        return list_pending_approvals(include_resolved=include_resolved)
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"strengthener approvals read failed: {exc}")


# ──────────────────────────────────────────────────────────────────────
# Resolve
# ──────────────────────────────────────────────────────────────────────
class ResolveRequest(BaseModel):
    hypothesis_id: str
    decision:      str        # approved / rejected / deferred
    rationale:     str = ""


class ResolveResponse(BaseModel):
    status:        str        # "ok"
    hypothesis_id: str
    decision:      str
    resolved_ts:   str


def _emit_forward_vector_created(hypothesis_id: str, verdict: dict) -> None:
    """Phase 2.1a handler: B's APPROVE_FOR_PIPELINE verdict approved →
    hypothesis enters /research/forward queue via the dual-track
    generator (P2.1b reads forward_vector_created events to gate
    LLM_SYNTHESIS-rooted forward vectors)."""
    from engine.agents.strengthener.approval_view import find_hypothesis_family
    from engine.research_store import emit
    try:
        from engine.research_store.hypothesis.store import find_by_id
        h = find_by_id(hypothesis_id)
        em = h.extraction_method.value if h else "unknown"
    except Exception:
        em = "unknown"
    emit.forward_vector_created(
        hypothesis_id     = hypothesis_id,
        verdict_type      = verdict.get("verdict_type", "APPROVE_FOR_PIPELINE"),
        b_confidence      = float(verdict.get("confidence", 0.5)),
        extraction_method = em,
        mechanism_family  = find_hypothesis_family(hypothesis_id),
    )


def _auto_extract_factor_spec(hypothesis_id: str) -> None:
    """Tier C-2d hook: after B's APPROVE_FOR_PIPELINE is human-
    approved, automatically run the factor_spec_extractor and
    persist the resulting SPEC as a NEW pending row in
    factor_specs.jsonl. The principal then reviews the SPEC in
    /approvals and explicitly approves IT before any dispatch
    runs (preserves the spec-approval gate per spec C-2 point 5).

    Failure here MUST NOT block the resolution row write — the
    principal's APPROVE on the hypothesis stands regardless. Just
    log + carry on; user can manually re-extract from a script if
    the auto-extract bug-out.

    Non-factor hypotheses (procedural, methodology, no provenance)
    silently no-op via is_factor_hypothesis gate in the extractor.
    """
    try:
        from engine.agents.strengthener.approval_view import (
            find_hypothesis_family,
        )
        from engine.agents.strengthener.factor_spec_store import (
            extract_and_persist_pending,
        )
        from engine.research_store.hypothesis.store import find_by_id
        h = find_by_id(hypothesis_id)
        if h is None:
            logger.warning(
                "_auto_extract_factor_spec: hypothesis %s not found",
                hypothesis_id,
            )
            return
        family = find_hypothesis_family(hypothesis_id) or "OTHER"
        sh = extract_and_persist_pending(h, family_hint=family)
        if sh:
            logger.info(
                "_auto_extract_factor_spec: extracted SPEC spec_hash=%s "
                "for hyp=%s (family=%s) → now pending in /approvals",
                sh, hypothesis_id, family,
            )
        else:
            logger.info(
                "_auto_extract_factor_spec: no SPEC extracted for hyp=%s "
                "(ineligible or extractor returned None)",
                hypothesis_id,
            )
    except Exception:
        logger.exception(
            "_auto_extract_factor_spec: failed for %s", hypothesis_id,
        )


def _handle_amendment_approval(hypothesis_id: str, verdict: dict) -> None:
    """Phase 2.0 step 13 handler: B's DOCTRINE_AMENDMENT_NEEDED verdict
    approved → write a draft amendment markdown file + emit
    memory_amendment_proposed.

    NOT auto-edits the memory file. Memory file edits go through
    Claude's Write tool (per project doctrine — autonomous file
    mutation banned). The draft is parked for the principal to apply
    manually.
    """
    from engine.agents.strengthener.approval_view import write_amendment_draft
    from engine.research_store import emit

    blocking = verdict.get("blocking_doctrine_id") or ""
    if not blocking:
        logger.warning(
            "_handle_amendment_approval: verdict missing blocking_doctrine_id "
            "for hyp %s; skipping draft + emit", hypothesis_id,
        )
        return

    draft_path = write_amendment_draft(
        hypothesis_id              = hypothesis_id,
        blocking_doctrine_id       = blocking,
        proposed_amendment_summary = verdict.get("proposed_amendment_summary") or "",
        b_reasoning                = verdict.get("reasoning") or "",
        b_confidence               = float(verdict.get("confidence", 0.5)),
    )

    emit.memory_amendment_proposed(
        hypothesis_id              = hypothesis_id,
        blocking_doctrine_id       = blocking,
        proposed_amendment_summary = verdict.get("proposed_amendment_summary") or "",
        b_reasoning                = verdict.get("reasoning") or "",
        draft_doc_path             = str(draft_path),
        b_confidence               = float(verdict.get("confidence", 0.5)),
    )


@router.post("/approvals/resolve", response_model=ResolveResponse)
def resolve_approval(req: ResolveRequest):
    """Record the principal's decision on a B verdict.

    decision='approved'  — principal wants this to advance (downstream
                           pipeline action is the principal's choice;
                           this endpoint does NOT auto-trigger F14b).
                           Phase 2.1a: ALSO emits forward_vector_created
                           so generate_forward_vectors can pick the
                           hypothesis up for /research/forward queue.
    decision='rejected'  — principal disagrees with B's APPROVE or
                           AMENDMENT proposal.
    decision='deferred'  — needs more thought; surfaces back to queue
                           after a cooldown (handled by view layer).

    Appends to resolutions.jsonl; latest row per hypothesis_id wins.
    """
    try:
        from engine.agents.strengthener.approval_view import (
            append_resolution, find_verdict, find_hypothesis_family,
        )
        r = append_resolution(
            hypothesis_id = req.hypothesis_id,
            decision      = req.decision,
            rationale     = req.rationale,
        )

        # Phase 2.1a + step 13: on `approved`, route by B's verdict_type.
        #   APPROVE_FOR_PIPELINE      → emit forward_vector_created
        #                                (brainstorm → /research/forward queue)
        #   DOCTRINE_AMENDMENT_NEEDED → write draft amendment file
        #                                + emit memory_amendment_proposed
        # Failure here MUST NOT roll back the resolution (resolution is
        # the authoritative record); log + continue.
        if r.decision == "approved":
            try:
                v = find_verdict(req.hypothesis_id)
                if v is None:
                    logger.warning(
                        "resolve_approval: no verdict found for %s; "
                        "skipping downstream emit",
                        req.hypothesis_id,
                    )
                else:
                    verdict_type = v.get("verdict_type", "")
                    if verdict_type == "APPROVE_FOR_PIPELINE":
                        _emit_forward_vector_created(req.hypothesis_id, v)
                        # Tier C-2d (2026-06-08): also auto-extract a
                        # factor backtest SPEC for downstream dispatch.
                        # The SPEC lands as a NEW pending row in
                        # /approvals (factor_specs queue) for the
                        # principal to review before any dispatch runs.
                        _auto_extract_factor_spec(req.hypothesis_id)
                    elif verdict_type == "DOCTRINE_AMENDMENT_NEEDED":
                        _handle_amendment_approval(req.hypothesis_id, v)
                    else:
                        logger.info(
                            "resolve_approval: no downstream wiring for "
                            "verdict_type=%s (hyp %s)",
                            verdict_type, req.hypothesis_id,
                        )
            except Exception as exc:
                logger.exception(
                    "resolve_approval: downstream emit failed for %s: %s",
                    req.hypothesis_id, exc,
                )

        return ResolveResponse(
            status        = "ok",
            hypothesis_id = r.hypothesis_id,
            decision      = r.decision,
            resolved_ts   = r.resolved_ts,
        )
    except ValueError as exc:
        # Invalid decision string
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"strengthener resolve failed: {exc}")


# ──────────────────────────────────────────────────────────────────────
# Tier C-2d (2026-06-08): factor SPEC approval queue
# ──────────────────────────────────────────────────────────────────────
class FactorSpecApprovalRow(BaseModel):
    spec_hash:            str
    source_hypothesis_id: str
    family_hint:          str
    persisted_ts:         str
    spec:                 dict
    resolved:             bool
    resolution:           Optional[dict] = None


class FactorSpecApprovalsDigest(BaseModel):
    n_pending:  int
    n_resolved: int
    rows:       list[FactorSpecApprovalRow]


@router.get("/factor_specs", response_model=FactorSpecApprovalsDigest)
def list_factor_specs(include_resolved: bool = Query(False)):
    """Return pending Tier C factor SPECs awaiting human approval.

    Populated by _auto_extract_factor_spec when the principal
    approves a B verdict (APPROVE_FOR_PIPELINE) in /approvals.
    Each row carries the LLM-extracted SPEC (signal_kind / universe /
    dates / inputs / weighting / pit_audits) — principal approves
    SPEC → dispatcher runs gates + template + emits verdict.
    """
    try:
        from engine.agents.strengthener.factor_spec_store import (
            list_pending_factor_specs,
        )
        return list_pending_factor_specs(
            include_resolved=include_resolved)
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"factor_specs read failed: {exc}")


class FactorSpecResolveRequest(BaseModel):
    spec_hash:  str
    decision:   str         # approved / rejected / deferred
    rationale:  str = ""


class FactorSpecResolveResponse(BaseModel):
    status:             str
    spec_hash:          str
    decision:           str
    resolved_ts:        str
    dispatch_event_id:  Optional[str] = None
    verdict_event_id:   Optional[str] = None
    template_verdict:   Optional[str] = None
    template_summary:   Optional[str] = None
    refusal_reason:     Optional[str] = None


@router.post("/factor_specs/resolve",
              response_model=FactorSpecResolveResponse)
def resolve_factor_spec_endpoint(req: FactorSpecResolveRequest):
    """Record the principal's decision on a pending factor SPEC.

    decision='approved' — invokes dispatch_factor_spec synchronously
                          (gates + template + emit). Response carries
                          dispatch_event_id (audit log row) + the
                          template verdict / summary. If a gate
                          refused, refusal_reason is set instead.
    decision='rejected' — record only; no dispatch
    decision='deferred' — record only; row may resurface in queue
                          after a cooldown
    """
    try:
        from engine.agents.strengthener.factor_spec_store import (
            resolve_factor_spec,
        )
        out = resolve_factor_spec(
            spec_hash = req.spec_hash,
            decision  = req.decision,
            rationale = req.rationale,
        )
        # Decompose dispatch_result for the response shape
        dr = out.get("dispatch_result") or {}
        tr = dr.get("template_result") or {}
        refusal = dr.get("refusal") or {}
        return FactorSpecResolveResponse(
            status            = "ok",
            spec_hash         = out["spec_hash"],
            decision          = out["decision"],
            resolved_ts       = out["resolved_ts"],
            dispatch_event_id = out.get("dispatch_event_id"),
            verdict_event_id  = out.get("verdict_event_id"),
            template_verdict  = tr.get("verdict"),
            template_summary  = tr.get("summary"),
            refusal_reason    = refusal.get("reason_code"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"factor_spec resolve failed: {exc}")
