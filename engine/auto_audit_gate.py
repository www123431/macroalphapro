"""
engine/auto_audit_gate.py — Layer 2 Deterministic Safety Gate (R-1.D, 2026-05-06)

The gate validates AuditProposal payloads written by Layer 1 LLM (R-1.C)
and decides:
  • gate_status = 'pass' | 'fail'           (proposal quality)
  • governance_required = True | False      (severity for human review)

These are TWO INDEPENDENT axes:
  • gate_status='pass' + governance_required=False → trivial route
  • gate_status='pass' + governance_required=True  → governance route (Tier 2 review)
  • gate_status='fail'                             → dead-letter (supervisor can read but not approve)

The gate is **zero-LLM** — pure deterministic Python. Each validation rule
returns a `gate_reasons` list of dicts: {rule, severity, message}.

Sync invocation: called at the end of `generate_proposal()` so each LLM
output is gated immediately. Re-gating after rule edits requires running
`scripts/run_auto_audit_gate.py` (separate utility, R-1.E).

The 10 validation rules + boundaries (locked with supervisor 2026-05-06):
  V1   Schema completeness        — 8 required fields + options ∈ [1,5] + recommendation_index range
  V2   files_to_touch whitelist   — ALL options scanned; FORBIDDEN→fail, FLAGGED→governance
  V3   amendment_kind valid       — must be in AMENDMENT_KINDS ∪ {'no_action'}
  V4   rationale_short length     — 20 ≤ len ≤ 500
  V5   diff_size_estimate cap     — recommended option ≤ 50 lines (hard cap)
  V6   evidence_refs non-empty    — len ≥ 1
  V7   estimated_effort_min       — 0 ≤ value ≤ 480 (8 hours)
  V8   risk_level vs amendment_kind consistency (5-row matrix)
  V9   PRODUCTION_SIGNAL detection — extended word list, full-payload search
  V10  Proposal-self-reference    — evidence_refs cannot contain "Proposal #"
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from engine.auto_audit_models import AuditProposal
from engine.memory import SessionFactory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Whitelist (must mirror engine/auto_audit_proposer.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
FORBIDDEN_PATHS = (
    "engine/auto_audit.py",
    "engine/auto_audit_models.py",
    "engine/auto_audit_rules.py",
    "engine/auto_audit_proposer.py",
    "engine/auto_audit_gate.py",
    "engine/auto_audit_promoter.py",   # R-1.E
    "engine/auto_audit_executor.py",   # R-1.E
    "engine/preregistration.py",
    "engine/memory.py",
    "engine/db_models.py",
)
FLAGGED_PATHS = (
    "engine/portfolio.py",
    "engine/signal.py",
    "engine/config.py",
    "engine/regime.py",
    "engine/anomaly_screener.py",
)


# ─────────────────────────────────────────────────────────────────────────────
# AMENDMENT_KINDS — imported from preregistration; 'no_action' is gate-only.
# ─────────────────────────────────────────────────────────────────────────────
def _allowed_amendment_kinds() -> set[str]:
    from engine.preregistration import AMENDMENT_KINDS
    return set(AMENDMENT_KINDS) | {"no_action"}


# V8 consistency matrix: amendment_kind → allowed risk_levels
RISK_BY_KIND: Dict[str, set[str]] = {
    "clarification":     {"LOW", "MID"},
    "scope_narrow":      {"LOW", "MID"},
    "threshold_tweak":   {"LOW", "MID", "HIGH"},     # most flexible
    "hypothesis_amend":  {"MID", "HIGH"},
    "endpoint_swap":     {"MID", "HIGH"},
    "superseded":        {"LOW", "MID", "HIGH"},
    "no_action":         {"LOW"},
}


# V9 extended word list — case-insensitive substring on JSON-stringified payload
PRODUCTION_SIGNAL_TRIGGERS = (
    "PRODUCTION_SIGNAL",
    "ql01_bab",
    "tsmom",
    "production signal",
    "production strategy",
    "betting against beta",
    "BAB",
)


# ─────────────────────────────────────────────────────────────────────────────
# Validation rule helpers
# ─────────────────────────────────────────────────────────────────────────────
def _v1_schema_completeness(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    required = {"summary", "diagnosis", "options", "recommendation_index",
                "amendment_kind", "rationale_short", "evidence_refs"}
    missing = required - set(payload.keys())
    if missing:
        out.append({"rule": "V1", "severity": "fail",
                    "message": f"missing required fields: {sorted(missing)}"})
        return out

    options = payload.get("options") or []
    if not isinstance(options, list) or not (1 <= len(options) <= 5):
        out.append({"rule": "V1", "severity": "fail",
                    "message": f"options must be list of length 1-5; got len={len(options) if isinstance(options, list) else 'non-list'}"})

    rec = payload.get("recommendation_index")
    if not isinstance(rec, int) or not (0 <= rec < len(options)):
        out.append({"rule": "V1", "severity": "fail",
                    "message": f"recommendation_index out of range: {rec} vs options len={len(options)}"})
    return out


def _v2_files_to_touch_whitelist(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """Returns (gate_reasons, governance_required). Scans ALL options."""
    out: List[Dict[str, Any]] = []
    gov_required = False
    options = payload.get("options") or []
    forbidden_hits: List[str] = []
    flagged_hits: List[str] = []
    for i, opt in enumerate(options):
        files = opt.get("files_to_touch") or []
        for f in files:
            if not isinstance(f, str):
                continue
            f_norm = f.strip().replace("\\", "/")
            if f_norm in FORBIDDEN_PATHS:
                forbidden_hits.append(f"option[{i}]:{f_norm}")
            elif f_norm in FLAGGED_PATHS:
                flagged_hits.append(f"option[{i}]:{f_norm}")
    if forbidden_hits:
        out.append({"rule": "V2", "severity": "fail",
                    "message": f"FORBIDDEN paths in proposal options: {forbidden_hits}"})
    if flagged_hits:
        gov_required = True
        out.append({"rule": "V2", "severity": "governance",
                    "message": f"FLAGGED paths require governance review: {flagged_hits}"})
    return out, gov_required


def _v3_amendment_kind(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    kind = payload.get("amendment_kind")
    allowed = _allowed_amendment_kinds()
    if kind not in allowed:
        out.append({"rule": "V3", "severity": "fail",
                    "message": f"amendment_kind {kind!r} not in allowed set {sorted(allowed)}"})
    return out


def _v4_rationale_length(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    r = payload.get("rationale_short") or ""
    if not isinstance(r, str):
        out.append({"rule": "V4", "severity": "fail",
                    "message": f"rationale_short must be str; got {type(r).__name__}"})
        return out
    n = len(r.strip())
    if n < 20:
        out.append({"rule": "V4", "severity": "fail",
                    "message": f"rationale_short too short ({n} chars; min 20)"})
    elif n > 500:
        out.append({"rule": "V4", "severity": "fail",
                    "message": f"rationale_short too long ({n} chars; max 500)"})
    return out


def _v5_diff_size_recommended(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    options = payload.get("options") or []
    rec = payload.get("recommendation_index")
    if not isinstance(rec, int) or not (0 <= rec < len(options)):
        return out  # V1 already caught this
    diff_size = options[rec].get("diff_size_estimate")
    if not isinstance(diff_size, int):
        out.append({"rule": "V5", "severity": "fail",
                    "message": f"recommended option diff_size_estimate not int: {diff_size}"})
    elif diff_size > 50:
        out.append({"rule": "V5", "severity": "fail",
                    "message": f"recommended option diff_size_estimate {diff_size} > hard cap 50; split into multiple amends"})
    elif diff_size > 30:
        out.append({"rule": "V5", "severity": "warn",
                    "message": f"recommended option diff_size_estimate {diff_size} > soft cap 30; review carefully"})
    return out


def _v6_evidence_refs_nonempty(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    refs = payload.get("evidence_refs") or []
    if not isinstance(refs, list) or len(refs) < 1:
        out.append({"rule": "V6", "severity": "fail",
                    "message": "evidence_refs must contain ≥1 reference"})
    return out


def _v7_effort_range(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    options = payload.get("options") or []
    for i, opt in enumerate(options):
        eff = opt.get("estimated_effort_min")
        if eff is None:
            continue  # not strictly required by schema
        if not isinstance(eff, int):
            out.append({"rule": "V7", "severity": "fail",
                        "message": f"option[{i}].estimated_effort_min not int: {eff}"})
        elif eff < 0:
            out.append({"rule": "V7", "severity": "fail",
                        "message": f"option[{i}].estimated_effort_min negative: {eff}"})
        elif eff > 480:
            out.append({"rule": "V7", "severity": "fail",
                        "message": f"option[{i}].estimated_effort_min {eff} > 480 (8h cap); use spec process for larger work"})
    return out


def _v8_risk_kind_consistency(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recommended option's risk_level must be consistent with amendment_kind."""
    out: List[Dict[str, Any]] = []
    kind = payload.get("amendment_kind")
    options = payload.get("options") or []
    rec = payload.get("recommendation_index")
    if not isinstance(rec, int) or not (0 <= rec < len(options)):
        return out
    risk = options[rec].get("risk_level")
    allowed = RISK_BY_KIND.get(kind)
    if allowed is None:
        return out  # V3 already failed
    if risk not in allowed:
        out.append({"rule": "V8", "severity": "fail",
                    "message": f"risk_level {risk!r} inconsistent with amendment_kind {kind!r}; allowed: {sorted(allowed)}"})
    return out


def _v9_production_signal_detection(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """Returns (gate_reasons, governance_required). Scans full JSON-stringified payload."""
    out: List[Dict[str, Any]] = []
    gov_required = False
    haystack = json.dumps(payload, ensure_ascii=False).lower()
    hits: List[str] = []
    for trigger in PRODUCTION_SIGNAL_TRIGGERS:
        if trigger.lower() in haystack:
            hits.append(trigger)
    if hits:
        gov_required = True
        out.append({"rule": "V9", "severity": "governance",
                    "message": f"PRODUCTION_SIGNAL-related triggers detected: {hits}; governance review required"})
    return out, gov_required


def _v10_proposal_self_reference(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """evidence_refs must not contain 'Proposal #' (or AuditProposal id reference)."""
    out: List[Dict[str, Any]] = []
    refs = payload.get("evidence_refs") or []
    pat = re.compile(r"\bproposal\s*#\s*\d+", re.IGNORECASE)
    bad: List[str] = []
    for r in refs:
        if isinstance(r, str) and pat.search(r):
            bad.append(r)
    if bad:
        out.append({"rule": "V10", "severity": "fail",
                    "message": f"evidence_refs cite other proposals (forbidden — cite finding/snapshot/file evidence): {bad}"})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def validate_payload(payload: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]], bool]:
    """
    Run all 10 validations on a parsed proposal payload.

    Returns (gate_pass, gate_reasons, governance_required).

    gate_reasons is the FULL list — including 'warn' (informational) and
    'governance' (sets gov flag) entries. Only 'fail' severity flips
    gate_pass to False.
    """
    if not isinstance(payload, dict):
        return False, [{"rule": "V0", "severity": "fail",
                        "message": f"payload must be dict, got {type(payload).__name__}"}], False

    reasons: List[Dict[str, Any]] = []
    gov_required = False

    reasons.extend(_v1_schema_completeness(payload))
    v2_reasons, v2_gov = _v2_files_to_touch_whitelist(payload)
    reasons.extend(v2_reasons)
    gov_required = gov_required or v2_gov
    reasons.extend(_v3_amendment_kind(payload))
    reasons.extend(_v4_rationale_length(payload))
    reasons.extend(_v5_diff_size_recommended(payload))
    reasons.extend(_v6_evidence_refs_nonempty(payload))
    reasons.extend(_v7_effort_range(payload))
    reasons.extend(_v8_risk_kind_consistency(payload))
    v9_reasons, v9_gov = _v9_production_signal_detection(payload)
    reasons.extend(v9_reasons)
    gov_required = gov_required or v9_gov
    reasons.extend(_v10_proposal_self_reference(payload))

    has_fail = any(r["severity"] == "fail" for r in reasons)
    return (not has_fail), reasons, gov_required


def gate_proposal(proposal_id: int) -> Dict[str, Any]:
    """
    Validate AuditProposal #proposal_id, persist gate_status + reasons +
    governance_required. Idempotent: re-gating overwrites prior result.

    Returns dict with {proposal_id, gate_status, governance_required,
                       n_reasons, fail_reasons}.
    """
    with SessionFactory() as s:
        prop = s.get(AuditProposal, proposal_id)
        if prop is None:
            return {"proposal_id": proposal_id, "error": "proposal_not_found"}
        if prop.generation_status != "success":
            # Cannot gate non-success proposals; mark them as 'fail' with reason
            prop.gate_status = "fail"
            prop.gate_failure_reasons_json = json.dumps([{
                "rule": "V0",
                "severity": "fail",
                "message": f"cannot gate proposal with generation_status={prop.generation_status}",
            }])
            s.commit()
            return {
                "proposal_id":         proposal_id,
                "gate_status":         "fail",
                "governance_required": False,
                "reason":              f"non-success generation_status={prop.generation_status}",
            }

        try:
            payload = json.loads(prop.parsed_payload_json) if prop.parsed_payload_json else None
        except Exception as exc:
            prop.gate_status = "fail"
            prop.gate_failure_reasons_json = json.dumps([{
                "rule": "V0",
                "severity": "fail",
                "message": f"parsed_payload_json malformed: {exc}",
            }])
            s.commit()
            return {"proposal_id": proposal_id, "gate_status": "fail",
                    "governance_required": False, "reason": "malformed payload"}

        gate_pass, reasons, gov_required = validate_payload(payload)
        prop.gate_status = "pass" if gate_pass else "fail"
        prop.governance_required = gov_required
        prop.gate_failure_reasons_json = json.dumps(reasons, ensure_ascii=False)
        s.commit()

    # Promoter (R-1.E): on gate-pass, write PendingApproval row synchronously.
    # No reason to delay; if gate passed the supervisor should see it.
    promotion: Dict[str, Any] = {}
    if gate_pass:
        try:
            from engine.auto_audit_promoter import promote_to_pending_approval
            promotion = promote_to_pending_approval(proposal_id)
        except Exception:
            logger.exception("gate_proposal: promotion raised on proposal %d", proposal_id)
            promotion = {"ok": False, "error": "promotion_raised"}

    fail_reasons = [r for r in reasons if r["severity"] == "fail"]
    return {
        "proposal_id":         proposal_id,
        "gate_status":         "pass" if gate_pass else "fail",
        "governance_required": gov_required,
        "n_reasons":           len(reasons),
        "fail_reasons":        fail_reasons[:10],
        "promotion":           promotion,
    }
