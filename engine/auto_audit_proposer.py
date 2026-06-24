"""
engine/auto_audit_proposer.py — Layer 1 LLM proposer for the Auto-Audit Loop
(R-1.C, 2026-05-06).

Reads an AuditFinding (a contradiction detected by R-1.B's deterministic rule
engine) and asks Gemini 2.5 Flash to draft 1-3 candidate remediation options
in a structured JSON envelope. The output is persisted to AuditProposal but
is **not** evaluated here — Layer 2 safety gate (R-1.D) and human supervisor
review (R-1.E) sit downstream.

Three-layer invariant:
    Layer 0 (rule engine, R-1.B)  : deterministic; finds contradictions
    Layer 1 (this file, R-1.C)    : LLM generates proposals (CAN propose)
    Layer 2 (safety gate, R-1.D)  : deterministic; validates proposals (CAN reject)
    + supervisor (R-1.E)          : human; final approval

The LLM is intentionally constrained:
  • Cannot propose edits to engine/auto_audit*.py (would let it self-disable)
  • Cannot propose edits to engine/preregistration.py (audit-trail integrity)
  • Cannot propose edits to engine/memory.py / engine/db_models.py (DB infra)
  • Must reference specific evidence from the contradiction snapshot
  • Must classify the proposed change against AMENDMENT_KINDS
  • Output schema is enforced by Gemini structured-output mode

Cost tracking is independent from the S6 anomaly screener — separate JSON
state file + separate budget cap (R_COST_BUDGET_USD), to avoid the audit
system competing with anomaly screener for the same dollars.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from engine.auto_audit_models import AuditFinding, AuditProposal
# LLM cost/config constants — previously in engine.config, removed in a refactor; each LLM
# module now defines its own (mirrors engine.etf_holdings_risk_monitor / fomc_surprise_override).
# Values copied from those live modules (2026-05-22 config-drift fix; not guessed).
COST_PER_1M_INPUT_TOKENS: float = 0.30
COST_PER_1M_OUTPUT_TOKENS: float = 2.50
LLM_MODEL_VERSION: str = "gemini-2.5-flash"
LLM_TEMPERATURE: float = 0.0
LLM_THINKING_BUDGET: int = 1500
from engine.memory import SessionFactory
# 2026-05-08: budget moved from constant import to runtime SystemConfig-backed
# helper so supervisor can adjust without code edits. Default $50/yr unchanged.
from engine.llm_budget import get_r_audit_budget_usd_per_year

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output schema — enforced by Gemini structured-output mode
# ─────────────────────────────────────────────────────────────────────────────
PROPOSAL_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary":   {"type": "string", "description": "1-line summary, 20-200 chars"},
        "diagnosis": {"type": "string", "description": "Why the contradiction occurred. Reference snapshot evidence."},
        "options": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "action":               {"type": "string"},
                    "pros":                 {"type": "array", "items": {"type": "string"}},
                    "cons":                 {"type": "array", "items": {"type": "string"}},
                    "estimated_effort_min": {"type": "integer"},
                    "risk_level":           {"type": "string", "enum": ["LOW", "MID", "HIGH"]},
                    "files_to_touch":       {"type": "array", "items": {"type": "string"}},
                    "diff_size_estimate":   {"type": "integer"},
                },
                "required": ["action", "risk_level", "files_to_touch", "diff_size_estimate"],
            },
        },
        "recommendation_index": {"type": "integer"},
        "amendment_kind": {
            "type": "string",
            "enum": ["clarification", "scope_narrow", "threshold_tweak",
                     "hypothesis_amend", "endpoint_swap", "superseded", "no_action"],
        },
        "rationale_short": {"type": "string", "description": "≥20 chars; consumed by amend_spec"},
        "evidence_refs":   {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "diagnosis", "options", "recommendation_index",
                 "amendment_kind", "rationale_short", "evidence_refs"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-rule extra-context registry
# ─────────────────────────────────────────────────────────────────────────────
# Each entry maps rule_name → callable(snapshot_dict) → {"facts": ..., "prompt_overrides": ...}.
# Rules without a registered provider get an empty context (just the snapshot
# itself + the generic skeleton).
ContextProviderFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def _load_registered_providers() -> Dict[str, ContextProviderFn]:
    """Lazy import: avoid circular deps with auto_audit_rules at module load."""
    try:
        from engine.auto_audit_rules import get_context_provider_registry
        return get_context_provider_registry()
    except Exception:
        logger.exception("auto_audit_proposer: failed to load context provider registry")
        return {}


EXTRA_CONTEXT_PROVIDERS: Dict[str, ContextProviderFn] = _load_registered_providers()


def register_context_provider(rule_name: str) -> Callable[[ContextProviderFn], ContextProviderFn]:
    """Decorator: register an extra-context provider for a given rule (runtime path)."""
    def _wrap(fn: ContextProviderFn) -> ContextProviderFn:
        EXTRA_CONTEXT_PROVIDERS[rule_name] = fn
        return fn
    return _wrap


def _get_extra_context(rule_name: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Returns {"facts": dict, "prompt_overrides": dict}. Empty when no provider."""
    fn = EXTRA_CONTEXT_PROVIDERS.get(rule_name)
    if fn is None:
        return {"facts": {}, "prompt_overrides": {}}
    try:
        out = fn(snapshot)
    except Exception:
        logger.exception("auto_audit_proposer: context provider for %s raised", rule_name)
        return {"facts": {}, "prompt_overrides": {}}
    return {
        "facts":            out.get("facts", {}),
        "prompt_overrides": out.get("prompt_overrides", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# File-edit whitelist (Layer 2 / R-1.D will enforce; we surface the rules to
# the LLM so it doesn't waste tokens proposing forbidden edits)
# ─────────────────────────────────────────────────────────────────────────────
LLM_FORBIDDEN_PATHS = (
    "engine/auto_audit.py",
    "engine/auto_audit_models.py",
    "engine/auto_audit_rules.py",
    "engine/auto_audit_proposer.py",
    "engine/auto_audit_gate.py",       # mirror gate.FORBIDDEN_PATHS
    "engine/auto_audit_promoter.py",   # mirror gate.FORBIDDEN_PATHS (R-1.E)
    "engine/auto_audit_executor.py",   # mirror gate.FORBIDDEN_PATHS (R-1.E)
    "engine/preregistration.py",
    "engine/memory.py",
    "engine/db_models.py",
)
LLM_FLAGGED_PATHS = (
    "engine/portfolio.py",
    "engine/signal.py",
    "engine/config.py",
    "engine/regime.py",
    "engine/anomaly_screener.py",
)
# All `pages/*.py` and `docs/*.md` are allowed without flag.


def _whitelist_text() -> str:
    forbid = "\n".join(f"  - {p}" for p in LLM_FORBIDDEN_PATHS)
    flag = "\n".join(f"  - {p}" for p in LLM_FLAGGED_PATHS)
    return (
        f"FORBIDDEN — proposals touching these are auto-rejected:\n{forbid}\n\n"
        f"FLAGGED — allowed but Layer 2 marks `governance_required=true` "
        f"(supervisor must explicitly approve):\n{flag}\n\n"
        f"OK — `docs/**/*.md` and `pages/**/*.py` (UI code) are allowed."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────
GENERIC_PROMPT_SKELETON = """\
You are an audit-loop assistant for a quantitative trading research platform
("Macro Alpha Pro"). A deterministic rule has detected a contradiction. Read
the contradiction details and propose 1-3 candidate remediation options for
human-supervisor review.

# Standing invariants (do not violate)
1. You PROPOSE, you do not DECIDE. A human supervisor reviews + a
   deterministic safety gate (Layer 2) validates your output afterward.
2. Your `files_to_touch` list MUST respect the whitelist below.
3. Each option must reference specific evidence from the contradiction
   snapshot in its `pros` / `cons` text.
4. Prefer minimal safe action when uncertain (drop unused, document the
   state, defer to supervisor with a no_action option).
5. Your `rationale_short` will be consumed verbatim by `amend_spec` if the
   supervisor approves — keep it 20-500 chars, factual, no marketing.

# File-edit whitelist
{whitelist}

# Contradiction
Rule:     {rule_name}
Severity: {severity}
Detected: {detected_at}

## Snapshot (deterministic-rule output, raw JSON)
```json
{snapshot_json}
```

## Rule-specific facts
{facts_block}

## Diagnosis hint (rule-specific)
{diagnosis_hint}

## Options hint (rule-specific)
{options_hint}

## Recommendation bias (rule-specific)
{recommendation_bias}

## Recent supervisor decisions on this rule (last 30 days)
{ignored_history}

# Output
Return JSON conforming to the response schema. The schema is enforced —
unknown fields will be dropped, missing required fields will fail.
"""


def _facts_block(facts: Dict[str, Any]) -> str:
    if not facts:
        return "(no rule-specific facts provided)"
    return "```json\n" + json.dumps(facts, indent=2, default=str)[:4000] + "\n```"


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor-text sanitization (R-1.E, 2026-05-06)
# Defends `_ignored_history_block` against prompt-injection when supervisor
# IGNORE rationale flows back into the LLM's prompt next time the rule fires.
# 4-layer defense: strip-control + length-cap + injection-pattern-redact + tag-wrap.
# ─────────────────────────────────────────────────────────────────────────────
import re as _re_sup_text

_SUPERVISOR_TEXT_MAX_LEN = 500
_INJECTION_PATTERNS = (
    "ignore previous instruction",
    "ignore previous prompts",
    "you are now",
    "system:",
    "</supervisor_text>",
    "</snapshot>",
    "</invariants>",
    "execute:",
    "respond with",
    "rewrite the schema",
    "disregard the rules",
)
_CTRL_RE = _re_sup_text.compile(r"[\x00-\x1f\x7f-\x9f​-‍﻿]")


def _sanitize_supervisor_text(text: Any) -> str:
    """
    4-layer defense for supervisor-typed text re-entering an LLM prompt:
      1. coerce to str + strip control chars + zero-width unicode
      2. case-insensitive injection-pattern redact
      3. length cap (500 chars)
      4. caller wraps in <supervisor_text> tags

    Returns the sanitised string (without tag wrap — caller adds tags).
    """
    if text is None:
        return ""
    s = str(text)
    s = _CTRL_RE.sub(" ", s)
    s_lower = s.lower()
    for pat in _INJECTION_PATTERNS:
        if pat in s_lower:
            # Replace each occurrence with a marker, preserving rough length
            idx = s_lower.find(pat)
            while idx != -1:
                s = s[:idx] + "[REDACTED:injection]" + s[idx + len(pat):]
                s_lower = s.lower()
                idx = s_lower.find(pat)
    if len(s) > _SUPERVISOR_TEXT_MAX_LEN:
        s = s[:_SUPERVISOR_TEXT_MAX_LEN] + "…[TRUNCATED]"
    return s


def _ignored_history_block(rule_name: str) -> str:
    """
    Last 30d IGNORED findings for this rule, with sanitised rationale if any.
    Helps LLM avoid repeating proposals supervisor already rejected. Each
    rationale runs through `_sanitize_supervisor_text` to defang potential
    prompt-injection content the supervisor may have typed.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    with SessionFactory() as s:
        rows = (
            s.query(AuditFinding)
             .filter(AuditFinding.rule_name == rule_name)
             .filter(AuditFinding.status == "IGNORED")
             .filter(AuditFinding.detected_at >= cutoff)
             .order_by(AuditFinding.detected_at.desc())
             .limit(3)
             .all()
        )
    if not rows:
        return "(no IGNORED findings in last 30d for this rule)"
    lines = []
    for r in rows:
        rationale = _sanitize_supervisor_text(r.notes or "")
        wrapped = (
            f'<supervisor_text source="finding_id={r.id}">{rationale}</supervisor_text>'
            if rationale else "(no rationale recorded)"
        )
        lines.append(f"- finding #{r.id} at {r.detected_at} (severity={r.severity}); "
                     f"rationale: {wrapped}")
    return "\n".join(lines)


def _build_prompt(finding: AuditFinding) -> tuple[str, Dict[str, Any]]:
    """Returns (prompt_text, {snapshot, facts, prompt_overrides}) for caching/audit."""
    snapshot = json.loads(finding.snapshot_json) if finding.snapshot_json else {}
    extra = _get_extra_context(finding.rule_name, snapshot)
    overrides = extra["prompt_overrides"]

    prompt = GENERIC_PROMPT_SKELETON.format(
        whitelist            = _whitelist_text(),
        rule_name            = finding.rule_name,
        severity             = finding.severity,
        detected_at          = str(finding.detected_at),
        snapshot_json        = json.dumps(snapshot, indent=2, default=str)[:4000],
        facts_block          = _facts_block(extra["facts"]),
        diagnosis_hint       = overrides.get("diagnosis_hint", "(no rule-specific hint)"),
        options_hint         = overrides.get("options_hint",   "(no rule-specific hint)"),
        recommendation_bias  = overrides.get("recommendation_bias", "(no bias)"),
        ignored_history      = _ignored_history_block(finding.rule_name),
    )
    return prompt, {"snapshot": snapshot, "facts": extra["facts"], "prompt_overrides": overrides}


# ─────────────────────────────────────────────────────────────────────────────
# Independent cost tracker (separate file from S6 anomaly_llm_detector)
# ─────────────────────────────────────────────────────────────────────────────
_COST_TRACKER_FILE = Path(__file__).parent.parent / ".streamlit" / "auto_audit_llm_cost.json"


def _load_cost_tracker() -> dict:
    if not _COST_TRACKER_FILE.exists():
        return {"total_usd": 0.0, "calls": 0, "by_date": {}}
    try:
        return json.loads(_COST_TRACKER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"total_usd": 0.0, "calls": 0, "by_date": {}}


def _save_cost_tracker(state: dict) -> None:
    _COST_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _COST_TRACKER_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _record_call_cost(cost_usd: float) -> dict:
    state = _load_cost_tracker()
    state["total_usd"] = round(state.get("total_usd", 0.0) + cost_usd, 6)
    state["calls"]     = state.get("calls", 0) + 1
    by_date = state.setdefault("by_date", {})
    key = str(datetime.date.today())
    by_date[key] = round(by_date.get(key, 0.0) + cost_usd, 6)
    _save_cost_tracker(state)
    return state


def get_cost_status() -> dict:
    """Public helper; mirrors anomaly_llm_detector.get_cost_status.

    Reads runtime budget from engine.llm_budget (SystemConfig-backed, falls
    back to engine.config.R_COST_BUDGET_USD default if no override set).
    """
    state = _load_cost_tracker()
    total = state.get("total_usd", 0.0)
    budget = get_r_audit_budget_usd_per_year()
    return {
        "total_usd":  total,
        "budget_usd": budget,
        "fraction":   total / budget if budget > 0 else 0,
        "calls":      state.get("calls", 0),
    }


def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * COST_PER_1M_INPUT_TOKENS +
            output_tokens * COST_PER_1M_OUTPUT_TOKENS) / 1_000_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Core: call Gemini for one finding
# ─────────────────────────────────────────────────────────────────────────────
def _call_gemini(prompt: str) -> dict:
    """Returns {parsed, raw_text, prompt_hash, input_tokens, output_tokens, cost_usd}."""
    p_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    from engine.key_pool import get_pool
    pool = get_pool()
    model = pool.get_model(
        model_name=LLM_MODEL_VERSION,
        response_schema=PROPOSAL_RESPONSE_SCHEMA,
        temperature=LLM_TEMPERATURE,
        thinking_budget=LLM_THINKING_BUDGET,
    )
    resp = model.generate_content(prompt)
    pool.report_success(has_content=True)

    text = getattr(resp, "text", None) or str(resp)
    usage = getattr(resp, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = ((getattr(usage, "candidates_token_count", 0) or 0)
               + (getattr(usage, "thoughts_token_count", 0) or 0))
    cost = _compute_cost(in_tok, out_tok)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.error("auto_audit_proposer: JSON parse failed; first 200 chars: %s", text[:200])
        raise

    return {
        "parsed":        parsed,
        "raw_text":      text,
        "prompt_hash":   p_hash,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "cost_usd":      cost,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def generate_proposal(finding_id: int) -> Dict[str, Any]:
    """
    Generate one LLM proposal for the given AuditFinding. Idempotent: if a
    proposal already exists (any generation_status), this is a no-op and
    returns the existing record summary.

    Returns dict with {finding_id, proposal_id, generation_status, cost_usd}.
    """
    with SessionFactory() as s:
        finding = s.get(AuditFinding, finding_id)
        if finding is None:
            return {"finding_id": finding_id, "error": "finding_not_found"}

        existing = s.query(AuditProposal).filter_by(finding_id=finding_id).first()
        if existing is not None:
            return {
                "finding_id":         finding_id,
                "proposal_id":        existing.id,
                "generation_status":  existing.generation_status,
                "cost_usd":           existing.cost_usd,
                "skipped":            "already_generated",
            }

        prompt, _ = _build_prompt(finding)

    # LLM call happens outside session — long-running, don't hold the lock.
    try:
        llm_out = _call_gemini(prompt)
        status = "success"
        failure_reason: Optional[str] = None
        parsed_payload = llm_out["parsed"]
        raw_text       = llm_out["raw_text"]
    except Exception as exc:
        logger.exception("auto_audit_proposer: LLM call failed for finding %d", finding_id)
        llm_out = {
            "prompt_hash":   hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens":  0,
            "output_tokens": 0,
            "cost_usd":      0.0,
        }
        status = "generation_failed"
        failure_reason = f"{type(exc).__name__}: {str(exc)[:200]}"
        parsed_payload = None
        raw_text = None

    if status == "success":
        _record_call_cost(llm_out["cost_usd"])

    with SessionFactory() as s:
        prop = AuditProposal(
            finding_id          = finding_id,
            generated_at        = datetime.datetime.utcnow(),
            model_version       = LLM_MODEL_VERSION,
            prompt_hash         = llm_out["prompt_hash"],
            input_tokens        = llm_out["input_tokens"],
            output_tokens       = llm_out["output_tokens"],
            cost_usd            = llm_out["cost_usd"],
            raw_response_text   = raw_text,
            parsed_payload_json = json.dumps(parsed_payload, ensure_ascii=False) if parsed_payload else None,
            generation_status   = status,
            failure_reason      = failure_reason,
        )
        s.add(prop)
        s.flush()
        # Back-link
        finding = s.get(AuditFinding, finding_id)
        finding.proposal_id = prop.id
        if status == "success":
            finding.status = "PROPOSED"
        s.commit()
        proposal_id = prop.id

    # Layer 2 gate (R-1.D): synchronous post-generation validation.
    # Fast deterministic Python — adds <10ms; no API risk. Doing it here
    # means each AuditProposal row reaches steady state in one transaction
    # cycle (no half-gated rows lying around).
    gate_summary: Dict[str, Any] = {}
    if status == "success":
        try:
            from engine.auto_audit_gate import gate_proposal as _gate_proposal
            gate_summary = _gate_proposal(proposal_id)
        except Exception:
            logger.exception("auto_audit_proposer: gate_proposal raised on proposal %d", proposal_id)
            gate_summary = {"gate_status": "error", "error": "gate raised"}

    return {
        "finding_id":          finding_id,
        "proposal_id":         proposal_id,
        "generation_status":   status,
        "cost_usd":            llm_out["cost_usd"],
        "input_tokens":        llm_out["input_tokens"],
        "output_tokens":       llm_out["output_tokens"],
        "failure_reason":      failure_reason,
        "gate_status":         gate_summary.get("gate_status"),
        "governance_required": gate_summary.get("governance_required", False),
    }


def generate_proposals_for_open_findings(cap: Optional[int] = None) -> Dict[str, Any]:
    """
    Iterate OPEN findings without a proposal, generate up to `cap` (defaults
    to engine.config.R_PROPOSAL_CAP_PER_RUN). Excess findings are marked
    via proposal row with generation_status='deferred_quota'.

    Hard stop if budget exceeded mid-run: remaining findings get
    generation_status='deferred_quota' and the function returns.
    """
    from engine.config import R_PROPOSAL_CAP_PER_RUN

    if cap is None:
        cap = R_PROPOSAL_CAP_PER_RUN

    summary = {
        "n_processed":     0,
        "n_succeeded":     0,
        "n_failed":        0,
        "n_deferred":      0,
        "total_cost_usd":  0.0,
        "budget_remaining": None,
    }

    with SessionFactory() as s:
        candidates = (
            s.query(AuditFinding.id)
             .filter(AuditFinding.status == "OPEN")
             .filter(AuditFinding.proposal_id.is_(None))
             .order_by(AuditFinding.detected_at.asc())
             .all()
        )
        candidate_ids = [c[0] for c in candidates]

    cost_status = get_cost_status()
    budget_remaining = cost_status["budget_usd"] - cost_status["total_usd"]
    summary["budget_remaining"] = round(budget_remaining, 4)

    for idx, fid in enumerate(candidate_ids):
        # Cap reached → defer remaining
        if idx >= cap:
            with SessionFactory() as s:
                # Mark deferred
                prop = AuditProposal(
                    finding_id        = fid,
                    generated_at      = datetime.datetime.utcnow(),
                    model_version     = LLM_MODEL_VERSION,
                    prompt_hash       = "deferred",
                    generation_status = "deferred_quota",
                    failure_reason    = f"cap={cap} reached",
                )
                s.add(prop)
                try:
                    s.flush()
                    finding = s.get(AuditFinding, fid)
                    finding.proposal_id = prop.id
                    s.commit()
                except Exception:
                    s.rollback()
            summary["n_deferred"] += 1
            continue

        # Budget exhausted → defer
        if budget_remaining <= 0:
            with SessionFactory() as s:
                prop = AuditProposal(
                    finding_id        = fid,
                    generated_at      = datetime.datetime.utcnow(),
                    model_version     = LLM_MODEL_VERSION,
                    prompt_hash       = "deferred",
                    generation_status = "deferred_quota",
                    failure_reason    = "annual budget exhausted",
                )
                s.add(prop)
                try:
                    s.flush()
                    finding = s.get(AuditFinding, fid)
                    finding.proposal_id = prop.id
                    s.commit()
                except Exception:
                    s.rollback()
            summary["n_deferred"] += 1
            continue

        result = generate_proposal(fid)
        summary["n_processed"] += 1
        if result.get("generation_status") == "success":
            summary["n_succeeded"]    += 1
            summary["total_cost_usd"] += result.get("cost_usd", 0.0)
            budget_remaining          -= result.get("cost_usd", 0.0)
        else:
            summary["n_failed"] += 1

    summary["total_cost_usd"] = round(summary["total_cost_usd"], 6)
    return summary
