"""
Auto-Spec Drafting Agent (P3.5 deliverable, 2026-05-07).

Sakana AI Scientist style — supervisor describes a hypothesis in natural
language; Gemini 2.5 Flash drafts a complete pre-registration spec
following the project's existing spec format. Supervisor reviews +
freezes hash. LLM acts as **scientific collaborator at the proposer
layer**, never at the decision layer.

Red lines (verified by safety gate)
-----------------------------------
1. **0-LLM-in-evaluation invariant** preserved. LLM only DRAFTS spec
   text. It cannot:
     - mark a spec ``status='active'`` (only supervisor approval can)
     - call ``register_spec()`` or ``amend_spec()`` directly
     - decide whether a backtest passed/failed
     - write code that runs in the production signal pipeline

2. **No silent overwrite**. Drafter refuses to draft for a spec_path
   that already exists in SpecRegistry — supervisor must explicitly
   choose ``amend`` flow instead.

3. **FORBIDDEN_PATHS**. Even at draft time, LLM cannot reference
   ``engine/auto_audit*.py``, ``engine/preregistration.py``,
   ``engine/memory.py``, or ``engine/db_models.py`` in proposed
   implementation steps. These are audit infrastructure and changing
   them could bypass HARKing detection.

4. **Citation anchor required**. At least one literature citation
   (Frazzini-Pedersen / López de Prado / etc.) must be in the draft.
   No hallucinated DOIs — cited papers must be from a known-allowed
   list (extensible by supervisor over time).

Public API
----------
    draft_spec(hypothesis_text, *, target_path=None) -> SpecDraft

Returns SpecDraft dataclass with .draft_markdown (full spec text),
.metadata (parsed structured fields), .safety_findings (any gate
warnings), .cost_usd, and .status ("ok" | "rejected" | "llm_error").

This is NOT auto-approval — caller (UI page in P3.5.3) presents the
draft to supervisor for review, edit, freeze.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Forbidden paths (safety invariant) ───────────────────────────────────────
# LLM must not reference these in proposed implementation steps. Mirrors
# the Tier R proposer gate (engine/auto_audit_proposer.py FORBIDDEN_PATHS)
# but applied at draft time so the supervisor never sees a draft that
# would bypass the audit infrastructure if implemented.
_FORBIDDEN_PATH_PATTERNS = (
    "engine/auto_audit",
    "engine/preregistration.py",
    "engine/memory.py",
    "engine/db_models.py",
    "engine/agents/spec_drafter.py",   # the drafter cannot rewrite itself
    ".streamlit/secrets.toml",
)

# ── Allowed citation anchors ─────────────────────────────────────────────────
# Whitelist of papers we expect drafts to anchor to. Adding a new paper
# requires supervisor amendment to this list (intentional friction —
# prevents LLM hallucinating fake citations). Format: lowercase author-year
# fragment that must appear somewhere in the draft.
_ALLOWED_CITATIONS = (
    # Multiple-testing / FDR
    "benjamini",         # Benjamini-Hochberg / Benjamini-Yekutieli
    "harvey",            # Harvey-Liu false discoveries
    "lópez de prado", "lopez de prado",
    # Factor / momentum
    "frazzini",          # Frazzini-Pedersen 2014 BAB
    "moskowitz",         # Moskowitz-Ooi-Pedersen 2012 TSMOM
    "asness",            # Asness QMJ / value-momentum
    "carhart",           # 4-factor
    "fama",              # FF3 / FF5
    "hou",               # Hou-Xue-Zhang q-factor
    # Risk / drawdown
    "ang",               # Ang-Bekaert
    "campbell",          # campbell-hentschel
    # Pre-registration / replication
    "olken", "camerer",
    # Regime / time series
    "hamilton",          # Hamilton 1989 MSM
    # Vol / risk
    "engle", "bollerslev",
    # AI / LLM-as-judge
    "zheng",             # Zheng 2023 LLM bias
    # Reflexion / agentic
    "shinn",             # Reflexion
    "park",              # Generative Agents
    # Roncalli / risk parity
    "roncalli", "maillard",
)

# ── Output schema (Gemini structured-output) ─────────────────────────────────
SPEC_DRAFT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Spec title in the project's existing pattern: 'Spec — <noun phrase>'. 30-100 chars.",
        },
        "tldr": {
            "type": "string",
            "description": "1-paragraph summary (50-300 chars). What hypothesis, what data, what verdict shape.",
        },
        "hypothesis": {
            "type": "object",
            "properties": {
                "h0": {"type": "string", "description": "Null hypothesis (no effect / no improvement). 30-200 chars."},
                "h1": {"type": "string", "description": "Alternative hypothesis (specific direction + magnitude). 30-200 chars."},
            },
            "required": ["h0", "h1"],
        },
        "decision_rule": {
            "type": "object",
            "properties": {
                "ship_criteria": {
                    "type": "string",
                    "description": "Concrete quantitative threshold(s) for SHIP. e.g. 'Deflated Sharpe Ratio > 0.5 AND BHY-adjusted p < 0.05 AND raw t > 2'.",
                },
                "marginal_criteria": {
                    "type": "string",
                    "description": "Concrete threshold(s) for MARGINAL. Optional — set to '' if no marginal tier intended.",
                },
                "fail_criteria": {
                    "type": "string",
                    "description": "Default fail criteria; usually 'all other outcomes'.",
                },
                "literature_conditional_exemption": {
                    "type": "boolean",
                    "description": "Whether literature-conditional ship rule applies (≥10y external lit support → BHY exemption allowed).",
                },
            },
            "required": ["ship_criteria", "fail_criteria", "literature_conditional_exemption"],
        },
        "n_trials_impact": {
            "type": "object",
            "properties": {
                "n_added": {
                    "type": "integer",
                    "description": "Number of independent trials this spec adds to EFFECTIVE_N_TRIALS. Standard = 1 per testable hypothesis. >1 if spec embeds multiple sub-hypotheses.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why n_added is what it is. e.g. '1 trial: single hypothesis test on COT-conditional BAB Sharpe'.",
                },
            },
            "required": ["n_added", "rationale"],
        },
        "data_requirements": {
            "type": "array",
            "description": "Each data source needed. Specify table / API / column.",
            "items": {"type": "string"},
        },
        "predictions": {
            "type": "array",
            "description": "Qualitative predictions (3-6 items). Frozen at registration; cannot be revised post-test without HARKing flag.",
            "items": {"type": "string"},
        },
        "implementation_steps": {
            "type": "array",
            "description": "3-7 numbered high-level steps. Reference real file paths (engine/*.py, etc.) but NOT forbidden paths.",
            "items": {"type": "string"},
        },
        "success_criteria": {
            "type": "string",
            "description": "Quantitative success thresholds beyond decision_rule. e.g. 'Sharpe > 0.5 over 5y OOS; max drawdown <30%; hit rate >50%'.",
        },
        "failure_modes": {
            "type": "array",
            "description": "Anticipated failure modes (3-5 items). Pre-registered so we don't post-hoc rationalize.",
            "items": {"type": "string"},
        },
        "out_of_scope": {
            "type": "array",
            "description": "Explicit 'we are NOT doing X' (3-5 items). Anti-scope-creep.",
            "items": {"type": "string"},
        },
        "literature_anchors": {
            "type": "array",
            "description": "Required citations (at least 1). Format: 'Author Year, Journal' — must reference real, well-known finance/ML papers.",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "risks_and_caveats": {
            "type": "string",
            "description": "Honest pre-registered caveats. e.g. 'short forward window; calendar-bound verdict; data source is calendar-fragile'.",
        },
    },
    "required": [
        "title", "tldr", "hypothesis", "decision_rule", "n_trials_impact",
        "data_requirements", "predictions", "implementation_steps",
        "success_criteria", "failure_modes", "out_of_scope",
        "literature_anchors", "risks_and_caveats",
    ],
}


# ── Output dataclass ─────────────────────────────────────────────────────────
@dataclass
class SpecDraft:
    """Result of spec_drafter run.

    status meanings:
      "ok"               — draft generated + safety gate clean; ready for supervisor review
      "rejected"         — safety gate blocked (forbidden path / overwrite / no citation)
      "llm_error"        — Gemini call failed (see error_msg)
      "spec_exists"      — target_path already in SpecRegistry; use amend flow

    .draft_markdown is the human-readable spec text rendered from .metadata
    in the project's standard spec format. Supervisor edits the markdown;
    metadata is the LLM's structured output for downstream programmatic use.
    """
    hypothesis_input:  str
    target_path:       str | None         = None
    status:            str                = "ok"
    metadata:          dict[str, Any]     = field(default_factory=dict)
    draft_markdown:    str                = ""
    safety_findings:   list[str]          = field(default_factory=list)
    cost_usd:          float              = 0.0
    input_tokens:      int                = 0
    output_tokens:     int                = 0
    error_msg:         str | None         = None


# ── Safety gate ──────────────────────────────────────────────────────────────

def _check_forbidden_paths(text: str) -> list[str]:
    """Return list of forbidden-path matches found in the draft text."""
    findings: list[str] = []
    for pat in _FORBIDDEN_PATH_PATTERNS:
        if pat.lower() in text.lower():
            findings.append(f"references forbidden path: {pat}")
    return findings


def _check_citation_anchor(citations: list[str]) -> bool:
    """Return True iff at least one citation matches the allowed-list."""
    if not citations:
        return False
    blob = " | ".join(str(c).lower() for c in citations)
    return any(anchor in blob for anchor in _ALLOWED_CITATIONS)


def _safety_gate(metadata: dict, draft_text: str) -> list[str]:
    """Run all safety checks. Returns list of findings; empty = pass."""
    findings: list[str] = []

    # 1. Forbidden paths in implementation steps + risks + draft
    haystack = " | ".join([
        " ".join(metadata.get("implementation_steps", []) or []),
        metadata.get("risks_and_caveats", "") or "",
        metadata.get("tldr", "") or "",
        draft_text,
    ])
    findings.extend(_check_forbidden_paths(haystack))

    # 2. Citation anchor required
    cites = metadata.get("literature_anchors") or []
    if not _check_citation_anchor(cites):
        findings.append(
            f"no citation matches allowed-list. Got: {cites!r}; "
            f"allowed anchors include: {_ALLOWED_CITATIONS[:5]}..."
        )

    # 3. n_trials_impact must be ≥ 1
    n_added = (metadata.get("n_trials_impact") or {}).get("n_added")
    if not isinstance(n_added, int) or n_added < 1:
        findings.append(f"n_trials_impact.n_added must be int ≥ 1; got {n_added!r}")

    # 4. literature_conditional_exemption requires explicit literature support
    dr = metadata.get("decision_rule") or {}
    if dr.get("literature_conditional_exemption") is True:
        # Must have explicit lit anchor with FP / Asness / Moskowitz scope
        strong_anchors = ("frazzini", "asness", "moskowitz", "fama", "carhart", "hou")
        if not any(a in (" ".join(cites)).lower() for a in strong_anchors):
            findings.append(
                "literature_conditional_exemption=True but citations don't include "
                "a strong factor-finance anchor (Frazzini/Asness/Moskowitz/Fama/Carhart/Hou)"
            )

    return findings


# ── Spec text rendering ──────────────────────────────────────────────────────
# Translates structured metadata into the project's canonical Markdown
# spec layout. Mirrors the visual structure of existing
# docs/spec_*.md files so supervisor review feels familiar.

def _render_markdown(metadata: dict, hypothesis_input: str,
                      target_path: str | None) -> str:
    """Render structured metadata into the project's spec markdown format."""
    today = datetime.date.today().isoformat()
    title = metadata.get("title", "Spec — (untitled)")
    tldr = metadata.get("tldr", "(missing)")
    h = metadata.get("hypothesis", {}) or {}
    dr = metadata.get("decision_rule", {}) or {}
    n_imp = metadata.get("n_trials_impact", {}) or {}
    data_req = metadata.get("data_requirements") or []
    preds = metadata.get("predictions") or []
    steps = metadata.get("implementation_steps") or []
    success = metadata.get("success_criteria", "(missing)")
    failures = metadata.get("failure_modes") or []
    oos = metadata.get("out_of_scope") or []
    cites = metadata.get("literature_anchors") or []
    risks = metadata.get("risks_and_caveats", "(missing)")

    def _bullet_list(items: list[str]) -> str:
        return "\n".join(f"- {i}" for i in items) if items else "- (none specified)"

    parts = [
        f"# {title}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Status | DRAFT — pending supervisor review |",
        f"| Drafted by | Auto-Spec Drafter (LLM proposer; supervisor freezes) |",
        f"| Drafted at | {today} |",
        f"| Target path | {target_path or '(unset; supervisor will assign)'} |",
        f"| Source hypothesis | `{hypothesis_input.strip()[:200]}` |",
        "",
        "---",
        "",
        "## TL;DR",
        tldr,
        "",
        "## 1. Hypothesis",
        f"- **H0**: {h.get('h0', '(missing)')}",
        f"- **H1**: {h.get('h1', '(missing)')}",
        "",
        "## 2. Decision rule",
        f"**SHIP**: {dr.get('ship_criteria', '(missing)')}",
        "",
        f"**MARGINAL**: {dr.get('marginal_criteria', '') or '(none)'}",
        "",
        f"**FAIL**: {dr.get('fail_criteria', 'all other outcomes')}",
        "",
        (
            "**Literature-conditional exemption**: ENABLED — see citations §6 for "
            "≥10y external support justifying BHY exemption."
            if dr.get("literature_conditional_exemption")
            else "**Literature-conditional exemption**: DISABLED — strict BHY FDR applies."
        ),
        "",
        "## 3. Multiple-testing impact",
        f"- **n_trials added**: {n_imp.get('n_added', '?')}",
        f"- **Rationale**: {n_imp.get('rationale', '(missing)')}",
        "",
        "## 4. Data requirements",
        _bullet_list(data_req),
        "",
        "## 5. Predictions (frozen at registration)",
        _bullet_list(preds),
        "",
        "## 6. Implementation steps",
        "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) if steps else "(none specified)",
        "",
        "## 7. Success criteria",
        success,
        "",
        "## 8. Pre-registered failure modes",
        _bullet_list(failures),
        "",
        "## 9. Out of scope",
        _bullet_list(oos),
        "",
        "## 10. Literature anchors",
        _bullet_list(cites),
        "",
        "## 11. Risks & caveats",
        risks,
        "",
        "---",
        "",
        "## Reviewer checklist (supervisor)",
        "",
        "- [ ] Hypothesis is falsifiable + has clear directional expectation",
        "- [ ] Decision rule is quantitative (no fuzzy 'looks promising')",
        "- [ ] n_trials impact is honest (don't lowball to dodge BHY penalty)",
        "- [ ] Literature anchors are real + justify the directional prior",
        "- [ ] Failure modes are pre-registered (no post-hoc rationalization)",
        "- [ ] Implementation steps don't reference forbidden paths",
        "- [ ] Predictions are specific enough that *future-you* can grade them",
        "",
        "If all checked: save this file to `docs/<filename>.md`, run "
        "`engine.preregistration.register_spec(<path>)`, and the spec hash "
        "joins SpecRegistry. EFFECTIVE_N_TRIALS auto-increments by n_added.",
    ]
    return "\n".join(parts)


# ── Public API ───────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = (
    "You are the Auto-Spec Drafter for Macro Alpha Pro, a production-grade "
    "applied AI-quant system MVP. You operate at the PROPOSER layer — the "
    "supervisor (a single human) reviews and freezes every draft before "
    "it joins the spec registry. You never decide whether a hypothesis "
    "is good; you draft the pre-registration spec given the hypothesis "
    "the supervisor provides.\n\n"
    "Discipline:\n"
    "  - Output only the structured JSON; no prose preamble or postscript.\n"
    "  - Cite real, well-known finance / ML papers in literature_anchors. "
    "Never fabricate DOIs.\n"
    "  - n_trials_impact.n_added is honest. A 1-hypothesis spec = 1 trial. "
    "Multi-arm test = number of arms. Don't lowball to dodge BHY penalty.\n"
    "  - implementation_steps reference engine/* paths but NEVER "
    "engine/auto_audit*, engine/preregistration.py, engine/memory.py, "
    "engine/db_models.py — these are audit infrastructure.\n"
    "  - predictions are specific enough that future-you can grade them. "
    "'Sharpe > 0.3 OOS' is good; 'should be positive' is too vague.\n"
    "  - failure_modes are pre-registered, not post-hoc. List 3-5 honest ways "
    "this could go wrong.\n"
    "  - All quantitative claims must be defensible. If a paper claims 'BAB "
    "Sharpe ~0.5-0.8 long-run', cite it; don't invent numbers.\n"
)


def draft_spec(
    hypothesis_text:  str,
    *,
    target_path:      str | None = None,
    daily_budget_usd: float      = 0.05,
    temperature:      float      = 0.3,
) -> SpecDraft:
    """Generate a draft pre-registration spec from a natural-language hypothesis.

    Args
    ----
    hypothesis_text : the supervisor's verbal hypothesis statement, in
                      English or Chinese. 50-1000 chars typical.
    target_path : optional; if supplied, drafter checks SpecRegistry and
                  refuses to draft when path already exists.
    daily_budget_usd : refuse to call LLM if today's spend would exceed this.
    temperature : Gemini temperature (low for grounded drafting).

    Returns SpecDraft with .status indicating outcome.
    """
    hypothesis_text = (hypothesis_text or "").strip()
    if not hypothesis_text:
        return SpecDraft(
            hypothesis_input=hypothesis_text,
            status="rejected",
            error_msg="empty hypothesis_text",
            safety_findings=["hypothesis_text was empty or whitespace-only"],
        )

    # ── Pre-flight: spec-already-exists check ────────────────────────────────
    if target_path:
        try:
            from engine.memory import SessionFactory
            from engine.db_models import SpecRegistry
            with SessionFactory() as s:
                exists = s.query(SpecRegistry).filter_by(spec_path=target_path).first()
                if exists is not None:
                    return SpecDraft(
                        hypothesis_input=hypothesis_text,
                        target_path=target_path,
                        status="spec_exists",
                        error_msg=(
                            f"target_path={target_path!r} already in SpecRegistry "
                            f"(id={exists.id} status={exists.status}). Use the amendment "
                            f"flow instead of drafting a new spec on top."
                        ),
                    )
        except Exception as exc:
            logger.warning("spec_drafter pre-flight DB check failed: %s", exc)

    # ── Build prompt ─────────────────────────────────────────────────────────
    prompt = (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        f"=== Supervisor's hypothesis ===\n"
        f"{hypothesis_text}\n\n"
        f"{f'=== Target spec path === {target_path}' if target_path else ''}\n\n"
        f"Draft the spec now. Output only the JSON structure."
    )

    # ── Call Gemini with structured output ───────────────────────────────────
    try:
        from engine.key_pool import get_pool
        pool = get_pool()
        model = pool.get_model(
            model_name="gemini-2.5-flash",
            response_schema=SPEC_DRAFT_RESPONSE_SCHEMA,
            temperature=temperature,
        )
        resp = model.generate_content(prompt)
        pool.report_success(has_content=True)
    except Exception as exc:
        logger.exception("spec_drafter: Gemini call failed")
        return SpecDraft(
            hypothesis_input=hypothesis_text,
            target_path=target_path,
            status="llm_error",
            error_msg=str(exc)[:200],
        )

    raw_text = getattr(resp, "text", None) or str(resp)
    usage    = getattr(resp, "usage_metadata", None)
    in_tok   = int(getattr(usage, "prompt_token_count", 0) or 0)
    out_tok  = int(
        (getattr(usage, "candidates_token_count", 0) or 0)
        + (getattr(usage, "thoughts_token_count", 0) or 0)
    )
    cost = (in_tok * 0.30 + out_tok * 2.50) / 1_000_000.0

    try:
        metadata = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as exc:
        return SpecDraft(
            hypothesis_input=hypothesis_text,
            target_path=target_path,
            status="llm_error",
            error_msg=f"response_schema parse failed: {str(exc)[:100]}",
            cost_usd=cost,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    # ── Render markdown + run safety gate ────────────────────────────────────
    draft_md = _render_markdown(metadata, hypothesis_text, target_path)
    findings = _safety_gate(metadata, draft_md)

    status = "ok" if not findings else "rejected"

    return SpecDraft(
        hypothesis_input=hypothesis_text,
        target_path=target_path,
        status=status,
        metadata=metadata,
        draft_markdown=draft_md,
        safety_findings=findings,
        cost_usd=cost,
        input_tokens=in_tok,
        output_tokens=out_tok,
        error_msg=(
            f"safety_gate blocked: {findings[:2]}" if findings else None
        ),
    )
