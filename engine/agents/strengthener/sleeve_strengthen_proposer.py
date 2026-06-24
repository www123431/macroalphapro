"""engine.agents.strengthener.sleeve_strengthen_proposer — Stage B P3a.

Active B worker — given ONE deployed sleeve's full context, asks
Sonnet to propose 0-3 concrete improvement-candidate Hypotheses
targeted at THAT sleeve. NOT reactive to D signals (that's P2's
sleeve_fix_proposer); this is the proactive scan: "for sleeve X
running production, what specific improvements would tighten it?"

Pattern-5-compliant: SINGLE LLM call + strict tool-use JSON schema.
No retry loop, no agent debate, no chained reasoning. Output 0-3
candidates or [] — empty is valid + preferred over weak suggestions
(same discipline as A's synthesis, per [[project-anti-rut-doctrine-
2026-06-07]]).

Cost: ~$0.05/sleeve × 13 deployed sleeves × weekly cadence ≈ $0.65/wk.

Output Hypothesis shape (B's contract with the strengthener review
pipeline downstream):
  extraction_method     = LLM_SYNTHESIS
  addresses_decay_in    = sleeve_id (always set — proposals are
                                       per-sleeve)
  mechanism_family      = sleeve's family (inherited, not invented)
  mechanism_subtype     = template + LLM-augmented
  improvement_kind      = one of 6 controlled enum values (lives in
                                tags, drives downstream test selection)
  synthesizes_paper_ids = (sleeve's canonical_paper_id,) ∪
                           any new methodology paper IDs the LLM cited
  required_data         = LLM-listed concrete data needs
  test_methodology      = LLM-written, must reference an existing
                           engine.* module to be testable

Improvement-kind enum (controls LLM creativity surface — without
this, LLM invents arbitrary "fix categories" that aren't testable):
  regime_filter         — add VIX / OAS / yield-curve overlay
  cost_aware_exec       — implementation-shortfall / queue model
  position_weighting    — risk parity / vol-target / weight cap
  replacement_seek      — search papers for fresher mechanism
  risk_overlay          — tail hedge / pair trade
  data_quality_patch    — fix PIT issue / lookahead / staleness
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
import uuid
from typing import Optional

from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


# Controlled improvement-kind enum. Keep in sync with _TOOL_DEFINITION
# schema's enum list — single source of truth.
IMPROVEMENT_KINDS = (
    "regime_filter",
    "cost_aware_exec",
    "position_weighting",
    "replacement_seek",
    "risk_overlay",
    "data_quality_patch",
)


# ────────────────────────────────────────────────────────────────────
# Input + output dataclasses
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class SleeveContext:
    """Full context bundle for ONE deployed sleeve. Built by P3b
    (per-sleeve scan runner) from library YAML + recent verdicts +
    decay alerts + doctrine."""
    sleeve_id:              str
    family:                 str           # MechanismFamily value as string
    canonical_paper_id:     str           # internal slug
    mechanism_economics:    str           # from YAML
    canonical_universe:     str
    typical_sample:         str
    deployed_summary:       str           # KPI snapshot one-liner
    # Recent state — empty tuples OK if nothing in window
    recent_family_red_ids:  tuple[str, ...] = ()   # factor_verdict_filed RED in family
    recent_decay_alert_ids: tuple[str, ...] = ()   # doctrine_signal for THIS sleeve
    doctrine_snippet_ids:   tuple[str, ...] = ()
    snapshot_ts:            str = ""


@_dc.dataclass(frozen=True)
class StrengthenProposal:
    """Single LLM-produced improvement-candidate. Step P3b adapts to a
    Hypothesis dataclass for persistence."""
    claim:                str
    improvement_kind:     str       # one of IMPROVEMENT_KINDS
    mechanism_subtype:    str
    predicted_magnitude:  str       # "marginal" / "moderate" / "high"
    required_data:        tuple[str, ...]
    test_methodology:     str
    references_paper_ids: tuple[str, ...] = ()    # cited papers beyond canonical
    expected_outcome_prior: str = ""
    rationale:            str = ""
    # Diagnostics
    generation_ts:        str = ""
    model:                str = ""


# ────────────────────────────────────────────────────────────────────
# System prompt
# ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are Employee B (Strengthener) — the proactive sleeve-improvement
proposer for a solo-quant research book.

Your job: read ONE deployed sleeve's full state below. Propose 0-3
concrete improvement-candidate hypotheses targeted at THIS sleeve.

CONSTRAINTS (load-bearing — break any and the candidate is dropped):

  - PREFER ZERO OUTPUT over weak output. The principal's prior is
    Hou-Xue-Zhang 65% non-replication AND McLean-Pontiff 32-58%
    post-publication decay. If the sleeve is healthy and no genuine
    improvement is visible from the input, return []. The principal
    will run this every week; producing weak candidates erodes
    review attention.

  - Every candidate MUST specify exactly one `improvement_kind` from
    the controlled enum. NO new categories. If your fix doesn't fit
    one of the six, it's not a testable improvement.

  - Every candidate's `test_methodology` MUST reference at least one
    existing engine.* module path (e.g.
    "engine.validation.decay_sentinel", "engine.execution.cost_model")
    so the principal can dry-run the test without writing new
    infrastructure. Hypotheses requiring new infra cost more than
    they're worth — defer them.

  - `required_data` MUST list concrete data sources by their cache
    name or vendor (e.g. "WRDS CRSP daily", "OptionMetrics IV
    surface"). Do NOT propose fixes requiring data the deployed
    sleeve doesn't already use unless you list the exact source.

  - `predicted_magnitude` MUST be one of "marginal" / "moderate" /
    "high". Use "marginal" for spec tweaks; "moderate" for adding a
    new factor element; "high" for replacement / structural change.

  - `expected_outcome_prior` MUST be honest. Default to
    "likely_REJECT_per_HXZ_65pct" unless cross-source evidence is
    unusually strong. Marginal improvements to a working sleeve
    typically don't survive deflated-SR + paired bootstrap.

  - `references_paper_ids` lists any methodology paper IDs you cite
    BEYOND the sleeve's canonical paper. Empty list is fine. NEVER
    invent paper IDs.

DISCIPLINE FILTERS (silently downgrade or drop):

  - If the proposal is "make sleeve X look more like sleeve Y where
    Y is already deployed", DOWNGRADE expected_outcome_prior — the
    principal is at risk of homogenizing deployed sleeves
    (anti-orthogonality, see [[project-anti-rut-doctrine-2026-06-07]]).

  - If recent_decay_alert_ids is non-empty AND the proposal does NOT
    address that decay, prefer a replacement_seek or
    data_quality_patch candidate over a regime_filter or
    cost_aware_exec — addressing the live alert is higher priority.

  - If the sleeve has NO recent family REDs AND NO decay alerts AND
    deployed_summary indicates healthy KPI, return [] more often
    than not. Healthy sleeves don't need to be improved every week.

OUTPUT: invoke the emit_strengthen_proposals tool exactly once with
the candidates list. Empty list is valid output.
"""


# ────────────────────────────────────────────────────────────────────
# Tool schema (strict JSON, enforced server-side by Anthropic tool_use)
# ────────────────────────────────────────────────────────────────────
_TOOL_DEFINITION = {
    "name": "emit_strengthen_proposals",
    "description": (
        "Emit 0-3 improvement-candidate hypotheses for the input "
        "deployed sleeve. Empty list is valid and preferred over "
        "weak candidates."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim":              {"type": "string"},
                        "improvement_kind":   {
                            "type": "string",
                            "enum": list(IMPROVEMENT_KINDS),
                        },
                        "mechanism_subtype":  {"type": "string"},
                        "predicted_magnitude": {
                            "type": "string",
                            "enum": ["marginal", "moderate", "high"],
                        },
                        "required_data":      {
                            "type": "array", "items": {"type": "string"},
                            "minItems": 1,
                        },
                        "test_methodology":   {"type": "string"},
                        "references_paper_ids": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "expected_outcome_prior": {"type": "string"},
                        "rationale":          {"type": "string"},
                    },
                    "required": [
                        "claim", "improvement_kind", "mechanism_subtype",
                        "predicted_magnitude", "required_data",
                        "test_methodology", "expected_outcome_prior",
                    ],
                    "additionalProperties": False,
                },
            },
            "skip_reason": {
                "type": "string",
                "description": ("If returning [] candidates, "
                                 "one-sentence reason why this "
                                 "sleeve has no actionable "
                                 "improvement this week."),
            },
        },
        "required": ["candidates"],
        "additionalProperties": False,
    },
}


# ────────────────────────────────────────────────────────────────────
# User-message builder
# ────────────────────────────────────────────────────────────────────
def _format_input(ctx: SleeveContext) -> str:
    parts = [
        f"SLEEVE_ID:           {ctx.sleeve_id}",
        f"FAMILY:              {ctx.family}",
        f"CANONICAL_PAPER:     {ctx.canonical_paper_id}",
        f"CANONICAL_UNIVERSE:  {ctx.canonical_universe}",
        f"TYPICAL_SAMPLE:      {ctx.typical_sample}",
        f"SNAPSHOT_TS:         {ctx.snapshot_ts}",
        "",
        "MECHANISM_ECONOMICS:",
        ctx.mechanism_economics.strip(),
        "",
        "DEPLOYED_KPI_SUMMARY:",
        ctx.deployed_summary.strip(),
        "",
    ]
    if ctx.recent_family_red_ids:
        parts.append(f"RECENT_FAMILY_RED_VERDICTS "
                      f"({len(ctx.recent_family_red_ids)} in window):")
        for rid in ctx.recent_family_red_ids[:10]:
            parts.append(f"  - {rid}")
        parts.append("")
    else:
        parts.append("RECENT_FAMILY_RED_VERDICTS: none in window")
        parts.append("")
    if ctx.recent_decay_alert_ids:
        parts.append(f"RECENT_DECAY_ALERTS FOR THIS SLEEVE "
                      f"({len(ctx.recent_decay_alert_ids)}):")
        for did in ctx.recent_decay_alert_ids[:5]:
            parts.append(f"  - {did}")
        parts.append("")
    else:
        parts.append("RECENT_DECAY_ALERTS FOR THIS SLEEVE: none")
        parts.append("")
    if ctx.doctrine_snippet_ids:
        parts.append(f"RELEVANT_DOCTRINE_SNIPPETS "
                      f"({len(ctx.doctrine_snippet_ids)}):")
        for sid in ctx.doctrine_snippet_ids[:8]:
            parts.append(f"  - {sid}")
        parts.append("")
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# Main call
# ────────────────────────────────────────────────────────────────────
def run_strengthen_proposer(
    ctx: SleeveContext,
    *,
    max_tokens: int = 4096,
) -> list[StrengthenProposal]:
    """Fire ONE strengthener_propose call. Returns 0-3 proposals.

    Returns [] on hard LLM failure / unparseable response / tool not
    called. The Pattern-5-compliant pattern: single structured call,
    no retry, no agent-to-agent debate. The orchestrator (P3b) logs
    skips and proceeds to the next sleeve.
    """
    user_msg = _format_input(ctx)

    try:
        result = llm_call(
            workload   = "strengthener_propose",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "strengthener_propose",
            tools      = [_TOOL_DEFINITION],
            max_tokens = max_tokens,
            scope      = "stage_b_p3a_active_proposer",
        )
    except Exception as exc:
        logger.warning("strengthen_proposer: llm_call failed for %s: %s",
                        ctx.sleeve_id, exc)
        return []

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_strengthen_proposals":
            payload = tc.input
            break
    if payload is None:
        logger.warning("strengthen_proposer: tool not called for %s; "
                        "raw text first 200 chars: %s",
                        ctx.sleeve_id, (result.text or "")[:200])
        return []

    raw_candidates = payload.get("candidates") or []
    if not isinstance(raw_candidates, list):
        logger.warning("strengthen_proposer: candidates not a list for "
                        "%s (%s)", ctx.sleeve_id, type(raw_candidates))
        return []

    out: list[StrengthenProposal] = []
    ts_now = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for raw in raw_candidates[:3]:
        try:
            kind = str(raw["improvement_kind"])
            if kind not in IMPROVEMENT_KINDS:
                logger.warning("strengthen_proposer: %s emitted unknown "
                                "improvement_kind=%r; dropping",
                                ctx.sleeve_id, kind)
                continue
            out.append(StrengthenProposal(
                claim                = str(raw["claim"]),
                improvement_kind     = kind,
                mechanism_subtype    = str(raw["mechanism_subtype"]),
                predicted_magnitude  = str(raw["predicted_magnitude"]),
                required_data        = tuple(raw.get("required_data") or ()),
                test_methodology     = str(raw["test_methodology"]),
                references_paper_ids = tuple(raw.get("references_paper_ids") or ()),
                expected_outcome_prior = str(raw.get("expected_outcome_prior", "")),
                rationale            = str(raw.get("rationale", "")),
                generation_ts        = ts_now,
                model                = result.model,
            ))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("strengthen_proposer: %s dropping malformed "
                            "candidate: %s", ctx.sleeve_id, exc)
            continue
    return out
