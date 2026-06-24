"""engine.agents.strengthener.review — Phase 2.0 step 11a.

Per-hypothesis second-pass review for Employee B. Pure LLM call (no
I/O); a separate runner (step 11c) handles loading hypotheses from
hypotheses.jsonl, looping, persisting verdicts, and creating
/approvals rows.

Architecturally per [[spec-research-session-orchestrator-2026-06-06]]
§"Employee B":

  - Pattern: single-agent LLM call with strict JSON schema tool_use
            (NOT Pattern 5 multi-agent debate)
  - Workload: strengthener_review (Sonnet 4.6)
  - Cost ceiling: $0.05/hypothesis review
  - Input: ONE Hypothesis + frozen context (deployed sleeves + active
           doctrine snippets + recent verdicts in same family)
  - Output: StrengthenerVerdict with exactly one of three verdict types

B's job is to be a SKEPTICAL second reviewer. A's job was to PROPOSE
across sources; B's job is to ASK whether the principal should
spend pipeline-budget actually testing the proposal. Default is
REJECT — burden is on the candidate to clear the bar.

The third path (DOCTRINE_AMENDMENT_NEEDED) is rare: only emit when
the candidate is strong but ACTIVE doctrine is what's blocking it,
AND the candidate provides enough evidence to question that
doctrine. The principal still owns the doctrine decision; B only
PROPOSES the amendment.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from enum import Enum
from typing import Optional

# Top-level import for monkeypatch in tests (same pattern as synthesis.py)
from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Verdict types — the contract B emits
# ────────────────────────────────────────────────────────────────────
class VerdictType(str, Enum):
    APPROVE_FOR_PIPELINE       = "APPROVE_FOR_PIPELINE"
    REJECT                     = "REJECT"
    DOCTRINE_AMENDMENT_NEEDED  = "DOCTRINE_AMENDMENT_NEEDED"


# ────────────────────────────────────────────────────────────────────
# Input shape — what the runner passes in per hypothesis
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class HypothesisRef:
    """Lean projection of the Hypothesis row we want B to reason over.
    Built by the runner (step 11c) from the full Hypothesis dataclass."""
    hypothesis_id:          str
    claim:                  str
    mechanism_family:       str
    mechanism_subtype:      str
    predicted_direction:    str
    predicted_magnitude:    str
    required_data:          tuple[str, ...]
    test_methodology:       str
    extraction_method:      str           # "llm_synthesis" / "llm_extract" / "human_authored"
    synthesizes_paper_ids:  tuple[str, ...]
    synthesizes_event_ids:  tuple[str, ...]
    addresses_decay_in:     Optional[str]
    created_ts:             str
    # Phase 2.2c: aggregate citation_verifier output. None = not yet
    # verified (old rows / synthesis before Phase 2.2b). B's prompt
    # surfaces it as a load-bearing signal: if any_unresolved=True,
    # this candidate likely cites a hallucinated paper and B should
    # weight that heavily toward REJECT.
    citation_quality:       "dict | None" = None


@_dc.dataclass(frozen=True)
class SleeveContextRef:
    """One deployed sleeve to surface in B's context. Lets B reason
    'A's candidate is too similar to deployed sleeve X' or 'X is
    decaying, candidate is a credible replacement'."""
    sleeve_id:           str
    family:              str
    ann_sharpe_live:     Optional[float]
    months_since_deploy: Optional[int]
    last_decay_alert:    Optional[str]


@_dc.dataclass(frozen=True)
class DoctrineContextRef:
    """One active doctrine snippet — typically pulled by family match
    or by topical similarity. Empty list is valid (B reviews without
    doctrine in scope)."""
    memory_file_id: str
    headline:       str
    snippet:        str          # ≤ 400 chars
    relevance_note: str          # why the runner thinks it's relevant


@_dc.dataclass(frozen=True)
class FamilyVerdictRef:
    """Recent factor verdict in the same family as the hypothesis.
    Lets B see 'this family has 8 RED verdicts in 30 days' even when
    no doctrine signal exists yet for it."""
    event_id:  str
    subject_id:str
    verdict:   str               # GREEN / MARGINAL / RED
    ts:        str
    summary:   str


@_dc.dataclass(frozen=True)
class StrengthenerInput:
    """Frozen snapshot per hypothesis review. The runner builds one
    of these per hypothesis it loops over."""
    hypothesis:        HypothesisRef
    deployed_sleeves:  tuple[SleeveContextRef, ...]
    doctrine_snippets: tuple[DoctrineContextRef, ...]
    family_verdicts:   tuple[FamilyVerdictRef, ...]
    snapshot_ts:       str


# ────────────────────────────────────────────────────────────────────
# Output shape
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class StrengthenerVerdict:
    """B's typed output. Persisted as approval_row payload (step 13)
    + tagged onto the Hypothesis."""
    hypothesis_id:               str
    verdict_type:                VerdictType
    one_line_summary:            str
    confidence:                  float         # 0.0-1.0
    reasoning:                   str           # 2-5 sentence justification
    similar_to_deployed:         Optional[str] # sleeve_id if "too similar"
    replaces_decaying:           Optional[str] # sleeve_id if good replacement
    blocking_doctrine_id:        Optional[str] # memory_file_id when verdict=DOCTRINE_AMENDMENT_NEEDED
    proposed_amendment_summary:  Optional[str] # ≤ 400 chars; only when verdict=DOCTRINE_AMENDMENT_NEEDED
    recommended_pipeline_action: Optional[str] # e.g. "run f14b strict gate"; only when APPROVE
    risk_flags:                  tuple[str, ...]

    # Call diagnostics
    review_ts: str
    model:     str


# ────────────────────────────────────────────────────────────────────
# System prompt
# ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are Employee B — the strengthener for a solo-quant research book.
You are the SECOND reviewer, after Employee A proposed this hypothesis
across sources. Your job is to be SKEPTICAL and decide whether the
principal should spend pipeline-budget on it.

The principal's prior:
  - Hou-Xue-Zhang 65% non-replication rate. Most proposals DIE in
    the strict gate. Don't waste pipeline budget on weak candidates.
  - Post-publication decay (McLean-Pontiff 32-58%, Linnainmaa-Roberts).
    A factor with public discovery should be expected to decay.
  - The strict gate (F14b autopilot) is expensive (~$2-5/candidate
    + capital risk). Burden is on the candidate to clear B's bar
    BEFORE running.

You emit exactly ONE of three verdict types via the emit_review tool:

  APPROVE_FOR_PIPELINE
    Candidate is strong enough to spend strict-gate budget on. Set
    confidence ≥ 0.6 and explain WHY it clears the bar (which
    weaknesses you weighed against which strengths). Default is NOT
    this — only choose it if the candidate is genuinely worth the
    principal's review attention.

  REJECT
    Default verdict. Examples that should land here:
      - Too similar to a deployed sleeve (set similar_to_deployed)
      - Same family is in a RED cluster (cite recent_family_verdicts)
      - Mechanism contradicts a deployed sleeve's known reason for
        success without explaining how it could co-exist
      - Required data we don't have
      - Test methodology is hand-wavy
      - Hypothesis cites no concrete predicted_magnitude
    REJECT is NOT a failure — it's saving pipeline budget.

  DOCTRINE_AMENDMENT_NEEDED
    Rare. Only emit when:
      - The candidate is genuinely strong (would otherwise APPROVE)
      - AND active doctrine snippet X is what's blocking it
      - AND the candidate provides enough evidence to question
        doctrine X
    Set blocking_doctrine_id = the memory_file_id; populate
    proposed_amendment_summary (≤ 400 chars) with the SPECIFIC
    change you'd propose. The principal still owns the doctrine
    decision; you only PROPOSE it.

Hard rules:
  - similar_to_deployed and replaces_decaying CANNOT both be set
  - blocking_doctrine_id required when verdict=DOCTRINE_AMENDMENT_NEEDED
  - proposed_amendment_summary required when verdict=DOCTRINE_AMENDMENT_NEEDED
  - recommended_pipeline_action required when verdict=APPROVE_FOR_PIPELINE
  - reasoning MUST cite SPECIFIC inputs (sleeve_id / event_id /
    memory_file_id), not generalities
  - one_line_summary MUST be ≤ 200 chars

COMFORT-BIAS GUARD (anti-mental-rut, 2026-06-07):
  - The principal works solo and is at risk of selection bias toward
    mechanisms similar to currently-deployed sleeves ("comfortable"
    candidates that confirm existing beliefs).
  - When evaluating, explicitly NOTE whether this candidate's
    mechanism feels "comfortable" (aligns with deployed sleeves)
    vs "uncomfortable" (orthogonal or contrarian relative to
    deployed).
  - If "comfortable": add a "comfort_bias_risk" entry to risk_flags
    AND require STRONGER evidence than usual to clear APPROVE bar.
    Comfortable candidates that aren't materially novel produce low
    marginal alpha — biased toward REJECT.
  - If "uncomfortable" (genuinely orthogonal): this does NOT
    automatically clear the bar, but bias slightly toward APPROVE
    on the margin — novel angles that the strict gate can falsify
    are net positive for the principal even if confidence is moderate.
  - This is NOT a softening of the REJECT default. It is a TIGHTENING
    against comfortable repetition and a marginal nudge to surface
    angles the principal isn't already considering.

Tone: institutional, skeptical, decisive. Never recommend anything
just to seem productive. REJECT with clear reasoning is the most
common valid output.

Call emit_review with your verdict. ALWAYS call it — no plain-text
escape.
"""


# ────────────────────────────────────────────────────────────────────
# Tool schema
# ────────────────────────────────────────────────────────────────────
_TOOL_DEFINITION = {
    "name": "emit_review",
    "description": (
        "Emit the strengthener's verdict on this hypothesis. Exactly "
        "one verdict per call. Conditional fields enforced in code."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict_type": {
                "type": "string",
                "enum": [v.value for v in VerdictType],
            },
            "one_line_summary":            {"type": "string", "maxLength": 200},
            "confidence":                  {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning":                   {"type": "string"},
            "similar_to_deployed":         {"type": ["string", "null"]},
            "replaces_decaying":           {"type": ["string", "null"]},
            "blocking_doctrine_id":        {"type": ["string", "null"]},
            "proposed_amendment_summary":  {"type": ["string", "null"], "maxLength": 400},
            "recommended_pipeline_action": {"type": ["string", "null"]},
            "risk_flags":                  {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
        },
        "required": [
            "verdict_type", "one_line_summary", "confidence", "reasoning",
            "risk_flags",
        ],
        "additionalProperties": False,
    },
}


# ────────────────────────────────────────────────────────────────────
# Prompt construction
# ────────────────────────────────────────────────────────────────────
def _format_input(si: StrengthenerInput) -> str:
    """Build the user message Sonnet reads. Sections are explicit so
    the LLM can address each piece of evidence in reasoning."""
    h = si.hypothesis

    lines: list[str] = []
    lines.append(f"REVIEW REQUEST — snapshot_ts {si.snapshot_ts}")
    lines.append("")

    lines.append("HYPOTHESIS UNDER REVIEW")
    lines.append("-" * 40)
    lines.append(f"  hypothesis_id        : {h.hypothesis_id}")
    lines.append(f"  extraction_method    : {h.extraction_method}")
    lines.append(f"  mechanism_family     : {h.mechanism_family} / {h.mechanism_subtype}")
    lines.append(f"  predicted_direction  : {h.predicted_direction}")
    lines.append(f"  predicted_magnitude  : {h.predicted_magnitude}")
    lines.append(f"  required_data        : {', '.join(h.required_data) or '(none specified)'}")
    lines.append(f"  test_methodology     : {h.test_methodology}")
    lines.append(f"  addresses_decay_in   : {h.addresses_decay_in or '(none)'}")
    lines.append(f"  synthesizes (papers) : {len(h.synthesizes_paper_ids)} paper(s)")
    lines.append(f"  synthesizes (events) : {len(h.synthesizes_event_ids)} event(s)")
    lines.append(f"  CLAIM:")
    lines.append(f"    {h.claim}")
    lines.append("")

    # Phase 2.2c: citation verification block. Surfacing this in the
    # user message lets B downweight candidates whose synthesizer
    # claims aren't substantiated by the cited papers' actual chunks.
    cq = h.citation_quality
    lines.append("CITATION VERIFICATION")
    lines.append("-" * 40)
    if cq is None:
        lines.append("  (not verified — pre-2.2b hypothesis or verifier "
                      "unavailable; treat citations as un-checked)")
    else:
        flag = "⚠️ LOW CONFIDENCE" if cq.get("low_confidence_flag") else "OK"
        lines.append(f"  status              : {flag}")
        lines.append(f"  papers cited        : {cq.get('n_papers_cited', 0)} "
                      f"(resolved {cq.get('n_resolved', 0)}, "
                      f"unresolved {cq.get('n_unresolved', 0)})")
        lines.append(f"  mean confidence     : {cq.get('mean_confidence', 0):.2f}")
        lines.append(f"  min confidence      : {cq.get('min_confidence', 0):.2f}")
        if cq.get("any_unresolved"):
            lines.append("  ⚠️ at least one cited paper NOT in the registry "
                          "— likely hallucinated citation. Weight this "
                          "heavily toward REJECT unless other evidence "
                          "strongly compels otherwise.")
    lines.append("")

    lines.append(f"DEPLOYED SLEEVES ({len(si.deployed_sleeves)})")
    lines.append("-" * 40)
    if not si.deployed_sleeves:
        lines.append("  (none)")
    for s in si.deployed_sleeves:
        bits = [s.sleeve_id, s.family]
        if s.ann_sharpe_live is not None:
            bits.append(f"Sharpe={s.ann_sharpe_live:.2f}")
        if s.months_since_deploy is not None:
            bits.append(f"{s.months_since_deploy}mo")
        if s.last_decay_alert:
            bits.append(f"decay@{s.last_decay_alert[:10]}")
        lines.append(f"  {' | '.join(bits)}")
    lines.append("")

    lines.append(f"ACTIVE DOCTRINE SNIPPETS ({len(si.doctrine_snippets)})")
    lines.append("-" * 40)
    if not si.doctrine_snippets:
        lines.append("  (none — review without doctrine in scope)")
    for d in si.doctrine_snippets:
        lines.append(f"  [{d.memory_file_id}] {d.headline}")
        lines.append(f"    relevance: {d.relevance_note}")
        lines.append(f"    snippet: {d.snippet[:300]}")
    lines.append("")

    lines.append(f"RECENT FAMILY VERDICTS ({len(si.family_verdicts)} — same family)")
    lines.append("-" * 40)
    if not si.family_verdicts:
        lines.append("  (none in recent window)")
    for v in si.family_verdicts:
        lines.append(f"  [{v.ts[:10]}] {v.subject_id:30s} {v.verdict:8s} {v.summary[:80]}")
    lines.append("")

    lines.append("Call emit_review with your verdict per the tool schema.")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# Validation of the LLM-emitted dict
# ────────────────────────────────────────────────────────────────────
def _parse_verdict(
    *,
    hypothesis_id: str,
    raw: dict,
    model: str,
    review_ts: str,
) -> Optional[StrengthenerVerdict]:
    """Parse + cross-field validate the LLM's tool input. Returns None
    if a required cross-field check fails (caller treats None as a
    rejected emit — same fail-safe path as a missing tool call)."""
    try:
        vt = VerdictType(raw["verdict_type"])
    except (KeyError, ValueError):
        logger.warning("strengthener: invalid/missing verdict_type")
        return None

    one_line = raw.get("one_line_summary", "")
    if not one_line or len(one_line) > 200:
        logger.warning("strengthener: one_line_summary missing or too long")
        return None

    reasoning = raw.get("reasoning", "")
    if not reasoning:
        logger.warning("strengthener: reasoning required")
        return None

    sim = raw.get("similar_to_deployed") or None
    repl = raw.get("replaces_decaying") or None
    if sim and repl:
        logger.warning("strengthener: similar_to_deployed AND replaces_decaying both set")
        return None

    blocking = raw.get("blocking_doctrine_id") or None
    amend    = raw.get("proposed_amendment_summary") or None
    pipeline = raw.get("recommended_pipeline_action") or None

    if vt == VerdictType.DOCTRINE_AMENDMENT_NEEDED:
        if not blocking or not amend:
            logger.warning(
                "strengthener: DOCTRINE_AMENDMENT_NEEDED requires "
                "blocking_doctrine_id + proposed_amendment_summary"
            )
            return None
    if vt == VerdictType.APPROVE_FOR_PIPELINE and not pipeline:
        logger.warning(
            "strengthener: APPROVE_FOR_PIPELINE requires "
            "recommended_pipeline_action"
        )
        return None

    try:
        conf = float(raw.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = 0.5

    flags = raw.get("risk_flags") or []
    if not isinstance(flags, list):
        flags = []

    return StrengthenerVerdict(
        hypothesis_id              = hypothesis_id,
        verdict_type               = vt,
        one_line_summary           = one_line,
        confidence                 = conf,
        reasoning                  = reasoning,
        similar_to_deployed        = sim,
        replaces_decaying          = repl,
        blocking_doctrine_id       = blocking,
        proposed_amendment_summary = amend,
        recommended_pipeline_action= pipeline,
        risk_flags                 = tuple(str(f)[:120] for f in flags[:5]),
        review_ts                  = review_ts,
        model                      = model,
    )


# ────────────────────────────────────────────────────────────────────
# Public entry — one LLM call per hypothesis
# ────────────────────────────────────────────────────────────────────
def run_strengthener_review(si: StrengthenerInput) -> Optional[StrengthenerVerdict]:
    """Make ONE strengthener_review call for the given hypothesis.

    Fail-safe: returns None on any unrecoverable LLM / parsing
    failure. The runner (step 11c) treats None as "review failed —
    skip, retry next pass". Never raises into the caller.
    """
    review_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        result = llm_call(
            workload   = "strengthener_review",
            system     = _SYSTEM_PROMPT,
            user       = _format_input(si),
            agent_id   = "strengthener_review",
            tools      = [_TOOL_DEFINITION],
            max_tokens = 2000,
        )
    except Exception as exc:
        logger.exception("strengthener: llm_call raised: %s", exc)
        return None

    # Find the emit_review tool call
    tool_call = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_review":
            tool_call = tc
            break
    if tool_call is None:
        logger.warning(
            "strengthener: model did not call emit_review; "
            "stop_reason=%s text=%s",
            result.stop_reason, (result.text or "")[:200],
        )
        return None

    raw = tool_call.input
    if not isinstance(raw, dict):
        try:
            raw = json.loads(raw)
        except Exception:
            logger.warning("strengthener: tool input not a dict / not parseable")
            return None

    return _parse_verdict(
        hypothesis_id = si.hypothesis.hypothesis_id,
        raw           = raw,
        model         = result.model,
        review_ts     = review_ts,
    )
