"""engine.agents.autopilot_pre_compute_da — Pre-compute DA gate for F14b.

Phase 2.0 step 6 (2026-06-06). Mirrors the structure of the existing
post-compute DA (engine.agents.autopilot_devils_advocate, Phase 4) but
runs BEFORE Composer + strict-gate math instead of AFTER.

Question this DA answers:
  "Given graveyard history + family doctrine + n_trials budget,
   is this candidate even worth spending the ~$0.005 compose call
   and 60-90s wall time to test?"

Wired into autopilot_live.run_top1 BETWEEN top-1 selection and
compose. If pre-compute DA returns worth_running=False, the F14b
cycle short-circuits with `candidate_skipped_pre_compute` event;
strict-gate spend saved.

Cost economics:
  Pre-compute DA call: ~$0.005 (Deepseek V4 Pro, ~1.5k in + 400 out)
  Saved on skip      : ~$0.005 (compose) + 60-90s wall + emit cycle
  Saved when worth_running=False  → roughly break-even direct $$ but
  also avoids polluting events.jsonl with a RED that adds n_trials
  noise (Lopez de Prado: every test counts toward family n_trials,
  so SKIPPING is strictly better than RUNNING+RED on weak candidates).

Doctrine alignment:
  - Pattern 5 ban: single-call + JSON-schema tool_use, NO multi-agent
    debate. Same shape as post-compute DA + Phase 1.5 filter + B
    review.
  - DA output is gating, not advisory: worth_running=False halts the
    cycle. This is the only place an LLM call can prevent strict-gate
    spend (acceptable risk because the cost cap on a wrong-skip is
    bounded — caller can re-run with --force-compute to bypass).
  - Default stance leans PROCEED (the candidate already cleared
    upstream gates: D signals + B review + principal approval); DA's
    job is to catch graveyard-redundancy + post-pub-decay + family-
    over-mining signals that slipped through.
"""
from __future__ import annotations

import dataclasses as _dc
import json
import logging
from typing import Optional

# Top-level import for monkeypatch in tests (matches synthesis.py /
# strengthener/review.py convention). Production code references
# llm_call via module-level binding so tests patch
# autopilot_pre_compute_da.llm_call cleanly.
from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


@_dc.dataclass(frozen=True)
class PreComputeVerdict:
    """Structured pre-compute DA output."""
    worth_running:    bool              # True = proceed to compose + strict gate
    attack_vector:    str               # if NOT worth, why not (one sentence)
    reasoning:        str               # 2-4 sentence expansion
    confidence:       float             # 0..1 — DA's confidence in this call
    raw_response:     str               # full LLM text for forensics


_SYSTEM_PROMPT = """\
You are the PRE-COMPUTE gate for a single-quant factor-test pipeline.

A candidate has reached the strict-gate execution stage. Before we
spend the ~$0.005 compose call and ~60-90s of wall time running it,
you have a final cheap-veto opportunity.

Default stance: PROCEED. The candidate already cleared:
  - D's graveyard / cluster screen (book monitor)
  - B's institutional skeptical review (strengthener)
  - The principal's approval gate (/approvals)

Your job is to catch what slipped through. Specifically:

  1. GRAVEYARD REDUNDANCY — does the candidate's spec essentially
     replicate a cell that already has multiple REDs in the same
     family + signal? If so, the strict gate will produce another
     RED + add n_trials noise. SKIP.

  2. METHODOLOGY DEAD-END — is the candidate's test methodology a
     known dead-end pattern (e.g. backward-looking regime detection,
     overfit weighting tail, sub-3y window)? SKIP.

  3. SOURCE-PAPER POST-PUB DECAY DOMINANT — if the source paper is
     > 8 years old AND the test methodology specifies an OOS window
     that opens after publication AND the family is "EARNINGS_DRIFT"
     / "POST_EARNINGS_DRIFT" / "VALUE" / similar over-mined family,
     the expected outcome is RED — SKIP unless the candidate provides
     a NEW mechanism for why decay won't dominate.

  4. n_TRIALS BUDGET CONCERN — if the family already has > 20 trials
     this quarter (you'll see family_recent_test_count), the deflated
     Sharpe bar is already at near-impossible levels. SKIP unless
     the candidate's predicted_magnitude is exceptional.

  5. CLEAR-PROCEED INDICATORS — if the candidate:
        - cites unique data sources we have
        - uses a methodology that's NOT been recently tested
        - has a Cochrane discount-rate story (behavioral/risk/friction)
        - addresses a known decay in a currently-deployed sleeve
     Then PROCEED with confidence.

Output via the emit_pre_compute_verdict tool. ONE veto vector + 2-4
sentence reasoning. ALWAYS call the tool — no plain-text escape.

Be SHARP and OPINIONATED. A False here saves real money; a False
when it should have been True wastes a real opportunity. Calibrate.
"""


_TOOL_DEFINITION = {
    "name": "emit_pre_compute_verdict",
    "description": (
        "Emit a structured pre-compute gating verdict for this F14b "
        "candidate. Exactly one call. worth_running=True proceeds to "
        "compose; False short-circuits and emits skip event."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "worth_running": {
                "type": "boolean",
                "description": (
                    "True if this candidate should proceed to compose + "
                    "strict gate; False to skip and save compute."
                ),
            },
            "attack_vector": {
                "type":  "string",
                "description": (
                    "If worth_running=False: ONE sentence naming the "
                    "specific reason to skip (e.g. 'EARNINGS_DRIFT "
                    "family has 7 recent REDs; this spec is the same "
                    "cell with weighting tweak'). If True: brief "
                    "statement of why this clears the gate (e.g. "
                    "'novel data source + addresses carry decay')."
                ),
            },
            "reasoning": {
                "type":  "string",
                "description": (
                    "2-4 sentences expanding the attack with concrete "
                    "reference to the candidate's spec / paper / "
                    "graveyard hits."
                ),
            },
            "confidence": {
                "type":  "number",
                "description": (
                    "Self-reported confidence (0..1) in this gate "
                    "decision. Low confidence on a worth_running=False "
                    "means the caller may want to --force-compute."
                ),
            },
        },
        "required": [
            "worth_running", "attack_vector", "reasoning", "confidence",
        ],
        "additionalProperties": False,
    },
}


def _build_user_message(
    *,
    spec,
    claim_text:                  str,
    graveyard_matches:           list,
    family_recent_test_count:    int,
    paper_age_years:             Optional[float],
    addresses_decay_in:          Optional[str],
) -> str:
    """Construct the user-message with all the evidence the pre-compute
    DA needs. Mirrors the post-compute DA shape minus metrics (we
    haven't run strict gate yet)."""
    primary = spec.legs[0] if spec.legs else None
    lines: list[str] = []
    lines.append("CANDIDATE TO GATE")
    lines.append("=================")
    lines.append(f"family:       {spec.family.value}")
    lines.append(f"signal_type:  {primary.signal_type.value if primary else 'NONE'}")
    lines.append(f"universe:     {spec.universe.asset_class.value}/{spec.universe.subset.value}")
    lines.append(f"weighting:    {spec.construction.weighting.value}")
    lines.append(f"rebalance:    {spec.construction.rebalance.value}")
    lines.append(f"source_hyp:   {(spec.source_hypothesis_id or '')[:12]}")
    if paper_age_years is not None:
        lines.append(f"paper_age:    {paper_age_years:.1f} years")
    if addresses_decay_in:
        lines.append(f"addresses_decay_in: {addresses_decay_in}")
    lines.append("")

    lines.append(f"GRAVEYARD HITS ({len(graveyard_matches)})")
    lines.append("-" * 40)
    if not graveyard_matches:
        lines.append("  (no graveyard matches found)")
    for m in graveyard_matches[:5]:
        # graveyard match shape is from find_redundancy_for_spec;
        # surface what makes it a hit
        if isinstance(m, dict):
            lines.append(
                f"  family={m.get('family')} "
                f"signal={m.get('signal_type')} "
                f"verdict={m.get('verdict', '?')} "
                f"score={m.get('score', '?')}"
            )
        else:
            lines.append(f"  {str(m)[:120]}")
    lines.append("")

    lines.append(f"FAMILY n_TRIALS THIS QUARTER: {family_recent_test_count}")
    lines.append("")

    lines.append("CLAIM FROM HYPOTHESIS")
    lines.append("-" * 40)
    lines.append((claim_text or "").strip()[:500])
    lines.append("")

    lines.append("Call emit_pre_compute_verdict now.")
    return "\n".join(lines)


def run_autopilot_pre_compute_da(
    *,
    spec,
    claim_text:                str,
    graveyard_matches:         list,
    family_recent_test_count:  int = 0,
    paper_age_years:           Optional[float] = None,
    addresses_decay_in:        Optional[str]   = None,
) -> Optional[PreComputeVerdict]:
    """Fire pre-compute DA on a candidate spec. Returns None on hard
    failure (LLM error, no tool call) — caller treats None as 'no
    gate decision, proceed with compose' (fail-OPEN). The cost of a
    wrong proceed is one strict-gate run; the cost of a wrong skip
    is a missed opportunity — biasing toward fail-open keeps the
    research pipeline making forward progress even when DA is down.
    """
    user_msg = _build_user_message(
        spec                      = spec,
        claim_text                = claim_text,
        graveyard_matches         = graveyard_matches,
        family_recent_test_count  = family_recent_test_count,
        paper_age_years           = paper_age_years,
        addresses_decay_in        = addresses_decay_in,
    )

    try:
        result = llm_call(
            workload   = "devils_advocate",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "devils_advocate",
            tools      = [_TOOL_DEFINITION],
            max_tokens = 1200,
            scope      = "autopilot_pre_compute_da",
        )
    except Exception as exc:
        logger.warning("pre_compute DA: llm_call failed: %s", exc)
        return None

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_pre_compute_verdict":
            payload = tc.input
            break
    if payload is None:
        logger.warning("pre_compute DA: model returned text without tool_call; raw=%s",
                        (result.text or "")[:200])
        return None

    try:
        return PreComputeVerdict(
            worth_running = bool(payload.get("worth_running", True)),
            attack_vector = str(payload.get("attack_vector", "")).strip(),
            reasoning     = str(payload.get("reasoning", "")).strip(),
            confidence    = float(payload.get("confidence", 0.5)),
            raw_response  = result.text or json.dumps(payload),
        )
    except Exception as exc:
        logger.warning("pre_compute DA: payload → PreComputeVerdict failed: %s", exc)
        return None


# ────────────────────────────────────────────────────────────────────
# Gate decision — used by autopilot_live wire-up + isolated for tests
# ────────────────────────────────────────────────────────────────────
def decide_pre_compute_gate(
    *,
    spec,
    claim_text:                str,
    graveyard_matches:         list,
    family_recent_test_count:  int = 0,
    paper_age_years:           Optional[float] = None,
    addresses_decay_in:        Optional[str] = None,
    skip:                      bool = False,
) -> tuple[bool, Optional[PreComputeVerdict]]:
    """Decision wrapper for autopilot_live.run_top1.

    Returns (proceed, verdict):
      (True,  None)            — bypass requested (skip=True)
      (True,  None)            — DA failed (None) → fail-OPEN, proceed
      (True,  PreComputeVerdict)  — DA approved (worth_running=True)
      (False, PreComputeVerdict)  — DA gated (worth_running=False) →
                                    caller MUST emit
                                    candidate_skipped_pre_compute + abandon
    """
    if skip:
        return True, None

    verdict = run_autopilot_pre_compute_da(
        spec                      = spec,
        claim_text                = claim_text,
        graveyard_matches         = graveyard_matches,
        family_recent_test_count  = family_recent_test_count,
        paper_age_years           = paper_age_years,
        addresses_decay_in        = addresses_decay_in,
    )
    if verdict is None:
        # Fail-OPEN: DA being down should not block research.
        return True, None
    return verdict.worth_running, verdict
