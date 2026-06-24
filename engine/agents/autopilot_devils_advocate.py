"""engine.agents.autopilot_devils_advocate — DA refuter for F14b verdicts.

Phase 4 of the 4-employee roadmap (2026-06-05). The full DA briefing
with chunk-id-grounded verbatim quotes (per da_briefing schema) is a
separate heavier project. This module is the LIGHTWEIGHT DA wired
into the F14b verdict pipeline:

  - Fires ONLY when F14b verdict is GREEN or MARGINAL
    (no point challenging a kill)
  - Single Deepseek V4 Pro call with capability evidence + metrics
  - Returns a structured critique (refuted bool + severity + attack)
  - Output drives a verdict-downgrade rule:
      severity=high   → GREEN → RED, MARGINAL → RED
      severity=medium → GREEN → MARGINAL, MARGINAL unchanged + 'da_caution'
      severity=low    → no downgrade, 'da_noted'
      not refuted     → no change, 'da_confirmed'

Cost: ~$0.005 per fire (Deepseek V4 Pro, ~2k input + ~500 output).
Wall: ~2-4s.

Doctrine alignment:
  - Per Pattern 5 ban (project memory): NOT a free-form debate. Single
    constrained-evidence prompt, single response. No agent chatter.
  - DA's verdict is ADVISORY: it modifies the F14b verdict according
    to a fixed rule, not by re-running anything.
  - Capital decisions still human regardless of DA output (per A+B
    hard line).
"""
from __future__ import annotations

import dataclasses as _dc
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@_dc.dataclass(frozen=True)
class DACritique:
    """Structured Devil's Advocate output for an F14b verdict."""
    refuted:        bool              # True = DA thinks the verdict is wrong (false positive)
    severity:       str               # "high" | "medium" | "low"
    attack_vector:  str               # short attack line (≤ 1 sentence)
    reasoning:      str               # 2-4 sentences expanding the attack
    confidence:     float             # 0..1 — DA's self-reported confidence in its critique
    raw_response:   str               # full LLM text for forensics


_SYSTEM_PROMPT = """\
You are a skeptical quantitative-finance reviewer. Your job: refute
positive factor-test verdicts that look like false positives.

A new factor-test verdict has just landed (GREEN or MARGINAL). Before
the human acts on it, you give an opposing view.

Default stance: REFUTE unless evidence is overwhelming. False positives
in factor research are common; survivors of the strict gate STILL fail
out-of-sample 60-80% of the time per academic surveys (Harvey-Liu-Zhu,
Hou-Xue-Zhang, McLean-Pontiff).

Specifically watch for:
  1. POST-PUBLICATION DECAY — most factors in academic papers Sharpe-decay
     after publication. If the underlying paper is > 5 years old AND the
     test only covers years AFTER publication, the OOS Sharpe is suspect.
  2. P-HACKING / SELECTION BIAS — Sharpe ~0.5 with t ~2 is the SUSPICIOUS
     zone (the bar that rewards parameter search). Strong factors clear
     |t|>3, weak ones don't clear |t|>2.
  3. DATA-MINING — if the spec specifies a particular weighting / rebalance
     / universe combination, ask: was this combination chosen ex-ante from
     the paper, or ex-post from the data?
  4. IMPLEMENTATION GAP — does the spec's construction match the original
     paper's construction? Differences (universe, lookback, lagging) can
     flip the sign of the alpha.
  5. SHORT WINDOW — n_obs < 60 monthly = < 5 years of data. Sharpe stderr
     at that length is so wide that any positive Sharpe is noise.
  6. MULTIPLE TESTING — DSR with n_trials=20 is the autopilot default,
     but the project's family-aware n_trials may be much higher; the
     deflation here is OPTIMISTIC.

Return your critique via the emit_critique tool. Be brief. One attack
vector + 2-4 sentence reasoning is enough; do NOT write paragraphs.
"""


_TOOL_DEFINITION = {
    "name": "emit_critique",
    "description": "Emit a structured Devil's Advocate critique of the verdict.",
    "input_schema": {
        "type": "object",
        "properties": {
            "refuted": {
                "type": "boolean",
                "description": "True if you think this verdict is likely a false positive; False if evidence is genuinely strong.",
            },
            "severity": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": (
                    "high = strong reason to downgrade (probably wrong); "
                    "medium = real concern (caution flag); "
                    "low = noted reservation (no action needed)."
                ),
            },
            "attack_vector": {
                "type": "string",
                "description": "One sentence naming the SPECIFIC attack (e.g., 'post-publication decay; OOS window starts after paper').",
            },
            "reasoning": {
                "type": "string",
                "description": "2-4 sentences expanding the attack with concrete reference to the verdict's metrics or spec.",
            },
            "confidence": {
                "type": "number",
                "description": "Your self-reported confidence (0..1) in this critique.",
            },
        },
        "required": ["refuted", "severity", "attack_vector", "reasoning", "confidence"],
        "additionalProperties": False,
    },
}


def _build_user_message(
    *,
    spec,
    metrics:    dict,
    verdict:    str,
    score:      int,
    claim_text: str,
    evidence_md: str,
) -> str:
    """Construct the user-message with all the evidence DA needs."""
    primary = spec.legs[0] if spec.legs else None
    lines = []
    lines.append("VERDICT TO CRITIQUE")
    lines.append("===================")
    lines.append(f"verdict:      {verdict}  (score {score}/4)")
    lines.append(f"family:       {spec.family.value}")
    lines.append(f"signal_type:  {primary.signal_type.value if primary else 'NONE'}")
    lines.append(f"universe:     {spec.universe.asset_class.value}/{spec.universe.subset.value}")
    lines.append(f"weighting:    {spec.construction.weighting.value}")
    lines.append(f"rebalance:    {spec.construction.rebalance.value}")
    lines.append("")
    lines.append("METRICS")
    lines.append("-------")
    for k, v in metrics.items():
        lines.append(f"  {k:<14} {v}")
    lines.append("")
    lines.append("CLAIM FROM SOURCE PAPER")
    lines.append("-----------------------")
    lines.append((claim_text or "").strip()[:600])
    lines.append("")
    lines.append("CAPABILITY EVIDENCE MARKDOWN (truncated)")
    lines.append("----------------------------------------")
    lines.append(evidence_md[:2000])
    lines.append("")
    lines.append("Call emit_critique with your refutation now.")
    return "\n".join(lines)


def run_autopilot_da(
    *,
    spec,
    metrics:        dict,
    verdict:        str,
    score:          int,
    claim_text:     str,
    evidence_md:    str = "",
) -> Optional[DACritique]:
    """Fire DA on a verdict. Returns None on hard failure (LLM error,
    no tool call) — caller treats None as "no critique, verdict stands
    unchanged" to fail-safe rather than fail-loud (DA being down should
    not block the whole F14b run).

    evidence_md is optional — DA can critique purely from metrics + spec
    + claim. Pass it when available for richer context, omit when DA
    fires BEFORE the markdown is written.
    """
    from engine.llm.call import call as llm_call

    user_msg = _build_user_message(
        spec        = spec,
        metrics     = metrics,
        verdict     = verdict,
        score       = score,
        claim_text  = claim_text,
        evidence_md = evidence_md,
    )

    try:
        result = llm_call(
            workload   = "devils_advocate",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "devils_advocate",
            tools      = [_TOOL_DEFINITION],
            max_tokens = 1200,
            scope      = "autopilot_da",
        )
    except Exception as exc:
        logger.warning("DA: llm_call failed: %s", exc)
        return None

    # Find the tool call
    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_critique":
            payload = tc.input
            break
    if payload is None:
        logger.warning("DA: model returned text instead of tool_call; raw=%s",
                        result.text[:200])
        return None

    try:
        return DACritique(
            refuted       = bool(payload.get("refuted", False)),
            severity      = str(payload.get("severity", "low")),
            attack_vector = str(payload.get("attack_vector", "")).strip(),
            reasoning     = str(payload.get("reasoning", "")).strip(),
            confidence    = float(payload.get("confidence", 0.5)),
            raw_response  = result.text or json.dumps(payload),
        )
    except Exception as exc:
        logger.warning("DA: payload → DACritique failed: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Verdict-downgrade rule (pure function for testability)
# ──────────────────────────────────────────────────────────────────────
def apply_critique_to_verdict(
    verdict:  str,
    score:    int,
    critique: Optional[DACritique],
) -> tuple[str, int, str]:
    """Apply DA critique to the F14b verdict per the rule below.

    Returns (new_verdict, new_score, da_tag) where da_tag is one of:
      'da_skipped'   — DA didn't fire (RED) or hard-failed (None)
      'da_confirmed' — DA reviewed and did not refute
      'da_noted'     — DA refuted with severity=low (no downgrade)
      'da_caution'   — DA refuted with severity=medium (downgrade GREEN→MARGINAL)
      'da_refuted'   — DA refuted with severity=high (downgrade to RED)
    """
    if critique is None:
        return verdict, score, "da_skipped"
    if not critique.refuted:
        return verdict, score, "da_confirmed"
    sev = critique.severity.lower()
    if sev == "high":
        return "RED", 0, "da_refuted"
    if sev == "medium":
        if verdict == "GREEN":
            return "MARGINAL", max(2, score - 1), "da_caution"
        return verdict, score, "da_caution"
    return verdict, score, "da_noted"
