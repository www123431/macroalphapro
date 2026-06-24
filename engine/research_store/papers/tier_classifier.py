"""engine.research_store.papers.tier_classifier — Stage C Phase A.

Sonnet-driven batch tier classification for papers_registry entries.

Per the three-libraries doctrine (locked 2026-06-07):
  T1 DOCTRINE: methodology paper whose SPECIFIC phrasings matter
               (DSR / Cochrane / multi-test). Full PDF + chunks
               required for downstream retrieval.
  T2 ANCHOR:   canonical mechanism paper, citation target.
               Title + 1-line summary enough — no chunks needed.
  T3 RECENT:   anything else. Title + abstract only.
  UNCLASSIFIED: low-confidence model output OR pre-Phase-A entry.

Pattern-5-compliant: SINGLE batched LLM call (not one-per-paper).
The model sees all papers together, picks tier per paper from the
controlled enum + 1-sentence rationale. Strict JSON tool_use.

This module emits a CLASSIFICATION PLAN (proposed tier per paper +
rationale). It does NOT mutate the registry — the caller (or a
follow-up commit) applies the plan after human review.

Cost: ~$0.10-0.30 per batch of 60 papers (≈ $0.002-0.005 per paper).
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
from typing import Optional

from engine.llm.call import call as llm_call
from engine.research_store.papers.schema import PaperTier

logger = logging.getLogger(__name__)


@_dc.dataclass(frozen=True)
class TierProposal:
    """One classifier output for one paper."""
    paper_id:    str
    tier:        PaperTier
    rationale:   str           # 1-sentence why
    confidence:  float         # 0-1; < 0.7 → UNCLASSIFIED


# ────────────────────────────────────────────────────────────────────
# Prompt + tool schema
# ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are classifying quant-finance papers into a THREE-TIER knowledge
library for a solo-quant AI-augmented research system.

The tiers and their EXACT semantics:

T1_DOCTRINE — METHODOLOGY papers whose specific phrasings matter for
  system behavior. Examples:
    * Bailey-Lopez de Prado 2014 DSR (defines our deflated-SR gate
      threshold; system literally retrieves the paper's exact bar)
    * McLean-Pontiff 2016 (defines our post-pub decay prior)
    * Harvey-Liu-Zhu 2016 (|t|>=3 hurdle that gates every factor)
    * Hou-Xue-Zhang 2020 (replication rate 35% → our prior)
    * Cochrane 2011 discount-rate framework
  Criteria: the SYSTEM depends on this paper's exact concepts +
  thresholds. ~20-30 papers MAX. Be selective.

T2_ANCHOR — Canonical MECHANISM papers that define a class of
  factor/strategy. Functions as CITATION TARGET, not retrieval source.
  Examples:
    * Fama-French 1992 / 1993 (the 3-factor model)
    * Jegadeesh-Titman 1993 (cross-sectional momentum)
    * Koijen-Moskowitz-Pedersen-Vrugt 2018 (Carry)
    * Carr-Wu 2009 (Variance Risk Premium)
    * Asness-Frazzini-Pedersen 2019 (Quality-Minus-Junk)
  Criteria: defined a class of strategy that's well-known + commonly
  referenced. Title + 1-line summary is enough for the system.

T3_RECENT — Anything else: recent publications, narrow applications,
  unproven contributions, working papers without breakout status.
  Default for substrate-crawl arrivals.

UNCLASSIFIED — Use ONLY when you genuinely can't tell from the
  metadata (e.g. very short abstract, ambiguous title). DO NOT use
  this as a way to avoid commitment.

CONSTRAINTS (load-bearing):
  - PREFER T3 over T2 over T1 when in doubt. T1 + T2 are precious
    classifications; mis-promoting a paper damages downstream
    retrieval quality.
  - You MUST classify every paper in the input.
  - rationale must be 1 sentence (under 200 chars).
  - confidence in [0.0, 1.0]. If < 0.7, the human will review.

OUTPUT: invoke the emit_tier_classifications tool EXACTLY ONCE with
the proposals list. Each paper gets one proposal.
"""


_TOOL_DEFINITION = {
    "name": "emit_tier_classifications",
    "description": ("Emit tier classifications for the batch of "
                    "papers. One proposal per input paper, in the "
                    "same order."),
    "input_schema": {
        "type": "object",
        "properties": {
            "proposals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "paper_id":   {"type": "string"},
                        "tier":       {
                            "type": "string",
                            "enum": ["T1_DOCTRINE", "T2_ANCHOR",
                                      "T3_RECENT", "UNCLASSIFIED"],
                        },
                        "rationale":  {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0, "maximum": 1.0,
                        },
                    },
                    "required": ["paper_id", "tier", "rationale",
                                  "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["proposals"],
        "additionalProperties": False,
    },
}


def _format_input(papers: list) -> str:
    """Build the user-message paper list (one block per paper)."""
    lines = [f"BATCH SIZE: {len(papers)} papers", ""]
    for i, p in enumerate(papers, 1):
        authors_str = ", ".join((p.authors or ())[:3])
        if len(p.authors or ()) > 3:
            authors_str += " et al."
        abstract = (p.abstract or "").strip()
        if len(abstract) > 600:
            abstract = abstract[:600] + "…"
        lines.append(f"--- PAPER {i} ---")
        lines.append(f"paper_id: {p.paper_id}")
        lines.append(f"title:    {p.title}")
        lines.append(f"authors:  {authors_str}")
        lines.append(f"year:     {p.year}")
        lines.append(f"venue:    {p.venue or '(none)'}")
        if abstract:
            lines.append(f"abstract: {abstract}")
        else:
            lines.append("abstract: (none)")
        lines.append("")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# Main entry — single LLM call, return per-paper proposals
# ────────────────────────────────────────────────────────────────────
def classify_papers_batch(
    papers: list,
    *,
    max_tokens: int = 8192,
    confidence_floor: float = 0.7,
) -> list[TierProposal]:
    """ONE LLM call. Returns a TierProposal per input paper, in input
    order (or empty list on hard failure).

    Papers with confidence < confidence_floor are coerced to
    UNCLASSIFIED so the human reviewer sees them flagged.
    """
    if not papers:
        return []

    try:
        result = llm_call(
            workload   = "papers_tier_classifier",
            system     = _SYSTEM_PROMPT,
            user       = _format_input(papers),
            agent_id   = "papers_tier_classifier",
            tools      = [_TOOL_DEFINITION],
            max_tokens = max_tokens,
            scope      = "stage_c_phase_a_tier_classification",
        )
    except Exception as exc:
        logger.warning("tier_classifier: llm_call failed: %s", exc)
        return []

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_tier_classifications":
            payload = tc.input
            break
    if payload is None:
        logger.warning("tier_classifier: tool not called; raw text: %s",
                        (result.text or "")[:200])
        return []

    raw_proposals = payload.get("proposals") or []
    if not isinstance(raw_proposals, list):
        return []

    # Index by paper_id for ordered output
    by_pid: dict[str, dict] = {}
    for raw in raw_proposals:
        pid = str(raw.get("paper_id") or "")
        if pid:
            by_pid[pid] = raw

    out: list[TierProposal] = []
    for p in papers:
        raw = by_pid.get(p.paper_id)
        if raw is None:
            # Model didn't classify this paper → UNCLASSIFIED with note
            out.append(TierProposal(
                paper_id   = p.paper_id,
                tier       = PaperTier.UNCLASSIFIED,
                rationale  = "(classifier did not return a proposal)",
                confidence = 0.0,
            ))
            continue
        try:
            tier_str = str(raw.get("tier") or "UNCLASSIFIED")
            tier = PaperTier(tier_str)
        except ValueError:
            tier = PaperTier.UNCLASSIFIED
        conf = float(raw.get("confidence") or 0.0)
        rationale = str(raw.get("rationale") or "")[:300]
        # Confidence floor: coerce low-confidence proposals to
        # UNCLASSIFIED so human reviewer sees them
        if conf < confidence_floor and tier != PaperTier.UNCLASSIFIED:
            rationale = (f"[conf={conf:.2f} below {confidence_floor} "
                          f"floor; original={tier.value}] {rationale}")
            tier = PaperTier.UNCLASSIFIED
        out.append(TierProposal(
            paper_id   = p.paper_id,
            tier       = tier,
            rationale  = rationale,
            confidence = conf,
        ))
    return out
