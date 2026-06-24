"""engine.agents.papers_curator.synthesis — Phase 2.0 step 3.

Cross-source hypothesis synthesis for Employee A. Reads a frozen
multi-source state snapshot and asks Claude Sonnet 4.6 to propose 0-3
candidate hypotheses that synthesize across the inputs — NOT extracted
from a single paper's claim field (that's hypothesis_extractor's job),
but synthesized from multi-paper + sleeve + decay + memory context.

Architecturally per
[[spec-research-session-orchestrator-2026-06-06]] §"Employee A":

  - Pattern: single-agent LLM call with strict JSON schema tool_use
            (NOT Pattern 5 multi-agent debate)
  - Workload: papers_curator_synthesis (Sonnet 4.6 — synthesis quality
              matters; R1 audit ruled out Deepseek for complex schemas)
  - Cost ceiling: $0.10/call
  - Input: deterministic SynthesisInput snapshot — NO live reads;
           the caller (chief_of_staff orchestrator, later) freezes
           state BEFORE this call
  - Output: list[SynthesizedCandidate] — richer than the persisted
            Hypothesis dataclass; step 4 will adapt + write to
            hypotheses.jsonl

This module is intentionally PURE w.r.t. stores: it does NOT read
papers / sleeves / memory directly. The context gatherer is a separate
piece (step 3b coming next), keeping synthesis testable with synthetic
inputs and the gatherer testable without LLM calls.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import re
from typing import Optional

# Top-level import so monkeypatch in tests works (see filter.py for the
# Pattern 5-ban-adjacent reason this isn't an inline import).
from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Input shape — what the orchestrator passes in
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class PaperSummaryRef:
    """One element of recent_summaries — the LLM sees title + thesis +
    paper_id + (synthesizable) source link to back-reference in output."""
    paper_id:           str
    title:              str
    authors_short:      str
    thesis:             str
    testable_hypothesis: str
    why_matters_for_us: str
    risk_flags_short:   tuple[str, ...]
    recommended_action: str          # INGEST / READ / SKIP


@_dc.dataclass(frozen=True)
class SleeveStateRef:
    """One element of deployed_sleeves — the LLM sees enough KPI to
    notice 'sleeve X is decaying; could paper Y address that?'"""
    sleeve_id:           str
    family:              str
    status:              str         # DEPLOYED / DECAY_WATCH / RAMP_DOWN / etc
    ann_sharpe_live:     Optional[float]
    months_since_deploy: Optional[int]
    last_decay_alert:    Optional[str]  # iso ts, or None


@_dc.dataclass(frozen=True)
class RecentEventRef:
    """One element of recent_events — verdict_filed / decay_alert /
    doctrine_signal lineage that gives the LLM 'what just happened'."""
    event_id:    str
    event_type:  str
    subject_id:  str
    family:      str
    verdict:     str
    summary:     str
    ts:          str


@_dc.dataclass(frozen=True)
class DoctrineHit:
    """One element of doctrine_snippets — a relevant memory_file slice
    so the LLM doesn't propose something that conflicts with a held
    rule."""
    memory_file_id: str
    headline:       str
    snippet:        str          # ≤ 400 chars from the memory body


@_dc.dataclass(frozen=True)
class AnchorRef:
    """Stage C Phase E (2026-06-07): one entry in the anchor library
    A sees during synthesis. T1_DOCTRINE + T2_ANCHOR papers from
    papers_registry, each carrying tier_anchor_summary (the 1-line
    meta-summary explaining what the paper anchors). A must verify
    each candidate against the orthogonality requirement these
    define."""
    paper_id:           str       # short form (first 8 chars) for prompt brevity
    tier:               str       # "T1_DOCTRINE" / "T2_ANCHOR"
    first_author:       str
    year:               int
    anchor_summary:     str       # the actual 1-line summary


@_dc.dataclass(frozen=True)
class SynthesisInput:
    """Frozen snapshot the LLM reads. Gatherer (step 3b) builds this;
    orchestrator (step 14) calls run_synthesis with it.

    Empty/minimal inputs are valid — the call just yields fewer
    candidates (or zero). The LLM is instructed to prefer empty output
    over weak output.
    """
    recent_summaries:  tuple[PaperSummaryRef, ...]
    deployed_sleeves:  tuple[SleeveStateRef, ...]
    recent_events:     tuple[RecentEventRef, ...]
    doctrine_snippets: tuple[DoctrineHit, ...]
    snapshot_ts:       str
    # Stage C Phase E (2026-06-07): canonical anchor library. Default
    # empty tuple keeps pre-Phase-E gatherer calls compatible.
    anchor_library:    tuple[AnchorRef, ...] = ()
    # Phase B (2026-06-14): belief layer summary — per-family
    # GREEN/MARGINAL/RED distribution from autopsy ledger. Empty
    # tuple = belief layer not consulted (pre-Phase-B compatibility).
    # See engine.research.belief_synthesis_context.build_belief_summary
    belief_layer_summary: tuple = ()


# ────────────────────────────────────────────────────────────────────
# Output shape — richer than persisted Hypothesis
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class SynthesizedCandidate:
    """Full LLM output for one candidate. Step 4 will adapt this to the
    persisted Hypothesis dataclass; some fields land on events instead
    of the registry row.
    """
    claim:                       str
    mechanism_family:            str   # MechanismFamily enum value
    mechanism_subtype:           str
    predicted_direction:         str   # HypothesisDirection enum value
    predicted_magnitude:         str
    required_data:               tuple[str, ...]
    test_methodology:            str

    # Generation provenance (lands on hypotheses.jsonl per step 1 schema)
    synthesizes_event_ids:       tuple[str, ...]
    synthesizes_paper_ids:       tuple[str, ...]
    addresses_decay_in:          Optional[str]

    # Generation-time metadata (lands on emit events, NOT registry row)
    cochrane_frame:              str   # behavioral / risk / friction
    novelty_vs_known:            str   # genuinely_new / extension / refinement
    estimated_n_trials_in_family: int
    graveyard_conflicts:         tuple[str, ...]   # family/signal n_REDs
    doctrine_conflicts:          tuple[str, ...]   # memory_file_id list
    expected_outcome_prior:      str   # honest "likely RED per HXZ 65%" etc

    # Call diagnostics
    generation_ts:               str
    model:                       str

    # Phase 2.2b (2026-06-07): citation verification results.
    #   citation_verifications: one CitationCheck per paper A cited
    #     (paper_resolved + confidence + supporting_chunks + notes)
    #   citation_quality:       aggregate dict B's prompt + audit event
    #     consume ({n_papers_cited, n_resolved, n_unresolved,
    #              mean_confidence, min_confidence, any_unresolved,
    #              low_confidence_flag}). Empty when not yet verified.
    citation_verifications:      tuple = ()
    citation_quality:            "Optional[dict]" = None

    # Stage C Phase E (2026-06-07): per-candidate orthogonality
    # statements against the anchor library. Each entry is
    # {anchor_paper_id, why_orthogonal} — the model must name at
    # least one anchor that this candidate would NOT just replicate
    # + explain the orthogonal angle in 1 sentence. Empty list →
    # expected_outcome_prior auto-downgraded (handled at adapter time).
    orthogonal_to_anchors:       tuple = ()


# ────────────────────────────────────────────────────────────────────
# Prompt + tool schema
# ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are Employee A — the papers curator for a solo-quant research book.
This task is CROSS-SOURCE SYNTHESIS, not per-paper extraction.

Your job: read the multi-source state snapshot below — recent paper
summaries, deployed sleeves with their KPI, recent verdict / decay
events, and active doctrine — and propose 0-3 hypothesis candidates
that SYNTHESIZE across at least two sources. NOT candidates that any
single paper already wrote out.

CONSTRAINTS (load-bearing):

  - PREFER zero output over weak output. The principal's prior is
    Hou-Xue-Zhang 65% non-replication; if no synthesis is genuinely
    worth the principal's review attention this week, return [].
  - Every candidate MUST cite ≥ 2 source rows from the input
    (synthesizes_paper_ids + synthesizes_event_ids combined ≥ 2).
  - Every candidate MUST report graveyard_conflicts honestly. Look
    at the doctrine_snippets — if a memory file says
    "equity single-name signals exhausted" and your candidate is an
    equity single-name signal, list that memory file.
  - Every candidate MUST report doctrine_conflicts honestly. If a
    candidate's mechanism requires a method the doctrine has banned
    (e.g. regime detection in book sizing), say so. The principal
    will decide whether to amend doctrine; you do not.
  - cochrane_frame MUST be one of: behavioral, risk, friction. A
    candidate without a discount-rate story is data-mining; don't
    propose it.
  - expected_outcome_prior MUST be honest. Default is
    "likely_RED_per_HXZ_65pct" unless the candidate has unusually
    strong cross-source evidence.

ORTHOGONALITY GATE (anti-mental-rut, 2026-06-07):
  - The principal works solo and is at risk of subtle confirmation
    bias toward mechanisms similar to currently-deployed sleeves.
  - For EACH candidate, identify ONE deployed sleeve that this
    candidate would NOT just replicate. State the orthogonal angle
    explicitly inside novelty_vs_known.
  - If the candidate's mechanism overlaps > 70% with ANY deployed
    sleeve (same family + same direction + same data window +
    similar magnitude), DOWNGRADE expected_outcome_prior by one
    tier — even if paper evidence looks strong. Comfortable
    repetition produces low marginal alpha.
  - This is a tightening, not a softening: candidates the principal
    "feels familiar with" need MORE evidence to clear the bar, not
    less.

ANCHOR LIBRARY ORTHOGONALITY (Stage C Phase E, 2026-06-07):
  The input includes an ANCHOR_LIBRARY section listing canonical
  quant-finance papers — T1_DOCTRINE (methodology that defines our
  gates) + T2_ANCHOR (mechanism-class definitions like Carry, TSMOM,
  Quality). Each anchor carries a 1-sentence summary stating its
  mechanism class + the orthogonality requirement it imposes.

  For EACH candidate you propose:
    - You MUST populate `orthogonal_to_anchors` with at least ONE
      entry: {anchor_paper_id, why_orthogonal}.
    - Pick the anchor whose mechanism class is CLOSEST to your
      candidate (don't dodge by picking an unrelated anchor).
    - The `why_orthogonal` sentence must explicitly state what makes
      your candidate NOT a replication of that anchor (e.g. different
      asset class, different time horizon, different data input,
      different mechanism direction).
    - If you genuinely cannot articulate an orthogonal angle vs the
      relevant T1/T2 anchors, you SHOULD NOT propose the candidate —
      return [] instead. Comfortable-recombinations-of-canonical-
      anchors produce zero marginal alpha at HXZ-65%-replication
      base rates.

  This is load-bearing: it forces you to NAME the canonical work
  you're extending/diverging-from rather than treating the substrate
  as a context-free idea pool.

BELIEF LAYER FEEDBACK (Phase B, 2026-06-14):
  The input may include a SYSTEM VERDICT HISTORY section listing
  per-family GREEN/MARGINAL/RED counts from this system's accumulated
  autopsy ledger + a directional hint per family. Use it as a
  CONDITIONAL PRIOR for candidate selection:

    - AVOID families: ≥80% RED — do NOT propose new specs in this
      family unless your synthesis has a STRUCTURALLY NEW angle (new
      data class, new mechanism, new asset universe). Routine variants
      will inherit the RED prior and produce zero marginal alpha.

    - MARGINAL-ONLY families: orthogonal alpha exists but ≥50% of obs
      are MARGINAL. Don't propose more standalone specs in these
      families. INSTEAD propose COMBINATIONS or CONDITIONAL variants
      (e.g. "X under regime Y", "X residualized by Z") that have a
      chance of crossing into GREEN.

    - EXPLORE families: ≥50% GREEN — these are the highest-value
      neighborhoods. Propose specs that exploit the family but with
      DIFFERENT spec content (e.g. different sub-period, different
      cost convention, different asset within the universe). Honor
      the sub-period-dup caveat: same spec with new hypothesis_id
      will NOT count toward independent obs.

    - THIN families (n<3): insufficient evidence — neither encourage
      nor discourage. Treat as exploration territory.

  Cite the belief-layer hint EXPLICITLY in `novelty_vs_known` when
  you propose against an AVOID/MARGINAL family. E.g. "system has
  seen 5 PROFITABILITY-family RED in 2001-2024 sample → this
  candidate uses [novel angle X] to break the dead pattern".

  If belief layer says AVOID + you can't articulate the angle: don't
  propose. The system has already empirically demonstrated this
  family doesn't yield GREEN at our threshold.

The principal's hard line: you propose; deterministic gates and a
human reviewer decide what runs. NEVER use tone that pressures the
principal to accept a candidate.

Call the emit_synthesis tool with your output. If no synthesis is
worth proposing, call it with candidates=[].
"""


_TOOL_DEFINITION = {
    "name": "emit_synthesis",
    "description": (
        "Emit 0-3 hypothesis candidates synthesized across the "
        "multi-source input. Empty list is valid and preferred over "
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
                        "claim":               {"type": "string"},
                        "mechanism_family":    {"type": "string"},
                        "mechanism_subtype":   {"type": "string"},
                        "predicted_direction": {
                            "type": "string",
                            "enum": ["positive", "negative", "zero"],
                        },
                        "predicted_magnitude": {"type": "string"},
                        "required_data":       {
                            "type": "array", "items": {"type": "string"},
                        },
                        "test_methodology":    {"type": "string"},
                        "synthesizes_paper_ids": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "synthesizes_event_ids": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "addresses_decay_in":  {"type": ["string", "null"]},
                        "cochrane_frame":      {
                            "type": "string",
                            "enum": ["behavioral", "risk", "friction"],
                        },
                        "novelty_vs_known":    {"type": "string"},
                        "estimated_n_trials_in_family": {"type": "integer"},
                        "graveyard_conflicts": {
                            "type": "array", "items": {"type": "string"},
                        },
                        "doctrine_conflicts":  {
                            "type": "array", "items": {"type": "string"},
                        },
                        "expected_outcome_prior": {"type": "string"},
                        # Stage C Phase E: orthogonality to anchor library
                        "orthogonal_to_anchors": {
                            "type": "array",
                            "minItems": 1,   # at least one anchor must be addressed
                            "maxItems": 5,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "anchor_paper_id": {"type": "string"},
                                    "why_orthogonal":  {"type": "string"},
                                },
                                "required": ["anchor_paper_id",
                                              "why_orthogonal"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "claim", "mechanism_family", "mechanism_subtype",
                        "predicted_direction", "predicted_magnitude",
                        "required_data", "test_methodology",
                        "synthesizes_paper_ids", "synthesizes_event_ids",
                        "cochrane_frame", "novelty_vs_known",
                        "estimated_n_trials_in_family",
                        "graveyard_conflicts", "doctrine_conflicts",
                        "expected_outcome_prior",
                        "orthogonal_to_anchors",
                    ],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["candidates"],
        "additionalProperties": False,
    },
}


# ────────────────────────────────────────────────────────────────────
# User-message builder
# ────────────────────────────────────────────────────────────────────
def _format_input(inp: SynthesisInput) -> str:
    lines = [f"SNAPSHOT TS: {inp.snapshot_ts}", ""]

    lines.append(f"RECENT PAPER SUMMARIES ({len(inp.recent_summaries)})")
    lines.append("-" * 40)
    for s in inp.recent_summaries:
        lines.append(f"paper_id: {s.paper_id}")
        lines.append(f"  title:        {s.title[:140]}")
        lines.append(f"  authors:      {s.authors_short}")
        lines.append(f"  thesis:       {s.thesis[:300]}")
        lines.append(f"  testable:     {s.testable_hypothesis[:200]}")
        lines.append(f"  why us:       {s.why_matters_for_us[:200]}")
        if s.risk_flags_short:
            lines.append(f"  risks:        {', '.join(s.risk_flags_short[:5])}")
        lines.append(f"  rec action:   {s.recommended_action}")
        lines.append("")

    lines.append(f"DEPLOYED SLEEVES ({len(inp.deployed_sleeves)})")
    lines.append("-" * 40)
    for sl in inp.deployed_sleeves:
        sharpe = f"{sl.ann_sharpe_live:+.2f}" if sl.ann_sharpe_live is not None else "n/a"
        mos = f"{sl.months_since_deploy}mo" if sl.months_since_deploy is not None else "?mo"
        decay = f" decay_alert={sl.last_decay_alert}" if sl.last_decay_alert else ""
        lines.append(f"  {sl.sleeve_id} ({sl.family}) {sl.status} "
                      f"sharpe={sharpe} {mos}{decay}")
    lines.append("")

    lines.append(f"RECENT EVENTS ({len(inp.recent_events)})")
    lines.append("-" * 40)
    for ev in inp.recent_events:
        lines.append(f"  {ev.event_id[:10]}  {ev.event_type:<26}  "
                      f"{ev.family:<16}  {ev.verdict:<10}  {ev.summary[:90]}")
    lines.append("")

    lines.append(f"DOCTRINE SNIPPETS ({len(inp.doctrine_snippets)})")
    lines.append("-" * 40)
    for d in inp.doctrine_snippets:
        lines.append(f"  [[{d.memory_file_id}]]  {d.headline}")
        lines.append(f"    {d.snippet[:300]}")
    lines.append("")

    # Stage C Phase E: anchor library — canonical T1+T2 papers each
    # candidate must be orthogonal to. Group by tier for readability.
    if inp.anchor_library:
        t1s = [a for a in inp.anchor_library if a.tier == "T1_DOCTRINE"]
        t2s = [a for a in inp.anchor_library if a.tier == "T2_ANCHOR"]
        lines.append(f"ANCHOR LIBRARY ({len(inp.anchor_library)} "
                      f"canonical papers — each candidate's "
                      f"`orthogonal_to_anchors` MUST cite ≥1)")
        lines.append("-" * 40)
        if t1s:
            lines.append(f"T1 DOCTRINE (methodology / gates) — {len(t1s)}:")
            for a in t1s:
                lines.append(f"  anchor_paper_id={a.paper_id}  "
                              f"{a.first_author} {a.year}")
                lines.append(f"    {a.anchor_summary[:240]}")
        if t2s:
            lines.append("")
            lines.append(f"T2 ANCHOR (mechanism classes) — {len(t2s)}:")
            for a in t2s:
                lines.append(f"  anchor_paper_id={a.paper_id}  "
                              f"{a.first_author} {a.year}")
                lines.append(f"    {a.anchor_summary[:240]}")
        lines.append("")

    # Phase B (2026-06-14): belief layer summary — what the system has
    # actually OBSERVED about each family's success rate. Sonnet uses
    # this to avoid mining DEAD families + explore neighborhoods of
    # robust GREEN ones. Empty tuple = pre-Phase-B compat (no section).
    if inp.belief_layer_summary:
        try:
            from engine.research.belief_synthesis_context import render_for_prompt
            for line in render_for_prompt(inp.belief_layer_summary):
                lines.append(line)
        except Exception:
            pass   # never break prompt assembly on belief failure

    lines.append("Call emit_synthesis now. Empty candidates list is valid.")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# Main call
# ────────────────────────────────────────────────────────────────────
def run_synthesis(
    inp: SynthesisInput,
    *,
    max_tokens: int = 4096,
) -> list[SynthesizedCandidate]:
    """Fire one papers_curator_synthesis call. Returns 0-3 candidates.

    Returns [] on hard LLM failure / unparseable response / tool not
    called (per Pattern 5-allowed pattern: structured single-call,
    no retry loop, no agent-to-agent debate). The orchestrator can
    log the skip and proceed.
    """
    user_msg = _format_input(inp)

    try:
        result = llm_call(
            workload   = "papers_curator_synthesis",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "papers_curator_synthesis",
            tools      = [_TOOL_DEFINITION],
            max_tokens = max_tokens,
            scope      = "phase_2_0_synthesis",
        )
    except Exception as exc:
        logger.warning("synthesis: llm_call failed: %s", exc)
        return []

    # Pull the tool call payload
    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_synthesis":
            payload = tc.input
            break
    if payload is None:
        logger.warning("synthesis: tool not called; raw text first 200 chars: %s",
                        (result.text or "")[:200])
        return []

    raw_candidates = payload.get("candidates") or []
    if not isinstance(raw_candidates, list):
        logger.warning("synthesis: candidates not a list (%s)", type(raw_candidates))
        return []

    out: list[SynthesizedCandidate] = []
    ts_now = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for raw in raw_candidates[:3]:    # hard cap of 3 even if model emits more
        try:
            out.append(SynthesizedCandidate(
                claim                       = str(raw["claim"]),
                mechanism_family            = str(raw["mechanism_family"]),
                mechanism_subtype           = str(raw.get("mechanism_subtype", "")),
                predicted_direction         = str(raw["predicted_direction"]),
                predicted_magnitude         = str(raw["predicted_magnitude"]),
                required_data               = tuple(raw.get("required_data") or ()),
                test_methodology            = str(raw["test_methodology"]),
                synthesizes_event_ids       = tuple(raw.get("synthesizes_event_ids") or ()),
                synthesizes_paper_ids       = tuple(raw.get("synthesizes_paper_ids") or ()),
                addresses_decay_in          = raw.get("addresses_decay_in") or None,
                cochrane_frame              = str(raw["cochrane_frame"]),
                novelty_vs_known            = str(raw["novelty_vs_known"]),
                estimated_n_trials_in_family = int(raw["estimated_n_trials_in_family"]),
                graveyard_conflicts         = tuple(raw.get("graveyard_conflicts") or ()),
                doctrine_conflicts          = tuple(raw.get("doctrine_conflicts") or ()),
                expected_outcome_prior      = str(raw.get("expected_outcome_prior", "")),
                # Stage C Phase E: orthogonality statements. Parse +
                # coerce to tuple of frozen dicts. Schema enforces
                # minItems=1 server-side; we also drop the candidate
                # if list is empty (defense in depth).
                orthogonal_to_anchors       = tuple(
                    {
                        "anchor_paper_id": str(o.get("anchor_paper_id") or ""),
                        "why_orthogonal":  str(o.get("why_orthogonal") or ""),
                    }
                    for o in (raw.get("orthogonal_to_anchors") or [])
                    if isinstance(o, dict)
                ),
                generation_ts               = ts_now,
                model                       = result.model,
            ))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("synthesis: dropping malformed candidate: %s", exc)
            continue
    return out
