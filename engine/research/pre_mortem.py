"""engine/research/pre_mortem.py — α Pre-Mortem Generator.

Phase α of the multi-agent brainstorm rebuild (2026-06-14). NOT a
multi-agent debate — a single specialized skeptic persona that runs
adversarial review BEFORE strict-gate dispatch. The output is a
structured failure-mode list the gate USES (not votes on).

ACADEMIC ANCHOR
================
  - Stigler 1973 "Conflict of Interest as a Cost of Verification" —
    adversarial review reveals defects deliberation hides.
  - Kahneman pre-mortem technique — imagine the project failed,
    enumerate why, BEFORE committing.
  - Lakatos research programs — a progressive hypothesis predicts
    new facts. A degenerative one patches old anomalies. Skeptic
    flags hypotheses that look degenerative.
  - McLean-Pontiff 2016 — 58% Sharpe drop post-pub; hypotheses
    re-mining known anomalies are prior-failed.
  - Bailey-Lopez de Prado 2014 — N trials in family inflates DSR
    threshold. Skeptic should foreground if N already high.

WHY NOT N-PERSONA BRAINSTORM
============================
The system is substrate-bound, not idea-bound (238 hyps, 9/week
verdict cap). N-persona converging on hypotheses bloats n_trials
and surfaces mainstream-converging ideas with sub-zero expected
alpha. Adversarial pre-mortem is the higher-ROI use of LLM budget:
catches BRITTLE hyps before we spend 1 week of verdict throughput
on them.

OUTPUT SCHEMA
=============
PreMortemReport
  failure_modes:    list[FailureMode] (3-7 items)
    severity:       "HIGH" / "MEDIUM" / "LOW"
    category:       enum (DATA_REGIME / OVERFITTING / SPANNING /
                    COST_REALISM / PUBLICATION_DECAY / GRAVEYARD_MATCH /
                    POWER / SURVIVOR_BIAS / OTHER)
    description:    1-2 sentence concrete failure mode
    check_suggestion: 1 sentence "the strict-gate should verify X"
                    (concrete enough that a gate engineer can wire it)
  overall_kill_recommendation: "KILL_BEFORE_TEST" / "TEST_WITH_CAVEATS" /
                                "PROCEED_NORMAL"
  rationale:        1-3 sentence summary

PERSISTENCE
===========
data/research/pre_mortems.jsonl — one row per assessment. Re-running
on the same hypothesis appends a new row (history kept; latest wins
on read). Lineage: parent_hypothesis_id field carries the hyp_id.

GRACEFUL DEGRADATION
====================
LLM call failure / tool not called / validation fail → returns None.
Caller can proceed with strict-gate without pre-mortem context.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
PRE_MORTEM_PATH = _REPO_ROOT / "data" / "research" / "pre_mortems.jsonl"

_SEVERITY_VALUES   = {"HIGH", "MEDIUM", "LOW"}
_KILL_REC_VALUES   = {"KILL_BEFORE_TEST", "TEST_WITH_CAVEATS", "PROCEED_NORMAL"}
_CATEGORY_VALUES   = {
    "DATA_REGIME",         # data window doesn't cover relevant regime
    "OVERFITTING",         # too many degrees of freedom for sample size
    "SPANNING",            # likely subsumed by anchor library
    "COST_REALISM",        # alpha disappears under realistic costs
    "PUBLICATION_DECAY",   # post-McLean-Pontiff post-pub Sharpe drop
    "GRAVEYARD_MATCH",     # similar idea already RED'd
    "POWER",               # n too small to detect realistic alpha
    "SURVIVOR_BIAS",       # universe inclusion criterion uses future info
    "OTHER",
}


@_dc.dataclass(frozen=True)
class FailureMode:
    severity:          str    # HIGH / MEDIUM / LOW
    category:          str    # enum
    description:       str
    check_suggestion:  str


@_dc.dataclass(frozen=True)
class PreMortemReport:
    pre_mortem_id:               str
    hypothesis_id:               str
    failure_modes:               tuple[FailureMode, ...]
    overall_kill_recommendation: str
    rationale:                   str
    assessed_ts:                 str
    model:                       str
    cost_usd:                    float
    n_obs_inputs:                int   # how many context items injected


# ── System prompt — what the skeptic IS ──────────────────────────────


_SYSTEM_PROMPT = """\
You are the SKEPTIC reviewing a quantitative trading hypothesis BEFORE
the strict-gate test runs. Your job is to enumerate concrete, specific
failure modes the strict-gate MIGHT MISS. You are NOT voting on whether
the hypothesis is "right" — you are pre-mortem'ing it.

ACADEMIC GROUNDING (cite when applicable)
==========================================
- Stigler 1973: adversarial review > deliberation for catching defects
- Kahneman pre-mortem: imagine the test ran and gave a misleading GREEN —
  what would be the cause? imagine it gave RED — was the test fair?
- McLean-Pontiff 2016: 58% Sharpe drop post-publication. If this idea
  echoes a published anomaly, factor that in
- Hou-Xue-Zhang 2020: ~50% of published anomalies fail rigorous
  replication. Default skeptical
- Bailey-Lopez de Prado 2014: each new trial in a family INFLATES
  the DSR threshold. If the family already has N trials, flag it
- Lakatos progressive vs degenerative: does this hypothesis predict
  a NEW fact, or patch an OLD anomaly? Patches are degenerative

CONTEXT YOU'RE GIVEN
====================
- The hypothesis claim + mechanism family + verbatim quotes
- Family belief layer (what we've already tested in this family)
- Graveyard nearest collisions (RED'd hyps that look similar)
- Bailey-LdP n_trials counter for the family
- The strict-gate's known silent bugs (B0-B7)

OUTPUT
======
3-7 FailureMode items via emit_pre_mortem tool. Each:
  - severity: HIGH if it would FALSIFY a GREEN; MEDIUM if it would
              significantly degrade Sharpe; LOW if it's a minor
              cleanliness concern
  - category: one of the enum values (DATA_REGIME / OVERFITTING /
              SPANNING / COST_REALISM / PUBLICATION_DECAY /
              GRAVEYARD_MATCH / POWER / SURVIVOR_BIAS / OTHER)
  - description: ONE concrete failure mode, naming the SPECIFIC issue
              (not "could be overfit" but "60-month rolling window
              with 12 hyperparameter knobs likely overfit; recommend
              walk-forward")
  - check_suggestion: ONE sentence the strict-gate engineer could
              wire (concrete check, not "be careful")

Plus:
  - overall_kill_recommendation: KILL_BEFORE_TEST (don't even spend
    the test budget) / TEST_WITH_CAVEATS (proceed but flag) /
    PROCEED_NORMAL (no special concerns)
  - rationale: 1-3 sentence summary

DO NOT
======
- Repeat what the strict-gate already does (FF5 spanning, OOS, borrow
  cost, NW SE, multi-cost stress)
- Give vague concerns ("watch for noise"); be specific
- Recommend KILL_BEFORE_TEST unless there's strong graveyard collision
  or known-impossible data ask
- Make up failure modes if you genuinely don't see any — return 0-2
  LOW-severity items + PROCEED_NORMAL instead of inflating concerns
"""


_TOOL_SCHEMA = {
    "name": "emit_pre_mortem",
    "description": "Emit the pre-mortem report as structured JSON.",
    "input_schema": {
        "type": "object",
        "required": ["failure_modes", "overall_kill_recommendation", "rationale"],
        "properties": {
            "failure_modes": {
                "type": "array",
                "minItems": 0,
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "required": ["severity", "category", "description",
                                 "check_suggestion"],
                    "properties": {
                        "severity":         {"type": "string", "enum": sorted(_SEVERITY_VALUES)},
                        "category":         {"type": "string", "enum": sorted(_CATEGORY_VALUES)},
                        "description":      {"type": "string", "maxLength": 500},
                        "check_suggestion": {"type": "string", "maxLength": 300},
                    },
                },
            },
            "overall_kill_recommendation": {
                "type": "string",
                "enum": sorted(_KILL_REC_VALUES),
            },
            "rationale": {"type": "string", "maxLength": 600},
        },
    },
}


# ── Context builders ─────────────────────────────────────────────────


def _hyp_block(hyp: dict) -> str:
    """Render hypothesis for the user message."""
    claim = hyp.get("claim") or {}
    if isinstance(claim, dict):
        claim_line = claim.get("one_line") or json.dumps(claim, ensure_ascii=False)[:300]
    else:
        claim_line = str(claim)[:300]
    quotes = hyp.get("verbatim_quotes") or []
    quote_lines = []
    for q in quotes[:3]:
        if isinstance(q, dict):
            quote_lines.append(f'  "{(q.get("text") or "")[:200]}" [chunk_id={q.get("chunk_id") or "?"}]')
        else:
            quote_lines.append(f'  "{str(q)[:200]}"')
    return (
        f"HYPOTHESIS\n"
        f"==========\n"
        f"hypothesis_id:     {hyp.get('hypothesis_id')}\n"
        f"mechanism_family:  {hyp.get('mechanism_family') or '(unknown)'}\n"
        f"mechanism_subtype: {hyp.get('mechanism_subtype') or '(unknown)'}\n"
        f"predicted_dir:     {hyp.get('predicted_direction') or '?'}\n"
        f"predicted_mag:     {hyp.get('predicted_magnitude') or '?'}\n"
        f"claim:             {claim_line}\n"
        f"source_paper_id:   {hyp.get('source_paper_id') or '?'}\n"
        + (f"verbatim quotes:\n" + "\n".join(quote_lines) + "\n" if quote_lines else "")
    )


def _belief_block(belief: Optional[dict]) -> str:
    if not belief:
        return "FAMILY BELIEF\n=============\n(no belief data — family is THIN, no prior autopsies)\n"
    n_obs = belief.get("n_obs", 0)
    g = belief.get("n_green", 0); m = belief.get("n_marginal", 0); r = belief.get("n_red", 0)
    hint = belief.get("direction_hint", "")
    return (
        f"FAMILY BELIEF (Phase B autopsy ledger)\n"
        f"======================================\n"
        f"family:         {belief.get('family')}\n"
        f"distribution:   {g}G / {m}M / {r}R  (n={n_obs})\n"
        f"direction hint: {hint}\n"
    )


def _graveyard_block(collisions: Optional[dict]) -> str:
    if not collisions or not collisions.get("top_collisions"):
        return "GRAVEYARD COLLISIONS\n====================\n(no similar RED outcomes found above min_score)\n"
    rows = collisions["top_collisions"][:3]
    out = ["GRAVEYARD COLLISIONS (Phase 8 collision detector)",
           "=" * 52,
           f"n_total_red found: {collisions.get('n_total_red')}"]
    for c in rows:
        out.append(
            f"  - score={c.get('score'):.2f} (family_match={c.get('family_match')}, "
            f"jaccard={c.get('jaccard'):.2f}) family={c.get('family')}: "
            f"{(c.get('claim_excerpt') or '')[:140]}"
        )
    return "\n".join(out) + "\n"


def _n_trials_block(n_trials_info: Optional[dict]) -> str:
    if not n_trials_info:
        return "BAILEY-LDP n_trials\n===================\n(not computed for this family)\n"
    return (
        f"BAILEY-LDP n_trials\n"
        f"===================\n"
        f"N = {n_trials_info.get('n_trials')}  "
        f"({n_trials_info.get('library_entries')} library + "
        f"{n_trials_info.get('exploration_buffer')} exploration buffer)\n"
        f"Each new trial in this family RAISES the DSR threshold; "
        f"if N is already high, factor that into KILL_BEFORE_TEST.\n"
    )


_KNOWN_SILENT_BUGS = """\
STRICT-GATE KNOWN SILENT BUGS (do NOT repeat these as failure modes —
the gate already knows about them)
=====================================================================
- FF5 + MOM spanning: ALREADY CHECKED
- Newey-West HAC SE lag 6: ALREADY USED
- OOS split: ALREADY DONE
- Borrow-cost realistic: ALREADY APPLIED (for shorts)
- Multi-cost stress 0/30/60/80bp: ALREADY REPORTED
- EW quintile L/S: KNOWN to halve under VW — only flag if hypothesis
  CRUCIALLY depends on quintile L/S structure
- Survivor-bias: PIT SP500 universe is SAFE; top-3000 by mktcap is
  partial-bias — flag if hypothesis uses top-3000 + small-cap regime
"""


# ── Main entry ───────────────────────────────────────────────────────


def generate_pre_mortem(
    hypothesis_id: str,
    *,
    persist: bool = True,
) -> Optional[PreMortemReport]:
    """Run the pre-mortem on one hypothesis. Returns the report or None
    on failure (graceful degradation).

    Pulls all context (hypothesis row + family belief + graveyard
    collisions + n_trials counter) automatically. Caller just supplies
    the hypothesis_id.
    """
    # 1) Load hypothesis row
    hyp_path = _REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"
    if not hyp_path.is_file():
        logger.warning("pre_mortem: hypotheses.jsonl missing")
        return None
    hyp = None
    for ln in hyp_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("hypothesis_id") == hypothesis_id:
            hyp = r
            break
    if hyp is None:
        logger.warning("pre_mortem: hypothesis %s not found", hypothesis_id)
        return None

    # 2) Pull family belief
    belief = None
    try:
        from engine.research.belief_synthesis_context import build_belief_summary
        family = hyp.get("mechanism_family") or ""
        if family:
            beliefs = build_belief_summary(min_obs_per_family=1)
            # Try exact + substring + synonym (mirror safety_rails_for_hypothesis)
            _SYN = {"VOL_RISK_PREMIUM": "VRP", "EARNINGS_DRIFT": "EVENT_DRIFT",
                    "POST_EARNINGS_DRIFT": "EVENT_DRIFT"}
            fam_lower = family.lower()
            match = next((b for b in beliefs if b.family == family), None)
            if match is None:
                syn = _SYN.get(family.upper())
                if syn:
                    match = next((b for b in beliefs if b.family == syn), None)
            if match is None:
                match = next((b for b in beliefs
                              if fam_lower in b.family.lower() or
                                 b.family.lower() in fam_lower), None)
            if match:
                belief = {
                    "family": match.family, "n_obs": match.n_obs,
                    "n_green": match.n_green, "n_marginal": match.n_marginal,
                    "n_red": match.n_red, "direction_hint": match.direction_hint,
                }
    except Exception:
        logger.warning("pre_mortem: belief lookup failed", exc_info=True)

    # 3) Graveyard collisions
    collisions = None
    try:
        # Use the api/main helper but call its underlying logic — duplicating
        # a tiny bit of inline scoring to avoid api import cycle.
        # Easier: just GET the endpoint via direct fn import.
        import sys, importlib
        if "api.main" in sys.modules:
            api_mod = sys.modules["api.main"]
        else:
            api_mod = importlib.import_module("api.main")
        collisions = api_mod.graveyard_collisions(hypothesis_id, top_k=3,
                                                    min_score=0.20)
    except Exception:
        logger.warning("pre_mortem: collision lookup failed", exc_info=True)

    # 4) n_trials
    n_trials_info = None
    try:
        from engine.research.family_trial_counter import (
            count_trials_in_family, count_library_entries_in_family,
            FAMILY_BUFFER_OVERRIDES, DEFAULT_EXPLORATION_BUFFER,
        )
        family = hyp.get("mechanism_family") or ""
        if family:
            n_trials_info = {
                "n_trials":           count_trials_in_family(family),
                "library_entries":    count_library_entries_in_family(family),
                "exploration_buffer": FAMILY_BUFFER_OVERRIDES.get(
                    family.lower(), DEFAULT_EXPLORATION_BUFFER),
            }
    except Exception:
        logger.warning("pre_mortem: n_trials lookup failed", exc_info=True)

    user_msg = "\n".join([
        _hyp_block(hyp),
        _belief_block(belief),
        _graveyard_block(collisions),
        _n_trials_block(n_trials_info),
        _KNOWN_SILENT_BUGS,
        "",
        "Now run the pre-mortem. Be SPECIFIC. Cite academic anchors "
        "where applicable. Recommend KILL_BEFORE_TEST only if there's "
        "strong evidence (high-score graveyard collision OR Bailey-LdP "
        "N very high OR data ask cannot be satisfied).",
    ])

    n_inputs = sum([
        1 if belief else 0,
        len(collisions.get("top_collisions") or []) if collisions else 0,
        1 if n_trials_info else 0,
    ])

    try:
        result = llm_call(
            workload   = "pre_mortem",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "pre_mortem",
            tools      = [_TOOL_SCHEMA],
            max_tokens = 2048,
            scope      = "alpha_pre_mortem",
        )
    except Exception as exc:
        logger.warning("pre_mortem: llm_call failed for %s: %s",
                        hypothesis_id, exc)
        return None

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_pre_mortem":
            payload = tc.input
            break
    if payload is None:
        logger.warning("pre_mortem: %s did not call emit_pre_mortem",
                        hypothesis_id)
        return None

    # Validate
    fm_raw = payload.get("failure_modes") or []
    failure_modes: list[FailureMode] = []
    for fm in fm_raw:
        try:
            sev = str(fm.get("severity"))
            cat = str(fm.get("category"))
            if sev not in _SEVERITY_VALUES or cat not in _CATEGORY_VALUES:
                continue
            failure_modes.append(FailureMode(
                severity         = sev,
                category         = cat,
                description      = str(fm.get("description"))[:500],
                check_suggestion = str(fm.get("check_suggestion"))[:300],
            ))
        except Exception:
            continue

    kill_rec = str(payload.get("overall_kill_recommendation") or "")
    if kill_rec not in _KILL_REC_VALUES:
        kill_rec = "PROCEED_NORMAL"

    report = PreMortemReport(
        pre_mortem_id              = str(uuid.uuid4()),
        hypothesis_id              = hypothesis_id,
        failure_modes              = tuple(failure_modes),
        overall_kill_recommendation = kill_rec,
        rationale                  = str(payload.get("rationale") or "")[:600],
        assessed_ts                = _dt.datetime.utcnow().strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        model                      = result.model,
        cost_usd                   = float(result.cost_usd),
        n_obs_inputs               = n_inputs,
    )

    if persist:
        try:
            PRE_MORTEM_PATH.parent.mkdir(parents=True, exist_ok=True)
            row = _dc.asdict(report)
            row["failure_modes"] = [_dc.asdict(fm) for fm in report.failure_modes]
            with PRE_MORTEM_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("pre_mortem: persist failed", exc_info=True)

    return report


# ── Read API for /api/research/pre_mortem/{hyp_id} ───────────────────


def list_for_hypothesis(hypothesis_id: str) -> list[dict]:
    """Return all pre-mortem rows for a hyp_id, newest first."""
    if not PRE_MORTEM_PATH.is_file():
        return []
    out: list[dict] = []
    for ln in PRE_MORTEM_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("hypothesis_id") == hypothesis_id:
            out.append(r)
    out.sort(key=lambda r: r.get("assessed_ts", ""), reverse=True)
    return out


def latest_for_hypothesis(hypothesis_id: str) -> Optional[dict]:
    rows = list_for_hypothesis(hypothesis_id)
    return rows[0] if rows else None
