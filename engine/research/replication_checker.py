"""engine/research/replication_checker.py — γ Replication Checker.

Phase γ of the single-agent specialist trio (α / β / γ). Where α
(pre_mortem) is GENERAL adversarial review and β
(cross_domain_transfer) is GENERATIVE cross-asset transfer, γ is
LIT-AWARE replication-failure detection.

ACADEMIC ANCHOR
================
- Hou-Xue-Zhang 2020 "Replicating Anomalies" (RFS): tested ~452
  cross-sectional anomalies from the literature; ~50% fail to replicate
  under their q-factor model after controlling for microcaps + NYSE
  breakpoints. The single most-cited "literature is overstated" paper.
- McLean-Pontiff 2016 "Does Academic Research Destroy Stock Return
  Predictability?": catalog of 97 anomalies, ~58% mean Sharpe drop
  post-publication. Companion paper to HXZ.
- Harvey-Liu-Zhu 2016 "...and the Cross-Section of Expected Returns":
  316 anomalies surveyed; suggests multiple-testing correction needed.
- Linnainmaa-Roberts 2018 "The History of the Cross-Section of Stock
  Returns": OOS replication of 36 anomalies; many die out-of-sample.

WHY γ EXISTS
============
α flags general failure modes (overfitting / data regime / cost
realism). β proposes new cross-asset transfers. NEITHER specifically
asks "has this exact mechanism class been tested in the literature
and found to fail?" That's γ's job — literature-evidence-grounded
prior on whether the hypothesis is worth our 9-verdicts/week budget.

OUTPUT SCHEMA
=============
ReplicationCheck
  replication_status: enum
    PROBABLY_DEAD        — HXZ/MP catalog matches show post-pub
                           Sharpe drop > 60% OR HXZ q-factor reject
    DECAYED_BUT_LIVE     — published, decayed 20-60%, may still pay
                           on enhance-margin pipeline
    WORTH_TESTING        — no strong replication failure evidence
    NOT_FOUND_IN_LIT     — genuinely novel mechanism, no prior
  flags: list[ReplicationFlag] (0-3 items)
    matched_paper:       author + year
    replication_evidence: which study, what they found
    estimated_alpha_decay_pct: 0.0-1.0 LLM estimate
    confidence:          0.0-0.99 LLM self-rating
  rationale: 1-3 sentence summary
  est_post_pub_sharpe_factor: 0.0-1.0 multiplier on published Sharpe
    (1.0 = no decay, 0.42 = McLean-Pontiff catalog avg)

PERSISTENCE
===========
data/research/replication_checks.jsonl
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
REPLICATION_PATH = _REPO_ROOT / "data" / "research" / "replication_checks.jsonl"

_STATUS_VALUES = {
    "PROBABLY_DEAD",
    "DECAYED_BUT_LIVE",
    "WORTH_TESTING",
    "NOT_FOUND_IN_LIT",
}


@_dc.dataclass(frozen=True)
class ReplicationFlag:
    matched_paper:             str
    replication_evidence:      str
    estimated_alpha_decay_pct: float
    confidence:                float


@_dc.dataclass(frozen=True)
class ReplicationCheck:
    check_id:                   str
    hypothesis_id:              str
    replication_status:         str
    flags:                      tuple[ReplicationFlag, ...]
    rationale:                  str
    est_post_pub_sharpe_factor: float
    assessed_ts:                str
    model:                      str
    cost_usd:                   float


_SYSTEM_PROMPT = """\
You are the REPLICATION CHECKER. Your job: given a quantitative
trading hypothesis, scan it against the literature on
replication-failure catalogs and emit a structured replication-
status verdict.

ACADEMIC GROUNDING (cite by name when matching)
================================================
- Hou-Xue-Zhang 2020 "Replicating Anomalies" (RFS): tested ~452
  cross-sectional anomalies; ~50% fail to replicate under their
  q-factor model with microcaps removed + NYSE breakpoints.
- McLean-Pontiff 2016 "Does Academic Research Destroy Stock
  Return Predictability?": ~58% mean Sharpe drop post-publication
  across 97 anomalies.
- Harvey-Liu-Zhu 2016: multiple-testing correction → many
  published anomalies are noise after Bonferroni.
- Linnainmaa-Roberts 2018: OOS replication; many factors die
  in pre-1963 / post-2000 windows.
- Fama-French 2018: 5-factor model spans many published anomalies.

YOUR TASK
=========
Given the hypothesis's CLAIM + MECHANISM_FAMILY + verbatim quotes
from the source paper:

1. Scan it against your knowledge of the published anomaly
   literature. Is this a re-statement of (or close to) a
   well-known published anomaly?
2. If yes: emit ReplicationFlag items naming the matched paper(s)
   + the replication evidence (which study tested it, what they
   found, what Sharpe-decay estimate). Confidence ≤ 0.85 unless
   you can name a specific catalog entry.
3. If no matched anomaly: emit 0 flags + NOT_FOUND_IN_LIT.

OUTPUT
======
- replication_status (REQUIRED enum):
    PROBABLY_DEAD       → HXZ catalog says it doesn't replicate
                          OR MP catalog says ≥ 60% post-pub decay
    DECAYED_BUT_LIVE    → published, decayed 20-60%, may still
                          pay on enhance-margin pipeline (paired
                          bootstrap can detect small alpha that
                          forward strict-gate would reject)
    WORTH_TESTING       → no strong replication failure evidence
                          in catalogs you know
    NOT_FOUND_IN_LIT    → genuinely novel; no prior to apply
- flags: 0-3 ReplicationFlag items (more = noisy, prefer few)
- est_post_pub_sharpe_factor: 0.0-1.0 multiplier on the
  hypothesis's predicted Sharpe (1.0 = no decay, 0.42 = MP
  catalog average for published anomalies)
- rationale: 1-3 sentences

CAUTION
=======
- Cite SPECIFIC papers by author + year. Do not vaguely say
  "this is in the literature."
- est_post_pub_sharpe_factor: if you're guessing (no specific
  catalog match), prefer 0.50-0.70 range (priors from MP catalog).
  Only go ≤ 0.30 with specific catalog evidence.
- If status is NOT_FOUND_IN_LIT, set est_post_pub_sharpe_factor
  to 1.0 (no prior decay) but flag rationale should warn that
  novelty = absence of replication evidence = higher uncertainty.
"""


_TOOL_SCHEMA = {
    "name": "emit_replication_check",
    "description": "Emit a structured replication-status check.",
    "input_schema": {
        "type": "object",
        "required": ["replication_status", "flags",
                     "est_post_pub_sharpe_factor", "rationale"],
        "properties": {
            "replication_status": {"type": "string", "enum": sorted(_STATUS_VALUES)},
            "flags": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "required": ["matched_paper", "replication_evidence",
                                 "estimated_alpha_decay_pct", "confidence"],
                    "properties": {
                        "matched_paper":              {"type": "string", "maxLength": 200},
                        "replication_evidence":       {"type": "string", "maxLength": 500},
                        "estimated_alpha_decay_pct":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "confidence":                 {"type": "number", "minimum": 0.0, "maximum": 0.99},
                    },
                },
            },
            "est_post_pub_sharpe_factor": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale":                  {"type": "string", "maxLength": 600},
        },
    },
}


# ── Hypothesis renderer ─────────────────────────────────────────────


def _hyp_block(hyp: dict) -> str:
    claim = hyp.get("claim") or {}
    if isinstance(claim, dict):
        claim_line = claim.get("one_line") or json.dumps(claim, ensure_ascii=False)[:300]
    else:
        claim_line = str(claim)[:300]
    quotes = hyp.get("verbatim_quotes") or []
    quote_lines = []
    for q in quotes[:3]:
        if isinstance(q, dict):
            quote_lines.append(f'  "{(q.get("text") or "")[:200]}"')
        else:
            quote_lines.append(f'  "{str(q)[:200]}"')
    return (
        f"HYPOTHESIS\n"
        f"==========\n"
        f"hypothesis_id:     {hyp.get('hypothesis_id')}\n"
        f"mechanism_family:  {hyp.get('mechanism_family') or '?'}\n"
        f"mechanism_subtype: {hyp.get('mechanism_subtype') or '?'}\n"
        f"predicted_dir:     {hyp.get('predicted_direction') or '?'}\n"
        f"predicted_mag:     {hyp.get('predicted_magnitude') or '?'}\n"
        f"claim:             {claim_line}\n"
        f"source_paper_id:   {hyp.get('source_paper_id') or '?'}\n"
        + (f"verbatim quotes:\n" + "\n".join(quote_lines) + "\n" if quote_lines else "")
    )


def check_replication_text(
    claim_text: str,
    *,
    mechanism_family: str = "",
    target_asset_class: str = "",
    persist_key: str = "",
) -> Optional[ReplicationCheck]:
    """Variant for brainstorm pre-vet — accepts raw claim text instead
    of looking up a hypothesis row. Same Sonnet call, schema, and
    output shape. Used by the brainstorm UI 'Pre-vet' button to ask
    'is this idea already in HXZ/MP catalog?' BEFORE PM spends time.

    persist_key is used as the hypothesis_id in the persisted row so the
    UI can re-fetch the result. Pass the brainstorm idea_id."""
    # Build a synthetic hypothesis dict the existing renderer can consume
    pseudo_hyp = {
        "hypothesis_id":   persist_key or "brainstorm_text",
        "mechanism_family": mechanism_family or "?",
        "mechanism_subtype": "(brainstorm draft)",
        "predicted_direction": "?",
        "predicted_magnitude": "?",
        "claim":           {"one_line": claim_text[:400]},
        "source_paper_id": "(brainstorm — no source paper)",
        "verbatim_quotes": [],
    }
    user_msg = "\n".join([
        _hyp_block(pseudo_hyp),
        f"TARGET ASSET CLASS: {target_asset_class or '?'}",
        "",
        "Now run the replication check on this BRAINSTORM IDEA (not "
        "a paper-extracted hypothesis). Same rules: cite SPECIFIC "
        "papers, NOT_FOUND_IN_LIT is fine and honest.",
    ])
    try:
        result = llm_call(
            workload   = "replication_checker",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "replication_checker",
            tools      = [_TOOL_SCHEMA],
            max_tokens = 1536,
            scope      = "gamma_brainstorm_prevet",
        )
    except Exception as exc:
        logger.warning("replication_checker(text): llm_call failed: %s", exc)
        return None
    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_replication_check":
            payload = tc.input
            break
    if payload is None:
        return None
    return _parse_and_persist(payload, persist_key or "brainstorm_text", result)


def _parse_and_persist(payload: dict, hypothesis_id: str, result) -> Optional[ReplicationCheck]:
    """Shared parser + persister between hypothesis-based + text-based check."""
    status = str(payload.get("replication_status") or "")
    if status not in _STATUS_VALUES:
        status = "WORTH_TESTING"
    flags: list[ReplicationFlag] = []
    for f in (payload.get("flags") or []):
        try:
            decay = float(f.get("estimated_alpha_decay_pct"))
            conf = float(f.get("confidence"))
            if not (0.0 <= decay <= 1.0) or not (0.0 <= conf <= 0.99):
                continue
            flags.append(ReplicationFlag(
                matched_paper             = str(f.get("matched_paper"))[:200],
                replication_evidence      = str(f.get("replication_evidence"))[:500],
                estimated_alpha_decay_pct = decay,
                confidence                = conf,
            ))
        except Exception:
            continue
    try:
        sharpe_factor = float(payload.get("est_post_pub_sharpe_factor"))
        if not (0.0 <= sharpe_factor <= 1.0):
            sharpe_factor = 0.50
    except (TypeError, ValueError):
        sharpe_factor = 0.50
    check = ReplicationCheck(
        check_id                   = str(uuid.uuid4()),
        hypothesis_id              = hypothesis_id,
        replication_status         = status,
        flags                      = tuple(flags),
        rationale                  = str(payload.get("rationale") or "")[:600],
        est_post_pub_sharpe_factor = sharpe_factor,
        assessed_ts                = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        model                      = result.model,
        cost_usd                   = float(result.cost_usd),
    )
    try:
        REPLICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = _dc.asdict(check)
        row["flags"] = [_dc.asdict(f) for f in check.flags]
        with REPLICATION_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("replication_checker: persist failed", exc_info=True)
    return check


# ── Main ────────────────────────────────────────────────────────────


def check_replication(
    hypothesis_id: str,
    *,
    persist: bool = True,
) -> Optional[ReplicationCheck]:
    """Run the replication-checker on one hypothesis."""
    hyp_path = _REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"
    if not hyp_path.is_file():
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
        return None

    user_msg = "\n".join([
        _hyp_block(hyp),
        "",
        "Now run the replication check. Cite SPECIFIC papers by "
        "author + year. Status NOT_FOUND_IN_LIT is fine and honest "
        "when the claim is genuinely novel — do NOT invent matches "
        "to fill the flags array.",
    ])

    try:
        result = llm_call(
            workload   = "replication_checker",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "replication_checker",
            tools      = [_TOOL_SCHEMA],
            max_tokens = 1536,
            scope      = "gamma_replication_checker",
        )
    except Exception as exc:
        logger.warning("replication_checker: llm_call failed for %s: %s",
                        hypothesis_id, exc)
        return None

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_replication_check":
            payload = tc.input
            break
    if payload is None:
        logger.warning("replication_checker: %s did not call tool",
                        hypothesis_id)
        return None

    status = str(payload.get("replication_status") or "")
    if status not in _STATUS_VALUES:
        status = "WORTH_TESTING"  # safe default

    flags: list[ReplicationFlag] = []
    for f in (payload.get("flags") or []):
        try:
            decay = float(f.get("estimated_alpha_decay_pct"))
            conf = float(f.get("confidence"))
            if not (0.0 <= decay <= 1.0) or not (0.0 <= conf <= 0.99):
                continue
            flags.append(ReplicationFlag(
                matched_paper             = str(f.get("matched_paper"))[:200],
                replication_evidence      = str(f.get("replication_evidence"))[:500],
                estimated_alpha_decay_pct = decay,
                confidence                = conf,
            ))
        except Exception:
            continue

    try:
        sharpe_factor = float(payload.get("est_post_pub_sharpe_factor"))
        if not (0.0 <= sharpe_factor <= 1.0):
            sharpe_factor = 0.50
    except (TypeError, ValueError):
        sharpe_factor = 0.50

    check = ReplicationCheck(
        check_id                   = str(uuid.uuid4()),
        hypothesis_id              = hypothesis_id,
        replication_status         = status,
        flags                      = tuple(flags),
        rationale                  = str(payload.get("rationale") or "")[:600],
        est_post_pub_sharpe_factor = sharpe_factor,
        assessed_ts                = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        model                      = result.model,
        cost_usd                   = float(result.cost_usd),
    )

    if persist:
        try:
            REPLICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
            row = _dc.asdict(check)
            row["flags"] = [_dc.asdict(f) for f in check.flags]
            with REPLICATION_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("replication_checker: persist failed", exc_info=True)

    return check


# ── Read API ─────────────────────────────────────────────────────────


def list_for_hypothesis(hypothesis_id: str) -> list[dict]:
    if not REPLICATION_PATH.is_file():
        return []
    out: list[dict] = []
    for ln in REPLICATION_PATH.read_text(encoding="utf-8").splitlines():
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
