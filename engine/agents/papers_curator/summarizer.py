"""engine.agents.papers_curator.summarizer — Phase 1.5b deeper summary.

Sits after filter.judge_paper. For each YES-judged candidate (and
on-demand for NO ones the user wants to dig into anyway), produces a
5-field structured summary the user reads in /research/papers/incoming.

Fields:
  thesis              1-2 sentences — what does the paper claim
  mechanism           1-2 sentences — economic story (why should this work)
  testable_hypothesis 1 sentence    — translated to OUR (family, signal_type)
                                      vocabulary so the Composer / autopilot
                                      can pick it up downstream
  why_matters_for_us  1 sentence    — relation to our deployed sleeves
                                      (adjacent / overlap / orthogonal / RED'd)
  risk_flags          list[str]     — short data/sample/decay flags
  recommended_action  enum          — INGEST | READ_AND_DISCARD | SKIP

`recommended_action` is the user's primary triage signal:
  INGEST            = strong enough to run through T7 paper-ingest pipeline
  READ_AND_DISCARD  = worth reading once for ideas; no auto-ingest
  SKIP              = even after deeper look, not worth time

Cost: ~$0.01/paper (Deepseek V4 Pro, ~3k in + ~600 out). 5-10 YES/day
× $0.01 ≈ $0.10/day. Storage: append-only summaries.jsonl, keyed by
(source, source_id).
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import re
from typing import Optional

from engine.agents.papers_curator.crawler import PaperCandidate
from engine.agents.papers_curator.filter import (
    FilterJudgment, _parse_judgment_json,
)
# See filter.py for why this import is at module top, not inline
from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


_VALID_ACTIONS = {"INGEST", "READ_AND_DISCARD", "SKIP"}


@_dc.dataclass(frozen=True)
class PaperSummary:
    """Structured summary; append-only, latest-by-summarized_ts wins."""
    source:              str
    source_id:           str
    thesis:              str
    mechanism:           str
    testable_hypothesis: str
    why_matters_for_us:  str
    risk_flags:          tuple[str, ...]
    recommended_action:  str            # INGEST | READ_AND_DISCARD | SKIP
    triggered_by:        str            # "auto_yes" | "user_request_no" | "user_request_recheck"
    summarized_ts:       str
    model:               str
    raw_response:        str
    # W4-piece-3 (2026-06-21): Stage 0 ClaimType routing tag. Tagged
    # PRE-LLM via engine.agents.papers_curator.claim_type_router.
    # Defaults UNKNOWN for back-compat with pre-W4 records that lack
    # the field. Confidence is the deterministic-router score; 1.0
    # means LLM-fallback was used.
    claim_type:             str   = "UNKNOWN"
    claim_type_confidence:  float = 0.0
    claim_type_router:      str   = ""

    def to_dict(self) -> dict:
        return _dc.asdict(self)


_SYSTEM_PROMPT = """\
You are writing a 5-field structured summary of a finance research
paper for a solo quant researcher. The summary helps them decide
whether to (a) auto-ingest the paper into their research pipeline,
(b) read it personally for ideas, or (c) skip it.

OUR CONTEXT — keep this in mind when writing "why_matters_for_us":
  - We run a small quant book with a handful of DEPLOYED sleeves
    (carry / TSMOM / equity value / commodity / FX / etc.). New
    candidates pass through F14b which tests them via deflated SR +
    OOS Sharpe.
  - We deliberately graveyard-block equity single-name signals (we
    have tested 12+ flavors, all RED). Cross-asset / macro is open.
  - We're solo — implementation cost matters. Papers requiring exotic
    data (paid Bloomberg fields, niche industry data) are usually
    SKIP unless the alpha is extreme.

Output EXACTLY ONE JSON OBJECT, NOTHING ELSE:

{
  "thesis":              "1-2 sentences — what does the paper claim",
  "mechanism":           "1-2 sentences — economic story why this should work",
  "testable_hypothesis": "1 sentence — exact testable claim in finance-factor language (mention factor family + signal type if applicable, e.g. CARRY/CARRY_ROLL_YIELD or EARNINGS_DRIFT/PEAD_SUE)",
  "why_matters_for_us":  "1 sentence — relation to our deployed sleeves (adjacent / overlap / orthogonal / equity-single-name-RED'd)",
  "risk_flags":          ["short sample window", "single asset class", "data not free", "post-publication decay risk", ...],
  "recommended_action":  "INGEST | READ_AND_DISCARD | SKIP"
}

NO markdown fences. NO prose. JUST the JSON.
"""


def _build_user_message(c: PaperCandidate, j: Optional[FilterJudgment]) -> str:
    cats = ", ".join(c.categories) if c.categories else "(unknown)"
    authors = ", ".join(c.authors[:3]) + (" et al" if len(c.authors) > 3 else "")
    lines = []
    lines.append(f"PAPER")
    lines.append(f"-----")
    lines.append(f"title:      {c.title}")
    lines.append(f"authors:    {authors}")
    lines.append(f"published:  {c.published_ts}")
    lines.append(f"arxiv cat:  {cats}")
    if j is not None:
        lines.append(f"")
        lines.append(f"PRIOR FILTER JUDGMENT (1-line triage already run):")
        lines.append(f"  is_tradable_factor: {j.is_tradable_factor}")
        lines.append(f"  category:           {j.category_guess}")
        lines.append(f"  reason:             {j.one_line_reason}")
    lines.append(f"")
    lines.append(f"ABSTRACT")
    lines.append(f"--------")
    lines.append(c.abstract[:3500])
    lines.append(f"")
    lines.append(f"Write the JSON summary now.")
    return "\n".join(lines)


def summarize_paper(
    c: PaperCandidate,
    j: Optional[FilterJudgment] = None,
    *,
    triggered_by: str = "auto_yes",
    max_tokens:   int = 3000,  # was 2000; real-data smoke 2026-06-06
                                # hit 1/6 truncation on a complex paper
                                # (crypto DRL pair trading, long risks
                                # list + nested mechanism). 3000 covers
                                # V4 Pro reasoning + verbose JSON without
                                # being wasteful.
) -> Optional[PaperSummary]:
    """Run Deepseek summary on a candidate. Returns None on LLM failure
    or unparseable response — caller leaves for next run."""
    # W4-piece-3 (2026-06-21): Stage 0 ClaimType routing. Classify
    # the paper BEFORE the summarizer LLM call so the resulting
    # summary is tagged. Defaults to UNKNOWN + 0.0 if router fails;
    # failure NEVER blocks summarization (router is observability,
    # not a gate).
    _ct_label = "UNKNOWN"
    _ct_conf  = 0.0
    _ct_router = ""
    try:
        from engine.agents.papers_curator.claim_type_router import (
            classify_hybrid,
        )
        _v = classify_hybrid(c.title or "", c.abstract or "",
                              llm_fallback=True)
        _ct_label  = _v.claim_type.value
        _ct_conf   = float(_v.confidence)
        # Router id: deterministic-only if no llm_rationale in hits,
        # else hybrid (det + LLM fallback fired).
        _ct_router = ("hybrid_det+llm_2026-06-21"
                       if "llm_rationale" in _v.hits
                       else "deterministic_v1_2026-06-21")
    except Exception as _ct_exc:
        logger.warning("summarizer: claim_type router failed (%s); "
                          "tagging UNKNOWN, summary will proceed", _ct_exc)

    try:
        result = llm_call(
            workload   = "papers_curator_summary",
            system     = _SYSTEM_PROMPT,
            user       = _build_user_message(c, j),
            agent_id   = "papers_curator_summary",
            max_tokens = max_tokens,
            scope      = f"papers_summary:{c.source}",
        )
    except Exception as exc:
        logger.warning("summarizer: llm_call failed for %s/%s: %s",
                        c.source, c.source_id, exc)
        return None

    payload = _parse_judgment_json(result.text)
    if payload is None:
        logger.warning("summarizer: unparseable response for %s/%s: %s",
                        c.source, c.source_id, (result.text or "")[:200])
        return None

    try:
        action = str(payload.get("recommended_action", "SKIP")).upper().strip()
        if action not in _VALID_ACTIONS:
            action = "SKIP"
        risks_raw = payload.get("risk_flags") or []
        if not isinstance(risks_raw, list):
            risks_raw = []
        risks = tuple(str(r)[:140] for r in risks_raw[:10])

        return PaperSummary(
            source                 = c.source,
            source_id              = c.source_id,
            thesis                 = str(payload.get("thesis", ""))[:600],
            mechanism              = str(payload.get("mechanism", ""))[:600],
            testable_hypothesis    = str(payload.get("testable_hypothesis", ""))[:400],
            why_matters_for_us     = str(payload.get("why_matters_for_us", ""))[:400],
            risk_flags             = risks,
            recommended_action     = action,
            triggered_by           = triggered_by,
            summarized_ts          = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            model                  = "deepseek-v4-pro",
            raw_response           = (result.text or "")[:1500],
            claim_type             = _ct_label,
            claim_type_confidence  = _ct_conf,
            claim_type_router      = _ct_router,
        )
    except Exception as exc:
        logger.warning("summarizer: payload → PaperSummary failed for %s/%s: %s",
                        c.source, c.source_id, exc)
        return None
