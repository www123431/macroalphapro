"""engine.agents.papers_curator.filter — Phase 1.5 LLM triage.

Sits between crawler and UI. For each freshly crawled PaperCandidate,
ask Deepseek V4 Pro: "is this a tradable-factor paper worth a quant
researcher's 30 seconds?"

Output triage:
  YES = paper proposes a NEW or REFINED tradable quant signal/mechanism
        with quantitative tests
  NO  = pure theory / survey / commentary / non-quantifiable / off-topic
        for our solo-quant book

Cost: ~$0.001 per paper (~500 tokens in, ~150 tokens out). 30-50
papers/day × $0.001 ≈ $0.05/day. Negligible.

Method note: we use plain JSON-in-text parse instead of tool_use because
R1 cost-route audit (2026-06-05) showed Deepseek tool_use compliance
fails ~53% of the time. For a 3-field judgment, deterministic prompt +
JSON parse is more reliable than the tool_use path.

This module does NOT touch the crawler cache; it WRITES to a separate
judgments.jsonl keyed by (source, source_id). Judgments are append-only
and immutable (re-judging produces a new line; reader picks latest).
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import re
from typing import Optional

from engine.agents.papers_curator.crawler import PaperCandidate
# Top-level import so monkeypatching `llm_call` in tests works
# (engine.llm.__init__ re-exports `call`, which shadows the submodule
# path, so setattr("engine.llm.call.call", ...) fails). Patch this
# module's local name instead.
from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


@_dc.dataclass(frozen=True)
class FilterJudgment:
    """One judgment per (source, source_id). Append-only — re-judging
    produces a new row; reader takes latest by judged_ts."""
    source:           str
    source_id:        str
    is_tradable_factor: bool
    confidence:       float            # 0..1
    one_line_reason:  str
    category_guess:   str              # "new_factor" | "refinement" | "theory" | "survey" | "commentary" | "off_topic" | "unknown"
    judged_ts:        str              # iso UTC
    model:            str              # "deepseek-v4-pro"
    raw_response:     str              # for forensics

    def to_dict(self) -> dict:
        return _dc.asdict(self)


_SYSTEM_PROMPT = """\
You are a triage agent for a solo-quant researcher. Your only job: for
each paper, decide if it is worth the researcher's 30 seconds of
attention.

Worth-it = the paper proposes a NEW or REFINED tradable quantitative
signal / factor / mechanism, OR provides a tradable-relevant robustness
result on an existing factor.

NOT worth-it includes:
  - Pure theory with no empirical test
  - Surveys / literature reviews
  - Editorials / commentaries / policy papers
  - Pure econometric methodology with no factor application
  - Macroeconomic forecasting where the output is not a tradable signal
  - Microstructure latency-arbitrage (we are not HFT)
  - Behavioral / experimental finance with no implementable signal
  - Papers about specific firms / events with no general factor
    (single-stock case studies)

You MUST respond with EXACTLY ONE JSON OBJECT and NOTHING ELSE:

{"is_tradable_factor": true|false,
 "confidence": 0.0-1.0,
 "one_line_reason": "concise reason (<= 25 words)",
 "category_guess": "new_factor" | "refinement" | "theory" | "survey" | "commentary" | "off_topic"}

NO markdown code fences. NO prose before or after. ONE LINE. Just the JSON.
"""


def _build_user_message(c: PaperCandidate) -> str:
    cats = ", ".join(c.categories) if c.categories else "(unknown)"
    authors = ", ".join(c.authors[:3]) + (" et al" if len(c.authors) > 3 else "")
    return (
        f"PAPER\n"
        f"-----\n"
        f"title:      {c.title}\n"
        f"authors:    {authors}\n"
        f"published:  {c.published_ts}\n"
        f"arxiv cat:  {cats}\n"
        f"\n"
        f"ABSTRACT\n"
        f"--------\n"
        f"{c.abstract[:2500]}\n"
        f"\n"
        f"Respond with the JSON object now.\n"
    )


_VALID_CATEGORIES = {"new_factor", "refinement", "theory", "survey",
                      "commentary", "off_topic", "unknown"}


def _parse_judgment_json(text: str) -> Optional[dict]:
    """Extract the JSON object from the LLM response. Tolerates code
    fences and surrounding prose because Deepseek occasionally ignores
    the 'just JSON' instruction. Returns None if no parseable JSON
    found."""
    if not text:
        return None
    # Strip common code-fence wrappers
    s = text.strip()
    if s.startswith("```"):
        # Find the closing fence
        body = re.split(r"^```(?:json)?\s*\n?", s, maxsplit=1, flags=re.MULTILINE)
        if len(body) >= 2:
            s = body[1]
            s = s.rsplit("```", 1)[0]
    # Find the first {...} balanced block
    m = re.search(r"\{[^{}]*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_paper(
    c: PaperCandidate,
    *,
    max_tokens: int = 1500,  # was 400; V4 Pro reasoning tokens
                              # truncated ~60% of outputs on real data
                              # (verified 2026-06-06 on 15 q-fin papers).
                              # 1500 gives slack for both reasoning AND
                              # the 4-field JSON without overcharging.
) -> Optional[FilterJudgment]:
    """Run Deepseek triage on one candidate. Returns FilterJudgment on
    success; None on LLM failure / unparseable response — caller
    treats None as "unjudged, leave for next run"."""
    try:
        result = llm_call(
            workload   = "papers_curator_filter",
            system     = _SYSTEM_PROMPT,
            user       = _build_user_message(c),
            agent_id   = "papers_curator_filter",
            max_tokens = max_tokens,
            scope      = f"papers_filter:{c.source}",
        )
    except Exception as exc:
        logger.warning("filter: llm_call failed for %s/%s: %s",
                        c.source, c.source_id, exc)
        return None

    payload = _parse_judgment_json(result.text)
    if payload is None:
        logger.warning("filter: unparseable response for %s/%s: %s",
                        c.source, c.source_id, (result.text or "")[:200])
        return None

    try:
        cat = str(payload.get("category_guess", "unknown")).lower().strip()
        if cat not in _VALID_CATEGORIES:
            cat = "unknown"
        return FilterJudgment(
            source             = c.source,
            source_id          = c.source_id,
            is_tradable_factor = bool(payload.get("is_tradable_factor", False)),
            confidence         = float(payload.get("confidence", 0.5)),
            one_line_reason    = str(payload.get("one_line_reason", ""))[:300],
            category_guess     = cat,
            judged_ts          = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            model              = "deepseek-v4-pro",
            raw_response       = (result.text or "")[:1000],
        )
    except Exception as exc:
        logger.warning("filter: payload → FilterJudgment failed for %s/%s: %s",
                        c.source, c.source_id, exc)
        return None
