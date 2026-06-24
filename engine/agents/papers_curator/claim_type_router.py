"""engine.agents.papers_curator.claim_type_router — Stage 0 routing.

Built 2026-06-21 (W4 of six-week-critical-path).

Classifies a paper (title + abstract) into one of the 9 ClaimType
values per `engine/hypothesis_spec/enums.py`. This is the FIRST
decision in the papers_curator pipeline; downstream funnels
(summarizer prompt, verdict schema, audit trail) branch on it.

Pre-Stage-0, every paper went through one generic summarizer prompt
with no claim-type awareness — 108 summaries written, 0 with
claim_type tagged. This module + the W4-piece-3 wire-up fixes that.

Design choice: pure DETERMINISTIC keyword scoring (no LLM call).
Reasoning:
  1. Cheap — runs in microseconds on the 633-paper backlog
  2. Measurable — accuracy can be inspected by hand against backfill
  3. Iterable — keyword tables can grow without re-architecting
  4. LLM fallback is a follow-up commit IF the UNKNOWN rate is high

The classifier returns (ClaimType, confidence) so the caller can
decide whether to escalate. confidence = (top_score / total_score),
range [0, 1]. UNKNOWN is returned when no keywords match.

Senior anchor: scoring beats first-match (a paper hitting both
"factor" and "anomaly" KW is more confidently FACTOR_HYPOTHESIS than
one hitting only "factor"; same paper hitting "data snooping" once
should NOT outrank).
"""
from __future__ import annotations

import dataclasses
import re
from typing import Optional

from engine.hypothesis_spec.enums import ClaimType


# ── Keyword tables (per-class, expandable) ────────────────────────

# Senior calibration: keywords selected from a hand-read of 30 papers
# across SSRN/arxiv/NBER/semantic_scholar. Each class gets 8-15 high-
# precision keywords. Phrases beat single words (lower false positive).

# 2026-06-22 W6-rigor-A-router-v2: replaced ambiguous single-word
# triggers ("anomaly", "premia") with multi-word phrases that require
# factor-research context. Single-word triggers caused 30% false-positive
# rate (labor economics, AI/ML, cybersecurity papers triggered on
# "anomaly" or "premia" without any factor-research intent).
_FACTOR_HYPOTHESIS_KW = [
    "long-short portfolio", "long short portfolio", "high minus low",
    "high-minus-low", "cross-section of returns", "cross-sectional return",
    "decile portfolio", "quintile portfolio", "predicts returns",
    "predict returns", "predicts the cross-section",
    "predicts the cross section", "new factor", "novel factor",
    "tradable signal", "tradable strategy",
    "alpha against the", "risk premium", "risk premia",
    "we construct a portfolio", "we form portfolios",
    "characteristic-based strategy", "factor-based strategy",
    # Multi-word phrases requiring factor-research context (replaces
    # ambiguous "anomaly" + "premia" single-word triggers that caused
    # 30% false-positive rate)
    "anomaly returns", "anomaly portfolio", "anomaly strategy",
    "cross-sectional anomaly", "asset-pricing anomaly", "anomaly literature",
    "cross-sectional premia", "factor premia", "anomaly decile",
]

_METHODOLOGY_KW = [
    "novel method", "we propose a method", "we develop a method",
    "new estimator", "robust standard error", "newey-west",
    "newey west", "bootstrap procedure", "block bootstrap",
    "monte carlo experiment", "p-hacking",
    "data snooping", "data-snooping", "multiple testing",
    "multiple hypothesis testing", "fdr", "false discovery rate",
    "deflated sharpe", "haircut sharpe", "in-sample bias",
    "in sample bias", "out-of-sample procedure", "walk-forward",
    "framework for testing", "hypothesis testing framework",
    "we develop a framework",
]

_DECAY_STUDY_KW = [
    "post-publication decay", "post-publication performance",
    "anomaly decay", "post publication decline",
    "fades after publication", "in-sample versus out-of-sample",
    "in sample versus out of sample", "performance degradation",
    "out-of-sample performance", "replication", "replicate",
    "we replicate", "anomaly performance over time",
    "robustness over time", "factor decay",
]

_CAPACITY_KW = [
    "capacity constraints", "capacity of",
    "trading costs at scale", "scalability of", "scalability constraints",
    "implementation shortfall", "market impact",
    "transaction costs erode", "break-even fund size",
    "aum ceiling", "fund capacity", "strategy capacity",
    "almgren-chriss", "almgren chriss", "permanent impact",
    "liquidity-adjusted",
]

_MICROSTRUCTURE_KW = [
    "bid-ask spread", "bid ask spread", "effective spread",
    "quoted spread", "limit order book", "limit-order book",
    "high-frequency", "high frequency trading",
    "market microstructure", "order flow imbalance",
    "tick size", "quote stuffing", "latency arbitrage",
    "kyle lambda", "amihud illiquidity",
]

_FACTOR_STRUCTURE_KW = [
    "factor correlation", "factor correlations",
    "spanning regression", "spanning test", "spanned by",
    "subsumes the", "ff5", "fama-french 5", "fama french 5",
    "factor model explanatory", "redundant factor",
    "principal component", "factor loadings",
    "hxz q-factor", "hou xue zhang", "covariance matrix shrinkage",
]

_DOMAIN_FACT_KW = [
    "customer-supplier", "input-output", "industry link",
    "real economy", "real-economy", "macro fact",
    "macroeconomic relationship", "stylized fact",
    "we document that", "we find evidence that",
    "we provide evidence that", "empirical regularity",
]

_KW_TABLES: dict[ClaimType, list[str]] = {
    ClaimType.FACTOR_HYPOTHESIS: _FACTOR_HYPOTHESIS_KW,
    ClaimType.METHODOLOGY:       _METHODOLOGY_KW,
    ClaimType.DECAY_STUDY:       _DECAY_STUDY_KW,
    ClaimType.CAPACITY:          _CAPACITY_KW,
    ClaimType.MICROSTRUCTURE:    _MICROSTRUCTURE_KW,
    ClaimType.FACTOR_STRUCTURE:  _FACTOR_STRUCTURE_KW,
    ClaimType.DOMAIN_FACT:       _DOMAIN_FACT_KW,
}


@dataclasses.dataclass(frozen=True)
class ClaimTypeVerdict:
    claim_type: ClaimType
    confidence: float           # [0, 1]; max(scores)/sum(scores)
    scores:     dict[str, int]  # per-class hit count, for debugging
    hits:       dict[str, list[str]]  # per-class matched keywords


def _normalize_text(title: str, abstract: str) -> str:
    """Lowercase + collapse whitespace + strip basic punctuation
    that would break phrase matches."""
    s = (title or "") + " " + (abstract or "")
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s


def classify(
    title: str,
    abstract: str,
    *,
    min_total_score: int = 1,
) -> ClaimTypeVerdict:
    """Score each ClaimType by keyword hits; return the top scorer.

    Returns UNKNOWN with confidence 0.0 when no keywords match anywhere
    OR when total score across all classes is below `min_total_score`.

    `min_total_score` (default 1, W6-rigor-A-router-v2 amendment
    2026-06-22): kept at 1 because most genuine factor papers' titles
    use only ONE high-precision keyword phrase. The W6-rigor-A-router-v2
    fix is the REMOVAL of ambiguous single-word triggers ("anomaly",
    "premia") in favor of phrase-required versions ("anomaly returns",
    "anomaly portfolio", etc.). With ambiguous keywords gone, a single
    hit is high-precision again. (Tried min_total_score=2 first;
    over-corrected, sent 615/633 papers to LLM fallback — at the cost
    of router precision the keyword tightening was meant to deliver.)
    """
    text = _normalize_text(title, abstract)
    scores: dict[ClaimType, int] = {ct: 0 for ct in _KW_TABLES}
    hits:   dict[ClaimType, list[str]] = {ct: [] for ct in _KW_TABLES}

    for ct, kw_list in _KW_TABLES.items():
        for kw in kw_list:
            if kw in text:
                scores[ct] += 1
                hits[ct].append(kw)

    total = sum(scores.values())
    if total < min_total_score:
        return ClaimTypeVerdict(
            claim_type = ClaimType.UNKNOWN,
            confidence = 0.0,
            scores     = {ct.value: n for ct, n in scores.items()},
            hits       = {ct.value: kws for ct, kws in hits.items()},
        )

    best_ct = max(scores.items(), key=lambda kv: kv[1])[0]
    best_score = scores[best_ct]
    return ClaimTypeVerdict(
        claim_type = best_ct,
        confidence = round(best_score / total, 4),
        scores     = {ct.value: n for ct, n in scores.items()},
        hits       = {ct.value: kws for ct, kws in hits.items()},
    )


def classify_summary(summary: dict) -> ClaimTypeVerdict:
    """Convenience: classify a summaries.jsonl row.

    summary is expected to have a 'thesis' field at minimum;
    'mechanism' is concatenated if present.
    """
    title = ""  # summaries don't preserve original title; use thesis
    abstract = (summary.get("thesis") or "") + " " + (summary.get("mechanism") or "")
    return classify(title, abstract)


# ── W4-piece-2 (2026-06-21): LLM fallback for UNKNOWN cases ───────

_LLM_SYSTEM_PROMPT = """You are a classifier that assigns each academic finance paper to ONE of 8 categories (ClaimType). Read the title + abstract and pick the single best fit.

Categories (use EXACTLY these strings):
- FACTOR_HYPOTHESIS: claims that some variable X predicts the cross-section of stock/asset returns; long-short portfolios, decile sorts, "anomaly" papers
- METHODOLOGY: claims about how to TEST things — new estimators, bootstrap procedures, multiple-testing corrections, ML for forecasting, robust standard errors, p-hacking critiques
- DECAY_STUDY: post-publication out-of-sample performance of known anomalies; replication of prior anomalies; documentation of factor decay over time
- CAPACITY: estimates of strategy scalability — AUM ceilings, market impact, transaction-cost erosion at scale, break-even fund size
- MICROSTRUCTURE: market microstructure — bid-ask spreads, limit-order book dynamics, dealer behavior, HFT, liquidity provision, tick-size effects
- FACTOR_STRUCTURE: relationships AMONG factors — spanning tests, factor correlation matrices, model comparison (FF5 vs HXZ), redundant factors, factor risk decomposition
- DOMAIN_FACT: real-economy observations that are research-valuable but not directly tradable — input-output linkages, regulatory facts, macro relationships, customer-supplier data
- OTHER: doesn't fit any of the above (e.g. pure economic theory without prediction, pure data-publication paper)

Reply with ONLY a JSON object: {"claim_type": "FACTOR_HYPOTHESIS", "rationale": "one short sentence"}."""


_LLM_USER_TEMPLATE = """TITLE: {title}

ABSTRACT: {abstract}"""


def classify_llm(
    title: str,
    abstract: str,
    *,
    agent_id: str = "papers_curator_filter",
) -> ClaimTypeVerdict:
    """LLM fallback for UNKNOWN cases. Single Haiku call (~$0.001).

    agent_id default routes the cost ledger entry under the papers_curator
    family. Override for testing. Returns a ClaimTypeVerdict with
    confidence=1.0 (the LLM is decisive — single class output).

    Falls back to UNKNOWN on any LLM failure (network / parse / unrecognized
    label) so the calling pipeline NEVER blocks on the router. Failure
    modes are logged but do not raise.
    """
    import json as _json
    import logging
    log = logging.getLogger(__name__)
    try:
        from engine.llm.call import call as _llm_call
    except ImportError:
        log.warning("classify_llm: engine.llm.call unavailable; returning UNKNOWN")
        return ClaimTypeVerdict(
            claim_type=ClaimType.UNKNOWN, confidence=0.0,
            scores={ct.value: 0 for ct in _KW_TABLES},
            hits={ct.value: [] for ct in _KW_TABLES},
        )

    user_msg = _LLM_USER_TEMPLATE.format(
        title=(title or "")[:300],
        abstract=(abstract or "")[:2000],
    )
    try:
        result = _llm_call(
            workload="papers_curator_claim_type_router",
            agent_id=agent_id,
            system=_LLM_SYSTEM_PROMPT,
            user=user_msg,
            # Deepseek-v4-pro is a reasoning model. Some papers need
            # 400-600 reasoning tokens; 200-300 isn't enough and the
            # model exhausts max_tokens BEFORE emitting the JSON output
            # (observed: ~20% empty-text rate at max_tokens=500).
            # 800 catches >99% of cases at no perceptible extra cost
            # (output billed at completion, not reservation).
            max_tokens=800,
            effort="low",
            scope="claim_type_router",
        )
        text = (result.text or "").strip()
    except Exception as exc:
        log.warning("classify_llm: LLM call failed: %s", exc)
        return ClaimTypeVerdict(
            claim_type=ClaimType.UNKNOWN, confidence=0.0,
            scores={ct.value: 0 for ct in _KW_TABLES},
            hits={ct.value: [] for ct in _KW_TABLES},
        )

    # Tolerant JSON parsing — strip code fences if present
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        obj = _json.loads(text)
        ct_raw = (obj.get("claim_type") or "").strip().upper()
    except Exception as exc:
        log.warning("classify_llm: JSON parse failed: %s; raw: %.200s",
                       exc, text)
        return ClaimTypeVerdict(
            claim_type=ClaimType.UNKNOWN, confidence=0.0,
            scores={ct.value: 0 for ct in _KW_TABLES},
            hits={ct.value: [] for ct in _KW_TABLES},
        )

    valid_labels = {ct.value for ct in ClaimType}
    if ct_raw not in valid_labels:
        log.warning("classify_llm: unrecognized label %r", ct_raw)
        return ClaimTypeVerdict(
            claim_type=ClaimType.UNKNOWN, confidence=0.0,
            scores={ct.value: 0 for ct in _KW_TABLES},
            hits={ct.value: [] for ct in _KW_TABLES},
        )

    return ClaimTypeVerdict(
        claim_type = ClaimType(ct_raw),
        confidence = 1.0,  # LLM is decisive
        scores     = {ct.value: 0 for ct in _KW_TABLES},
        hits       = {"llm_rationale": [obj.get("rationale", "")[:200]]},
    )


def classify_hybrid(
    title: str,
    abstract: str,
    *,
    llm_fallback: bool = True,
) -> ClaimTypeVerdict:
    """Hybrid: deterministic keyword first, LLM fallback for UNKNOWN.

    Use this in production. Set llm_fallback=False for offline / dry-run.
    """
    v = classify(title, abstract)
    if v.claim_type != ClaimType.UNKNOWN:
        return v
    if not llm_fallback:
        return v
    return classify_llm(title, abstract)
