"""engine.research_store.hypothesis.classifier — rules-based hypothesis_type
classifier.

Why
===
The first authorized cron run (2026-06-11) burned ~$0.09 LLM cost +
30s wall on 3 candidates that the extractor correctly refused — they
were Harvey-Liu-Zhu 2016 methodology + Fama-French 2015 factor-analysis
claims wearing legitimate family tags. The fix: distinguish factor
PROPOSALS from factor ANALYSIS / methodology / sleeve improvement at
the SCHEMA level, then filter to PROPOSALS in burndown_ranker.

Classification approach
=======================
RULES-BASED, not LLM. Reasons:
  * the rules cover the ~95% of obvious cases for free
  * the remaining ~5% (genuinely ambiguous claims) default to UNKNOWN
    which falls out of cron — conservative is the right bias here
  * LLM classification at scale costs N × $0.01 + variance; rules are
    reproducible + auditable

Rule order (first match wins)
=============================
  1. tag-based:
     - source:doctrine_signal      → SLEEVE_IMPROVEMENT
     - created_by contains sleeve_fix_proposer → SLEEVE_IMPROVEMENT
     - created_by contains autopilot          → may be PROPOSAL (no skip)
  2. claim pattern: methodology
       multiple testing / FDR / Bonferroni / publication bias /
       t-ratio threshold / minimum t-stat
  3. claim pattern: factor_analysis
       'becomes redundant' / 'is subsumed' / 'subsumes the alpha' /
       'loses significance' / 'is captured by' / 'spanned by'
  4. extraction_method == LLM_SYNTHESIS WITHOUT a clear proposal
     pattern → UNKNOWN (cron skips; synthesis output gets review)
  5. default → FACTOR_PROPOSAL

Test coverage
=============
Each rule + a few real-world claims from the existing 234-hypothesis
queue (sample). See tests/test_hypothesis_classifier.py.
"""
from __future__ import annotations

import re
from typing import Any


# ── Rule patterns ──────────────────────────────────────────────────


_METHODOLOGY_PATTERNS = (
    r"\bt[\-\s]?ratio\s+(threshold|benchmark|cutoff)\b",
    r"\bminimum\s+t[\-\s]?ratio\b",
    r"\bmultiple[\-\s]+testing\b",
    r"\bfalse[\-\s]+discovery\s+rate\b",
    r"\bp[\-\s]?value\s+threshold\b",
    r"\bbonferroni\b",
    r"\bholm[\-\s]+procedure\b",
    r"\b(BHY|Benjamini[\-\s]Hochberg)\b",
    r"\bpublication\s+bias\b",
    r"\bdata\s+snooping\b",
    r"\bdeflated\s+sharpe\b",
    r"\bhaircut\s+sharpe\b",
    r"\bbacktest\s+overfit",
    r"\bfalse\s+positives?\b",

    # B.2 (2026-06-11) — backtest overfitting / expected-max Sharpe under H0 / etc.
    # Surfaced when scanning 35 OTHER+factor_proposal claims; these are all
    # Bailey-Lopez de Prado / Harvey-Liu-Zhu methodology, not tradable claims.
    r"\bexpected\s+maximum\s+(sharpe|IS|in[\-\s]sample)\b",
    r"\bspuriously\s+high\s+(IS|in[\-\s]sample|sharpe)\b",
    r"\bsearching\s+over\s+a\s+\d+[\-\s]parameter\b",
    r"\bN\s+independent\s+(backtests|strategy\s+configurations|trials)\b",
    r"\bin[\-\s]sample[/\s]+out[\-\s]of[\-\s]sample\s+splits?\b",
    r"\boptimizing\s+IS\s+performance\b",
    r"\brandom\s+walk\s+of\s+~?\d+\s+daily\s+prices\b",
    r"\bstatistical\s+bias\s+(alone|due\s+to)\b",
)

# Capacity / microstructure / cost-research patterns (B.2)
# These are not tradable claims — they measure trading frictions, market impact,
# break-even fund sizes etc. Treated as methodology since hypothesis_type taxonomy
# doesn't have a separate capacity bucket (ClaimType.CAPACITY does, but at the
# hypothesis_type layer we collapse capacity/microstructure into methodology).
_METHODOLOGY_PATTERNS_CAPACITY = (
    r"\bbreak[\-\s]even\s+fund\s+(size|capacity)\b",
    r"\bmarket\s+impact\s+(?:of\s+)?(?:institutional\s+)?(trades?|orders?)\b",
    r"\bprice\s+impact\s+(function|is\s+convex|reverse[sd]?)\b",
    r"\bimplementation\s+shortfall\b",
    r"\b(?:institutional|live)\s+trade\s+data\b",
    r"\btrading\s+cost\s+estimates?\s+(?:derived\s+from|are|in\s+the\s+literature)\b",
    r"\bbasis\s+points\s+(?:and|out\s+of|of|per|reversed)\b.*\bmarket\s+impact\b",
    r"\bshort\s+covering.*price\s+impact\b",
    r"\bconvex\s+in\s+trade\s+size\b",
)
_METHODOLOGY_PATTERNS = _METHODOLOGY_PATTERNS + _METHODOLOGY_PATTERNS_CAPACITY

_FACTOR_ANALYSIS_PATTERNS = (
    r"\bbecomes?\s+(redundant|insignificant|negative|positive\s+and\s+significant)\b",
    r"\bis\s+subsumed\s+by\b",
    r"\bsubsumes?\s+the\s+(predictive\s+power|effect|return\s+predictability|alpha)\b",
    r"\bsubsumes?\s+(its\s+|the\s+)?(?:predictive\s+power\s+of\s+)?\b\w+\s+(?:and|in)\b",
    r"\bloses?\s+(its\s+|all\s+)?(significance|alpha|explanatory\s+power)\b",
    r"\bno\s+longer\s+(significant|predicts|explains)\b",
    r"\bis\s+(fully\s+|completely\s+)?(captured|explained|absorbed)\s+by\b",
    r"\bis\s+(redundant|spanned)\s+(?:by|in)\b",
    r"\bis\s+statistically\s+(redundant|insignificant)\b",
    r"\bdrops?\s+to\s+(zero|near\s+zero|insignificance)\b",
    r"\bsmall\s+stocks\s+with\s+returns\s+that\s+behave\s+like\b",
    r"\balternative\s+\w+[\-\s]factor\s+model\b",
    r"\bwhen\s+\w+\s+is\s+added\s+to\b.*\b(model|regression|factors?)\b",
    r"\bwhen\s+\w+\s+is\s+(controlled|orthogonalized)\s+(?:for|against)\b",

    # B.1 (2026-06-11) — added after second authorized cron run showed
    # 2 CUSTOM_CODE_REQUIRED verdicts from microcap-anomaly-critique
    # claims. These are critiques of existing factor literature:
    # "[anomalies] are driven by microcaps" / "[liquidity vars] are
    # susceptible to inflation". They require comparative microcap-
    # stratified studies, not single-signal backtests.
    r"\banomaly\s+profits?\b",
    r"\b(microcap|small[\-\s]cap)[\-\s](driven|stratified|sensitive)\b",
    r"\bare\s+the\s+primary\s+driver(s)?\s+of\b",
    r"\bare\s+(susceptible|vulnerable)\s+to\s+\w+[\-\s]driven\s+inflation\b",
    r"\b\d+\s+(out\s+of|of)\s+\d+\s+\([\d\.]+%\)\s+[\w\s\-]{0,60}?(variables?|factors?|anomalies)\s+are\s+(insignificant|spurious|inflated)\b",
    r"\busing\s+(NYSE[\-\s]?only|equal[\-\s]weighted)\s+breakpoints?\b",

    # B.2 (2026-06-11) — anomaly cataloging / post-publication decay /
    # cross-sectional factor correlation studies / pre-1963 vs post-1963
    # sample-period analyses. Surfaced from OTHER+factor_proposal sample.
    r"\b(out\s+of|of)\s+\d+\s+anomalies\b",
    r"\bpost[\-\s]publication\s+(return\s+)?(decay|decline)\b",
    r"\bdecay\s+(?:by|of)\s+approximately\s+\d+[\-�C\s]*\d*%\b",
    r"\b(pre|post|before|after)[\-\s]?19\d\d\s+(period|sample|years?)\b",
    r"\bcorrelations?\s+between\s+(?:a\s+)?(?:newly\s+|already[\-\s]?)?discovered\s+anomal(?:y|ies)\b",
    r"\bcross[\-\s]sectional\s+correlation\s+among\s+(equity\s+risk\s+)?factor\s+returns?\b",
    r"\bq[\-\s]?factor\s+model\s+(explains|leaves|of\s+|drives)\b",
    r"\bGRS\s+statistics?\b",
    r"\b(2\s*[×x]\s*3|2\s*[×x]\s*2|2\s*[×x]\s*2\s*[×x]\s*2\s*[×x]\s*2)\s+sorts?\b",
    r"\bsurvive\s+(replication|trading\s+costs)\s+at\b",
    r"\baccounting[\-\s]based\s+anomalies\b",
)

# Tag prefixes / created_by markers
# Phase 1 (2026-06-11) — `source:active_b_sleeve_scan` added after
# discovering sleeve_strengthen_scan produces 17 enhance-class
# hypotheses that the cron was about to silently route through the
# forward strict-gate path. See [[feedback-forward-vs-enhance-
# statistical-separation-2026-06-11]].
_SLEEVE_IMPROVEMENT_TAGS = (
    "source:doctrine_signal",
    "source:active_b_sleeve_scan",
)
_SLEEVE_IMPROVEMENT_CREATED_BY_PARTS = (
    "sleeve_fix_proposer",
    "sleeve_strengthen_scan",
)


# ── Compile regex once ─────────────────────────────────────────────


_METHOD_RE   = tuple(re.compile(p, re.IGNORECASE) for p in _METHODOLOGY_PATTERNS)
_ANALYSIS_RE = tuple(re.compile(p, re.IGNORECASE) for p in _FACTOR_ANALYSIS_PATTERNS)


# ── Public API ─────────────────────────────────────────────────────


def classify_hypothesis_type(h: dict[str, Any]) -> str:
    """Return a HypothesisType.value string for the raw hypothesis dict.

    Stable + deterministic + no LLM call. Caller doesn't need to
    construct a Hypothesis dataclass — operates on the raw jsonl row.
    """
    claim = (h.get("claim") or "").strip()
    tags = tuple(h.get("tags") or ())
    created_by = (h.get("created_by") or "").lower()

    # 1. Tag / origin-based: SLEEVE_IMPROVEMENT
    if any(any(t.startswith(p) for p in _SLEEVE_IMPROVEMENT_TAGS) for t in tags):
        return "sleeve_improvement"
    if any(part in created_by for part in _SLEEVE_IMPROVEMENT_CREATED_BY_PARTS):
        return "sleeve_improvement"
    # 1.5 Phase 1 (2026-06-11): addresses_decay_in non-null marks
    # enhance-class hypothesis (the claim is "fix sleeve X"). Stronger
    # signal than tags — only enhance proposers set this field.
    if h.get("addresses_decay_in"):
        return "sleeve_improvement"

    if not claim:
        # Without claim text we cannot classify content — be conservative
        return "unknown"

    # 2. Methodology check
    if any(rx.search(claim) for rx in _METHOD_RE):
        return "methodology"

    # 3. Factor-analysis check
    if any(rx.search(claim) for rx in _ANALYSIS_RE):
        return "factor_analysis"

    # 4. Synthesis without a clear proposal marker → UNKNOWN
    if h.get("extraction_method") == "llm_synthesis":
        # Most A-employee synthesis outputs ARE proposals, but the few
        # that are sleeve-improvements / amendments hit (1) above. If
        # nothing matched and synthesis_event_ids reference
        # decay_alert / doctrine_signal, the claim is likely amendment.
        synth_evt = tuple(h.get("synthesizes_event_ids") or ())
        if synth_evt:
            return "factor_proposal"
        # No event provenance + no claim pattern: be cautious
        return "unknown"

    # 5. Default: factor_proposal
    return "factor_proposal"


def hypothesis_type_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Diagnostic: count classifications across a list of hypothesis rows."""
    out: dict[str, int] = {}
    for h in rows:
        t = classify_hypothesis_type(h)
        out[t] = out.get(t, 0) + 1
    return dict(sorted(out.items()))
