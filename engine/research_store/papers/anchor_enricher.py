"""engine.research_store.papers.anchor_enricher — Stage C Phase B.

For each T2_ANCHOR paper, enrich the registry entry with:
  1. Full abstract from CrossRef (backfills missing/truncated abstracts)
  2. Sonnet-generated 1-line meta-summary explaining what makes this
     paper a canonical citation anchor for its mechanism class

Why this exists (per Stage C "three libraries" doctrine):
  T2_ANCHOR papers function as CITATION TARGETS, not retrieval
  sources. A's synthesis prompt cites them by `title + 1-line
  summary` to say things like "we already deployed Carry per KMPV
  2018 — this candidate is the orthogonal extension to bond futures."
  Without the 1-line summary, A only has the title — insufficient
  context to use the paper as an actionable anchor.

  Full PDF chunks are NOT needed (those are T1_DOCTRINE work).

Cost discipline:
  - CrossRef API is free + no auth; gentle 1 req/sec
  - Sonnet 1-line summary ~$0.001/paper × 23 anchors = $0.025 total
  - Idempotent: skip papers already enriched (tier_anchor_summary
    non-empty) unless --force
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


_CROSSREF_BASE = "https://api.crossref.org/works"
_USER_AGENT = ("MacroAlphaPro-AnchorEnricher/1.0 "
                "(mailto:${USER_EMAIL})")
_CROSSREF_THROTTLE_S = 0.5


# ────────────────────────────────────────────────────────────────────
# CrossRef abstract pull
# ────────────────────────────────────────────────────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ENT = {"&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&nbsp;": " "}


def _strip_markup(s: str) -> str:
    if not s:
        return ""
    s = _HTML_TAG_RE.sub(" ", s)
    for k, v in _ENT.items():
        s = s.replace(k, v)
    return " ".join(s.split())


@_dc.dataclass(frozen=True)
class CrossrefFetchResult:
    """Output of one CrossRef lookup."""
    found:          bool
    abstract:       str
    venue:          str       # container-title
    error:          str = ""


def fetch_crossref_metadata(doi: str,
                              *, timeout: float = 15.0) -> CrossrefFetchResult:
    """Single CrossRef GET. Returns abstract (markup-stripped) + venue.
    Empty fields on miss/error (callers can fall back)."""
    if not doi:
        return CrossrefFetchResult(False, "", "", "no_doi")
    time.sleep(_CROSSREF_THROTTLE_S)
    url = f"{_CROSSREF_BASE}/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": _USER_AGENT,
        "Accept":     "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return CrossrefFetchResult(False, "", "", f"http_{e.code}")
    except Exception as e:
        return CrossrefFetchResult(False, "", "", f"err:{type(e).__name__}")

    msg = d.get("message") or {}
    abstract = _strip_markup(msg.get("abstract", ""))
    venue_list = msg.get("container-title") or []
    venue = venue_list[0] if venue_list else ""
    return CrossrefFetchResult(True, abstract, venue, "")


# ────────────────────────────────────────────────────────────────────
# Sonnet 1-line anchor summary
# ────────────────────────────────────────────────────────────────────
_SUMMARY_SYSTEM_PROMPT = """\
You are writing a ONE-LINE meta-summary of a canonical quant-finance
paper for use as a CITATION ANCHOR in a research system.

Goal: when the system says "this candidate overlaps with [PAPER]",
the user should immediately know WHY this paper is the anchor for
that mechanism class.

Constraints:
  - EXACTLY one sentence. Max 200 chars.
  - Lead with the MECHANISM CLASS the paper defines / anchors.
  - Include the year + first author last name.
  - DON'T summarize methodology details. DO describe what's at stake
    if a new candidate IS or ISN'T orthogonal to this anchor.

Examples (good):
  "Cross-sectional momentum anchor (Jegadeesh-Titman 1993): defines
   12-1 monthly winners-minus-losers; any new equity momentum
   candidate must demonstrate orthogonal information beyond this."

  "Carry factor anchor (Koijen-Moskowitz-Pedersen-Vrugt 2018):
   defines cross-asset carry as a unified premium; new carry-style
   candidates must show non-trivial cross-asset orthogonality."

  "Quality-Minus-Junk anchor (Asness-Frazzini-Pedersen 2019):
   defines quality factor via 4-pillar (profitability, growth,
   safety, payout); any new 'quality' variant must outperform this
   composite."

Examples (BAD — don't do):
  "Jegadeesh-Titman 1993 examined returns to winner/loser portfolios"
   (no mechanism-class framing; not anchor-actionable)
  "Important paper on momentum showing 12-month formation period"
   (vague; no orthogonality framing)

Output: invoke the emit_anchor_summary tool exactly once.
"""

_SUMMARY_TOOL = {
    "name": "emit_anchor_summary",
    "description": "Emit a 1-line meta-summary for an anchor paper.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": 220},
        },
        "required": ["summary"],
        "additionalProperties": False,
    },
}


def generate_anchor_summary(*, title: str, authors: tuple[str, ...],
                              year: int, abstract: str) -> str:
    """Single Sonnet call → 1-line anchor summary. Returns "" on
    any failure."""
    authors_str = ", ".join(authors[:4]) if authors else "(unknown)"
    user = (
        f"PAPER\n"
        f"-----\n"
        f"title:    {title}\n"
        f"authors:  {authors_str}\n"
        f"year:     {year}\n"
        f"abstract: {abstract[:1500] if abstract else '(none)'}\n"
    )
    try:
        result = llm_call(
            workload   = "papers_anchor_summary",
            system     = _SUMMARY_SYSTEM_PROMPT,
            user       = user,
            agent_id   = "papers_anchor_summary",
            tools      = [_SUMMARY_TOOL],
            max_tokens = 512,
            scope      = "stage_c_phase_b_anchor_enrichment",
        )
    except Exception as exc:
        logger.warning("anchor_summary: llm_call failed: %s", exc)
        return ""
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_anchor_summary":
            return str(tc.input.get("summary") or "").strip()
    return ""


# ────────────────────────────────────────────────────────────────────
# Main entry — orchestrate per-paper enrichment
# ────────────────────────────────────────────────────────────────────
def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


@_dc.dataclass(frozen=True)
class EnrichmentResult:
    """Per-paper enrichment outcome."""
    paper_id:           str
    title:              str
    crossref_found:     bool
    abstract_updated:   bool       # True if we backfilled abstract
    new_abstract:       str        # the longer CrossRef abstract (if any)
    summary_generated:  bool
    summary:            str
    errors:             tuple[str, ...] = ()


def enrich_paper(paper, *, force: bool = False) -> EnrichmentResult:
    """Enrich ONE registry entry: CrossRef fetch → optional abstract
    backfill → Sonnet 1-line summary. Idempotent on summary unless
    force=True.
    """
    errors: list[str] = []
    # Idempotent skip
    if not force and paper.tier_anchor_summary:
        return EnrichmentResult(
            paper_id=paper.paper_id, title=paper.title,
            crossref_found=False, abstract_updated=False,
            new_abstract="",
            summary_generated=False, summary=paper.tier_anchor_summary,
            errors=(),
        )

    # 1. CrossRef abstract pull (if DOI exists)
    cr = fetch_crossref_metadata(paper.doi)
    if not cr.found and cr.error:
        errors.append(f"crossref:{cr.error}")
    abstract_updated = False
    effective_abstract = paper.abstract
    if cr.found and cr.abstract and len(cr.abstract) > len(paper.abstract or ""):
        effective_abstract = cr.abstract
        abstract_updated = True

    # 2. Sonnet summary
    summary = generate_anchor_summary(
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        abstract=effective_abstract,
    )
    summary_generated = bool(summary)
    if not summary_generated:
        errors.append("summary:empty")

    return EnrichmentResult(
        paper_id=paper.paper_id, title=paper.title,
        crossref_found=cr.found,
        abstract_updated=abstract_updated,
        new_abstract=(effective_abstract if abstract_updated else ""),
        summary_generated=summary_generated,
        summary=summary,
        errors=tuple(errors),
    )
