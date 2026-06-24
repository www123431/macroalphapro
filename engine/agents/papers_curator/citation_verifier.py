"""engine.agents.papers_curator.citation_verifier — Phase 2.2.

For each paper A's synthesis CLAIMS to cite, verify the claim is
actually substantiated by chunks of that paper. Closes the
hallucination gap: without this, A could attribute "Bollerslev-Tauchen
say VRP works on EM equity" to a paper that didn't study EM equity,
and B + the principal would have no easy way to catch it.

Architecture (Pattern 5–safe):
  - Single deterministic registry + chroma lookup per cited paper
  - ONE cheap LLM call per paper (Deepseek filter workload) asking:
    "do these chunks substantiate the claim?"
  - Output: CitationCheck per paper — confidence 0..1 + supporting
    chunk_ids + 1-line LLM note

Cost: ≤ $0.001 per check × typically 1-3 papers per candidate × 3
candidates per synthesis = ≤ $0.009 per synthesis run. At weekly
cadence: ~$0.50/yr. Negligible.

Where this lands:
  - synthesis_runner (Phase 2.2b, next commit) calls verify_citations
    on each SynthesizedCandidate after run_synthesis()
  - Results land on SynthesizedCandidate.citation_verifications tuple
  - B's review prompt (Phase 2.2c, future) surfaces verifications so
    B can downweight candidates with weak citations
  - papers_curator_synthesis_run event metrics include avg confidence
    so "how often does A hallucinate citations?" is queryable

Fail-OPEN: each step (resolve / fetch / LLM) returns a degraded
CitationCheck on failure rather than blocking the candidate. The
caller decides what to do with low confidence; we don't autonomously
drop synthesis output.
"""
from __future__ import annotations

import dataclasses as _dc
import json
import logging
from typing import Optional

# Top-level for monkeypatch in tests (consistent with synthesis.py,
# strengthener/review.py, autopilot_pre_compute_da.py).
from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Output shape
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class CitationCheck:
    """One verification result per paper A cited.

    Fields:
      paper_id:           what A claimed to cite (registry paper_id)
      paper_resolved:     did we find this paper in the registry?
                           False → confidence 0, paper likely hallucinated
      chunks_queried:     number of relevant chunks fetched from chroma
      confidence:         0..1 LLM's confidence the chunks substantiate
                           the claim (0 if paper unresolved or chroma
                           returned empty)
      supporting_chunks:  chunk_ids the LLM flagged as supporting; can
                           be empty even when confidence > 0 if the LLM
                           reasoned without citing specific chunks
      verifier_notes:     1-line LLM explanation for forensic / B's
                           prompt context (≤ 240 chars)
    """
    paper_id:          str
    paper_resolved:    bool
    chunks_queried:    int
    confidence:        float
    supporting_chunks: tuple[str, ...]
    verifier_notes:    str


# ────────────────────────────────────────────────────────────────────
# Registry lookup
# ────────────────────────────────────────────────────────────────────
def _resolve_paper_to_doi(paper_id: str) -> Optional[str]:
    """Look up the paper in the registry, return its doi (chroma's
    metadata key). Returns None if paper not in registry (likely
    hallucinated)."""
    try:
        from engine.research_store.papers import load_registry, latest_per_doi
        reg = load_registry()
        # Try direct paper_id match
        by_id = {e.paper_id: e for e in reg}
        paper = by_id.get(paper_id)
        if paper is None:
            # Try via latest_per_doi version chain
            latest = latest_per_doi(reg)
            for e in latest.values():
                if e.paper_id == paper_id:
                    paper = e
                    break
        if paper is None:
            return None
        return paper.doi or None
    except Exception as exc:
        logger.warning("citation_verifier: registry lookup failed for %s: %s",
                        paper_id, exc)
        return None


# ────────────────────────────────────────────────────────────────────
# Chroma fetch
# ────────────────────────────────────────────────────────────────────
def _fetch_relevant_chunks(doi: str, claim: str, *, k: int = 3) -> list[dict]:
    """Query papers_chroma for the top-K chunks of this DOI most
    relevant to the claim. Returns list of {chunk_id, text} (text
    truncated to 800 chars to keep LLM prompt bounded).

    Empty list on failure — caller treats as 'cannot verify'."""
    try:
        from engine.research_store.red_lessons.papers_chroma import get_collection
        coll = get_collection()
        res = coll.query(
            query_texts = [claim],
            n_results   = k,
            where       = {"doi": doi},
        )
        ids  = (res.get("ids")  or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        out: list[dict] = []
        for cid, txt in zip(ids, docs):
            out.append({"chunk_id": cid, "text": (txt or "")[:800]})
        return out
    except Exception as exc:
        logger.warning("citation_verifier: chroma fetch failed for doi=%s: %s",
                        doi, exc)
        return []


# ────────────────────────────────────────────────────────────────────
# Verifier LLM call — cheap (Deepseek filter workload)
# ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a citation verifier. Given a research claim and the top-K
chunks of a paper, decide whether the chunks substantiate the claim.

Default stance: NEUTRAL. Most claims are partially supported (some
mechanism overlap + some extrapolation). Use the full 0-1 scale:

  0.0 - 0.3   chunks contradict the claim OR contain nothing relevant
              (likely hallucinated citation)
  0.4 - 0.6   chunks partially support but claim extends beyond the paper
  0.7 - 0.9   chunks clearly support; minor extrapolation
  1.0         direct verbatim support

You MUST cite the supporting chunk_ids you found (empty list if 0.0-0.3).
Your verifier_notes MUST be ONE line ≤ 240 chars — no paragraphs.

Call emit_verification with your result. ALWAYS call it.
"""


_TOOL_DEFINITION = {
    "name": "emit_verification",
    "description": "Emit the citation verification result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "confidence":        {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "supporting_chunks": {
                "type":  "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
            "verifier_notes":    {"type": "string", "maxLength": 240},
        },
        "required": ["confidence", "supporting_chunks", "verifier_notes"],
        "additionalProperties": False,
    },
}


def _format_user_msg(*, claim: str, paper_id: str, chunks: list[dict]) -> str:
    lines = [
        "CLAIM TO VERIFY",
        "===============",
        claim.strip()[:600],
        "",
        f"PAPER (top-{len(chunks)} relevant chunks)",
        "=" * 40,
    ]
    for ch in chunks:
        lines.append(f"  [{ch['chunk_id']}]")
        lines.append(f"    {ch['text']}")
        lines.append("")
    lines.append("Call emit_verification per the tool schema.")
    return "\n".join(lines)


def _llm_verify(*, claim: str, paper_id: str, chunks: list[dict]) -> Optional[dict]:
    """One Deepseek call. Returns dict {confidence, supporting_chunks,
    verifier_notes} on success; None on failure (caller picks
    degraded outcome)."""
    try:
        result = llm_call(
            workload   = "papers_curator_filter",
            system     = _SYSTEM_PROMPT,
            user       = _format_user_msg(claim=claim, paper_id=paper_id, chunks=chunks),
            agent_id   = "papers_curator_filter",
            tools      = [_TOOL_DEFINITION],
            max_tokens = 400,
            scope      = "citation_verifier",
        )
    except Exception as exc:
        logger.warning("citation_verifier: llm_call failed: %s", exc)
        return None

    for tc in (result.tool_calls or ()):
        if tc.name == "emit_verification":
            payload = tc.input
            if isinstance(payload, dict):
                return payload
            try:
                return json.loads(payload)
            except Exception:
                return None
    logger.warning("citation_verifier: model did not call emit_verification")
    return None


# ────────────────────────────────────────────────────────────────────
# Top-level — one CitationCheck per paper_id
# ────────────────────────────────────────────────────────────────────
def verify_one_citation(*, claim: str, paper_id: str,
                          k: int = 3) -> CitationCheck:
    """Verify a single citation. Always returns a CitationCheck (no
    None) — degraded outcomes use the resolved/queried fields to
    communicate why confidence dropped."""
    doi = _resolve_paper_to_doi(paper_id)
    if doi is None:
        return CitationCheck(
            paper_id=paper_id, paper_resolved=False,
            chunks_queried=0, confidence=0.0,
            supporting_chunks=(),
            verifier_notes=f"paper_id {paper_id} not in registry — likely hallucinated",
        )

    chunks = _fetch_relevant_chunks(doi, claim, k=k)
    if not chunks:
        return CitationCheck(
            paper_id=paper_id, paper_resolved=True,
            chunks_queried=0, confidence=0.0,
            supporting_chunks=(),
            verifier_notes="paper resolved but no chunks in chroma — fulltext not ingested",
        )

    payload = _llm_verify(claim=claim, paper_id=paper_id, chunks=chunks)
    if payload is None:
        return CitationCheck(
            paper_id=paper_id, paper_resolved=True,
            chunks_queried=len(chunks), confidence=0.5,
            supporting_chunks=(),
            verifier_notes="LLM verifier unavailable — default to neutral 0.5",
        )

    try:
        conf = float(payload.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = 0.5
    sup = payload.get("supporting_chunks") or []
    if not isinstance(sup, list):
        sup = []
    notes = str(payload.get("verifier_notes", ""))[:240]

    return CitationCheck(
        paper_id=paper_id, paper_resolved=True,
        chunks_queried=len(chunks), confidence=conf,
        supporting_chunks=tuple(str(c)[:80] for c in sup[:5]),
        verifier_notes=notes,
    )


def verify_citations(
    *,
    claim:    str,
    paper_ids: tuple[str, ...],
    k:        int = 3,
) -> tuple[CitationCheck, ...]:
    """Verify each paper_id A cited. Empty input returns () — caller
    treats that as 'no citations to check', not a failure."""
    if not paper_ids:
        return ()
    out: list[CitationCheck] = []
    for pid in paper_ids:
        out.append(verify_one_citation(claim=claim, paper_id=pid, k=k))
    return tuple(out)


# ────────────────────────────────────────────────────────────────────
# Roll-up helpers — what B's prompt + audit event consume
# ────────────────────────────────────────────────────────────────────
def aggregate_citation_quality(
    checks: tuple[CitationCheck, ...],
) -> dict:
    """Roll up per-paper checks into one quality score for the
    candidate. Used by B's prompt context + audit event metrics.

    Returns:
      {
        "n_papers_cited":      int,
        "n_resolved":          int,
        "n_unresolved":        int,   # likely hallucinated
        "mean_confidence":     float,
        "min_confidence":      float,
        "any_unresolved":      bool,
        "low_confidence_flag": bool,  # True if mean < 0.5 OR any unresolved
      }
    """
    if not checks:
        return {
            "n_papers_cited":      0,
            "n_resolved":          0,
            "n_unresolved":        0,
            "mean_confidence":     1.0,   # vacuously OK
            "min_confidence":      1.0,
            "any_unresolved":      False,
            "low_confidence_flag": False,
        }
    resolved   = [c for c in checks if c.paper_resolved]
    unresolved = [c for c in checks if not c.paper_resolved]
    confs      = [c.confidence for c in checks]
    mean_conf  = sum(confs) / len(confs)
    min_conf   = min(confs)
    return {
        "n_papers_cited":      len(checks),
        "n_resolved":          len(resolved),
        "n_unresolved":        len(unresolved),
        "mean_confidence":     round(mean_conf, 3),
        "min_confidence":      round(min_conf, 3),
        "any_unresolved":      bool(unresolved),
        "low_confidence_flag": bool(unresolved) or mean_conf < 0.5,
    }
