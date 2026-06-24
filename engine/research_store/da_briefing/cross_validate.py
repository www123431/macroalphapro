"""engine.research_store.da_briefing.cross_validate — chunk_id resolution + verbatim check.

`validate_verdict_cross_store()` runs the deep cross-validation that
`DAVerdict.validate()` skips because it would require importing the
papers stores at dataclass eval time.

Catches:
  - chunk_id that doesn't resolve in papers_chroma
  - quote_text that is NOT a verbatim substring of the chunk
  - paper_id that doesn't resolve in papers_registry
  - paper_id whose status is NOT INGESTED
"""
from __future__ import annotations

import logging

from engine.research_store.da_briefing.schema import DAClaim, DAVerdict

logger = logging.getLogger(__name__)


def validate_verdict_cross_store(verdict: DAVerdict) -> list[str]:
    """Deep cross-store check. Returns list of error strings (empty = OK).

    On ChromaDB / papers_registry unreachable (e.g. in tests), returns
    an error message describing the unreachable state — caller decides
    whether to raise or warn.
    """
    errs: list[str] = []

    # Gather all (chunk_id, paper_id, quote_text) tuples
    all_claims = verdict.refutes + verdict.supports + verdict.conditional

    # ── papers_registry resolution + INGESTED check ─────────────────
    try:
        from engine.research_store.papers import (
            FulltextStatus, load_registry, latest_per_doi,
        )
        reg = load_registry()
        by_id = {e.paper_id: e for e in reg}
        # Also build a per-doi index of latest, since claims may reference
        # the latest paper_id while the entry chain has older versions
        latest_per = latest_per_doi(reg)
        latest_paper_ids = {e.paper_id for e in latest_per.values()}

        for i, claim in enumerate(all_claims):
            entry = by_id.get(claim.paper_id)
            if entry is None:
                errs.append(
                    f"claim[{i}] (stance={claim.stance.value}): paper_id "
                    f"{claim.paper_id} does not resolve in papers_registry"
                )
                continue
            if entry.fulltext_status != FulltextStatus.INGESTED:
                errs.append(
                    f"claim[{i}]: paper {claim.paper_id} has fulltext_status="
                    f"{entry.fulltext_status.value}; must be INGESTED for "
                    f"chunk references to be valid"
                )
    except Exception as e:
        errs.append(f"papers_registry unreachable for cross-validate: {e}")

    # ── papers_chroma chunk_id + verbatim quote check ────────────────
    try:
        from engine.research_store.red_lessons.papers_chroma import get_collection
        coll = get_collection()
        chunk_ids = list({c.chunk_id for c in all_claims})
        if chunk_ids:
            got = coll.get(ids=chunk_ids)
            got_ids   = set(got.get("ids") or [])
            got_docs  = dict(zip(got.get("ids") or [], got.get("documents") or []))
            for i, claim in enumerate(all_claims):
                if claim.chunk_id not in got_ids:
                    errs.append(
                        f"claim[{i}]: chunk_id {claim.chunk_id} not in "
                        f"papers_chroma — LLM may have fabricated"
                    )
                    continue
                doc_text = got_docs.get(claim.chunk_id, "")
                if claim.quote_text not in doc_text:
                    errs.append(
                        f"claim[{i}]: quote_text is NOT a verbatim substring "
                        f"of chunk {claim.chunk_id} — paraphrase / "
                        f"ellipsis / fabrication suspected"
                    )
    except Exception as e:
        errs.append(f"papers_chroma unreachable for cross-validate: {e}")

    # ── target_hypothesis_id resolves ───────────────────────────────
    try:
        from engine.research_store.hypothesis import find_by_id
        hyp = find_by_id(verdict.target_hypothesis_id)
        if hyp is None:
            errs.append(
                f"target_hypothesis_id {verdict.target_hypothesis_id} does "
                f"not resolve in hypotheses store"
            )
    except Exception as e:
        errs.append(f"hypotheses store unreachable for cross-validate: {e}")

    return errs
