"""scripts/extract_paper_hypotheses.py — T3 driver.

Walks papers_registry for fulltext_status=INGESTED papers, fetches each
paper's chunks from papers_chroma, batches them, calls the LLM
extractor, post-validates candidates, persists Hypothesis records to
data/research_store/hypotheses.jsonl.

Run:
  python scripts/extract_paper_hypotheses.py --limit 1            # smoke test
  python scripts/extract_paper_hypotheses.py --limit 1 --paper-id <UUID>
  python scripts/extract_paper_hypotheses.py --write              # all papers
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.papers import (
    FulltextStatus, latest_per_doi, load_registry,
)
from engine.research_store.red_lessons.mechanism_families import MechanismFamily
from engine.research_store.red_lessons.papers_chroma import get_collection
from engine.research_store.hypothesis import (
    Hypothesis, VerbatimQuote, save_hypothesis,
)
from engine.research_store.hypothesis.schema import (
    ExtractionMethod, HypothesisDirection, HypothesisReviewState,
)
from engine.agents.hypothesis_extractor import (
    extract_hypotheses_from_chunks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("extract_paper_hypotheses")


BATCH_SIZE = 12   # chunks per LLM call. Balance: small enough to fit
                  # in context comfortably; large enough to surface
                  # cross-paragraph claims.


def _fetch_chunks_for_paper(paper_doi: str) -> list[dict]:
    """Return all chunks for the paper, sorted by paragraph_idx."""
    coll = get_collection()
    res = coll.get(where={"doi": paper_doi})
    ids       = res.get("ids") or []
    docs      = res.get("documents") or []
    metas     = res.get("metadatas") or []
    chunks = []
    for cid, doc, meta in zip(ids, docs, metas):
        chunks.append({
            "chunk_id":      cid,
            "text":          doc,
            "section":       meta.get("section", ""),
            "paragraph_idx": meta.get("paragraph_idx", 0),
        })
    chunks.sort(key=lambda c: c["paragraph_idx"])
    return chunks


def _candidate_to_hypothesis(
    cand,
    *,
    source_paper_id: str,
    created_by: str,
) -> Hypothesis:
    """Convert post-validated HypothesisCandidate → Hypothesis dataclass."""
    quotes = tuple(
        VerbatimQuote(
            chunk_id       = q["chunk_id"],
            quote_text     = q["quote_text"],
            section_ref    = q.get("section_ref", ""),
            relevance_note = q.get("relevance_note", ""),
        )
        for q in cand.verbatim_quotes
    )
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return Hypothesis(
        hypothesis_id        = Hypothesis.new_id(),
        source_paper_id      = source_paper_id,
        version              = 1,
        parent_hypothesis_id = None,
        source_chunk_ids     = cand.source_chunk_ids,
        verbatim_quotes      = quotes,
        claim                = cand.claim,
        mechanism_family     = MechanismFamily(cand.mechanism_family),
        mechanism_subtype    = cand.mechanism_subtype,
        predicted_direction  = HypothesisDirection(cand.predicted_direction),
        predicted_magnitude  = cand.predicted_magnitude,
        required_data        = cand.required_data,
        test_methodology     = cand.test_methodology,
        extraction_method    = ExtractionMethod.LLM_EXTRACT,
        review_state         = HypothesisReviewState.PROPOSED,
        created_ts           = now_iso,
        updated_ts           = now_iso,
        created_by           = created_by,
        tags                 = ("t3_llm_extraction",),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually persist Hypothesis records")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N papers (test)")
    ap.add_argument("--paper-id", default=None,
                    help="process only this paper_id (for targeted smoke test)")
    ap.add_argument("--max-batches", type=int, default=99,
                    help="process at most N chunk batches per paper")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip papers that already have hypotheses in store "
                         "(idempotent resume)")
    args = ap.parse_args()

    reg = list(latest_per_doi(load_registry()).values())
    ingested = [e for e in reg if e.fulltext_status == FulltextStatus.INGESTED]
    if args.paper_id:
        ingested = [e for e in ingested if e.paper_id == args.paper_id]
    if args.skip_done:
        from engine.research_store.hypothesis import (
            latest_per_paper, load_hypotheses,
        )
        done_paper_ids = set(latest_per_paper(load_hypotheses()).keys())
        before = len(ingested)
        ingested = [e for e in ingested if e.paper_id not in done_paper_ids]
        logger.info("skip-done filter: %d → %d papers (skipped %d already done)",
                    before, len(ingested), before - len(ingested))
    if args.limit:
        ingested = ingested[: args.limit]
    logger.info("papers to process: %d", len(ingested))

    total_candidates = 0
    total_kept       = 0
    total_dropped    = 0
    n_persisted      = 0

    for paper in ingested:
        chunks = _fetch_chunks_for_paper(paper.doi)
        if not chunks:
            logger.warning("paper %s has no chunks", paper.title[:60])
            continue
        logger.info("[%s %d] %d chunks", paper.title[:50], paper.year, len(chunks))

        # Batch
        batches = [chunks[i:i + BATCH_SIZE]
                   for i in range(0, len(chunks), BATCH_SIZE)]
        if len(batches) > args.max_batches:
            logger.warning("paper has %d batches, capping at %d",
                           len(batches), args.max_batches)
            batches = batches[: args.max_batches]

        paper_metadata = {
            "title":    paper.title,
            "authors":  list(paper.authors),
            "year":     paper.year,
            "venue":    paper.venue,
            "doi":      paper.doi,
            "paper_id": paper.paper_id,
        }

        paper_kept    = []
        paper_dropped = 0
        for bi, batch in enumerate(batches, start=1):
            logger.info("  batch %d/%d (%d chunks)", bi, len(batches), len(batch))
            try:
                result = extract_hypotheses_from_chunks(
                    paper_metadata=paper_metadata,
                    chunks=batch,
                )
            except Exception as e:
                import traceback
                logger.error("  batch %d failed: %s\n%s",
                             bi, e, traceback.format_exc())
                continue

            logger.info("  → kept %d, dropped %d (notes: %s)",
                        len(result.candidates),
                        result.n_dropped_post_validation,
                        result.notes[:120])

            paper_kept.extend(result.candidates)
            paper_dropped += result.n_dropped_post_validation
            total_dropped += result.n_dropped_post_validation
            total_candidates += len(result.candidates) + result.n_dropped_post_validation

        total_kept += len(paper_kept)

        # Persist if --write
        if args.write:
            # Post-extraction dedup per paper (2026-06-15):
            # cluster near-duplicate claims via token-Jaccard, then
            # cap at PER_PAPER_CAP. Target 3-5 hyps/paper instead of
            # the empirical 6-12 we were getting. See
            # engine/research_store/hypothesis/dedup.py for rules.
            from engine.research_store.hypothesis.dedup import dedup_paper_hyps
            # Convert candidates to hypothesis dicts up-front so dedup can score quality.
            hyp_objs = [
                _candidate_to_hypothesis(
                    cand,
                    source_paper_id=paper.paper_id,
                    created_by="engine.agents.hypothesis_extractor:claude-sonnet-4-6",
                )
                for cand in paper_kept
            ]
            # Make dicts for dedup
            import dataclasses as _dc
            hyp_dicts = []
            for hyp in hyp_objs:
                d = _dc.asdict(hyp) if hasattr(hyp, "__dataclass_fields__") else hyp.__dict__
                d["_hyp_obj"] = hyp
                hyp_dicts.append(d)
            kept_dicts, dropped_dicts = dedup_paper_hyps(hyp_dicts)
            if dropped_dicts:
                logger.info(
                    "  → post-extract dedup: %d → %d (dropped %d)",
                    len(hyp_dicts), len(kept_dicts), len(dropped_dicts),
                )
            for d in kept_dicts:
                hyp = d["_hyp_obj"]
                try:
                    save_hypothesis(hyp, validate_strict=False,
                                    skip_cross_checks=True)
                    n_persisted += 1
                except Exception as e:
                    logger.error("save_hypothesis failed: %s", e)

    print()
    print("=" * 64)
    print("T3 EXTRACTION AUDIT")
    print("=" * 64)
    print(f"Papers processed:          {len(ingested)}")
    print(f"Total candidates returned: {total_candidates}")
    print(f"Kept (post-validation):    {total_kept}")
    print(f"Dropped:                   {total_dropped}")
    if args.write:
        print(f"Persisted to jsonl:        {n_persisted}")
    else:
        print(f"\nDRY RUN — pass --write to persist")


if __name__ == "__main__":
    main()
