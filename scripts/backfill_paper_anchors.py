"""scripts/backfill_paper_anchors.py — P2: paper metadata + full-text per lesson.

For each REDLesson currently at version=1 with paper_motivation=None:
  1. Determine the anchor paper string (lesson-specific hint > family anchor)
  2. OpenAlex lookup → PaperRef (doi/title/authors/year/abstract/venue)
  3. Attempt full-text PDF acquisition (OpenAlex OA → NBER → arXiv)
  4. If PDF acquired, chunk + embed into papers_chroma collection
  5. Write a NEW lesson version=2 with paper_motivation filled +
     `paper_fulltext_*` tags reflecting outcome

Run:
  python scripts/backfill_paper_anchors.py            # dry-run audit
  python scripts/backfill_paper_anchors.py --write    # persist v2 lessons
  python scripts/backfill_paper_anchors.py --write --skip-pdf  # metadata-only

Honest scope: this is an automated first pass. Quality of `key_claim` +
`our_finding` fields is rough — a human review pass refines them later.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.red_lessons import (
    REDLesson, PaperRef, save_lesson, load_lessons, LESSONS_PATH,
    MechanismFamily, MECHANISM_FAMILY_DOCS, ReviewState, LessonStrength,
)
from engine.research_store.red_lessons.store import latest_per_candidate
from engine.research_store.red_lessons.openalex_client import (
    lookup_anchor, parse_anchor_string,
    extract_doi, extract_authors, extract_year, extract_title,
    extract_venue, extract_abstract,
    load_cache, save_cache,
)
from engine.research_store.red_lessons.paper_acquisition import (
    acquire_pdf, extract_pdf_text,
)
from engine.research_store.red_lessons.papers_chroma import (
    chunk_paper, ingest_chunks, collection_stats,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("backfill_paper_anchors")


def _lesson_anchor_hint(lesson: REDLesson) -> str:
    """Decide which anchor-paper string to look up for a lesson.

    Priority:
      1. lesson.candidate_name contains a paper hint (e.g. 'Novy-Marx 2013')
         — use it as-is
      2. Otherwise use the family anchor from MECHANISM_FAMILY_DOCS
    """
    name = lesson.candidate_name or ""
    # Detect paper hint: contains a 4-digit year between 1970-2030
    if re.search(r"\b(19[7-9]\d|20[0-3]\d)\b", name):
        return name

    fam_doc = MECHANISM_FAMILY_DOCS.get(lesson.mechanism_family) or {}
    anchor = fam_doc.get("anchor_paper", "")
    # OTHER's anchor is placeholder; skip
    if "must be filled in lesson-specific context" in anchor:
        return ""
    return anchor


def _build_paper_ref(work: dict, our_finding: str, key_claim: str = "") -> PaperRef:
    authors = extract_authors(work)
    abstract = extract_abstract(work)
    if not key_claim:
        # Use abstract first ~280 chars as auto-claim. Better claims come
        # from later P4 LLM extraction.
        key_claim = (abstract or extract_title(work))[:280]
    return PaperRef(
        title       = extract_title(work),
        year        = extract_year(work) or 0,
        authors     = authors,
        key_claim   = key_claim,
        our_finding = our_finding,
        doi         = extract_doi(work),
        venue       = extract_venue(work),
        section_ref = "",
    )


def _bump_lesson(lesson: REDLesson, *,
                 paper_motivation: PaperRef | None,
                 fulltext_outcome: dict,
                 created_by: str = "engine.backfill_paper_anchors") -> REDLesson:
    """Produce a v=N+1 lesson with paper_motivation filled + new tags."""
    new_tags = list(lesson.tags)

    if paper_motivation is None:
        new_tags.append("p2_no_paper_found")
    else:
        new_tags = [t for t in new_tags if t != "needs_paper_anchor"]
        new_tags.append("p2_paper_anchored")

    if fulltext_outcome.get("ok"):
        new_tags.append(f"p2_fulltext_ok:{fulltext_outcome['source_kind']}")
        new_tags.append(f"p2_n_chunks:{fulltext_outcome['n_chunks']}")
    elif fulltext_outcome.get("attempted"):
        new_tags.append("p2_fulltext_unavailable")
        if fulltext_outcome.get("note"):
            # Store the failure note as a tag (truncated)
            new_tags.append(f"p2_fulltext_note:{fulltext_outcome['note'][:80]}")

    return REDLesson(
        **{**lesson.__dict__,
           "lesson_id":        REDLesson.new_id(),
           "version":          lesson.version + 1,
           "parent_lesson_id": lesson.lesson_id,
           "paper_motivation": paper_motivation,
           "updated_ts":       "2026-06-03T12:00:00Z",
           "created_by":       created_by,
           "tags":             tuple(dict.fromkeys(new_tags)),  # dedupe preserving order
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="persist v2 lessons + ingest PDFs to ChromaDB")
    ap.add_argument("--skip-pdf", action="store_true",
                    help="metadata only; skip full-text acquisition")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N lessons (for testing)")
    args = ap.parse_args()

    all_lessons = load_lessons()
    if not all_lessons:
        print("no lessons in store — run P1 first")
        sys.exit(1)
    latest = latest_per_candidate(all_lessons)
    targets = [L for L in latest.values()
               if L.paper_motivation is None and L.version == 1]
    if args.limit:
        targets = targets[:args.limit]
    print(f"loaded {len(all_lessons)} lessons; {len(targets)} need paper anchors")

    cache = load_cache()

    # Dedupe by anchor hint — many lessons share the same family anchor
    anchor_to_work: dict[str, dict | None] = {}
    anchor_to_pdf_outcome: dict[str, dict] = {}

    metadata_results: list[tuple[REDLesson, PaperRef | None, dict]] = []
    audit_anchor_hits: Counter = Counter()

    for i, lesson in enumerate(targets, start=1):
        anchor_hint = _lesson_anchor_hint(lesson)
        if not anchor_hint:
            metadata_results.append((lesson, None, {"attempted": False}))
            audit_anchor_hits["no_anchor_string"] += 1
            continue

        # Metadata via OpenAlex (cached per anchor string)
        if anchor_hint not in anchor_to_work:
            anchor_to_work[anchor_hint] = lookup_anchor(anchor_hint, cache=cache)
        work = anchor_to_work[anchor_hint]
        if work is None:
            metadata_results.append((lesson, None, {"attempted": False}))
            audit_anchor_hits["openalex_miss"] += 1
            continue

        our_finding = (
            f"Tested via {lesson.candidate_name} — verdict {lesson.verdict}; "
            f"failure_modes={[m.value for m in lesson.failure_modes]}"
        )[:300]
        paper_ref = _build_paper_ref(work, our_finding=our_finding)
        audit_anchor_hits["openalex_hit"] += 1

        # Full-text acquisition (one attempt per unique anchor)
        fulltext_outcome: dict = {"attempted": False}
        if not args.skip_pdf:
            if anchor_hint not in anchor_to_pdf_outcome:
                result = acquire_pdf(work, title=paper_ref.title, year=paper_ref.year)
                outcome: dict = {
                    "attempted":  True,
                    "ok":         False,
                    "source_kind":result.source_kind,
                    "source_url": result.source_url,
                    "note":       result.note,
                    "n_chunks":   0,
                }
                if result.ok:
                    full_text = extract_pdf_text(result.pdf_bytes)
                    if full_text and len(full_text) > 2000:
                        chunks = chunk_paper(
                            full_text,
                            doi             = paper_ref.doi or anchor_hint,
                            title           = paper_ref.title,
                            year            = paper_ref.year,
                            authors         = paper_ref.authors,
                            venue           = paper_ref.venue,
                            source_kind     = result.source_kind,
                            candidate_names = (lesson.candidate_name,),
                        )
                        if chunks:
                            if args.write:
                                ingest_chunks(chunks)
                            outcome["ok"] = True
                            outcome["n_chunks"] = len(chunks)
                    elif not full_text:
                        outcome["note"] = (outcome["note"] or "") + "; pdf extract failed"
                anchor_to_pdf_outcome[anchor_hint] = outcome
            fulltext_outcome = anchor_to_pdf_outcome[anchor_hint]

        metadata_results.append((lesson, paper_ref, fulltext_outcome))

    save_cache(cache)

    # Audit
    print()
    print("=" * 72)
    print("PAPER-ANCHOR AUDIT — P2")
    print("=" * 72)
    print(f"\nTotal lessons targeted:          {len(targets)}")
    print(f"  OpenAlex hit:                  {audit_anchor_hits['openalex_hit']}")
    print(f"  OpenAlex miss / no anchor:     {audit_anchor_hits['openalex_miss'] + audit_anchor_hits['no_anchor_string']}")

    distinct_anchors = len([w for w in anchor_to_work.values() if w is not None])
    print(f"\nDistinct papers resolved:         {distinct_anchors}")

    pdf_attempts = [o for o in anchor_to_pdf_outcome.values() if o.get("attempted")]
    pdf_ok       = [o for o in pdf_attempts if o.get("ok")]
    print(f"\nFull-text attempts (deduped):     {len(pdf_attempts)}")
    print(f"  PDF acquired + ingested:       {len(pdf_ok)}")
    if pdf_ok:
        kinds = Counter(o["source_kind"] for o in pdf_ok)
        for k, n in kinds.most_common():
            print(f"    via {k:15s}    {n}")
        total_chunks = sum(o.get("n_chunks", 0) for o in pdf_ok)
        print(f"  total chunks ingested:         {total_chunks}")
    print(f"  PDF unavailable / failed:      {len(pdf_attempts) - len(pdf_ok)}")

    if args.write:
        # Persist v2 lessons
        n_v2 = 0
        for lesson, paper_ref, outcome in metadata_results:
            v2 = _bump_lesson(lesson, paper_motivation=paper_ref, fulltext_outcome=outcome)
            try:
                save_lesson(v2, validate_strict=False)
                n_v2 += 1
            except Exception as e:
                logger.error("save v2 lesson failed for %s: %s", lesson.candidate_name, e)
        print(f"\nWROTE {n_v2} v2 lessons to {LESSONS_PATH}")
        if not args.skip_pdf:
            stats = collection_stats()
            print(f"papers_chroma collection: {stats}")
    else:
        print(f"\nDRY RUN — pass --write to persist")


if __name__ == "__main__":
    main()
