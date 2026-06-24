"""scripts/crosslink_lessons_to_papers.py — Q-E: reverse index lessons → papers.

For each RED Lesson with paper_motivation:
  1. Look up the paper in registry by DOI
  2. If exists: amend with referenced_by_lessons += [lesson_id] +
     add RED_MOTIVATION shelf (if not already present)
  3. If NOT in registry: create new entry from the lesson's PaperRef +
     shelf RED_MOTIVATION

Idempotent — re-running won't double-link (we union by lesson_id).

Run:
  python scripts/crosslink_lessons_to_papers.py            # dry-run
  python scripts/crosslink_lessons_to_papers.py --write    # persist
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.papers import (
    FulltextStatus, PaperRegistryEntry, Shelf, REGISTRY_PATH,
    amend_entry, find_by_doi, load_registry, save_entry,
)
from engine.research_store.red_lessons import load_lessons
from engine.research_store.red_lessons.store import latest_per_candidate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("crosslink")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="persist amendments + new entries")
    args = ap.parse_args()

    # Load lessons (latest per candidate_name)
    all_lessons = load_lessons()
    latest_lessons = list(latest_per_candidate(all_lessons).values())
    with_paper = [L for L in latest_lessons if L.paper_motivation is not None]
    without_paper = [L for L in latest_lessons if L.paper_motivation is None]
    logger.info("lessons total=%d  with_paper=%d  without_paper=%d",
                len(latest_lessons), len(with_paper), len(without_paper))

    # Load registry (latest per DOI semantics — store helper handles)
    registry = load_registry()
    # build doi -> latest entry mapping
    by_doi: dict[str, PaperRegistryEntry] = {}
    for e in registry:
        if not e.doi:
            continue
        key = e.doi.lower()
        prior = by_doi.get(key)
        if prior is None or e.version > prior.version:
            by_doi[key] = e
    logger.info("registry entries=%d  unique-by-doi=%d",
                len(registry), len(by_doi))

    # Group lessons by paper DOI
    doi_to_lessons: dict[str, list] = defaultdict(list)
    lessons_no_doi: list = []
    for L in with_paper:
        d = (L.paper_motivation.doi or "").strip().lower()
        if d:
            doi_to_lessons[d].append(L)
        else:
            lessons_no_doi.append(L)

    logger.info("lesson-paper groups: %d unique DOIs + %d lessons without DOI",
                len(doi_to_lessons), len(lessons_no_doi))

    # ── B1+B2 recovery: for lessons WITHOUT paper_motivation, try to find
    #    a registry paper via mechanism_family motivation/doctrine shelf.
    #    This recovers ~13 of the 20 "no-paper" lessons that failed P2's
    #    OpenAlex lookup but whose family anchor IS already in registry
    #    (KMPV / Frazzini-Pedersen / Bernard-Thomas etc).
    family_to_paper_doi: dict[str, str] = {}
    for e in by_doi.values():
        # If a paper has DOCTRINE_METHOD or GREEN_MOTIVATION shelf, treat it as
        # the family-canonical paper. Index by the mechanism_family it
        # belongs to. We use the first-author surname or known patterns to
        # link family — but simpler: read the entry's referenced_by_sleeves /
        # referenced_by_factors to determine family membership.
        pass
    # Instead use the family → known-paper-author mapping from
    # MECHANISM_FAMILY_DOCS as a hint and find matching registry entry.
    from engine.research_store.red_lessons.mechanism_families import (
        MECHANISM_FAMILY_DOCS,
    )
    from engine.research_store.red_lessons.openalex_client import (
        parse_anchor_string,
    )

    recovered_b1b2: list = []  # (lesson, registry_entry)
    bucket3_engineering: list = []
    for L in without_paper:
        family = L.mechanism_family
        anchor_str = (MECHANISM_FAMILY_DOCS.get(family) or {}).get("anchor_paper", "")
        if not anchor_str or "must be filled" in anchor_str:
            bucket3_engineering.append(L)
            continue
        parsed = parse_anchor_string(anchor_str)
        if parsed is None or not parsed.authors:
            bucket3_engineering.append(L)
            continue
        # Find registry entry whose authors share ≥1 surname with parsed
        expected_lower = [a.lower() for a in parsed.authors]
        match = None
        for e in by_doi.values():
            e_authors_lower = [a.lower() for a in e.authors]
            if any(a in e_authors_lower for a in expected_lower):
                # Prefer entry whose title also resembles parsed.title
                if parsed.title and parsed.title.lower() in (e.title or "").lower():
                    match = e
                    break
                if match is None:
                    match = e
        if match is not None:
            recovered_b1b2.append((L, match))
            # Add lesson_id to that paper's doi group
            doi_key = match.doi.lower()
            doi_to_lessons[doi_key].append(L)
        else:
            bucket3_engineering.append(L)

    logger.info("B1+B2 recovery: %d lessons matched to registry by family-author overlap; "
                "B3 engineering: %d lessons with no academic anchor",
                len(recovered_b1b2), len(bucket3_engineering))

    # Plan amendments + new entries
    amendments: list[PaperRegistryEntry] = []
    new_entries: list[PaperRegistryEntry] = []
    skipped_no_doi = lessons_no_doi  # we don't create entries without DOI

    for doi, lessons in doi_to_lessons.items():
        lesson_ids = [L.lesson_id for L in lessons]
        candidate_names = [L.candidate_name for L in lessons]

        existing = by_doi.get(doi)
        if existing is not None:
            # Amend: add RED_MOTIVATION shelf + referenced_by_lessons
            shelf_notes_add = {}
            if Shelf.RED_MOTIVATION not in existing.shelves:
                shelf_notes_add[Shelf.RED_MOTIVATION.value] = (
                    f"Backfilled Q-E 2026-06-03: motivated "
                    f"{len(lessons)} RED Lesson candidate(s) — "
                    f"{', '.join(candidate_names[:3])}"
                    f"{', ...' if len(candidate_names) > 3 else ''}."
                )
            amended = amend_entry(
                existing,
                add_shelves       = (Shelf.RED_MOTIVATION,),
                add_shelf_notes   = shelf_notes_add,
                add_lessons       = lesson_ids,
                add_tags          = ("backfill_qe_2026-06-03",),
                updated_ts        = "2026-06-03T15:00:00Z",
                created_by        = "engine.crosslink_lessons_to_papers",
                note_append       = f"Q-E: linked to {len(lesson_ids)} lesson(s).",
            )
            amendments.append(amended)
        else:
            # Create new entry from the first lesson's PaperRef
            first = lessons[0]
            pr = first.paper_motivation
            entry = PaperRegistryEntry(
                paper_id              = PaperRegistryEntry.new_id(),
                version               = 1,
                parent_paper_id       = None,
                doi                   = pr.doi,
                title                 = pr.title,
                year                  = pr.year,
                authors               = pr.authors,
                venue                 = pr.venue,
                abstract              = "",  # PaperRef doesn't carry abstract
                fulltext_status       = FulltextStatus.METADATA_ONLY,
                pdf_source_kind       = "",
                pdf_source_url        = "",
                n_chunks              = 0,
                ingested_ts           = "",
                referenced_by_lessons    = tuple(lesson_ids),
                referenced_by_factors    = (),
                referenced_by_sleeves    = (),
                referenced_by_doctrines  = (),
                shelves               = (Shelf.RED_MOTIVATION,),
                shelf_notes           = {
                    Shelf.RED_MOTIVATION.value: (
                        f"Created Q-E 2026-06-03 from RED Lesson "
                        f"paper_motivation. Linked to "
                        f"{len(lessons)} lesson(s): "
                        f"{', '.join(candidate_names[:3])}"
                        f"{', ...' if len(candidate_names) > 3 else ''}"
                    )
                },
                created_ts            = "2026-06-03T15:00:00Z",
                updated_ts            = "2026-06-03T15:00:00Z",
                created_by            = "engine.crosslink_lessons_to_papers",
                tags                  = ("backfill_qe_2026-06-03",
                                         "source:red_lessons_p2"),
                note                  = (
                    f"Auto-created from RED Lesson P2 paper_motivation; "
                    f"motivated {len(lessons)} RED candidate(s)."
                ),
            )
            new_entries.append(entry)

    # Audit
    print()
    print("=" * 72)
    print("CROSSLINK Q-E AUDIT — lessons → papers reverse index")
    print("=" * 72)
    print(f"\nLessons total:                     {len(latest_lessons)}")
    print(f"  with paper_motivation (P2):      {len(with_paper)}")
    print(f"  without paper_motivation:        {len(without_paper)}")
    print(f"     B1+B2 recovered via registry: {len(recovered_b1b2)}")
    print(f"     B3 engineering inventions:    {len(bucket3_engineering)}")
    print(f"\nLesson-paper groups by DOI:        {len(doi_to_lessons)}")
    print(f"  Lessons with paper but no DOI:   {len(lessons_no_doi)}")

    print(f"\nRegistry updates:")
    print(f"  Amendments (paper already in registry):  {len(amendments)}")
    print(f"  New entries (paper not in registry yet): {len(new_entries)}")

    print(f"\nB3 engineering lessons (no academic anchor — tag will be added):")
    for L in bucket3_engineering:
        print(f"    {L.candidate_name:50s}  family={L.mechanism_family.value}")

    print("\nNew entries (sample 5):")
    for e in new_entries[:5]:
        print(f"  {e.title[:55]:57s} {e.authors[:2]} {e.year}")
        print(f"     shelves: {[s.value for s in e.shelves]}")
        print(f"     lessons: {list(e.referenced_by_lessons)[:3]}")

    print("\nAmendments (sample 5):")
    for e in amendments[:5]:
        print(f"  {e.title[:55]:57s} v{e.version}")
        print(f"     shelves now: {[s.value for s in e.shelves]}")
        print(f"     +lessons:    {len(e.referenced_by_lessons)} total")

    if args.write:
        n_w = 0
        for e in new_entries + amendments:
            try:
                save_entry(e, validate_strict=False)
                n_w += 1
            except Exception as exc:
                logger.error("save failed for %s: %s", e.title[:40], exc)
        print(f"\nWROTE {n_w} entries to registry.")

        # Tag B3 lessons with bucket-specific tags. Two cases:
        #   - mechanism_family == OTHER → truly engineering invention; tag
        #     "engineering_invention" + "qe_no_paper_anchor"
        #   - mechanism_family != OTHER → has an academic anchor in the
        #     literature but that anchor isn't seeded in registry yet; tag
        #     "qe_no_paper_anchor" + "missing_canonical_paper_in_registry"
        #     (P2.5 manual pass should add the family anchor)
        from engine.research_store.red_lessons import (
            MechanismFamily, REDLesson, save_lesson,
        )
        n_lessons_tagged = 0
        for L in bucket3_engineering:
            if "qe_no_paper_anchor" in L.tags:
                continue  # idempotent
            extra = ["qe_no_paper_anchor"]
            if L.mechanism_family == MechanismFamily.OTHER:
                extra.append("engineering_invention")
            else:
                extra.append("missing_canonical_paper_in_registry")
            new_tags = tuple(list(L.tags) + extra)
            v_next = REDLesson(
                **{**L.__dict__,
                   "lesson_id":        REDLesson.new_id(),
                   "version":          L.version + 1,
                   "parent_lesson_id": L.lesson_id,
                   "updated_ts":       "2026-06-03T15:00:00Z",
                   "created_by":       "engine.crosslink_lessons_to_papers",
                   "tags":             new_tags,
                })
            try:
                save_lesson(v_next, validate_strict=False)
                n_lessons_tagged += 1
            except Exception as exc:
                logger.error("save B3 lesson tag failed for %s: %s",
                             L.candidate_name, exc)
        print(f"WROTE {n_lessons_tagged} B3 lesson tag amendments to red_lessons.")
    else:
        print(f"\nDRY RUN — pass --write to persist")


if __name__ == "__main__":
    main()
