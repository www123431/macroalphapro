"""scripts/backfill_papers_from_library_yaml.py — Q-C: library YAML refs → registry.

For each mechanism YAML in data/research/mechanism_library/:
  1. Read canonical_paper_id + key_followup_ids
  2. Look each up in _canonical_papers_tier1_2.yaml master index
  3. Map sleeve's status_in_our_book → shelf
  4. For canonical: shelf = motivation_shelf (DEPLOYED → GREEN_MOTIVATION etc)
  5. For followups: shelf = critique_shelf (e.g. GREEN_CRITIQUE for deployed)
  6. Update registry (amend if exists, create if new)

The master index already has verified DOIs — high-quality source.

Run:
  python scripts/backfill_papers_from_library_yaml.py            # dry-run
  python scripts/backfill_papers_from_library_yaml.py --write    # persist
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.papers import (
    FulltextStatus, PaperRegistryEntry, REGISTRY_PATH,
    Shelf, amend_entry, find_by_doi, load_registry, save_entry,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("backfill_papers_from_library_yaml")


LIB_DIR             = _REPO_ROOT / "data" / "research" / "mechanism_library"
MASTER_INDEX_FILE   = LIB_DIR / "_canonical_papers_tier1_2.yaml"


STATUS_TO_MOTIVATION_SHELF = {
    "DEPLOYED":           Shelf.GREEN_MOTIVATION,
    "PENDING_DEPLOY":     Shelf.YELLOW_MOTIVATION,
    "UNTESTED":           Shelf.YELLOW_MOTIVATION,
    "RED":                Shelf.RED_MOTIVATION,
}

STATUS_TO_CRITIQUE_SHELF = {
    "DEPLOYED":           Shelf.GREEN_CRITIQUE,
    "PENDING_DEPLOY":     Shelf.YELLOW_MOTIVATION,  # not deployed yet → no critique-of-deployed
    "UNTESTED":           Shelf.YELLOW_MOTIVATION,
    "RED":                Shelf.RED_CRITIQUE,
}


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("yaml read failed for %s: %s", path, e)
        return None


def _parse_authors(author_str: str) -> tuple[str, ...]:
    """Master-index authors are like 'Bernard, Thomas' or 'Hou, Xue, Zhang'.
    Comma-separated surnames. Split on comma + whitespace."""
    if not author_str:
        return ()
    parts = [p.strip() for p in str(author_str).split(",")]
    return tuple(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="persist amendments + new entries")
    args = ap.parse_args()

    # Load master paper index
    master = _read_yaml(MASTER_INDEX_FILE)
    if not master or "papers" not in master:
        print(f"ERROR: master index empty / missing 'papers' key: {MASTER_INDEX_FILE}")
        sys.exit(1)
    papers_index: dict[str, dict] = master["papers"]
    logger.info("loaded %d papers in master index", len(papers_index))

    # Load mechanism YAMLs (skip _* prefix files and _drafts dir)
    yaml_files = [
        Path(f) for f in glob.glob(str(LIB_DIR / "*.yaml"))
        if "_" not in Path(f).name[0]
    ]
    logger.info("loaded %d mechanism yaml files", len(yaml_files))

    # Build paper_id → {shelf → [sleeve_ids]} + paper_id metadata
    paper_to_shelves: dict[str, dict[Shelf, list[str]]] = defaultdict(lambda: defaultdict(list))
    broken_refs: list[tuple[str, str]] = []   # (sleeve_id, paper_id)

    for yf in yaml_files:
        d = _read_yaml(yf)
        if not d:
            continue
        sleeve_id = d.get("id") or yf.stem
        status = (d.get("status_in_our_book") or "").upper()
        motivation_shelf = STATUS_TO_MOTIVATION_SHELF.get(status)
        critique_shelf   = STATUS_TO_CRITIQUE_SHELF.get(status)
        if motivation_shelf is None:
            logger.warning("sleeve %s has unrecognized status %r; skipping",
                           sleeve_id, status)
            continue

        canonical = d.get("canonical_paper_id")
        if canonical:
            if canonical in papers_index:
                paper_to_shelves[canonical][motivation_shelf].append(sleeve_id)
            else:
                broken_refs.append((sleeve_id, canonical))

        for fu in (d.get("key_followup_ids") or []):
            if fu in papers_index:
                paper_to_shelves[fu][critique_shelf].append(sleeve_id)
            else:
                broken_refs.append((sleeve_id, fu))

    # Load registry
    registry = load_registry()
    by_doi = {e.doi.lower(): e for e in registry if e.doi}
    logger.info("registry has %d entries", len(registry))

    # Apply
    new_entries: list[PaperRegistryEntry] = []
    amendments:  list[PaperRegistryEntry] = []

    for paper_id, by_shelf in paper_to_shelves.items():
        meta = papers_index[paper_id]
        doi = (meta.get("doi") or "").strip()
        existing = by_doi.get(doi.lower()) if doi else None

        all_sleeves = [s for sleeves in by_shelf.values() for s in sleeves]
        shelf_notes_add = {
            shelf.value: (
                f"Backfilled Q-C 2026-06-03 from library yaml. "
                f"sleeves: {', '.join(sleeves[:3])}"
                f"{', ...' if len(sleeves) > 3 else ''}"
            )
            for shelf, sleeves in by_shelf.items()
        }

        if existing is not None:
            # Amend with shelves + sleeve refs
            amended = amend_entry(
                existing,
                add_shelves       = tuple(by_shelf.keys()),
                add_shelf_notes   = shelf_notes_add,
                add_sleeves       = all_sleeves,
                add_tags          = ("backfill_qc_2026-06-03",),
                updated_ts        = "2026-06-03T14:00:00Z",
                created_by        = "engine.backfill_papers_from_library_yaml",
                note_append       = f"Q-C: linked to {len(all_sleeves)} sleeve(s).",
            )
            amendments.append(amended)
        else:
            # Create new entry from master index metadata
            authors = _parse_authors(meta.get("author", ""))
            entry = PaperRegistryEntry(
                paper_id              = PaperRegistryEntry.new_id(),
                version               = 1,
                parent_paper_id       = None,
                doi                   = doi,
                title                 = meta.get("title", ""),
                year                  = int(meta.get("year", 0)) if meta.get("year") else 0,
                authors               = authors,
                venue                 = meta.get("journal", ""),
                abstract              = meta.get("notes", ""),
                fulltext_status       = FulltextStatus.METADATA_ONLY,
                pdf_source_kind       = "",
                pdf_source_url        = "",
                n_chunks              = 0,
                ingested_ts           = "",
                referenced_by_lessons    = (),
                referenced_by_factors    = (),
                referenced_by_sleeves    = tuple(all_sleeves),
                referenced_by_doctrines  = (),
                shelves               = tuple(by_shelf.keys()),
                shelf_notes           = shelf_notes_add,
                created_ts            = "2026-06-03T14:00:00Z",
                updated_ts            = "2026-06-03T14:00:00Z",
                created_by            = "engine.backfill_papers_from_library_yaml",
                tags                  = ("backfill_qc_2026-06-03",
                                         f"library_paper_id:{paper_id}",
                                         f"tier:{meta.get('tier', '')}"),
                note                  = (
                    f"Created Q-C 2026-06-03 from master index entry "
                    f"'{paper_id}'. Tier {meta.get('tier', '?')}."
                ),
            )
            new_entries.append(entry)

    # Audit
    print()
    print("=" * 76)
    print("LIBRARY YAML Q-C AUDIT")
    print("=" * 76)
    print(f"\nMaster index papers:      {len(papers_index)}")
    print(f"Mechanism yamls:          {len(yaml_files)}")
    print(f"Distinct papers linked:   {len(paper_to_shelves)}")
    print(f"  Already in registry → amendments: {len(amendments)}")
    print(f"  Not in registry → new entries:    {len(new_entries)}")
    print(f"\nBroken paper_id refs (in YAML but missing from master index): {len(broken_refs)}")
    for sleeve_id, paper_id in broken_refs[:8]:
        print(f"    sleeve={sleeve_id:35s}  missing_paper_id={paper_id}")

    if amendments or new_entries:
        print(f"\nNew entries (sample 5):")
        for e in new_entries[:5]:
            print(f"  {e.title[:55]:57s} {e.authors[:2]} {e.year}")
            print(f"     shelves: {[s.value for s in e.shelves]}")
            print(f"     sleeves: {list(e.referenced_by_sleeves)}")
        print(f"\nAmendments (sample 5):")
        for e in amendments[:5]:
            print(f"  {e.title[:55]:57s} v{e.version}")
            print(f"     shelves now: {[s.value for s in e.shelves]}")
            print(f"     +sleeves:    {list(e.referenced_by_sleeves)}")

    if args.write:
        n_w = 0
        for e in new_entries + amendments:
            try:
                save_entry(e, validate_strict=False)
                n_w += 1
            except Exception as exc:
                logger.error("save failed for %s: %s", e.title[:40], exc)
        print(f"\nWROTE {n_w} entries (new + amendments) to registry.")
    else:
        print(f"\nDRY RUN — pass --write to persist")


if __name__ == "__main__":
    main()
