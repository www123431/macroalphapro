"""scripts/backfill_papers_from_factory_ledger.py — Q-B: GREEN/YELLOW backfill.

For each factor in data/validation/factory_ledger.jsonl:
  1. Group by candidate_name (factory_ledger has duplicates from iterating)
  2. Determine the most-recent verdict per name (GREEN > YELLOW > RED priority
     when status flipped; we use the highest-confidence light)
  3. Classify mechanism family → look up matching doctrine seed paper
  4. If paper already in registry: amend (add shelf + referenced_by_factor link)
  5. If paper NOT in registry (no doctrine match) AND candidate is GREEN/YELLOW:
     log as "engineering invention without academic anchor"; do NOT create a
     paperless registry entry — Q-E cross-link will track them via lesson refs.

Run:
  python scripts/backfill_papers_from_factory_ledger.py            # dry-run
  python scripts/backfill_papers_from_factory_ledger.py --write    # persist
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.papers import (
    Shelf, amend_entry, find_by_doi, load_registry, save_entry,
)
from engine.research_store.papers.schema import PaperRegistryEntry
from engine.research_store.red_lessons.backfill_heuristics import (
    classify_mechanism,
)
from engine.research_store.red_lessons.mechanism_families import (
    MechanismFamily, MECHANISM_FAMILY_DOCS,
)
from engine.research_store.red_lessons.openalex_client import (
    parse_anchor_string,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("backfill_papers_from_factory_ledger")


FACTORY_LEDGER_PATH = _REPO_ROOT / "data" / "validation" / "factory_ledger.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                logger.warning("skipping malformed line: %s", e)
    return out


def _latest_per_name(records: list[dict]) -> dict[str, dict]:
    """Group records by name; keep the one with the strongest verdict.

    Strength priority: GREEN > YELLOW > RED > NULL. Within same verdict,
    take the LATEST ts.
    """
    by_name: dict[str, dict] = {}
    priority = {"GREEN": 3, "YELLOW": 2, "RED": 1}

    for r in records:
        name = r.get("name", "")
        if not name:
            continue
        light = (r.get("light") or "").upper()
        prior = by_name.get(name)
        if prior is None:
            by_name[name] = r
            continue
        prior_light = (prior.get("light") or "").upper()
        if priority.get(light, 0) > priority.get(prior_light, 0):
            by_name[name] = r
        elif priority.get(light, 0) == priority.get(prior_light, 0):
            # Same verdict — take latest ts
            if (r.get("ts") or "") > (prior.get("ts") or ""):
                by_name[name] = r
    return by_name


def _shelf_for_light(light: str) -> Shelf | None:
    """Map verdict light to motivation shelf. RED is handled separately
    via RED Lessons; here we capture GREEN + YELLOW."""
    light = (light or "").upper()
    if light == "GREEN":
        return Shelf.GREEN_MOTIVATION
    if light == "YELLOW":
        return Shelf.YELLOW_MOTIVATION
    return None


def _find_doctrine_paper_for_family(family: MechanismFamily,
                                    registry: list[PaperRegistryEntry]
                                    ) -> PaperRegistryEntry | None:
    """Find a paper in registry whose authors / title hint matches the
    family's anchor_paper string.

    Heuristic: parse the family's anchor string; find a registry entry
    where the registry-entry-authors share a surname with the parsed
    anchor authors.
    """
    family_anchor = MECHANISM_FAMILY_DOCS.get(family, {}).get("anchor_paper", "")
    if not family_anchor or "must be filled" in family_anchor:
        return None
    parsed = parse_anchor_string(family_anchor)
    if parsed is None or not parsed.authors:
        return None
    expected_authors_lower = [a.lower() for a in parsed.authors]
    candidates: list[tuple[int, PaperRegistryEntry]] = []
    for entry in registry:
        entry_authors_lower = [a.lower() for a in entry.authors]
        overlap = sum(1 for a in expected_authors_lower
                      if a in entry_authors_lower)
        if overlap >= 1:
            # Also bias by title similarity if available
            score = overlap * 10
            if parsed.title and parsed.title.lower() in (entry.title or "").lower():
                score += 5
            candidates.append((score, entry))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="persist amendments + new entries")
    args = ap.parse_args()

    records = _read_jsonl(FACTORY_LEDGER_PATH)
    latest = _latest_per_name(records)
    logger.info("loaded %d factory_ledger records → %d unique names",
                len(records), len(latest))

    registry = load_registry()
    logger.info("registry has %d entries", len(registry))

    # Group candidates by (family, mapped_paper)
    # paper_id → {shelf → [(candidate_name, light)]}
    paper_to_factors: dict[str, dict[Shelf, list[str]]] = defaultdict(lambda: defaultdict(list))
    paper_to_entry: dict[str, PaperRegistryEntry] = {}
    unmatched: list[tuple[str, str, MechanismFamily]] = []  # (name, light, family)

    for name, record in latest.items():
        light = (record.get("light") or "").upper()
        shelf = _shelf_for_light(light)
        if shelf is None:
            continue   # RED handled by RED Lessons separately

        family, subtype = classify_mechanism(name, None)
        entry = _find_doctrine_paper_for_family(family, registry)
        if entry is None:
            unmatched.append((name, light, family))
            continue

        paper_to_factors[entry.paper_id][shelf].append(name)
        paper_to_entry[entry.paper_id] = entry

    # Audit
    print()
    print("=" * 72)
    print("FACTORY LEDGER Q-B AUDIT")
    print("=" * 72)
    print(f"\nTotal unique factory_ledger candidates: {len(latest)}")
    print(f"  GREEN: {sum(1 for r in latest.values() if (r.get('light') or '').upper() == 'GREEN')}")
    print(f"  YELLOW: {sum(1 for r in latest.values() if (r.get('light') or '').upper() == 'YELLOW')}")
    print(f"  RED:   {sum(1 for r in latest.values() if (r.get('light') or '').upper() == 'RED')}")

    print(f"\nMatched to doctrine paper: {len(paper_to_factors)} unique papers")
    for paper_id, by_shelf in paper_to_factors.items():
        entry = paper_to_entry[paper_id]
        print(f"\n  {entry.title[:55]:57s} ({', '.join(entry.authors[:3])} {entry.year})")
        for shelf, names in by_shelf.items():
            print(f"    + shelf {shelf.value}: {len(names)} factor(s)")
            for n in names[:6]:
                print(f"        {n}")

    print(f"\nUnmatched (no doctrine paper for mechanism family):  {len(unmatched)}")
    by_family = Counter(u[2].value for u in unmatched)
    for fam, n in by_family.most_common():
        print(f"  {fam:30s}  {n}")
    print("\nUnmatched candidates (Q-E cross-link will track these via lessons):")
    for name, light, family in unmatched[:20]:
        print(f"    {light:6s}  {name:50s}  family={family.value}")

    # Write amendments
    if args.write:
        n_amended = 0
        for paper_id, by_shelf in paper_to_factors.items():
            entry = paper_to_entry[paper_id]
            new_shelves = list(by_shelf.keys())
            all_factor_names: list[str] = []
            for names in by_shelf.values():
                all_factor_names.extend(names)
            shelf_notes_add: dict[str, str] = {}
            for shelf, names in by_shelf.items():
                shelf_notes_add[shelf.value] = (
                    f"Backfilled Q-B 2026-06-03: motivation for "
                    f"{len(names)} {shelf.value.split('_')[0]} factor(s) "
                    f"in factory_ledger ({', '.join(names[:3])}"
                    f"{', ...' if len(names) > 3 else ''})."
                )
            amended = amend_entry(
                entry,
                add_shelves       = new_shelves,
                add_shelf_notes   = shelf_notes_add,
                add_factors       = all_factor_names,
                add_tags          = ("backfill_qb_2026-06-03",),
                updated_ts        = "2026-06-03T13:00:00Z",
                created_by        = "engine.backfill_papers_from_factory_ledger",
                note_append       = (
                    "Q-B amendment: linked to "
                    f"{len(all_factor_names)} factor(s)."
                ),
            )
            try:
                save_entry(amended, validate_strict=False)
                n_amended += 1
            except Exception as e:
                logger.error("save amended entry failed for %s: %s",
                             entry.title[:40], e)
        print(f"\nWROTE {n_amended} amendments (v+1 entries) to registry.")
    else:
        print(f"\nDRY RUN — pass --write to persist amendments")


if __name__ == "__main__":
    main()
