"""scripts/seed_canonical_papers.py — Stage C T1+T2 expansion.

Imports `data/research_store/canonical_paper_seeds.yaml` into
papers_registry. For each seed entry:
  1. Skip if DOI already in registry (idempotent across re-runs)
  2. Else fetch abstract + venue from CrossRef (~0.5s/paper)
  3. Create PaperRegistryEntry with intended_tier set + status=
     METADATA_ONLY (no PDF needed for T2 anchor function per the
     three-libraries doctrine)

Downstream (separate scripts):
  - classify_papers_into_tiers.py confirms tier via Sonnet
  - enrich_t2_anchors.py adds tier_anchor_summary to T2s

Usage:
  python scripts/seed_canonical_papers.py            # dry-run
  python scripts/seed_canonical_papers.py --write    # persist
  python scripts/seed_canonical_papers.py --json     # machine output
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from engine.research_store.papers.store import (
    load_registry, save_entry,
)
from engine.research_store.papers.schema import (
    PaperRegistryEntry, FulltextStatus, PaperTier,
)
from engine.research_store.papers.shelves import Shelf
from engine.research_store.papers.anchor_enricher import (
    fetch_crossref_metadata,
)


SEEDS_PATH = (_REPO_ROOT / "data" / "research_store"
                / "canonical_paper_seeds.yaml")


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_seeds() -> list[dict]:
    with SEEDS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("seeds", [])


def _existing_dois() -> set[str]:
    """Lowercase set of DOIs already in registry (latest version per
    paper_id, all DOIs collapsed)."""
    raw = load_registry()
    out: set[str] = set()
    for p in raw:
        d = (p.doi or "").strip().lower()
        if d:
            out.add(d)
    return out


def _seed_to_entry(seed: dict, *, abstract: str, venue: str) -> PaperRegistryEntry:
    now = _utc_iso()
    tier_str = str(seed.get("intended_tier") or "UNCLASSIFIED")
    try:
        tier = PaperTier(tier_str)
    except ValueError:
        tier = PaperTier.UNCLASSIFIED

    shelf_strs = list(seed.get("shelves") or ["doctrine_method"])
    shelves: list[Shelf] = []
    for s in shelf_strs:
        try:
            shelves.append(Shelf(s))
        except ValueError:
            shelves.append(Shelf.OTHER)
    if not shelves:
        shelves = [Shelf.DOCTRINE_METHOD]

    note = str(seed.get("note") or "")

    return PaperRegistryEntry(
        paper_id              = PaperRegistryEntry.new_id(),
        version               = 1,
        parent_paper_id       = None,
        doi                   = str(seed["doi"]).strip().lower(),
        title                 = str(seed["title"]).strip(),
        year                  = int(seed["year"]),
        authors               = tuple(seed.get("authors") or ()),
        venue                 = venue or str(seed.get("venue") or ""),
        abstract              = abstract,
        fulltext_status       = FulltextStatus.METADATA_ONLY,
        pdf_source_kind       = "",
        pdf_source_url        = "",
        n_chunks              = 0,
        ingested_ts           = "",
        referenced_by_lessons    = (),
        referenced_by_factors    = (),
        referenced_by_sleeves    = (),
        referenced_by_doctrines  = (),
        shelves                  = tuple(shelves),
        shelf_notes              = ({Shelf.OTHER.value:
                                       "canonical_paper_seed"}
                                       if Shelf.OTHER in shelves else {}),
        created_ts            = now,
        updated_ts            = now,
        created_by            = "scripts.seed_canonical_papers",
        tags                  = ("canonical_seed", "stage_c_t1_t2_expansion"),
        note                  = note,
        # Pre-set tier from intended_tier — classifier will confirm
        # or override on its next pass
        tier                  = tier,
        tier_classified_ts    = now,
        tier_rationale        = ("canonical seed list — pre-classified "
                                  "from intended_tier in YAML"),
        tier_anchor_summary   = "",   # populated by enrich_t2_anchors
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true",
                    help="Persist to registry (default: dry-run)")
    p.add_argument("--json", action="store_true",
                    help="JSON output")
    args = p.parse_args()

    seeds = _load_seeds()
    existing = _existing_dois()

    plan = {
        "n_seeds":     len(seeds),
        "n_skipped":   0,
        "n_to_import": 0,
        "n_imported":  0,
        "items": [],
        "errors": [],
    }

    for seed in seeds:
        doi = str(seed.get("doi", "")).strip().lower()
        title = (seed.get("title") or "")[:60]
        intended = str(seed.get("intended_tier") or "UNCLASSIFIED")

        if not doi:
            plan["errors"].append(f"missing doi: {title}")
            continue
        if doi in existing:
            plan["items"].append({
                "doi": doi, "title": title,
                "intended_tier": intended,
                "action": "skip_already_in_registry",
            })
            plan["n_skipped"] += 1
            continue

        # CrossRef metadata fetch — gentle throttle handled inside
        cr = fetch_crossref_metadata(doi)
        abstract = cr.abstract if cr.found else ""
        venue    = cr.venue    if cr.found else ""
        cr_err   = cr.error    if cr.error else ""

        item = {
            "doi":          doi,
            "title":        title,
            "intended_tier": intended,
            "crossref_found": cr.found,
            "abstract_chars": len(abstract),
            "venue":        venue,
            "crossref_error": cr_err,
        }

        if not args.write:
            item["action"] = "would_import"
            plan["items"].append(item)
            plan["n_to_import"] += 1
            continue

        # Build entry + persist
        try:
            entry = _seed_to_entry(seed,
                                     abstract=abstract,
                                     venue=venue)
            save_entry(entry)
            item["action"] = "imported"
            item["paper_id"] = entry.paper_id[:8]
            plan["n_imported"] += 1
        except Exception as exc:
            item["action"] = f"error: {exc}"
            plan["errors"].append(f"{doi}: {exc}")
        plan["items"].append(item)

    if args.json:
        sys.stdout.write(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    print(f"Canonical paper seed import:")
    print(f"  total seeds:           {plan['n_seeds']}")
    print(f"  skipped (in registry): {plan['n_skipped']}")
    if args.write:
        print(f"  imported:              {plan['n_imported']}")
    else:
        print(f"  would import:          {plan['n_to_import']}")
    if plan["errors"]:
        print(f"  errors: {len(plan['errors'])}")
        for e in plan["errors"]:
            print(f"    {e}")
    print()
    # Per-item summary
    for it in plan["items"]:
        flag = ("CR-no-abstract" if it.get("crossref_found")
                 and it.get("abstract_chars", 0) == 0
                 else ("CR-found" if it.get("crossref_found")
                        else f"CR-err:{it.get('crossref_error','')}"
                        if it.get("crossref_error") else "skip"))
        action = it.get("action", "?")
        print(f"  [{it['intended_tier']:14}] [{flag:18}] [{action}] {it['title']}")
    if not args.write:
        print("\nDRY RUN — re-run with --write to persist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
