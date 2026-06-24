"""scripts/seed_library_papers.py — P2.5 library PDF acquisition.

Walks all registry entries with fulltext_status=METADATA_ONLY and, for
each whose DOI appears in LIBRARY_PDF_OVERRIDES, attempts the same
acquire→title-validate→chunk→ingest pipeline used by seed_doctrine_papers.

On success: writes an AMENDMENT to the registry entry (v+1) with
fulltext_status=INGESTED + pdf_source_kind=manual_override + n_chunks
updated + ingested_ts stamped.

On failure: leaves the registry entry unchanged.

Run:
  python scripts/seed_library_papers.py            # dry-run
  python scripts/seed_library_papers.py --write    # persist + ingest
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
    FulltextStatus, REGISTRY_PATH, amend_entry, load_registry,
    latest_per_doi, save_entry,
)
from engine.research_store.papers.library_pdf_overrides import (
    LIBRARY_PDF_OVERRIDES, PdfOverride,
)
from engine.research_store.red_lessons.paper_acquisition import (
    USER_AGENT, _download, _hostname_ok, extract_pdf_text,
)
from engine.research_store.red_lessons.papers_chroma import (
    chunk_paper, collection_stats, ingest_chunks,
)

# We re-use the title validator from the doctrine seed driver
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from seed_doctrine_papers import _pdf_title_matches   # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("seed_library_papers")


def _try_url(url: str, bypass_whitelist: bool = False) -> tuple[bytes, str] | None:
    if not url:
        return None
    if not bypass_whitelist and not _hostname_ok(url):
        logger.warning("URL fails hostname check: %s", url[:100])
        return None
    data = _download(url)
    if data:
        return data, url
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="persist amendments + ingest chunks")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N entries (testing)")
    args = ap.parse_args()

    registry = load_registry()
    latest = list(latest_per_doi(registry).values())
    metadata_only = [e for e in latest
                     if e.fulltext_status == FulltextStatus.METADATA_ONLY]
    logger.info("registry latest=%d  metadata_only=%d",
                len(latest), len(metadata_only))

    targets = []
    for e in metadata_only:
        if not e.doi:
            continue
        ovr = LIBRARY_PDF_OVERRIDES.get(e.doi.lower())
        if ovr is not None:
            targets.append((e, ovr))
    if args.limit:
        targets = targets[: args.limit]
    logger.info("override-match candidates: %d", len(targets))

    successes = []   # list of amended PaperRegistryEntry
    failures  = []   # list of (entry, reason)

    for entry, ovr in targets:
        logger.info("[%s %d] %s", entry.authors[:1][0] if entry.authors else "?",
                    entry.year, entry.title[:60])
        # Try the manual URL
        result = _try_url(ovr.manual_pdf_url, bypass_whitelist=ovr.bypass_whitelist)
        if result is None:
            failures.append((entry, "download_failed"))
            continue
        pdf_bytes, pdf_url = result
        if not _pdf_title_matches(pdf_bytes, entry.title):
            failures.append((entry, "title_mismatch_with_registry_entry"))
            continue
        full_text = extract_pdf_text(pdf_bytes)
        if not full_text or len(full_text) < 2000:
            failures.append((entry, "extract_failed_or_too_short"))
            continue
        chunks = chunk_paper(
            full_text,
            doi             = entry.doi,
            title           = entry.title,
            year            = entry.year,
            authors         = entry.authors,
            venue           = entry.venue,
            source_kind     = "manual_override",
            candidate_names = (),
        )
        if not chunks:
            failures.append((entry, "no_chunks_after_filter"))
            continue
        if args.write:
            ingest_chunks(chunks)

        # Amend the registry entry — bump version + flip status to INGESTED
        # Note: amend_entry only supports the shelves/refs additions; for
        # acquisition fields we need to do it directly.
        from engine.research_store.papers import PaperRegistryEntry
        ingested_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        amended = PaperRegistryEntry(
            **{**entry.__dict__,
               "paper_id":         PaperRegistryEntry.new_id(),
               "version":          entry.version + 1,
               "parent_paper_id":  entry.paper_id,
               "fulltext_status":  FulltextStatus.INGESTED,
               "pdf_source_kind":  "manual_override",
               "pdf_source_url":   pdf_url,
               "n_chunks":         len(chunks),
               "ingested_ts":      ingested_ts,
               "updated_ts":       ingested_ts,
               "created_by":       "engine.seed_library_papers",
               "tags":             entry.tags + ("library_pdf_p25_2026-06-04",),
               "note":             (entry.note or "") +
                                   f" | P2.5: ingested via library_pdf_overrides "
                                   f"({pdf_url[:80]})",
            }
        )
        successes.append(amended)

    print()
    print("=" * 72)
    print("LIBRARY PDF SEED AUDIT")
    print("=" * 72)
    print(f"\nMetadata-only entries:    {len(metadata_only)}")
    print(f"Override-match candidates: {len(targets)}")
    print(f"Successful ingests:        {len(successes)}")
    print(f"Failures:                  {len(failures)}")
    print()
    print("Successes:")
    for e in successes:
        print(f"  {(e.authors[0] if e.authors else '?'):12s} {e.year}  "
              f"{e.title[:50]:52s}  chunks={e.n_chunks}")
    print()
    print("Failures:")
    for e, reason in failures:
        print(f"  {(e.authors[0] if e.authors else '?'):12s} {e.year}  "
              f"{e.title[:45]:47s}  reason={reason}")

    if args.write:
        for amended in successes:
            try:
                save_entry(amended, validate_strict=False)
            except Exception as exc:
                logger.error("save failed for %s: %s", amended.title[:40], exc)
        print(f"\nWROTE {len(successes)} amendments to {REGISTRY_PATH}")
        print(f"papers_chroma: {collection_stats()}")
    else:
        print(f"\nDRY RUN — pass --write to persist")


if __name__ == "__main__":
    main()
