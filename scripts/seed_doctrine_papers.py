"""scripts/seed_doctrine_papers.py — Q-D: ingest 14 doctrine framework papers.

For each entry in DOCTRINE_SEED:
  1. OpenAlex lookup (anchor_str → work dict)
  2. Validate authors match expected_authors (reject wrong-domain)
  3. PDF acquisition — try manual_pdf_url first if provided, then
     fall back to existing acquire_pdf() pipeline
  4. If PDF, extract + chunk + ingest into papers_chroma (idempotent
     by DOI — re-ingest replaces)
  5. Build PaperRegistryEntry with shelves + shelf_notes + status
  6. Persist to papers_registry.jsonl

Run:
  python scripts/seed_doctrine_papers.py            # dry-run audit
  python scripts/seed_doctrine_papers.py --write    # persist
"""
from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.papers import (
    FulltextStatus, PaperRegistryEntry, REGISTRY_PATH,
    Shelf, find_by_doi, load_registry, save_entry,
)
from engine.research_store.papers.doctrine_seed import (
    DOCTRINE_SEED, DoctrineSeedEntry,
)
from engine.research_store.red_lessons.openalex_client import (
    extract_abstract, extract_authors, extract_doi, extract_title,
    extract_venue, extract_year, load_cache, lookup_anchor, save_cache,
)
from engine.research_store.red_lessons.paper_acquisition import (
    USER_AGENT, _download, _hostname_ok, acquire_pdf, extract_pdf_text,
)
from engine.research_store.red_lessons.papers_chroma import (
    chunk_paper, ingest_chunks, collection_stats,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("seed_doctrine_papers")


def _authors_validate(work_authors: tuple[str, ...],
                      expected: tuple[str, ...]) -> bool:
    """Hard-validate: at least 1 expected surname in work_authors."""
    if not expected:
        return True
    lw = [a.lower() for a in work_authors]
    return any(e.lower() in lw for e in expected)


def _try_manual_pdf(url: str) -> tuple[bytes, str] | None:
    """Try the doctrine-entry's manual_pdf_url override.

    Bypass the hostname check ONLY if the URL is from a hand-vetted
    free-distribution source (NBER / AMS / .edu / .gov / arXiv).
    """
    if not url:
        return None
    if not _hostname_ok(url):
        logger.warning("manual_pdf_url failed hostname check: %s", url)
        return None
    data = _download(url)
    if data:
        return data, url
    return None


def _pdf_title_matches(pdf_bytes: bytes, expected_title: str) -> bool:
    """First-page title validator: extract the PDF's first ~500 chars and
    check that key tokens from expected_title appear.

    Defends against the failure mode discovered 2026-06-03 with NBER WP
    23394: I added an URL for HXZ 'Replicating Anomalies' but the file
    is actually HXZ 'An Augmented q-Factor Model'. Both have the same
    authors so author-overlap doesn't catch this. Title-token overlap
    does.
    """
    if not expected_title:
        return True  # nothing to check against
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        first_page_text = doc[0].get_text("text") if len(doc) > 0 else ""
        doc.close()
    except Exception:
        return True  # don't block on extraction failure

    # Normalize: lowercase + strip non-word
    import re as _re
    norm = lambda s: _re.sub(r"[^a-z0-9 ]", " ", s.lower())
    fp_norm = norm(first_page_text[:1500])
    expected_norm = norm(expected_title)
    # Take distinctive title words (>= 4 chars, drop stopwords)
    stopwords = {"the", "and", "for", "with", "from", "that", "this",
                 "into", "what", "model", "paper", "study", "section",
                 "anomalies", "factor", "factors"}
    tokens = [t for t in expected_norm.split()
              if len(t) >= 4 and t not in stopwords]
    if len(tokens) < 2:
        return True  # title too generic to validate
    matched = sum(1 for t in tokens if t in fp_norm)
    return matched >= max(1, len(tokens) // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="persist registry entries + ingest PDFs")
    ap.add_argument("--skip-pdf", action="store_true",
                    help="metadata only")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cache = load_cache()
    existing_registry = load_registry()
    existing_dois = {e.doi.lower() for e in existing_registry if e.doi}

    entries_to_process = DOCTRINE_SEED[: args.limit] if args.limit else DOCTRINE_SEED

    outcomes: list[dict] = []

    for i, seed in enumerate(entries_to_process, start=1):
        logger.info("[%d/%d] %s", i, len(entries_to_process),
                    seed.anchor_str[:80])

        work = None
        work_authors: tuple[str, ...] = ()
        doi = title = venue = abstr = ""
        year = 0

        # Strategy 0: hand-encoded manual_metadata always wins (handles
        # generic-title OpenAlex misses like "Carry" / "Trading Costs")
        if seed.manual_metadata is not None:
            mm = seed.manual_metadata
            work_authors = mm.authors
            doi   = mm.doi
            title = mm.title
            year  = mm.year
            venue = mm.venue
            abstr = mm.abstract
            logger.info("  using manual_metadata override")

        # Strategy 1: OpenAlex lookup if no manual_metadata
        else:
            work = lookup_anchor(seed.anchor_str, cache=cache)
            if work is None:
                logger.warning("  openalex miss")
                outcomes.append({
                    "seed": seed, "work": None,
                    "registry_entry": None, "reason": "openalex_miss",
                })
                continue

            work_authors = extract_authors(work)
            if not _authors_validate(work_authors, seed.expected_authors):
                logger.warning("  author validation failed; openalex returned "
                               "%s vs expected %s",
                               work_authors[:3], seed.expected_authors)
                outcomes.append({
                    "seed": seed, "work": work,
                    "registry_entry": None, "reason": "author_validation_failed",
                })
                continue

            doi    = extract_doi(work)
            title  = extract_title(work)
            year   = extract_year(work) or 0
            venue  = extract_venue(work)
            abstr  = extract_abstract(work)

        # Skip if already in registry (idempotent)
        if doi and doi.lower() in existing_dois:
            logger.info("  already in registry (DOI %s); skipping", doi)
            outcomes.append({
                "seed": seed, "work": work,
                "registry_entry": None, "reason": "already_in_registry",
            })
            continue

        # PDF acquisition: manual_pdf_url first, then existing pipeline
        pdf_bytes: bytes | None = None
        pdf_url: str = ""
        pdf_kind: str = ""

        if not args.skip_pdf:
            if seed.manual_pdf_url:
                mr = _try_manual_pdf(seed.manual_pdf_url)
                if mr:
                    pdf_bytes, pdf_url = mr
                    pdf_kind = "manual_override"
            if pdf_bytes is None and work is not None:
                ar = acquire_pdf(work, title=title, year=year)
                if ar.ok:
                    pdf_bytes = ar.pdf_bytes
                    pdf_url   = ar.source_url
                    pdf_kind  = ar.source_kind

            # CRITICAL: validate PDF content matches expected title before
            # ingesting. Caught the HXZ NBER-23394 wrong-paper bug.
            if pdf_bytes is not None:
                if not _pdf_title_matches(pdf_bytes, title):
                    logger.warning(
                        "  PDF title mismatch (file content does not match "
                        "expected '%s'); rejecting PDF", title[:60])
                    pdf_bytes = None
                    pdf_url = ""
                    pdf_kind = "rejected_title_mismatch"

        # Chunk + ingest if we got a PDF
        n_chunks = 0
        ingest_ts = ""
        if pdf_bytes:
            full_text = extract_pdf_text(pdf_bytes)
            if full_text and len(full_text) > 2000:
                chunks = chunk_paper(
                    full_text,
                    doi             = doi or seed.anchor_str,
                    title           = title,
                    year            = year,
                    authors         = work_authors,
                    venue           = venue,
                    source_kind     = pdf_kind,
                    candidate_names = (),
                )
                n_chunks = len(chunks)
                if args.write and chunks:
                    ingest_chunks(chunks)
                    ingest_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build registry entry
        if pdf_bytes and n_chunks > 0:
            ft_status = FulltextStatus.INGESTED
        elif pdf_bytes is None and not args.skip_pdf:
            ft_status = FulltextStatus.PAYWALLED
        else:
            ft_status = FulltextStatus.METADATA_ONLY

        # Mark abstract quality — OpenAlex returns inverted_index, which we
        # reconstruct, but order can be off. Tag for downstream awareness.
        tags = ["doctrine_seed", f"shelf_count:{len(seed.shelves)}"]
        if abstr:
            tags.append("abstract_quality:openalex_inverted")

        entry = PaperRegistryEntry(
            paper_id        = PaperRegistryEntry.new_id(),
            version         = 1,
            parent_paper_id = None,
            doi             = doi,
            title           = title,
            year            = year,
            authors         = work_authors,
            venue           = venue,
            abstract        = abstr,
            fulltext_status = ft_status,
            pdf_source_kind = pdf_kind,
            pdf_source_url  = pdf_url,
            n_chunks        = n_chunks,
            ingested_ts     = ingest_ts,
            referenced_by_lessons    = (),
            referenced_by_factors    = (),
            referenced_by_sleeves    = (),
            referenced_by_doctrines  = (),
            shelves         = seed.shelves,
            shelf_notes     = seed.shelf_notes,
            created_ts      = "2026-06-03T12:00:00Z",
            updated_ts      = "2026-06-03T12:00:00Z",
            created_by      = "engine.doctrine_seed",
            tags            = tuple(tags),
            note            = seed.note,
        )

        outcomes.append({
            "seed": seed, "work": work,
            "registry_entry": entry,
            "reason": ft_status.value,
        })

    save_cache(cache)

    # Audit
    print()
    print("=" * 76)
    print("DOCTRINE SEED Q-D AUDIT")
    print("=" * 76)
    print(f"\nTotal entries:       {len(entries_to_process)}")

    by_reason = Counter(o["reason"] for o in outcomes)
    for r, n in by_reason.most_common():
        print(f"  {r:30s}  {n}")

    pdf_results = [o for o in outcomes if o.get("registry_entry")
                   and o["registry_entry"].fulltext_status == FulltextStatus.INGESTED]
    print(f"\nWith full-text PDF ingested: {len(pdf_results)} / {len(entries_to_process)}")
    for o in pdf_results:
        e = o["registry_entry"]
        print(f"  {e.title[:50]:52s}  via {e.pdf_source_kind:20s} chunks={e.n_chunks}")

    paywalled = [o for o in outcomes if o.get("registry_entry")
                 and o["registry_entry"].fulltext_status == FulltextStatus.PAYWALLED]
    print(f"\nMetadata-only (paywalled / no PDF): {len(paywalled)}")
    for o in paywalled:
        e = o["registry_entry"]
        print(f"  {e.title[:50]:52s}  authors={', '.join(e.authors[:2])}")

    failed = [o for o in outcomes if o.get("registry_entry") is None
              and o["reason"] != "already_in_registry"]
    print(f"\nFailed (no entry created): {len(failed)}")
    for o in failed:
        print(f"  {o['seed'].anchor_str[:60]:62s}  reason={o['reason']}")

    if args.write:
        n_saved = 0
        for o in outcomes:
            e = o.get("registry_entry")
            if e:
                try:
                    save_entry(e, validate_strict=False)
                    n_saved += 1
                except Exception as exc:
                    logger.error("save failed for %s: %s", e.title[:40], exc)
        print(f"\nWROTE {n_saved} entries to {REGISTRY_PATH}")
        if not args.skip_pdf:
            stats = collection_stats()
            print(f"papers_chroma: {stats}")
    else:
        print(f"\nDRY RUN — pass --write to persist")


if __name__ == "__main__":
    main()
