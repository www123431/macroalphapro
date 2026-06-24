"""scripts/enrich_t2_anchors.py — Stage C Phase B CLI.

For each T2_ANCHOR paper in papers_registry: pull abstract from
CrossRef + generate Sonnet 1-line anchor summary. Persists via
amend_entry.

Usage:
  python scripts/enrich_t2_anchors.py            # dry-run
  python scripts/enrich_t2_anchors.py --write    # persist
  python scripts/enrich_t2_anchors.py --force    # re-enrich already-done
  python scripts/enrich_t2_anchors.py --max 3    # cap for testing
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.papers.store import load_registry, save_entry
from engine.research_store.papers.schema import PaperTier
from engine.research_store.papers.amend import amend_entry
from engine.research_store.papers.anchor_enricher import (
    enrich_paper, EnrichmentResult,
)


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true",
                    help="Persist enrichments (default: dry-run)")
    p.add_argument("--force", action="store_true",
                    help="Re-enrich papers that already have a summary")
    p.add_argument("--max", type=int, default=0,
                    help="Cap T2 papers processed (0 = all)")
    args = p.parse_args()

    raw = load_registry()
    # Latest-per-paper_id dedup
    by_pid: dict = {}
    for r in raw:
        prior = by_pid.get(r.paper_id)
        if prior is None or r.version > prior.version:
            by_pid[r.paper_id] = r
    latest = list(by_pid.values())

    # DOI dedup — prefer entry with tier set
    by_doi: dict = {}
    no_doi: list = []
    for p in latest:
        d = (p.doi or "").strip().lower()
        if not d:
            no_doi.append(p)
            continue
        if d not in by_doi:
            by_doi[d] = p
        elif (p.tier != PaperTier.UNCLASSIFIED
              and by_doi[d].tier == PaperTier.UNCLASSIFIED):
            by_doi[d] = p
    functional = list(by_doi.values()) + no_doi

    t2_anchors = [p for p in functional if p.tier == PaperTier.T2_ANCHOR]
    if args.max > 0:
        t2_anchors = t2_anchors[:args.max]

    print(f"T2 anchors found: {len(t2_anchors)}", file=sys.stderr)
    if not args.force:
        already = sum(1 for p in t2_anchors if p.tier_anchor_summary)
        print(f"  already enriched: {already} (will skip; use --force "
              "to re-do)", file=sys.stderr)

    results: list[EnrichmentResult] = []
    for i, paper in enumerate(t2_anchors, 1):
        print(f"  [{i}/{len(t2_anchors)}] {paper.title[:50]}...",
              file=sys.stderr)
        r = enrich_paper(paper, force=args.force)
        results.append(r)
        # Print outcome inline
        flags = []
        if r.summary_generated: flags.append("SUMMARY")
        if r.abstract_updated: flags.append("ABSTRACT")
        if r.errors: flags.append(f"ERR({','.join(r.errors)})")
        if not flags: flags = ["skip"]
        print(f"     → {' / '.join(flags)}", file=sys.stderr)

    # Apply if --write
    n_persisted = 0
    if args.write:
        # We need to look up the paper object again from `by_pid` by
        # paper_id, since enrich_paper just received the entry, not
        # the original list reference
        pid_to_paper = {p.paper_id: p for p in t2_anchors}
        for r in results:
            if not r.summary_generated:
                continue
            paper = pid_to_paper.get(r.paper_id)
            if paper is None:
                continue
            try:
                # Build amendment — set_abstract uses the CrossRef
                # backfill only when we actually got a longer one
                new_entry = amend_entry(
                    prior=paper,
                    set_tier_anchor_summary=r.summary,
                    set_abstract=(r.new_abstract
                                    if r.abstract_updated else None),
                    updated_ts=_utc_iso(),
                    created_by="engine.research_store.papers."
                                "anchor_enricher",
                    note_append=("Stage C Phase B: T2 anchor enriched"
                                  + (" (+CrossRef abstract)"
                                     if r.abstract_updated else "")),
                )
                save_entry(new_entry)
                n_persisted += 1
            except Exception as exc:
                print(f"  ERR persist {r.paper_id[:8]}: {exc}",
                      file=sys.stderr)

    # Final summary
    print("", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    n_summarized = sum(1 for r in results if r.summary_generated)
    n_abstract_back = sum(1 for r in results if r.abstract_updated)
    n_err = sum(1 for r in results if r.errors)
    print(f"T2 anchors processed: {len(results)}", file=sys.stderr)
    print(f"  summaries generated: {n_summarized}", file=sys.stderr)
    print(f"  abstracts backfilled: {n_abstract_back}", file=sys.stderr)
    print(f"  errors:               {n_err}", file=sys.stderr)
    if args.write:
        print(f"  persisted:            {n_persisted}", file=sys.stderr)
    else:
        print("  DRY RUN — re-run with --write", file=sys.stderr)

    # Stdout: per-paper summaries (the actual fuel — readable + auditable)
    print()
    print("# T2 anchor summaries\n")
    for paper, r in zip(t2_anchors, results):
        if r.summary:
            print(f"## {paper.title[:80]}")
            print(f"   _{paper.authors[0] if paper.authors else '?'} "
                  f"{paper.year}_  ·  `{paper.paper_id[:8]}`")
            print(f"   > {r.summary}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
