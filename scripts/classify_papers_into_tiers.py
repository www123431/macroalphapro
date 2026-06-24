"""scripts/classify_papers_into_tiers.py — Stage C Phase A CLI.

Runs Sonnet batch classifier on papers_registry entries; outputs a
markdown review table; optionally applies the classifications
(--write) by emitting a registry amendment per paper.

Default is DRY-RUN — print the proposed tiers, you review, then
re-run with --write to persist.

Usage:
  python scripts/classify_papers_into_tiers.py
  python scripts/classify_papers_into_tiers.py --only-unclassified
  python scripts/classify_papers_into_tiers.py --write
  python scripts/classify_papers_into_tiers.py --json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.papers.store import load_registry
from engine.research_store.papers.tier_classifier import (
    classify_papers_batch, TierProposal,
)
from engine.research_store.papers.schema import PaperTier


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _markdown_table(proposals: list[TierProposal],
                     papers_by_id: dict) -> str:
    lines = [
        "## Tier-classification proposals",
        "",
        f"Generated {_utc_iso()}. Review then re-run with `--write` "
        "to persist.",
        "",
        "| Tier | Conf | Year | First author | Title (60c) | Rationale |",
        "|---|---:|---:|---|---|---|",
    ]
    # Sort: T1 first (most precious), then T2, then T3, then UNCLASSIFIED
    tier_order = {
        PaperTier.T1_DOCTRINE: 0,
        PaperTier.T2_ANCHOR: 1,
        PaperTier.T3_RECENT: 2,
        PaperTier.UNCLASSIFIED: 3,
    }
    for prop in sorted(proposals, key=lambda p: (tier_order[p.tier],
                                                    -p.confidence)):
        paper = papers_by_id.get(prop.paper_id)
        if paper is None:
            continue
        first_author = (paper.authors or [""])[0]
        title = (paper.title or "")[:60]
        rationale = prop.rationale[:80]
        lines.append(
            f"| **{prop.tier.value}** | {prop.confidence:.2f} | "
            f"{paper.year} | {first_author} | {title} | {rationale} |"
        )
    lines.append("")
    # Summary counts
    by_tier: dict[str, int] = {}
    for p in proposals:
        by_tier[p.tier.value] = by_tier.get(p.tier.value, 0) + 1
    lines.append("### Counts")
    for t in ("T1_DOCTRINE", "T2_ANCHOR", "T3_RECENT", "UNCLASSIFIED"):
        lines.append(f"  {t}: {by_tier.get(t, 0)}")
    return "\n".join(lines)


def _apply_proposals_to_registry(proposals: list[TierProposal],
                                    papers_by_id: dict) -> int:
    """Use the amend pipeline to write tier + tier_classified_ts +
    tier_rationale on each paper as a new registry version. Returns
    count actually amended."""
    from engine.research_store.papers.amend import amend_entry
    from engine.research_store.papers.store import save_entry

    now = _utc_iso()
    written = 0
    for prop in proposals:
        if prop.tier == PaperTier.UNCLASSIFIED:
            continue   # don't bake UNCLASSIFIED — leave for re-review
        paper = papers_by_id.get(prop.paper_id)
        if paper is None:
            continue
        # Skip if already set to same tier (idempotent)
        if (paper.tier == prop.tier
            and paper.tier_classified_ts):
            continue
        try:
            new_entry = amend_entry(
                prior              = paper,
                set_tier           = prop.tier,
                set_tier_rationale = prop.rationale,
                set_tier_classified_ts = now,
                updated_ts         = now,
                created_by         = "engine.research_store.papers."
                                       "tier_classifier",
                note_append        = (f"tier set to {prop.tier.value} "
                                        f"(conf={prop.confidence:.2f})"),
            )
            save_entry(new_entry)
            written += 1
        except Exception as exc:
            print(f"  ERR {prop.paper_id[:8]}: {exc}", file=sys.stderr)
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only-unclassified", action="store_true",
                    help="Skip papers that already have a tier set "
                          "(default: re-classify everything for review)")
    p.add_argument("--write", action="store_true",
                    help="Persist proposed tiers (default: dry-run)")
    p.add_argument("--json", action="store_true",
                    help="Output JSON instead of markdown")
    p.add_argument("--max-papers", type=int, default=0,
                    help="Cap batch size for cost control (0 = all)")
    args = p.parse_args()

    raw_registry = load_registry()
    # DEDUP by paper_id: keep only the LATEST version. Otherwise the
    # classifier sees the same paper multiple times (registry has
    # version history; many papers have 2-3 amendments).
    by_id: dict = {}
    for p in raw_registry:
        prior = by_id.get(p.paper_id)
        if prior is None or p.version > prior.version:
            by_id[p.paper_id] = p
    registry = list(by_id.values())

    papers = registry
    if args.only_unclassified:
        papers = [p for p in registry
                   if p.tier == PaperTier.UNCLASSIFIED]
    if args.max_papers > 0:
        papers = papers[:args.max_papers]

    if not papers:
        print("No papers to classify.")
        return 0

    # Chunk into batches of 20 — single 57-paper call timed out at the
    # 60s anthropic default. 20 papers ≈ 5-8k tokens, returns in <30s.
    BATCH_SIZE = 20
    chunks = [papers[i:i+BATCH_SIZE]
              for i in range(0, len(papers), BATCH_SIZE)]
    print(f"Classifying {len(papers)} papers in {len(chunks)} batches "
          f"of ≤{BATCH_SIZE}...", file=sys.stderr)

    proposals = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  batch {i}/{len(chunks)} ({len(chunk)} papers)...",
              file=sys.stderr)
        batch_props = classify_papers_batch(chunk)
        if not batch_props:
            print(f"  WARN: batch {i} returned empty (timeout / "
                   "tool not called); skipping",
                   file=sys.stderr)
            continue
        proposals.extend(batch_props)

    if not proposals:
        print("ERROR: all batches returned empty", file=sys.stderr)
        return 1
    print(f"  → {len(proposals)} proposals returned ({len(papers) - len(proposals)} skipped)",
          file=sys.stderr)

    papers_by_id = {p.paper_id: p for p in registry}

    if args.json:
        sys.stdout.write(json.dumps({
            "n_proposals": len(proposals),
            "proposals": [
                {
                    "paper_id":   pr.paper_id,
                    "title":      papers_by_id[pr.paper_id].title,
                    "tier":       pr.tier.value,
                    "confidence": pr.confidence,
                    "rationale":  pr.rationale,
                }
                for pr in proposals
                if pr.paper_id in papers_by_id
            ],
        }, indent=2, ensure_ascii=False))
    else:
        sys.stdout.write(_markdown_table(proposals, papers_by_id))

    if args.write:
        n = _apply_proposals_to_registry(proposals, papers_by_id)
        print(f"\nWROTE {n} amendments to registry.", file=sys.stderr)
    else:
        print("\nDRY RUN — re-run with --write to persist.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
