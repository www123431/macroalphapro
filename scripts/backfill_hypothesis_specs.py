"""scripts/backfill_hypothesis_specs.py — extract typed HypothesisSpec
records from every claim in hypotheses.jsonl.

Idempotent: skips hypotheses that already have a stored spec (latest
version, unless --force). Resumable on crash.

Usage:
  python scripts/backfill_hypothesis_specs.py                 # 全量
  python scripts/backfill_hypothesis_specs.py --limit 10      # 跑 10 条试
  python scripts/backfill_hypothesis_specs.py --force         # 重新提取
  python scripts/backfill_hypothesis_specs.py --family CARRY  # 只一个家族

Output:
  - appends to data/research_store/hypothesis_specs.jsonl
  - prints summary: family distribution, confidence histogram,
    failed extractions

Cost estimate: ~$0.005 per hypothesis × 209 = ~$1.05 total.
Time: ~3s per hypothesis × 209 = ~10-12 minutes.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="extract only the first N hypotheses (test mode)")
    ap.add_argument("--force", action="store_true",
                    help="re-extract hypotheses that already have a spec")
    ap.add_argument("--family", type=str, default=None,
                    help="restrict to one mechanism_family (case-insensitive)")
    args = ap.parse_args()

    # Imports deferred so the CLI parser runs even if engine has import issues
    from engine.research_store.hypothesis import load_hypotheses
    from engine.hypothesis_spec.extractor import extract_spec
    from engine.hypothesis_spec.store import append, latest_for
    from engine.hypothesis_spec.hash import spec_hash
    from engine.research_store.manifest import current_git_sha

    # Load latest version per hypothesis_id
    hyps_raw = load_hypotheses()
    latest_by_id: dict = {}
    for h in hyps_raw:
        prior = latest_by_id.get(h.hypothesis_id)
        if prior is None or h.version > prior.version:
            latest_by_id[h.hypothesis_id] = h
    all_hyps = sorted(latest_by_id.values(),
                      key=lambda h: getattr(h, "created_ts", "") or "")

    # Filter
    if args.family:
        fam = args.family.upper()
        all_hyps = [h for h in all_hyps
                    if h.mechanism_family.value.upper() == fam]

    # Skip already-extracted unless --force
    todo = []
    skipped = 0
    for h in all_hyps:
        existing = latest_for(h.hypothesis_id)
        if existing is not None and not args.force:
            skipped += 1
            continue
        todo.append(h)

    if args.limit:
        todo = todo[: args.limit]

    print(f"Hypotheses total:       {len(all_hyps)}")
    print(f"Already have spec:      {skipped} (skip; use --force to redo)")
    print(f"To extract this run:    {len(todo)}")
    if args.limit:
        print(f"  (capped by --limit {args.limit})")
    print()

    if not todo:
        print("Nothing to do.")
        return 0

    git_sha = current_git_sha() or ""

    # Process
    families    = Counter()
    claim_types = Counter()   # B.2-A4: track claim_type distribution
    confidence  = []
    failed: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    for i, h in enumerate(todo, start=1):
        elapsed = time.perf_counter() - t0
        eta = (elapsed / i) * (len(todo) - i) if i > 1 else 0
        print(f"[{i:>3}/{len(todo)}]  elapsed={elapsed:.0f}s · eta={eta:.0f}s · "
              f"{h.hypothesis_id[:8]}  family={h.mechanism_family.value:<25}",
              end=" ", flush=True)
        try:
            spec = extract_spec(
                source_hypothesis_id = h.hypothesis_id,
                claim_text           = h.claim,
                mechanism_family     = h.mechanism_family.value,
                mechanism_subtype    = h.mechanism_subtype,
                git_sha              = git_sha,
            )
        except Exception as exc:
            failed.append((h.hypothesis_id, f"exc:{exc}"[:200]))
            print("EXC")
            continue
        if spec is None:
            failed.append((h.hypothesis_id, "extractor returned None"))
            print("FAIL")
            continue
        try:
            h_hash = append(spec)
        except Exception as exc:
            failed.append((h.hypothesis_id, f"persist:{exc}"[:200]))
            print("PERSIST_ERR")
            continue
        families[spec.family.value] += 1
        claim_types[spec.claim_type.value] += 1
        confidence.append(spec.extraction.confidence)
        print(f"ok  ct={spec.claim_type.value[:14]:<14} "
              f"fam={spec.family.value[:10]:<10} "
              f"conf={spec.extraction.confidence:.2f}  hash={h_hash[:8]}")

    total = time.perf_counter() - t0
    print()
    print("=" * 64)
    print(f"Done.  total={total:.0f}s  ok={len(confidence)}  fail={len(failed)}")
    print()
    print("Claim type distribution (B.2-A4):")
    for ct, n in claim_types.most_common():
        print(f"  {n:>4}  {ct}")
    print()
    print("Family distribution (FACTOR_HYPOTHESIS only meaningful):")
    for f, n in families.most_common():
        print(f"  {n:>4}  {f}")
    print()
    if confidence:
        import statistics
        print(f"Confidence:  mean={statistics.mean(confidence):.2f}  "
              f"median={statistics.median(confidence):.2f}  "
              f"min={min(confidence):.2f}  max={max(confidence):.2f}")
        bins = [0, 0.3, 0.5, 0.7, 0.85, 1.01]
        labels = ["<0.30", "0.30-0.50", "0.50-0.70", "0.70-0.85", "0.85-1.00"]
        counts = [0] * (len(bins) - 1)
        for c in confidence:
            for i in range(len(bins) - 1):
                if bins[i] <= c < bins[i + 1]:
                    counts[i] += 1
                    break
        print("Confidence histogram:")
        for label, n in zip(labels, counts):
            bar = "█" * (n * 40 // max(1, max(counts)))
            print(f"  {label:>9}  {n:>4}  {bar}")
    if failed:
        print()
        print(f"Failures ({len(failed)}):")
        for hid, reason in failed[:10]:
            print(f"  {hid[:12]}  {reason}")
        if len(failed) > 10:
            print(f"  ... +{len(failed) - 10} more")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
