"""scripts/generate_forward_vectors.py — T5 driver.

Generate ForwardVectorV2 records from untested hypotheses.

Run:
  python scripts/generate_forward_vectors.py            # dry-run (audit)
  python scripts/generate_forward_vectors.py --write    # persist
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.forward_vectors import (
    FORWARD_VECTORS_PATH, generate_forward_vectors, save_forward_vector,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("generate_forward_vectors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="persist forward vectors to jsonl")
    ap.add_argument("--top", type=int, default=20,
                    help="print top N in audit")
    args = ap.parse_args()

    vectors = generate_forward_vectors()

    print()
    print("=" * 72)
    print("FORWARD VECTORS AUDIT")
    print("=" * 72)
    print(f"\nTotal forward vectors generated: {len(vectors)}")

    priority_dist = Counter(fv.priority.value for fv in vectors)
    print("\nPriority distribution:")
    for k, v in priority_dist.most_common():
        print(f"  {v:3d}  {k}")

    family_dist = Counter(fv.mechanism_family.value for fv in vectors)
    print("\nMechanism family distribution:")
    for k, v in family_dist.most_common():
        print(f"  {v:3d}  {k}")

    print(f"\nTop {args.top} (sorted by priority):")
    for i, fv in enumerate(vectors[: args.top], start=1):
        print(f"\n[{i}] {fv.priority.value.upper()}  family={fv.mechanism_family.value}/{fv.mechanism_subtype}")
        print(f"    paper: {fv.paper_title[:70]}")
        print(f"    claim: {fv.claim[:120]}")
        print(f"    direction={fv.predicted_direction} | magnitude={fv.predicted_magnitude[:60]}")
        if fv.required_data:
            print(f"    required_data: {list(fv.required_data)[:2]}")

    if args.write:
        n_w = 0
        for fv in vectors:
            try:
                save_forward_vector(fv, validate_strict=False)
                n_w += 1
            except Exception as e:
                logger.error("save failed: %s", e)
        print(f"\nWROTE {n_w} forward vectors to {FORWARD_VECTORS_PATH}")
    else:
        print(f"\nDRY RUN — pass --write to persist")


if __name__ == "__main__":
    main()
