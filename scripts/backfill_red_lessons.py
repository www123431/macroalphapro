"""scripts/backfill_red_lessons.py — P1 backfill: historical RED → REDLesson.

Reads:
  data/research/gate_runs.jsonl           (23 records, 11 RED + 2 YELLOW)
  data/validation/factory_ledger.jsonl    (44 records, 32 RED + 5 YELLOW,
                                           ~40 unique names)

Produces:
  data/research_store/red_lessons.jsonl   (append-only; dedupe by candidate)

Run:
  python scripts/backfill_red_lessons.py            # default: dry-run + audit
  python scripts/backfill_red_lessons.py --write    # actually save

All lessons land with review_state=claude_drafted; downstream P2 (paper
anchors) and human review pass advance them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Make repo root importable when called from anywhere
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store.red_lessons import (
    save_lesson,
    load_lessons,
    LESSONS_PATH,
)
from engine.research_store.red_lessons.backfill_heuristics import (
    lesson_from_gate_run,
    lesson_from_factory_ledger,
)
from engine.research_store.red_lessons.store import latest_per_candidate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("backfill_red_lessons")


GATE_RUNS_PATH       = _REPO_ROOT / "data" / "research"   / "gate_runs.jsonl"
FACTORY_LEDGER_PATH  = _REPO_ROOT / "data" / "validation" / "factory_ledger.jsonl"


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
                logger.warning("skipping malformed line in %s: %s", path, e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Actually persist to red_lessons.jsonl. Without this "
                         "flag, dry-run + print summary only.")
    ap.add_argument("--session-id", default="backfill_p1_2026-06-03",
                    help="session_id tag for the backfill operation")
    args = ap.parse_args()

    gate_runs = _read_jsonl(GATE_RUNS_PATH)
    factory   = _read_jsonl(FACTORY_LEDGER_PATH)

    logger.info("loaded %d gate_runs + %d factory_ledger records",
                len(gate_runs), len(factory))

    # Build candidates
    candidates: list = []
    for r in gate_runs:
        L = lesson_from_gate_run(r, session_id=args.session_id)
        if L is not None:
            candidates.append(L)
    for r in factory:
        L = lesson_from_factory_ledger(r, session_id=args.session_id)
        if L is not None:
            candidates.append(L)

    logger.info("constructed %d candidate lessons (RED + YELLOW only)",
                len(candidates))

    # Dedupe: factory_ledger has duplicate entries by name (multiple runs
    # of the same candidate). Keep ONE per candidate_name (the last —
    # presumed latest stats). gate_runs entries take precedence when same
    # name appears in both (gate_runs is upstream / earlier).
    seen: dict[str, "REDLesson"] = {}
    for L in candidates:
        # Last-write-wins on duplicate name
        seen[L.candidate_name] = L
    final = list(seen.values())
    logger.info("deduplicated to %d unique candidates", len(final))

    # Audit summary
    family_counts = Counter(L.mechanism_family.value for L in final)
    fm_counts = Counter()
    for L in final:
        for m in L.failure_modes:
            fm_counts[m.value] += 1
    verdict_counts = Counter(L.verdict for L in final)

    print()
    print("=" * 70)
    print("BACKFILL AUDIT — REDLesson P1")
    print("=" * 70)
    print(f"\nTotal unique lessons:  {len(final)}")
    print(f"\nVerdict distribution:")
    for v, n in verdict_counts.most_common():
        print(f"  {n:3d}  {v}")
    print(f"\nMechanism family distribution:")
    for f, n in family_counts.most_common():
        print(f"  {n:3d}  {f}")
    print(f"\nFailure mode distribution (multi-label, may sum > N):")
    for f, n in fm_counts.most_common():
        print(f"  {n:3d}  {f}")

    # Audit: how many ended up with the F8 fallback only?
    fallback_only = [L for L in final
                     if len(L.failure_modes) == 1
                     and L.failure_modes[0].value == "F8_OVERFIT_INDUCED"
                     and "FALLBACK" in L.failure_evidence.get("F8_OVERFIT_INDUCED", "")]
    print(f"\nFallback-only (need human review most urgently): {len(fallback_only)}")
    for L in fallback_only:
        print(f"    {L.candidate_name}")

    # Audit: OTHER family
    other = [L for L in final if L.mechanism_family.value == "OTHER"]
    print(f"\nMechanismFamily=OTHER (need mechanism re-classification): {len(other)}")
    for L in other:
        print(f"    {L.candidate_name} → subtype={L.mechanism_subtype}")

    print()
    print(f"Each lesson has tags: {final[0].tags if final else 'N/A'}")
    print(f"All review_state=claude_drafted (need human review pass)")
    print()

    if args.write:
        # Avoid double-backfill: refuse to write if any of these candidate_names
        # already exist in the store
        existing = {L.candidate_name for L in load_lessons()}
        conflicts = [L for L in final if L.candidate_name in existing]
        if conflicts:
            print(f"ABORTING: {len(conflicts)} candidate(s) already in store:")
            for L in conflicts[:10]:
                print(f"    {L.candidate_name}")
            print("Use --force (not implemented) or manually delete from store first.")
            sys.exit(1)

        for L in final:
            try:
                save_lesson(L, validate_strict=False)
            except Exception as e:
                logger.error("failed to save %s: %s", L.candidate_name, e)
        print(f"WROTE {len(final)} lessons to {LESSONS_PATH}")
    else:
        print(f"DRY RUN — pass --write to persist to {LESSONS_PATH}")


if __name__ == "__main__":
    main()
