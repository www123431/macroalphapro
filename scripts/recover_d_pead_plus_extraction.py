"""
scripts/recover_d_pead_plus_extraction.py — Sprint I extraction recovery pass.

Background: 2026-05-13 evening, ~3401 records lost from extraction parquet due
to git stash race (data/d_pead_plus/_llm_extracted_features.parquet was tracked
in git; concurrent `git stash` during extraction reverted working tree to
HEAD's 20-record version; extraction process saved on top of stale base;
git stash pop kept working tree changes; ~3401 records silently dropped).

This script reads the current parquet (after main extraction completes), diffs
against the full transcripts_index (11580 unique transcript_ids), and re-extracts
the missing IDs. Uses run_extraction_rest with skip_already_extracted=True so
it naturally only re-processes the gap.

DOCTRINE COMPLIANCE:
- Uses SAME hash-locked prompt + model + temperature + schema → spec id=74
  hash 6d8e614e preserved
- Recovery is faithful re-execution of original spec, NOT a re-run with
  modified parameters. No HARKing concern.
- skip_already_extracted=True idempotency guarantee

USAGE:
  py -3.11 scripts/recover_d_pead_plus_extraction.py
  py -3.11 scripts/recover_d_pead_plus_extraction.py --max-recover 100  (smoke)

EXIT CODES:
  0 — recovery completed successfully (or nothing to recover)
  1 — recovery partial (some IDs still missing after retry)
  2 — pre-flight check failed
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("recover_d_pead_plus")


def identify_missing_transcript_ids() -> tuple[set[int], int, int]:
    """Diff cached parquet vs merged transcripts index. Returns (missing_ids, n_total, n_cached)."""
    from engine.d_pead_plus.llm_extractor import load_existing_extractions
    from engine.d_pead_plus.transcripts_loader import load_cached_transcripts

    cached = load_existing_extractions()
    n_cached = len(cached)

    idx, text = load_cached_transcripts()
    merged = idx.merge(
        text[["transcript_id", "full_text"]],
        on="transcript_id", how="inner",
    )
    merged_ids = set(merged["transcript_id"].astype(int).tolist())
    n_total = len(merged_ids)   # unique

    cached_ids: set[int] = set()
    if not cached.empty:
        cached_ids = set(cached["transcript_id"].astype(int).tolist())

    missing = merged_ids - cached_ids
    return missing, n_total, n_cached


def _run_one_pass(missing_ids: set[int], pass_num: int) -> int:
    """Run one recovery extraction pass on the given missing set.
    Returns n_records_added (cache delta)."""
    from engine.d_pead_plus.llm_extractor_rest import run_extraction_rest
    from engine.d_pead_plus.transcripts_loader import load_cached_transcripts
    from engine.d_pead_plus.llm_extractor import load_existing_extractions

    idx, text = load_cached_transcripts()
    idx_pass  = idx[idx["transcript_id"].astype(int).isin(missing_ids)].copy()
    text_pass = text[text["transcript_id"].astype(int).isin(missing_ids)].copy()

    if idx_pass.empty:
        logger.info("Pass %d: no missing IDs to process", pass_num)
        return 0

    n_before = len(load_existing_extractions())
    logger.info("Pass %d: attempting %d records (cache=%d before)",
                pass_num, len(idx_pass), n_before)

    run_extraction_rest(
        idx_pass,
        text_pass,
        skip_already_extracted = True,
    )

    n_after = len(load_existing_extractions())
    delta = n_after - n_before
    logger.info("Pass %d: cache=%d → %d (+%d records)", pass_num, n_before, n_after, delta)
    return delta


def run_recovery(
    max_recover: Optional[int] = None,
    smoke:       bool          = False,
    max_passes:  int           = 3,
) -> tuple[int, int]:
    """Execute multi-pass recovery. Returns (initial_missing, final_recovered).

    Multi-pass strategy:
      Pass 1: process all missing transcript_ids
      Pass 2: identify still-missing, retry those (catches 429 / transient failures)
      Pass 3: final retry for anything still missing

    Records still missing after all passes are LLM-intrinsic failures
    (corrupted transcripts, persistent JSON parse errors, etc.) — these are
    spec-bound, not fixable without modifying prompt (which would violate HARKing).
    """
    missing_ids_initial, n_total, n_cached_initial = identify_missing_transcript_ids()
    logger.info("Pre-flight: index=%d unique transcripts, cached=%d, missing=%d",
                n_total, n_cached_initial, len(missing_ids_initial))

    if not missing_ids_initial:
        logger.info("Nothing to recover — parquet matches index.")
        return 0, 0

    if max_recover is not None:
        missing_ids_initial = set(list(sorted(missing_ids_initial))[:max_recover])
        logger.info("max_recover=%d → recovering first %d missing", max_recover, len(missing_ids_initial))

    if smoke:
        logger.info("SMOKE mode: would process %d records across up to %d passes",
                    len(missing_ids_initial), max_passes)
        return len(missing_ids_initial), 0

    est_cost     = len(missing_ids_initial) * 0.000773
    est_time_min = len(missing_ids_initial) * 1.7 / 60
    logger.info("Total estimated cost: $%.2f / time: %.1f min (across %d passes max)",
                est_cost, est_time_min, max_passes)

    # Multi-pass loop
    current_missing = set(missing_ids_initial)
    total_recovered = 0
    for pass_num in range(1, max_passes + 1):
        if not current_missing:
            logger.info("Pass %d: all recovered, stopping early", pass_num)
            break

        logger.info("=" * 60)
        logger.info("=== RECOVERY PASS %d / %d (n_missing=%d) ===",
                    pass_num, max_passes, len(current_missing))
        logger.info("=" * 60)

        delta = _run_one_pass(current_missing, pass_num)
        total_recovered += delta

        # Refresh missing set
        new_missing, _, _ = identify_missing_transcript_ids()
        new_missing = new_missing & missing_ids_initial   # only ones we started with
        progress_this_pass = len(current_missing) - len(new_missing)
        logger.info("Pass %d net progress: %d recovered, %d remaining",
                    pass_num, progress_this_pass, len(new_missing))

        if progress_this_pass == 0:
            logger.warning("Pass %d zero progress — remaining %d are LLM-intrinsic failures, stopping",
                           pass_num, len(new_missing))
            break

        current_missing = new_missing

    # Final summary
    final_missing, _, n_cached_final = identify_missing_transcript_ids()
    n_remaining_from_initial = len(final_missing & missing_ids_initial)
    recovery_rate = (len(missing_ids_initial) - n_remaining_from_initial) / len(missing_ids_initial)

    logger.info("=" * 60)
    logger.info("RECOVERY COMPLETE")
    logger.info("  Initial missing:    %d", len(missing_ids_initial))
    logger.info("  Final missing:      %d (%.1f%% recovered)",
                n_remaining_from_initial, recovery_rate * 100)
    logger.info("  Cache: %d → %d (delta +%d)",
                n_cached_initial, n_cached_final, n_cached_final - n_cached_initial)
    logger.info("=" * 60)

    return len(missing_ids_initial), total_recovered


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Sprint I extraction recovery pass")
    p.add_argument("--max-recover", type=int, default=None,
                   help="Max records to recover this pass (default all missing)")
    p.add_argument("--smoke", action="store_true",
                   help="Dry-run: identify missing only, no API calls")
    args = p.parse_args()

    # Pre-flight: extraction not still running?
    try:
        from engine.d_pead_plus.llm_extractor import load_existing_extractions
        cached = load_existing_extractions()
    except Exception as exc:
        logger.error("Pre-flight failed: %s", exc)
        return 2

    # Check parquet age — if last modified < 5 min ago, extraction probably still active
    parquet_path = Path("data/d_pead_plus/_llm_extracted_features.parquet")
    if parquet_path.exists():
        age_s = (datetime.datetime.now().timestamp() - parquet_path.stat().st_mtime)
        if age_s < 300 and not args.smoke:
            logger.warning(
                "Parquet last modified %.0fs ago — extraction may still be running. "
                "Re-run after extraction completes to avoid races. (Use --smoke to dry-run safely.)",
                age_s,
            )
            return 2

    n_attempted, n_succeeded = run_recovery(
        max_recover = args.max_recover,
        smoke       = args.smoke,
    )

    if args.smoke:
        logger.info("Smoke complete: %d records would be recovered", n_attempted)
        return 0

    if n_attempted == 0:
        logger.info("Recovery: nothing to do.")
        return 0

    success_rate = n_succeeded / n_attempted if n_attempted > 0 else 0.0
    logger.info("Recovery summary: %d attempted, %d succeeded (%.1f%%)",
                n_attempted, n_succeeded, success_rate * 100)

    if success_rate < 0.95:
        logger.warning("Success rate < 95%% — investigate failures before proceeding to Phase 4")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
