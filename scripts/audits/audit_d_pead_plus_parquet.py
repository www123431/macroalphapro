"""
scripts/audit_d_pead_plus_parquet.py — Pre-Phase 4 integrity audit.

Verifies data/d_pead_plus/_llm_extracted_features.parquet before
fit_dev / backtest / verdict run. Catches:
  - Schema drift (missing/extra columns)
  - PK uniqueness (no duplicate transcript_id)
  - Feature range bounds (tone [-1,1], confidence/evasion/complexity [0,1])
  - Coverage by quarter (dev periods sufficient)
  - prompt_hash consistency (all rows extracted with same locked prompt)

Exits non-zero if integrity fail → Phase 4 SHOULD NOT proceed until resolved.

USAGE:
  py -3.11 scripts/audit_d_pead_plus_parquet.py
  py -3.11 scripts/audit_d_pead_plus_parquet.py --strict  (fail on warnings too)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("audit_d_pead_plus_parquet")


# Expected schema (from engine.d_pead_plus.llm_extractor.LLMExtractionRecord)
EXPECTED_COLUMNS: set[str] = {
    "transcript_id", "permno", "rdq", "company_name", "call_date",
    "tone_score", "forward_confidence", "macro_headwind_flag",
    "evasion_score", "linguistic_complexity",
    "prompt_hash", "model_version", "extract_ts_utc",
    "input_tokens", "output_tokens", "cost_usd",
}

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "tone_score":             (-1.0, 1.0),
    "forward_confidence":     (0.0, 1.0),
    "evasion_score":          (0.0, 1.0),
    "linguistic_complexity":  (0.0, 1.0),
}

LOCKED_PROMPT_HASH: str = "f01e18fbf998ec19c541193d8127403902bbc4850610306a764ac923d60f89a6"
LOCKED_MODEL:       str = "gemini-2.5-flash"


def audit_parquet(strict: bool = False) -> int:
    """Run integrity audit. Returns 0=PASS / 1=WARN / 2=FAIL."""
    from engine.d_pead_plus.llm_extractor import FEATURES_PARQUET

    if not FEATURES_PARQUET.exists():
        logger.error("AUDIT FAIL: parquet missing at %s", FEATURES_PARQUET)
        return 2

    try:
        df = pd.read_parquet(FEATURES_PARQUET)
    except Exception as exc:
        logger.error("AUDIT FAIL: parquet unreadable: %s", exc)
        return 2

    n_rows = len(df)
    logger.info("Loaded %d rows from %s", n_rows, FEATURES_PARQUET)

    issues: list[tuple[str, str]] = []  # (severity, message)
    def warn(msg: str): issues.append(("WARN", msg))
    def fail(msg: str): issues.append(("FAIL", msg))

    # Check 1: row count vs spec lower bound
    SPEC_MIN_ROWS = 6750   # spec assumes ~6750 firm-quarters super-powered
    if n_rows < SPEC_MIN_ROWS:
        fail(f"Row count {n_rows} < spec minimum {SPEC_MIN_ROWS} (statistical power inadequate)")
    else:
        logger.info("✓ Row count %d ≥ spec minimum %d", n_rows, SPEC_MIN_ROWS)

    # Check 2: schema
    actual_cols = set(df.columns)
    missing = EXPECTED_COLUMNS - actual_cols
    extra   = actual_cols - EXPECTED_COLUMNS
    if missing:
        fail(f"Missing columns: {sorted(missing)}")
    if extra:
        warn(f"Extra columns (probably fine): {sorted(extra)}")
    if not missing:
        logger.info("✓ Schema: all %d expected columns present", len(EXPECTED_COLUMNS))

    # Check 3: PK uniqueness
    n_unique_tid = df["transcript_id"].nunique()
    n_dup = n_rows - n_unique_tid
    if n_dup > 0:
        fail(f"PK violation: {n_dup} duplicate transcript_ids")
    else:
        logger.info("✓ PK uniqueness: all %d transcript_ids unique", n_unique_tid)

    # Check 4: feature range bounds
    for col, (lo, hi) in FEATURE_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            fail(f"{col}: {out_of_range} values outside [{lo}, {hi}]")
        else:
            logger.info("✓ %s range OK [%.2f, %.2f] (observed [%.3f, %.3f])",
                        col, lo, hi, df[col].min(), df[col].max())

    # Check 5: macro_headwind_flag is boolean (or bool-coercible 0/1)
    if "macro_headwind_flag" in df.columns:
        unique_vals = set(df["macro_headwind_flag"].dropna().unique().tolist())
        valid_bool_vals = {True, False, 0, 1, 0.0, 1.0}
        invalid = unique_vals - valid_bool_vals
        if invalid:
            fail(f"macro_headwind_flag has non-bool values: {invalid}")
        else:
            n_true = df["macro_headwind_flag"].astype(bool).sum()
            logger.info("✓ macro_headwind_flag: %d True / %d total (%.1f%%)",
                        n_true, n_rows, n_true / n_rows * 100)

    # Check 6: prompt_hash consistency (HARKing detection)
    if "prompt_hash" in df.columns:
        unique_hashes = df["prompt_hash"].unique()
        if len(unique_hashes) != 1:
            fail(f"PROMPT_HASH drift: {len(unique_hashes)} different hashes present "
                 f"(should be 1 — indicates HARKing risk)")
        elif unique_hashes[0] != LOCKED_PROMPT_HASH:
            warn(f"prompt_hash differs from LOCKED: got {unique_hashes[0][:16]} expected {LOCKED_PROMPT_HASH[:16]}")
        else:
            logger.info("✓ prompt_hash consistent + matches LOCKED %s", LOCKED_PROMPT_HASH[:16])

    # Check 7: model_version consistency
    if "model_version" in df.columns:
        unique_models = df["model_version"].unique()
        if len(unique_models) != 1:
            fail(f"model_version drift: {sorted(unique_models)}")
        elif unique_models[0] != LOCKED_MODEL:
            warn(f"model_version differs from LOCKED: got {unique_models[0]} expected {LOCKED_MODEL}")
        else:
            logger.info("✓ model_version consistent: %s", LOCKED_MODEL)

    # Check 8: coverage by quarter (dev vs OOS)
    if "rdq" in df.columns:
        df_q = df.copy()
        df_q["rdq"] = pd.to_datetime(df_q["rdq"])
        df_q["quarter"] = df_q["rdq"].dt.to_period("Q")
        coverage = df_q["quarter"].value_counts().sort_index()
        logger.info("Coverage by rdq quarter:")
        for q, n in coverage.items():
            logger.info("  %s: %d records", q, n)

        # Dev: 2024Q2 / Q3 / Q4
        dev_quarters = [pd.Period("2024Q2"), pd.Period("2024Q3"), pd.Period("2024Q4")]
        dev_total = sum(coverage.get(q, 0) for q in dev_quarters)
        # OOS: 2025Q1 onward
        oos_total = sum(n for q, n in coverage.items() if q >= pd.Period("2025Q1"))

        logger.info("Dev (2024-Q2/Q3/Q4):  %d records (spec target ~3000)", dev_total)
        logger.info("OOS (2025-Q1+):       %d records (spec target ~4500)", oos_total)

        DEV_MIN = 1500    # half of spec target
        OOS_MIN = 2000
        if dev_total < DEV_MIN:
            fail(f"Dev coverage {dev_total} < {DEV_MIN} (Gate 4 statistical power impaired)")
        elif dev_total < 3000:
            warn(f"Dev coverage {dev_total} < spec target 3000 (Gate 4 confidence reduced)")

        if oos_total < OOS_MIN:
            fail(f"OOS coverage {oos_total} < {OOS_MIN} (Gate 4 OOS test inadequate)")
        elif oos_total < 4500:
            warn(f"OOS coverage {oos_total} < spec target 4500 (Gate 4 OOS noise higher)")

    # Check 9: NaN check
    null_counts = df.isna().sum()
    nulls_present = null_counts[null_counts > 0]
    if len(nulls_present) > 0:
        for col, n in nulls_present.items():
            warn(f"{col}: {n} NaN values")
    else:
        logger.info("✓ No NaN values in any column")

    # Summary
    logger.info("=" * 60)
    fails = [m for s, m in issues if s == "FAIL"]
    warns = [m for s, m in issues if s == "WARN"]
    logger.info("AUDIT SUMMARY: %d FAIL / %d WARN", len(fails), len(warns))
    for m in fails:
        logger.error("  FAIL: %s", m)
    for m in warns:
        logger.warning("  WARN: %s", m)

    if fails:
        logger.error("=" * 60)
        logger.error("AUDIT FAIL — Phase 4 should NOT proceed until resolved")
        return 2
    if warns and strict:
        logger.warning("AUDIT WARN with --strict — Phase 4 should NOT proceed")
        return 1
    if warns:
        logger.warning("AUDIT PASS with warnings — Phase 4 may proceed but review warnings")
        return 1
    logger.info("AUDIT PASS — Phase 4 can proceed safely")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Pre-Phase-4 parquet integrity audit")
    p.add_argument("--strict", action="store_true", help="Fail on warnings too (not just errors)")
    args = p.parse_args()
    return audit_parquet(strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
