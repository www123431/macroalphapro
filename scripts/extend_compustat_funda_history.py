"""scripts/extend_compustat_funda_history.py — Tier C-2e.2.

One-time historical backfill of Compustat annual fundamentals from
1962 (Compustat coverage start) through current. Writes to
`data/cache/_compustat_funda_long_history.parquet` — DOES NOT
overwrite the existing `_compustat_funda.parquet` (2011-2024) used
by other consumers.

Why a new file:
  The existing _compustat_funda.parquet is small (25K rows, 2011-
  2024) and likely keyed by callers (BARRA Phase 2, etc.). Replacing
  it with a 6x larger long-history version risks silently slowing
  those consumers + changing their semantics. The Tier C cross_sec
  template reads from the NEW long-history file only.

Idempotency:
  If the output parquet exists, --force is required to overwrite.
  Default behavior: skip with a log message.

Cost (per WRDS):
  - One query, full universe, 1962-2024
  - DEFAULT_FUNDA_COLS (14 columns, no metadata bloat)
  - Expected: 5-15 minutes wall time, ~150-300K rows
  - PIT filter applied (indfmt='INDL' AND consol='C' etc.)

Output schema (DataFrame written to parquet):
  gvkey:    str   — Compustat firm key
  datadate: date  — fiscal year end
  fyear:    int   — fiscal year
  tic:      str   — ticker (as-of fiscal year end)
  at, ceq, lt, ni, oibdp, sale, cogs, xsga, xrd, ppent, dlc, dltt,
  dvc — see DEFAULT_FUNDA_COLS docstring

NOT in this script (deferred to downstream consumers):
  - 120-day public-availability lag (apply at signal-compute time)
  - CCM gvkey↔permno link join (already cached at
    _crsp_ccm_link.parquet)
  - Per-signal derived columns (gp_at, book_to_market, etc.) —
    computed in the cross_sec template
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `engine.*` importable when running as a top-level script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logger = logging.getLogger("extend_compustat_funda_history")
logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s %(levelname)s %(message)s")


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = (_REPO_ROOT / "data" / "cache"
                  / "_compustat_funda_long_history.parquet")


def _check_existing(force: bool) -> bool:
    """True iff we should proceed with the pull. False if existing
    cache is fine + --force not set."""
    if _OUTPUT_PATH.is_file():
        df = pd.read_parquet(_OUTPUT_PATH)
        msg = (f"Existing cache: {_OUTPUT_PATH.name} | shape={df.shape} | "
                 f"date range {df.datadate.min()} → {df.datadate.max()}")
        if force:
            logger.info("--force set; will OVERWRITE existing cache. %s",
                          msg)
            return True
        logger.info("Cache already exists, skipping. %s", msg)
        logger.info("Pass --force to refresh.")
        return False
    return True


def _pull(start: str, end: str) -> pd.DataFrame:
    from engine.data.fetchers.wrds_compustat import fetch_funda
    logger.info("Fetching Compustat funda %s → %s (this may take "
                  "5-15 minutes)...", start, end)
    df = fetch_funda(start, end)
    if df is None or df.empty:
        raise RuntimeError(
            "fetch_funda returned empty — check WRDS credentials + "
            "network. See engine/data/fetchers/wrds_compustat.py probe()."
        )
    logger.info("Fetched %d rows | gvkeys=%d | date range %s → %s",
                  len(df), df.gvkey.nunique(),
                  df.datadate.min(), df.datadate.max())
    return df


def _validate(df: pd.DataFrame) -> None:
    """Sanity checks before persisting."""
    required = {"gvkey", "datadate", "at", "ceq", "ni", "sale", "cogs"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"fetch_funda result missing required columns: {missing}"
        )
    # Reasonable size — at least 50K rows for a 60+ year pull
    if len(df) < 50_000:
        logger.warning("Pull only %d rows; expected >= 50K for "
                          "1962-2024 full history. Verify completeness.",
                          len(df))
    # Date sanity
    dmin, dmax = df.datadate.min(), df.datadate.max()
    if dmin > pd.Timestamp("1970-01-01"):
        logger.warning("Earliest datadate=%s > 1970-01-01 — historical "
                          "coverage may be shorter than expected", dmin)
    if dmax < pd.Timestamp("2020-01-01"):
        logger.warning("Latest datadate=%s < 2020-01-01 — recent "
                          "coverage incomplete", dmax)
    n_null_at = int(df["at"].isna().sum()) if "at" in df.columns else 0
    logger.info("Validation pass | %d rows | datadate %s → %s | "
                  "%d (%.1f%%) null `at`",
                  len(df), dmin, dmax,
                  int(n_null_at), 100 * n_null_at / len(df))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="1962-01-01",
                       help="Earliest datadate (default 1962-01-01 — "
                            "Compustat coverage start)")
    ap.add_argument("--end", default="2024-12-31",
                       help="Latest datadate (default 2024-12-31)")
    ap.add_argument("--force", action="store_true",
                       help="Overwrite existing output parquet")
    args = ap.parse_args()

    if not _check_existing(args.force):
        return 0

    df = _pull(args.start, args.end)

    # Persist FIRST so a validation-step bug doesn't lose the
    # expensive 5-15 min WRDS pull (caught 2026-06-08 — validation
    # used `df.at` which is the pandas .at indexer, not column
    # access; crashed after successful fetch with no save).
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_PATH, index=False)
    sz_mb = _OUTPUT_PATH.stat().st_size / 1e6
    logger.info("WROTE %s | %.1f MB", _OUTPUT_PATH, sz_mb)

    _validate(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
