"""scripts/add_knowable_at_to_funda_pit.py — Tier C L2-1 Phase 1.5.

Adds a `knowable_at` column to data/cache/_compustat_funda_pit.parquet
by JOINing with comp.fundq.rdq (release date), so the accessor (Phase
1.6) can filter PIT data as plain SQL-style "WHERE knowable_at
<= clock.now" — eliminating application-layer lag arithmetic.

PIPELINE
========
1. Read existing _compustat_funda_pit.parquet (9.3M rows from
   Phase 1.2 v2 chunked pull)
2. Pull comp.fundq rdq for the same gvkey/datadate range
3. LEFT JOIN funda on (gvkey, datadate) → fundq.rdq
4. knowable_at = COALESCE(rdq, datadate + 120 days)
   - rdq present: TRUE first-public date (the actual release)
   - rdq missing: 120d conservative approximation
5. Persist back to same parquet path with knowable_at column

DESIGN PHILOSOPHY (PIT BITEMPORAL DATA MODELING)
================================================
Per docs/spec_pit_data_accessor.md §A.1, this implements the
industry-standard pattern: every PIT data row carries its own
knowable_at timestamp, computed at data-engineering time. The
APPLICATION layer (accessor) becomes a pure filter, NOT a lag-
computation engine.

After this script + Phase 1.6 accessor refactor:
  _FUNDA_PUBLIC_LAG_DAYS = 120 application constant → DELETED
  accessor.funda_panel becomes: filter rows by knowable_at <= clock.now
  No more "where did 120 come from" in application code.

DEPENDENCIES
============
Phase 1.2 must have completed: _compustat_funda_pit.parquet exists.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logger = logging.getLogger("add_knowable_at_to_funda_pit")
logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s %(levelname)s %(message)s")


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PIT_PATH  = _REPO_ROOT / "data" / "cache" / "_compustat_funda_pit.parquet"

_FALLBACK_LAG_DAYS = 120


def _pull_fundq_rdq(start: str, end: str) -> pd.DataFrame:
    """Pull (gvkey, datadate, rdq) from comp.fundq for the window.
    Year-chunked for the same connection-stability reason as
    Phase 1.2 v2."""
    from engine.line_c import wrds_direct
    start_year = int(start[:4])
    end_year   = int(end[:4])
    chunks: list[pd.DataFrame] = []
    for yr in range(start_year, end_year + 1):
        sql = (
            f"SELECT gvkey, datadate, rdq FROM comp.fundq "
            f"WHERE datadate BETWEEN '{yr}-01-01' AND '{yr}-12-31' "
            f"AND indfmt='INDL' AND consol='C' "
            f"AND popsrc='D' AND datafmt='STD' "
            f"AND rdq IS NOT NULL"
        )
        try:
            chunk = wrds_direct.raw_sql(sql, account="${WRDS_USER_2}")
        except Exception as exc:
            logger.warning("fundq chunk year=%d failed: %s", yr, exc)
            continue
        if chunk is None or chunk.empty:
            continue
        chunks.append(chunk)
        logger.info("fundq year=%d: %d rdq rows", yr, len(chunk))
    if not chunks:
        raise RuntimeError("All fundq chunks empty/failed")
    df = pd.concat(chunks, ignore_index=True)
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["rdq"]      = pd.to_datetime(df["rdq"])
    logger.info("Total fundq rdq: %d rows, %d gvkeys, %s → %s",
                  len(df), df.gvkey.nunique(),
                  df.rdq.min(), df.rdq.max())
    return df


def _add_knowable_at(funda: pd.DataFrame,
                       fundq_rdq: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN funda on fundq (gvkey, datadate) to get rdq.
    knowable_at = COALESCE(rdq, datadate + 120 days)."""
    funda["datadate"] = pd.to_datetime(funda["datadate"])
    merged = funda.merge(fundq_rdq[["gvkey", "datadate", "rdq"]],
                            on=["gvkey", "datadate"], how="left")

    fallback = merged["datadate"] + pd.Timedelta(
        days=_FALLBACK_LAG_DAYS)
    raw_knowable = merged["rdq"].fillna(fallback)

    # DEFENSIVE FLOOR (fix 2026-06-08 for 1790 Compustat-data
    # rows where rdq < datadate — impossible publication timing
    # would create silent look-ahead). Enforce knowable_at >=
    # datadate + 1 day at minimum.
    floor = merged["datadate"] + pd.Timedelta(days=1)
    n_floored = (raw_knowable < floor).sum()
    merged["knowable_at"] = raw_knowable.where(
        raw_knowable >= floor, floor)
    if n_floored > 0:
        logger.warning(
            "Floored %d (%.4f%%) rows where rdq < datadate "
            "(Compustat data quirk; enforced knowable_at >= "
            "datadate+1d to prevent silent look-ahead).",
            n_floored, 100 * n_floored / len(merged),
        )

    n_rdq      = merged["rdq"].notna().sum()
    n_fallback = merged["rdq"].isna().sum()
    logger.info(
        "knowable_at coverage: %d rows from rdq (%.1f%%) + "
        "%d fallback +120d (%.1f%%)",
        n_rdq, 100 * n_rdq / len(merged),
        n_fallback, 100 * n_fallback / len(merged),
    )

    # Validation: knowable_at should NEVER be < datadate after floor
    bad = (merged["knowable_at"] < merged["datadate"]).sum()
    if bad > 0:
        raise RuntimeError(
            f"BUG: {bad} rows STILL have knowable_at < datadate "
            f"after floor — investigate")
    # Validation: knowable_at - datadate should be reasonable
    lag_days = (merged["knowable_at"] - merged["datadate"]).dt.days
    logger.info("Publication lag (days): mean=%.1f median=%.1f "
                  "p10=%.0f p90=%.0f",
                  lag_days.mean(), lag_days.median(),
                  lag_days.quantile(0.10), lag_days.quantile(0.90))

    return merged.drop(columns=["rdq"])   # rdq merged into knowable_at


def main():
    if not _PIT_PATH.is_file():
        raise FileNotFoundError(
            f"PIT funda cache missing: {_PIT_PATH}. "
            f"Run scripts/extend_compustat_funda_pit_history.py first.")
    funda = pd.read_parquet(_PIT_PATH)
    if "knowable_at" in funda.columns:
        logger.info("knowable_at column already present in %s; "
                      "delete file + re-run if you want to refresh.",
                      _PIT_PATH.name)
        return 0

    logger.info("Loaded PIT funda: %d rows", len(funda))
    start = funda["datadate"].min().strftime("%Y-%m-%d")
    end   = funda["datadate"].max().strftime("%Y-%m-%d")
    logger.info("Pulling comp.fundq rdq for %s → %s...", start, end)

    fundq_rdq = _pull_fundq_rdq(start, end)
    enriched  = _add_knowable_at(funda, fundq_rdq)
    enriched.to_parquet(_PIT_PATH, index=False)
    sz_mb = _PIT_PATH.stat().st_size / 1e6
    logger.info("WROTE %s | %.1f MB | knowable_at column added",
                  _PIT_PATH, sz_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
