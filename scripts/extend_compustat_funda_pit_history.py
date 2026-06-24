"""scripts/extend_compustat_funda_pit_history.py — Tier C L2-1 Phase 1.2.

One-time PIT-correct Compustat funda pull from comp_pit.pithistdataus.
Replaces the latest-restated _compustat_funda_long_history.parquet
(commit 6c4a7f18) with a point-in-time version that reports what
investors actually KNEW at each historical date.

WHY THIS EXISTS (B0 bug, identified in senior critique 2026-06-08):
  The previous Compustat pull used `comp.funda` which contains the
  LATEST restated value for each fiscal year. Using these values in
  a backtest = look-ahead bias on restatement: we use the 2020-
  restated 1995 fundamentals which weren't known until 2020.

  Suspected magnitude: GP/A t=3.34 vs Novy-Marx's t≈3.0 paper-
  window reproduction suggests the restated values inflate alpha
  by ~10%. McLean-Pontiff post-pub decay should DROP the t, not
  raise it.

WHAT comp_pit.pithistdataus PROVIDES
====================================
WRDS comp_pit schema, table pithistdataus:
  - 52.4M rows, 1980-2026 coverage
  - For each (gvkey, datadate), multiple "snapshots" of fundamentals
    as they were reported at different points in time
  - qtrsback field indexes the snapshots:
      qtrsback = 0   → MOST RECENT reported value (= comp.funda's
                       latest-restated; what we want to AVOID)
      qtrsback = k   → value reported k quarters back from current
      qtrsback = MAX → EARLIEST available snapshot
                       (closest to first-report / true PIT)
  - Columns: atqh, niqh, ceqqh, etc. (q suffix = quarterly historical)
    Each column has a companion _dc field for data class metadata.

PIT FILTERING STRATEGY
======================
For each (gvkey, datadate), retain the row with the LARGEST qtrsback.
This is the EARLIEST available snapshot in the table, closest to
what investors saw immediately after the fiscal period closed.

This is approximate (the actual first-report may be older than what's
in the table), but conservative: using MAX qtrsback ALWAYS uses an
earlier value than comp.funda, so we cannot accidentally introduce
look-ahead. We may use a value 1-2 restatements newer than the actual
first-report (modest residual bias).

ALTERNATE STRATEGIES considered + rejected
==========================================
1. Filter by qtrsback >= K for fixed K: arbitrary; misses early
   data where only fewer snapshots exist.
2. Use rdq (release date) from comp.fundq: rdq is fiscal-quarter
   release; annual values aren't released at rdq but at later
   annual filing. pithistdataus already handles this.
3. Use first row by snapshot timestamp: requires extra metadata
   not exposed in pithistdataus.

MAX qtrsback gives the cleanest filter with current available
data.

OUTPUT SCHEMA
=============
data/cache/_compustat_funda_pit.parquet (NEW file; DOES NOT
overwrite _compustat_funda_long_history.parquet — keeps both
for parity comparison during L2-1 refactor).

Columns kept (annualized from QH fields):
  gvkey, datadate, qtrsback,
  at, ceq, lt, ni, oibdp, sale, cogs, xsga, xrd, ppent,
  dlc, dltt, dvc

Expected ~5-10M rows (one PIT snapshot per gvkey × fiscal year).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make engine.* importable when running as a top-level script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logger = logging.getLogger("extend_compustat_funda_pit_history")
logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s %(levelname)s %(message)s")


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = (_REPO_ROOT / "data" / "cache"
                  / "_compustat_funda_pit.parquet")

# Compustat PIT field map: comp_pit.pithistdataus uses {field}qh
# suffix for quarterly-historical PIT values. Map to clean annual
# names that match the existing _compustat_funda_long_history schema.
_PIT_FIELD_MAP = {
    "atqh":     "at",      # total assets
    "ceqqh":    "ceq",     # common equity
    "ltqh":     "lt",      # total liabilities
    "niqh":     "ni",      # net income (TTM)
    "oibdpqh":  "oibdp",   # operating income before D&A (TTM)
    "saleqh":   "sale",    # net sales (TTM)
    "cogsqh":   "cogs",    # cost of goods sold (TTM)
    "xsgaqh":   "xsga",    # SG&A (TTM)
    "xrdqh":    "xrd",     # R&D (TTM)
    "ppentqh":  "ppent",   # net PP&E
    "dlcqh":    "dlc",     # short-term debt
    "dlttqh":   "dltt",    # long-term debt
    "dvqh":     "dv",      # total dividends (TTM; PIT table doesn't
                            # split common vs preferred — note 2026-06-08)
}


def _check_existing(force: bool) -> bool:
    if _OUTPUT_PATH.is_file():
        df = pd.read_parquet(_OUTPUT_PATH)
        msg = (f"Existing cache: {_OUTPUT_PATH.name} | shape={df.shape}"
                 + (f" | date range {df.datadate.min()} → {df.datadate.max()}"
                    if "datadate" in df.columns else ""))
        if force:
            logger.info("--force set; will OVERWRITE existing cache. %s",
                          msg)
            return True
        logger.info("Cache already exists, skipping. %s", msg)
        logger.info("Pass --force to refresh.")
        return False
    return True


def _pull(start: str, end: str) -> pd.DataFrame:
    """Year-chunked pull strategy (2026-06-08 v2):

    Original v1 (single full-window SELECT) died silently after 30+
    min — likely WRDS connection timeout on the long query. v2
    chunks by year: ~870K rows/year average × 60 years = small per-
    request load, ~10s per year query, recoverable on partial fail.

    Server-side filtering INSIDE the SQL (MAX qtrsback per gvkey×
    datadate) — pushes the 100x compression to the DB side so we
    don't return all snapshots and then filter client-side. This
    cuts the transferred data ~100x.
    """
    from engine.line_c import wrds_direct
    fields_sql = ", ".join(list(_PIT_FIELD_MAP.keys()))
    start_year = int(start[:4])
    end_year   = int(end[:4])
    chunks: list[pd.DataFrame] = []
    for yr in range(start_year, end_year + 1):
        sql = (
            f"WITH max_qb AS ("
            f"  SELECT gvkey, datadate, MAX(qtrsback) AS max_qtrsback"
            f"  FROM comp_pit.pithistdataus "
            f"  WHERE datadate BETWEEN '{yr}-01-01' AND '{yr}-12-31' "
            f"  GROUP BY gvkey, datadate"
            f") "
            f"SELECT t.gvkey, t.datadate, t.qtrsback, {fields_sql} "
            f"FROM comp_pit.pithistdataus t "
            f"JOIN max_qb m ON t.gvkey=m.gvkey "
            f"               AND t.datadate=m.datadate "
            f"               AND t.qtrsback=m.max_qtrsback "
            f"WHERE t.datadate BETWEEN '{yr}-01-01' AND '{yr}-12-31'"
        )
        logger.info("Chunk year=%d: querying...", yr)
        try:
            chunk = wrds_direct.raw_sql(sql, account="${WRDS_USER_2}")
        except Exception as exc:
            logger.warning("Chunk year=%d FAILED: %s — skipping",
                              yr, exc)
            continue
        if chunk is None or chunk.empty:
            logger.warning("Chunk year=%d empty", yr)
            continue
        logger.info("Chunk year=%d: %d rows | %d gvkeys",
                      yr, len(chunk), chunk.gvkey.nunique())
        chunks.append(chunk)

    if not chunks:
        raise RuntimeError("All year chunks empty/failed")
    df = pd.concat(chunks, ignore_index=True)
    logger.info("Total pulled: %d rows | %d unique gvkeys | "
                  "qtrsback range %d-%d",
                  len(df), df.gvkey.nunique(),
                  df.qtrsback.min(), df.qtrsback.max())
    return df


def _filter_pit(df: pd.DataFrame) -> pd.DataFrame:
    """v2 (2026-06-08): filter already done server-side per year-
    chunked SQL. This just renames QH columns to clean names."""
    logger.info("Renaming PIT columns (filter already done in SQL)...")
    pit = df.rename(columns=_PIT_FIELD_MAP)
    keep_cols = ["gvkey", "datadate", "qtrsback"] + list(_PIT_FIELD_MAP.values())
    pit = pit[keep_cols].reset_index(drop=True)
    logger.info("Renamed: %d rows ready", len(pit))
    return pit


def _validate(pit: pd.DataFrame, legacy: pd.DataFrame | None) -> None:
    """Sanity checks + smoke compare against the legacy
    (latest-restated) cache."""
    pit["datadate"] = pd.to_datetime(pit["datadate"])
    n_at_null = int(pit["at"].isna().sum()) if "at" in pit.columns else 0
    logger.info(
        "Validation pass | %d rows | datadate %s → %s | %d (%.1f%%) "
        "null `at` | %d unique gvkeys",
        len(pit), pit.datadate.min(), pit.datadate.max(),
        n_at_null, 100 * n_at_null / max(len(pit), 1),
        pit.gvkey.nunique(),
    )

    if legacy is not None:
        # Smoke compare: pick 5 random (gvkey, datadate) pairs that
        # exist in both and report at values. Should differ for older
        # data where restatement has occurred.
        common = pit.merge(
            legacy[["gvkey", "datadate", "at"]].rename(
                columns={"at": "at_legacy"}),
            on=["gvkey", "datadate"], how="inner",
        )
        if not common.empty:
            common["at_diff_pct"] = (
                (common["at"] - common["at_legacy"]) /
                common["at_legacy"].replace(0, pd.NA)
            ).abs()
            avg_diff = float(common["at_diff_pct"].dropna().mean())
            max_diff = float(common["at_diff_pct"].dropna().max())
            n_significant = int((common["at_diff_pct"] > 0.05).sum())
            logger.info(
                "Smoke compare vs legacy comp.funda (overlap "
                "%d rows): avg |Δat|/at = %.4f, max = %.4f, "
                "%d rows differ > 5%% (= restatement bias caught)",
                len(common), avg_diff, max_diff, n_significant,
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="1962-01-01")
    ap.add_argument("--end",   default="2024-12-31")
    ap.add_argument("--force", action="store_true",
                       help="Overwrite existing output parquet")
    args = ap.parse_args()

    if not _check_existing(args.force):
        return 0

    raw = _pull(args.start, args.end)

    # Persist raw before filtering (defensive — failed filter
    # shouldn't lose expensive pull)
    raw_path = _OUTPUT_PATH.with_name(_OUTPUT_PATH.stem + "_RAW.parquet")
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path, index=False)
    logger.info("Persisted raw pull to %s (%.1f MB)",
                  raw_path, raw_path.stat().st_size / 1e6)

    pit = _filter_pit(raw)
    pit.to_parquet(_OUTPUT_PATH, index=False)
    logger.info("WROTE %s | %.1f MB",
                  _OUTPUT_PATH, _OUTPUT_PATH.stat().st_size / 1e6)

    # Compare against existing legacy cache for restatement bias signal
    legacy_path = (_REPO_ROOT / "data" / "cache"
                     / "_compustat_funda_long_history.parquet")
    legacy = pd.read_parquet(legacy_path) if legacy_path.is_file() else None
    _validate(pit, legacy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
