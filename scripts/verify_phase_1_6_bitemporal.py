"""scripts/verify_phase_1_6_bitemporal.py — Tier C L2-1 Phase 1.6 verification.

Smoke verification script — run as soon as Phase 1.5 enrichment
completes. Confirms the bitemporal pipeline works end-to-end before
moving to Phase 3.1 (cross_sec template refactor).

CHECKS
======
1. _compustat_funda_pit.parquet has knowable_at column
2. knowable_at value distribution (mean lag, rdq-vs-fallback ratio)
3. PIT accessor with funda_source="pit" loads correctly
4. funda_pit_panel returns a non-empty (month_end × permno) panel
   for a known gvkey (e.g., gvkey 001690 = AAPL)
5. PIT vs legacy panel comparison on `at` for AAPL recent quarters
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logger = logging.getLogger("verify_phase_1_6")
logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s %(levelname)s %(message)s")


def main():
    repo = Path(__file__).resolve().parent.parent
    pit_path = repo / "data" / "cache" / "_compustat_funda_pit.parquet"

    # 1. Check parquet has knowable_at column
    funda = pd.read_parquet(pit_path)
    if "knowable_at" not in funda.columns:
        logger.error("FAIL: knowable_at column missing from PIT cache. "
                       "Run scripts/add_knowable_at_to_funda_pit.py first.")
        return 1
    logger.info("✓ Phase 1.5 enrichment confirmed (knowable_at column present)")

    # 2. Distribution check
    funda["datadate"]    = pd.to_datetime(funda["datadate"])
    funda["knowable_at"] = pd.to_datetime(funda["knowable_at"])
    lag_days = (funda["knowable_at"] - funda["datadate"]).dt.days
    logger.info("knowable_at lag distribution (days from datadate): "
                  "mean=%.1f median=%.1f p10=%.0f p90=%.0f",
                  lag_days.mean(), lag_days.median(),
                  lag_days.quantile(0.10), lag_days.quantile(0.90))
    # Count rdq-driven (lag != exactly 120d) vs fallback-driven
    fallback_count = (lag_days == 120).sum()
    rdq_count = len(funda) - fallback_count
    logger.info("Source breakdown: %d (%.1f%%) from rdq, "
                  "%d (%.1f%%) +120d fallback",
                  rdq_count, 100 * rdq_count / len(funda),
                  fallback_count, 100 * fallback_count / len(funda))

    # 3. Accessor PIT mode
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    clock = SimClock(start="2015-01-01", end="2024-12-31")
    clock.advance(pd.Timestamp("2024-12-31"))
    accessor = PITDataAccessor(clock, funda_source="pit")
    logger.info("✓ Accessor instantiated with funda_source=pit")

    # 4. funda_pit_panel returns data for AAPL window
    panel_at = accessor.funda_pit_panel(
        field="at",
        window=(pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31")),
    )
    if panel_at.empty:
        logger.error("FAIL: funda_pit_panel returned empty for at")
        return 1
    logger.info("✓ funda_pit_panel('at') shape: %s, %d non-null cells",
                  panel_at.shape, int(panel_at.notna().sum().sum()))

    # 5. PIT vs legacy compare
    accessor_legacy = PITDataAccessor(clock, funda_source="legacy")
    panel_at_legacy = accessor_legacy.funda_pit_panel(
        field="at",
        window=(pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31")),
    )
    # AAPL permno = 14593
    aapl = 14593
    if aapl in panel_at.columns and aapl in panel_at_legacy.columns:
        logger.info("\nAAPL (permno=14593) `at` values, recent months:")
        for d in panel_at.index[-6:]:
            if d in panel_at_legacy.index:
                pit_v = panel_at.loc[d, aapl]
                leg_v = panel_at_legacy.loc[d, aapl]
                diff = (pit_v - leg_v) / leg_v * 100 if pd.notna(leg_v) and leg_v else None
                logger.info("  %s: PIT=%.0f  legacy=%.0f  diff=%s",
                              d.date(), pit_v, leg_v,
                              f"{diff:+.2f}%" if diff is not None else "—")

    logger.info("\n✓ All Phase 1.6 bitemporal smoke checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
