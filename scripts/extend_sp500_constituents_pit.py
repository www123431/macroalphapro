"""scripts/extend_sp500_constituents_pit.py — Tier C L2-1 Phase 1.4.

One-shot PIT S&P 500 constituents pull from crsp.dsp500list.
Unlocks survivor-bias-free universe selection (fix for B2).

WHY THIS EXISTS
===============
Tier C templates currently support `us_equities_top_3000` (top-3000
by mktcap) but NOT `us_equities_sp500` (S&P 500 constituents PIT).
For S&P 500 anomalies (e.g., index addition/deletion effect, post-
2000 large-cap value tilt studies), we need point-in-time
membership — which crsp.dsp500list provides.

DATA STRUCTURE
==============
crsp.dsp500list (one row = one permno's membership interval):
  permno   integer    CRSP identifier
  start    date       inclusion date in S&P 500
  ending   date       exclusion date (NULL/MAX_DATE = still in)

For any backtest month-end t, S&P 500 membership = set of permnos
where start <= t <= ending. ~500 permnos at any given t.

This is FULL HISTORY 1925-2024, ~2K rows.

USAGE BY TIER C PIT ACCESSOR
=============================
PITDataAccessor.universe_sp500_constituents(as_of: pd.Timestamp)
  → set[int] of permnos in SP500 as of as_of
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logger = logging.getLogger("extend_sp500_constituents_pit")
logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s %(levelname)s %(message)s")


_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = (_REPO_ROOT / "data" / "cache"
                  / "_sp500_constituents_pit.parquet")


def main():
    if _OUTPUT_PATH.is_file():
        df_existing = pd.read_parquet(_OUTPUT_PATH)
        logger.info("Cache exists: %s | shape=%s; skipping. "
                      "Delete file + re-run to refresh.",
                      _OUTPUT_PATH.name, df_existing.shape)
        return 0

    from engine.line_c import wrds_direct
    logger.info("Fetching crsp.dsp500list (small ~2K rows)...")
    df = wrds_direct.raw_sql(
        "SELECT permno, start, ending FROM crsp.dsp500list "
        "ORDER BY permno, start",
        account="${WRDS_USER_2}",
    )
    if df is None or df.empty:
        raise RuntimeError("dsp500list pull empty — check WRDS creds")

    df["permno"] = df["permno"].astype(int)
    df["start"]  = pd.to_datetime(df["start"])
    df["ending"] = pd.to_datetime(df["ending"])
    # NaT in ending → still in S&P 500. Sentinel: 2100-01-01
    df["ending"] = df["ending"].fillna(pd.Timestamp("2100-01-01"))

    logger.info(
        "Fetched %d membership intervals | %d unique permnos | "
        "%s → %s",
        len(df), df.permno.nunique(),
        df["start"].min(), df.ending.replace(
            pd.Timestamp("2100-01-01"), pd.NaT).max(),
    )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_PATH, index=False)
    sz_kb = _OUTPUT_PATH.stat().st_size / 1024.0
    logger.info("WROTE %s | %.1f KB", _OUTPUT_PATH, sz_kb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
