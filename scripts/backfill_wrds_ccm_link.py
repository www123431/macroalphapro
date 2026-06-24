"""scripts/backfill_wrds_ccm_link.py — cache the CRSP-Compustat link table.

The single substrate gap blocking C1-C3 Composer components for equity
factors. Without crsp.ccmxpf_lnkhist (gvkey↔permno linkage), the cached
compustat_funda (book equity, profitability fields) cannot be joined
to cached CRSP returns. After this backfill, building VALUE_BOOK_TO_MARKET
/ QUALITY_QMJ / PROFITABILITY_GROSS becomes pure offline work.

Per F13 audit (2026-06-05): 25+ FACTOR_HYPOTHESIS specs in our corpus
need Compustat fundamentals to build. All are blocked on this single
table. One-shot WRDS query → ~70k rows → ~MB parquet.

Output: data/cache/_crsp_ccm_link.parquet
Schema: gvkey, lpermno (a.k.a. permno), linkdt, linkenddt, linktype, linkprim
        (filtered to LU/LC + P/C as the de-facto WRDS standard for
         "primary, usable" links)

Run:
  python scripts/backfill_wrds_ccm_link.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_OUT_PATH = REPO_ROOT / "data" / "cache" / "_crsp_ccm_link.parquet"


def main() -> int:
    # Probe WRDS first — fail loud if unauthenticated, don't silently produce empty
    try:
        from engine.line_c import wrds_direct
    except ImportError as exc:
        logger.error("wrds_direct unavailable: %s", exc)
        return 2

    logger.info("Probing WRDS via wrds_direct...")
    t0 = time.time()
    try:
        probe = wrds_direct.raw_sql(
            "SELECT 1 AS ok FROM crsp.ccmxpf_lnkhist LIMIT 1",
            account="${WRDS_USER_2}",
        )
    except Exception as exc:
        logger.error("WRDS probe failed: %s", exc)
        return 3
    if probe is None or probe.empty:
        logger.error("WRDS probe returned empty — auth or schema issue")
        return 3
    logger.info("Probe OK in %.1fs", time.time() - t0)

    # Full LU/LC primary link extract. Filter to LU (USEDIT — universe of
    # securities to use) + LC (primary-link confirmed by CRSP-Compustat),
    # linkprim P (primary identifier) + C (overrides primary). This is the
    # standard "join-safe" subset — see CRSP/Compustat Linking Reference.
    sql = """
        SELECT
            gvkey,
            lpermno  AS permno,
            linkdt,
            linkenddt,
            linktype,
            linkprim
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
          AND lpermno IS NOT NULL
    """
    logger.info("Fetching crsp.ccmxpf_lnkhist (filtered to LU/LC + P/C)...")
    t0 = time.time()
    try:
        df = wrds_direct.raw_sql(sql, account="${WRDS_USER_2}")
    except Exception as exc:
        logger.error("Link table fetch failed: %s", exc)
        return 4
    elapsed = time.time() - t0

    if df is None or df.empty:
        logger.error("Link table returned empty — investigate query/filters")
        return 4

    # NaT-tolerant date handling: linkenddt can be 'E' (end-of-time
    # sentinel) in raw Compustat — WRDS API typically returns NaT, but
    # be explicit so downstream comparisons don't surprise.
    import pandas as pd
    for col in ("linkdt", "linkenddt"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Coerce IDs to consistent types
    df["gvkey"]  = df["gvkey"].astype(str).str.zfill(6)
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT_PATH, index=False)
    logger.info(
        "Saved %s — %d rows, %.1fMB — fetch took %.1fs",
        _OUT_PATH, len(df), _OUT_PATH.stat().st_size / 1024 / 1024, elapsed,
    )

    # Brief sanity stats
    n_gvkeys = df["gvkey"].nunique()
    n_permnos = df["permno"].dropna().nunique()
    open_links = df["linkenddt"].isna().sum()
    logger.info(
        "  %d unique gvkeys / %d unique permnos / %d still-open links",
        n_gvkeys, n_permnos, open_links,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
