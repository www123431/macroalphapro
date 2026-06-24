"""
engine/d_pead_plus/universe.py — Top-1500 mcap point-in-time universe.

Spec id=74 §2.1 LOCK (amendment 1 2026-05-13):
  Compustat fundq quarter-end snapshot per gvkey:
    Take latest fundq row per gvkey with rdq <= quarter_end
    mcap_proxy = prccq × cshoq (price × shares outstanding, USD millions)
    Rank top-1500 by mcap_proxy
  US filter via crsp.msenames shrcd IN (10, 11), exchcd IN (1, 3)
    (share class filter via msenames; msenames data lag less critical than
     msf monthly returns; share class changes monthly not daily)

Original spec (pre-amendment 1) used crsp.msf for universe; pivoted due to
NUS WRDS CRSP data lag (max 2024-12-31).

DOCTRINE: This module is part of the DECISION LAYER context — must remain
deterministic and free of LLM calls.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# Spec id=74 §2.1 LOCKED
UNIVERSE_TOP_N_LOCKED:     int             = 1500
LOCKED_EXCH_CODES:         tuple[int, ...] = (1, 3)        # NYSE, NASDAQ
LOCKED_SHARE_CODES:        tuple[int, ...] = (10, 11)      # US common, excludes ADRs


@dataclass(frozen=True)
class UniverseQuarter:
    """Top-1500 universe membership for one quarter end."""
    quarter_end:    datetime.date
    permnos:        list[int]
    n_firms:        int
    mcap_median_m:  float       # USD millions
    mcap_p10_m:     float       # 10th percentile mcap (smallest in universe)


def fetch_top_1500_universe_for_quarter(
    quarter_end: datetime.date,
    *,
    conn = None,
) -> UniverseQuarter:
    """Pull top-1500 mcap NYSE/NASDAQ US-common-stock as of quarter-end.

    Args:
        quarter_end: e.g. datetime.date(2024, 6, 30)
        conn: optional pre-opened WRDS connection; if None, opens a new one

    Returns:
        UniverseQuarter with permnos list + summary mcap stats
    """
    own_conn = conn is None
    if conn is None:
        from engine.universe_singlename.crsp_loader import _open_wrds_connection
        conn = _open_wrds_connection()
    try:
        # Amendment 1: use Compustat fundq for mcap proxy (CRSP data lag bypass)
        # Latest fundq row per gvkey with rdq within ~120 days of quarter_end
        rdq_lookback_start = (quarter_end - datetime.timedelta(days=120)).isoformat()
        sql_fundq = (
            f"SELECT f.gvkey, f.rdq, f.prccq, f.cshoq, "
            f"       (f.prccq * f.cshoq) AS mcap_proxy_m "
            f"FROM comp.fundq f "
            f"WHERE f.rdq BETWEEN '{rdq_lookback_start}' AND '{quarter_end.isoformat()}' "
            f"  AND f.prccq IS NOT NULL AND f.cshoq IS NOT NULL "
            f"  AND f.prccq > 0 AND f.cshoq > 0 "
            f"  AND f.indfmt = 'INDL' AND f.consol = 'C' AND f.popsrc = 'D' AND f.datafmt = 'STD' "
            f"ORDER BY f.gvkey, f.rdq DESC"
        )
        df_fundq = conn.raw_sql(sql_fundq)
        if df_fundq.empty:
            logger.warning("fetch_top_1500_universe_for_quarter %s: empty fundq", quarter_end)
            return UniverseQuarter(quarter_end=quarter_end, permnos=[], n_firms=0,
                                    mcap_median_m=float("nan"), mcap_p10_m=float("nan"))

        # Latest row per gvkey
        df_fundq["gvkey"] = df_fundq["gvkey"].astype(str)
        df_fundq["mcap_proxy_m"] = pd.to_numeric(df_fundq["mcap_proxy_m"], errors="coerce")
        df_fundq = df_fundq.dropna(subset=["mcap_proxy_m"])
        latest = df_fundq.groupby("gvkey", as_index=False).first()  # ORDER BY rdq DESC gave us latest first

        # Link gvkey → permno via crsp.ccmxpf_lnkhist
        gvkey_csv = ",".join(f"'{g}'" for g in latest["gvkey"].tolist())
        sql_link = (
            f"SELECT lnk.gvkey, lnk.lpermno AS permno "
            f"FROM crsp.ccmxpf_lnkhist lnk "
            f"WHERE lnk.gvkey IN ({gvkey_csv}) "
            f"  AND lnk.linktype IN ('LU', 'LC') "
            f"  AND lnk.linkdt <= '{quarter_end.isoformat()}' "
            f"  AND (lnk.linkenddt IS NULL OR lnk.linkenddt >= '{quarter_end.isoformat()}') "
            f"  AND lnk.lpermno IS NOT NULL"
        )
        df_link = conn.raw_sql(sql_link)
        df_link["gvkey"] = df_link["gvkey"].astype(str)
        df_link["permno"] = df_link["permno"].astype(int)
        df_link = df_link.drop_duplicates(subset=["gvkey"], keep="first")

        # Join and filter via msenames (US common, NYSE/NASDAQ)
        merged = latest.merge(df_link, on="gvkey", how="inner")
        if merged.empty:
            return UniverseQuarter(quarter_end=quarter_end, permnos=[], n_firms=0,
                                    mcap_median_m=float("nan"), mcap_p10_m=float("nan"))

        permno_csv = ",".join(str(int(p)) for p in merged["permno"].tolist())
        sql_msenames = (
            f"SELECT DISTINCT permno FROM crsp.msenames "
            f"WHERE permno IN ({permno_csv}) "
            f"  AND shrcd IN {LOCKED_SHARE_CODES} "
            f"  AND exchcd IN {LOCKED_EXCH_CODES}"
        )
        df_us = conn.raw_sql(sql_msenames)
        df_us["permno"] = df_us["permno"].astype(int)
        us_permnos = set(df_us["permno"].tolist())
        merged = merged[merged["permno"].isin(us_permnos)]

        # Rank top-N by mcap
        merged = merged.sort_values("mcap_proxy_m", ascending=False).head(UNIVERSE_TOP_N_LOCKED)
        df = merged[["permno", "mcap_proxy_m"]].rename(columns={"mcap_proxy_m": "mcap_k"})
        df["mcap_k"] = df["mcap_k"] * 1000.0  # convert to thousands USD for downstream consistency
    finally:
        if own_conn:
            conn.close()

    if df.empty:
        logger.warning("fetch_top_1500_universe_for_quarter %s: empty result", quarter_end)
        return UniverseQuarter(
            quarter_end=quarter_end, permnos=[], n_firms=0,
            mcap_median_m=float("nan"), mcap_p10_m=float("nan"),
        )

    df["permno"] = df["permno"].astype(int)
    df["mcap_m"] = pd.to_numeric(df["mcap_k"], errors="coerce") / 1000.0  # k → millions
    df = df.dropna(subset=["mcap_m"])

    permnos = df["permno"].tolist()
    return UniverseQuarter(
        quarter_end   = quarter_end,
        permnos       = permnos,
        n_firms       = len(permnos),
        mcap_median_m = float(df["mcap_m"].median()),
        mcap_p10_m    = float(df["mcap_m"].quantile(0.10)),
    )


def fetch_universe_for_window(
    window_start: datetime.date,
    window_end:   datetime.date,
) -> dict[datetime.date, UniverseQuarter]:
    """Pull top-1500 universe for each quarter end in [window_start, window_end].

    Spec id=74 §2.2 LOCK window: 2024-04-01 to 2026-06-30 (9 quarter-ends).
    """
    # Generate quarter-end dates within window
    quarter_ends: list[datetime.date] = []
    q_end_months = (3, 6, 9, 12)
    q_end_days   = {3: 31, 6: 30, 9: 30, 12: 31}

    cur_year = window_start.year
    while cur_year <= window_end.year:
        for m in q_end_months:
            d = datetime.date(cur_year, m, q_end_days[m])
            if window_start <= d <= window_end:
                quarter_ends.append(d)
        cur_year += 1

    logger.info("fetch_universe_for_window: %d quarter-ends from %s to %s",
                len(quarter_ends), window_start, window_end)

    from engine.universe_singlename.crsp_loader import _open_wrds_connection
    conn = _open_wrds_connection()
    try:
        out: dict[datetime.date, UniverseQuarter] = {}
        for qe in quarter_ends:
            u = fetch_top_1500_universe_for_quarter(qe, conn=conn)
            out[qe] = u
            logger.info("  %s: %d firms, median mcap $%.0fM",
                        qe, u.n_firms, u.mcap_median_m)
    finally:
        conn.close()
    return out


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2024-04-01")
    p.add_argument("--end",   type=str, default="2026-06-30")
    args = p.parse_args()
    univ = fetch_universe_for_window(
        datetime.date.fromisoformat(args.start),
        datetime.date.fromisoformat(args.end),
    )
    print(f"\nPulled {len(univ)} quarters:")
    for qe, u in sorted(univ.items()):
        print(f"  {qe}: n={u.n_firms} median_mcap=${u.mcap_median_m:.0f}M p10=${u.mcap_p10_m:.0f}M")
