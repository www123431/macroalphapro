"""
engine/path_c/earnings_panel.py — Path C #1 PEAD firm-quarter panel builder.

Pre-registration: docs/spec_path_c_earnings_pead_v1.md (id=57) §2.2 + §2.3

Pulls + joins:
  - I/B/E/S ibes.det_epsus (analyst forecasts, for consensus_median + dispersion)
  - I/B/E/S ibes.act_epsus (realized actuals)
  - Compustat comp.fundq (rdq announcement date, gvkey + fiscal period)
  - CRSP linkage via crsp.ccmxpf_lnkhist + ibes.id (gvkey ↔ permno ↔ I/B/E/S ticker)

Output schema (firm-quarter long form):
  permno, ticker_ibes, gvkey, fiscal_yearq, rdq,
  actual_eps, consensus_median, consensus_dispersion, n_analysts,
  market_cap_at_q (point-in-time at quarter-start, for top-N filter)

This module is RAW PANEL BUILDER ONLY. SUE computation lives in
engine/path_c/sue_signal.py (Sprint 3). Decile portfolio formation lives in
engine/path_c/pead_backtest.py (Sprint 4).

Two modes:
  - mock_mode=True (default when WRDS unavailable): deterministic synthetic panel
  - mock_mode=False: real WRDS query, decorated with with_wrds_retry

Disk cache: parquet per query at data/path_c/_*.parquet.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.path_c import (
    CONSENSUS_LOCK_WINDOW_DAYS,
    MIN_ANALYSTS_REQUIRED,
)
from engine.universe_singlename.crsp_loader import (
    is_wrds_available as _crsp_is_wrds_available,
    _open_wrds_connection as _crsp_open_wrds_connection,
)
from engine.universe_singlename.wrds_retry import with_wrds_retry

logger = logging.getLogger(__name__)


# ── Storage ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _REPO_ROOT / "data" / "path_c"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_EARNINGS_PANEL_CACHE_PATH = _CACHE_DIR / "_earnings_panel.parquet"


# ── Public types ────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class EarningsPanelResult:
    """Output of bulk_fetch_earnings_panel.

    `panel` columns:
      - permno (int)
      - ticker_ibes (str): I/B/E/S ticker symbol
      - gvkey (int): Compustat firm identifier
      - fiscal_yearq (str): e.g. "2014Q1"
      - rdq (datetime.date): Compustat quarterly earnings report date
      - actual_eps (float): realized EPS per I/B/E/S act_epsus
      - consensus_median (float): median of analyst forecasts in 90d pre-rdq window
      - consensus_dispersion (float): std of analyst forecasts (same window)
      - n_analysts (int): count of unique analysts in window
      - market_cap_at_q (float): point-in-time market cap at quarter-start, USD millions
    """
    panel:           pd.DataFrame
    mode:            str          # "mock" or "wrds"
    n_firm_quarters: int
    exclusion_stats: dict          # breakdown by reason
    window_start:    datetime.date
    window_end:      datetime.date


# ── WRDS availability check (reuse crsp_loader's) ──────────────────────────
def is_wrds_available() -> bool:
    """Delegate to crsp_loader's WRDS availability check.

    Returns True if wrds Python lib + credentials are configured. False means
    callers must use mock_mode=True or fail.
    """
    return _crsp_is_wrds_available()


# ── Mock-mode panel generator ───────────────────────────────────────────────
def _mock_earnings_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Generate a deterministic synthetic firm-quarter earnings panel.

    Reproducible across runs + machines (ticker hash → RNG seed). Produces:
      - quarterly rdq dates roughly aligned with calendar quarter-ends + ~25-45d
      - actual EPS sampled from N(0.50, 0.30) per firm (lognormal-adjacent)
      - consensus_median = actual + bias (sampled per firm) + N(0, 0.05) noise
      - dispersion sampled from |N(0.10, 0.03)|
      - n_analysts from 3-15 uniform
      - market_cap_at_q drifting GBM around $5B-$50B
    """
    if not tickers:
        return pd.DataFrame()

    rows = []
    # Calendar quarters covered by [start_date, end_date]
    q_starts = pd.date_range(start=start_date, end=end_date, freq="QS").to_pydatetime()
    if len(q_starts) == 0:
        return pd.DataFrame()

    for ticker in sorted(set(tickers)):
        seed = int(hashlib.md5(ticker.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        # Per-firm baseline (eps level, analyst-bias offset, mktcap baseline)
        base_eps   = float(rng.normal(0.50, 0.30))
        bias_eps   = float(rng.normal(0.00, 0.10))
        base_mcap  = float(rng.uniform(5_000.0, 50_000.0))  # USD millions

        gvkey_synth  = abs(seed) % 999_999
        permno_synth = (abs(seed) // 1000) % 99_999 + 10000  # 5-digit-ish permno

        for q_start in q_starts:
            q_start_date = q_start.date()
            quarter_label = f"{q_start_date.year}Q{(q_start_date.month - 1) // 3 + 1}"
            # rdq is ~25-45 days after quarter-end (standard 10-Q filing window)
            # Quarter-end = q_start + 3 months - 1 day approximately
            q_end_offset_days = 90 + int(rng.integers(25, 46))
            rdq = q_start_date + datetime.timedelta(days=q_end_offset_days)
            if rdq > end_date:
                continue

            actual_eps   = base_eps + float(rng.normal(0.0, 0.08))
            forecasts    = actual_eps + bias_eps + rng.normal(0.0, 0.06, size=int(rng.integers(3, 16)))
            consensus_median = float(np.median(forecasts))
            dispersion       = float(np.std(forecasts, ddof=1)) if len(forecasts) > 1 else 0.0
            n_analysts       = int(len(forecasts))
            # Market cap drift: slow GBM-like across quarters
            quarter_idx = (q_start_date.year - start_date.year) * 4 + (q_start_date.month - 1) // 3
            mcap = base_mcap * float(np.exp(rng.normal(0.02, 0.10) * quarter_idx / 4.0))

            rows.append({
                "permno":               permno_synth,
                "ticker_ibes":          ticker,
                "gvkey":                gvkey_synth,
                "fiscal_yearq":         quarter_label,
                "rdq":                  rdq,
                "actual_eps":           actual_eps,
                "consensus_median":     consensus_median,
                "consensus_dispersion": dispersion,
                "n_analysts":           n_analysts,
                "market_cap_at_q":      mcap,
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["fiscal_yearq", "ticker_ibes"]).reset_index(drop=True)


# ── Real WRDS queries (activated when WRDS configured) ─────────────────────
# SQL templates — parameterized via raw_sql params to prevent injection.

# Compustat quarterly fundamentals: rdq + gvkey + fiscal period
_COMP_FUNDQ_RDQ_SQL = """
SELECT
    gvkey,
    fyearq,
    fqtr,
    datadate,
    rdq,
    cshoq,           -- common shares outstanding (millions)
    prccq            -- close price at fiscal period end
FROM comp.fundq
WHERE rdq BETWEEN %(start_date)s AND %(end_date)s
  AND gvkey IN %(gvkeys)s
  AND indfmt = 'INDL'
  AND datafmt = 'STD'
  AND popsrc = 'D'
  AND consol = 'C'
ORDER BY gvkey, fyearq, fqtr
"""

# I/B/E/S detail: analyst forecasts for EPS. Filtered to 90d-pre-rdq window
# at the join step (SQL pulls everything in range, Python filters per-firm).
#
# fpi codes in I/B/E/S = forecast HORIZON at time of forecast:
#   '6' = Quarter 1-ahead, '7' = Q2-ahead, ..., '11' = Q6-ahead
# We INCLUDE all quarterly horizons (6..11) — a forecast for Q1-2014 made when
# the analyst saw it as 2Q-ahead has fpi='7' at that time; we still want it.
# Initial implementation used fpi='6' only (rigor audit 2026-05-12 caught this
# as too narrow — clarification amendment +0 trials, no methodology change).
_IBES_DET_EPSUS_SQL = """
SELECT
    ticker,
    fpedats,         -- forecast period end date (the fiscal quarter being forecast)
    anndats,         -- forecast announcement date
    analys,          -- analyst code
    value            -- forecast EPS
FROM ibes.det_epsus
WHERE fpedats BETWEEN %(start_date)s AND %(end_date)s
  AND ticker IN %(tickers)s
  AND fpi IN ('6', '7', '8', '9', '10', '11')   -- all quarterly horizons
  AND value IS NOT NULL
ORDER BY ticker, fpedats, analys, anndats
"""

# I/B/E/S actuals: realized EPS
_IBES_ACT_EPSUS_SQL = """
SELECT
    ticker,
    pends,           -- period end date
    anndats,         -- announcement date of actual
    value            -- realized EPS
FROM ibes.act_epsus
WHERE pends BETWEEN %(start_date)s AND %(end_date)s
  AND ticker IN %(tickers)s
  AND pdicity = 'QTR'   -- quarterly actuals
  AND value IS NOT NULL
ORDER BY ticker, pends
"""

# I/B/E/S ↔ CRSP linkage: ibes.id provides ticker → cusip; crsp.stocknames
# provides cusip ↔ permno. Single join in SQL.
_IBES_PERMNO_LINK_SQL = """
SELECT DISTINCT
    i.ticker,
    s.permno
FROM ibes.id AS i
JOIN crsp.stocknames AS s
  ON SUBSTRING(i.cusip FROM 1 FOR 8) = SUBSTRING(s.ncusip FROM 1 FOR 8)
WHERE i.ticker IN %(tickers)s
  AND s.nameenddt >= %(start_date)s
  AND s.namedt   <= %(end_date)s
"""

# CRSP ↔ Compustat linkage: gvkey ↔ permno
_CRSP_COMP_LINK_SQL = """
SELECT
    gvkey,
    lpermno AS permno,
    linkdt,
    linkenddt
FROM crsp.ccmxpf_lnkhist
WHERE lpermno IN %(permnos)s
  AND linktype IN ('LU', 'LC')
  AND linkprim IN ('P', 'C')
  AND (linkenddt IS NULL OR linkenddt >= %(start_date)s)
  AND linkdt <= %(end_date)s
"""


@with_wrds_retry(max_attempts=3, base_delay=5.0)
def _real_earnings_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Build firm-quarter earnings panel from WRDS.

    Pipeline:
      1. Open WRDS connection
      2. ticker (I/B/E/S) → permno via ibes.id ⨝ crsp.stocknames on CUSIP
      3. permno → gvkey via crsp.ccmxpf_lnkhist
      4. Pull comp.fundq for those gvkeys, filtered to rdq in window
      5. Pull ibes.det_epsus for those tickers, fpedats in window
      6. Pull ibes.act_epsus for those tickers, pends in window
      7. Per firm-quarter:
         - Match fundq.datadate ↔ ibes.fpedats (=quarter-end)
         - Filter det_epsus to anndats ∈ [rdq - 90d, rdq - 1d]
         - Group by analyst, keep most-recent forecast per analyst in window
         - consensus_median = median(per_analyst_most_recent)
         - dispersion = std (ddof=1) of per_analyst_most_recent
         - n_analysts = count
      8. Compute market_cap_at_q = cshoq × prccq (millions × $/share)
      9. Return long-form DataFrame matching EarningsPanelResult.panel schema
    """
    if not is_wrds_available():
        raise RuntimeError(
            "WRDS not configured. Pass mock_mode=True for skeleton testing, "
            "or install wrds + configure credentials for real-data path."
        )
    if not tickers:
        return pd.DataFrame()

    needed_tickers = sorted(set(tickers))
    conn = _crsp_open_wrds_connection()
    try:
        # Step 1: ticker → permno via I/B/E/S → CRSP CUSIP linkage
        logger.info("path_c.earnings_panel: resolving %d tickers → permno via ibes.id",
                    len(needed_tickers))
        link_df = conn.raw_sql(
            _IBES_PERMNO_LINK_SQL,
            params={
                "tickers":    tuple(needed_tickers),
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
            },
        )
        if link_df.empty:
            logger.warning("path_c.earnings_panel: no permno linkages found")
            return pd.DataFrame()
        permnos = sorted(set(int(p) for p in link_df["permno"].dropna().tolist()))

        # Step 2: permno → gvkey via crsp.ccmxpf_lnkhist
        comp_link_df = conn.raw_sql(
            _CRSP_COMP_LINK_SQL,
            params={
                "permnos":    tuple(permnos),
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
            },
            date_cols=["linkdt", "linkenddt"],
        )
        if comp_link_df.empty:
            logger.warning("path_c.earnings_panel: no gvkey linkages found")
            return pd.DataFrame()
        gvkeys = sorted(set(comp_link_df["gvkey"].dropna().astype(str).tolist()))

        # Step 3: pull comp.fundq for rdq + market cap inputs
        # Buffer rdq window to include forecasts that arrive before window_start
        rdq_buffer_start = start_date - datetime.timedelta(days=CONSENSUS_LOCK_WINDOW_DAYS + 30)
        fundq_df = conn.raw_sql(
            _COMP_FUNDQ_RDQ_SQL,
            params={
                "start_date": rdq_buffer_start.isoformat(),
                "end_date":   end_date.isoformat(),
                "gvkeys":     tuple(gvkeys),
            },
            date_cols=["datadate", "rdq"],
        )
        if fundq_df.empty:
            logger.warning("path_c.earnings_panel: no fundq rows")
            return pd.DataFrame()
        fundq_df = fundq_df.dropna(subset=["rdq", "datadate"])
        fundq_df["market_cap_at_q"] = (
            fundq_df["cshoq"].astype(float) * fundq_df["prccq"].astype(float)
        )

        # Step 4: pull I/B/E/S detail forecasts (90d-buffer covered by fpedats range)
        det_df = conn.raw_sql(
            _IBES_DET_EPSUS_SQL,
            params={
                "start_date": rdq_buffer_start.isoformat(),
                "end_date":   end_date.isoformat(),
                "tickers":    tuple(needed_tickers),
            },
            date_cols=["fpedats", "anndats"],
        )
        if det_df.empty:
            logger.warning("path_c.earnings_panel: no det_epsus rows")
            return pd.DataFrame()

        # Step 5: pull I/B/E/S actuals
        act_df = conn.raw_sql(
            _IBES_ACT_EPSUS_SQL,
            params={
                "start_date": rdq_buffer_start.isoformat(),
                "end_date":   end_date.isoformat(),
                "tickers":    tuple(needed_tickers),
            },
            date_cols=["pends", "anndats"],
        )
        if act_df.empty:
            logger.warning("path_c.earnings_panel: no act_epsus rows")
            return pd.DataFrame()

        # Step 6: join + per-firm-quarter consensus aggregation
        # Build (gvkey → permno) and (permno → ticker_ibes) lookups
        gvkey_to_permno = {}
        for _, row in comp_link_df.iterrows():
            if pd.notna(row.get("gvkey")) and pd.notna(row.get("permno")):
                gvkey_to_permno[str(row["gvkey"])] = int(row["permno"])
        permno_to_ticker = {}
        for _, row in link_df.iterrows():
            if pd.notna(row.get("permno")) and pd.notna(row.get("ticker")):
                permno_to_ticker[int(row["permno"])] = str(row["ticker"]).strip()

        fundq_df["permno"]      = fundq_df["gvkey"].astype(str).map(gvkey_to_permno)
        fundq_df["ticker_ibes"] = fundq_df["permno"].map(permno_to_ticker)
        fundq_df = fundq_df.dropna(subset=["permno", "ticker_ibes"])

        # Build per-firm-quarter rows
        out_rows = []
        for _, q_row in fundq_df.iterrows():
            ticker  = str(q_row["ticker_ibes"])
            rdq     = q_row["rdq"]
            if hasattr(rdq, "date"):
                rdq = rdq.date()
            if not (start_date <= rdq <= end_date):
                continue
            datadate = q_row["datadate"]
            if hasattr(datadate, "date"):
                datadate = datadate.date()
            quarter_label = f"{datadate.year}Q{(datadate.month - 1) // 3 + 1}"

            # Match I/B/E/S fpedats ≈ Compustat datadate (allow ±5 day slack)
            firm_det = det_df[det_df["ticker"] == ticker].copy()
            firm_det = firm_det[
                (firm_det["fpedats"].dt.date >= datadate - datetime.timedelta(days=5))
                & (firm_det["fpedats"].dt.date <= datadate + datetime.timedelta(days=5))
            ]
            # Window: anndats ∈ [rdq - 90d, rdq - 1d]
            window_start = rdq - datetime.timedelta(days=CONSENSUS_LOCK_WINDOW_DAYS)
            window_end   = rdq - datetime.timedelta(days=1)
            firm_det = firm_det[
                (firm_det["anndats"].dt.date >= window_start)
                & (firm_det["anndats"].dt.date <= window_end)
            ]
            # Per analyst, keep most recent forecast in window
            if firm_det.empty:
                continue
            firm_det = (
                firm_det.sort_values("anndats")
                       .drop_duplicates(subset=["analys"], keep="last")
            )
            n_analysts = int(firm_det["analys"].nunique())
            if n_analysts < MIN_ANALYSTS_REQUIRED:
                continue
            forecasts = firm_det["value"].astype(float).values
            consensus_median = float(np.median(forecasts))
            dispersion       = float(np.std(forecasts, ddof=1)) if len(forecasts) > 1 else 0.0
            if dispersion == 0.0:
                continue

            # Find matching actual
            firm_act = act_df[act_df["ticker"] == ticker].copy()
            firm_act = firm_act[
                (firm_act["pends"].dt.date >= datadate - datetime.timedelta(days=5))
                & (firm_act["pends"].dt.date <= datadate + datetime.timedelta(days=5))
            ]
            if firm_act.empty:
                continue
            actual_eps = float(firm_act["value"].iloc[0])

            out_rows.append({
                "permno":               int(q_row["permno"]),
                "ticker_ibes":          ticker,
                "gvkey":                int(q_row["gvkey"]),
                "fiscal_yearq":         quarter_label,
                "rdq":                  rdq,
                "actual_eps":           actual_eps,
                "consensus_median":     consensus_median,
                "consensus_dispersion": dispersion,
                "n_analysts":           n_analysts,
                "market_cap_at_q":      float(q_row["market_cap_at_q"]) if pd.notna(q_row["market_cap_at_q"]) else np.nan,
            })

        if not out_rows:
            return pd.DataFrame()
        return pd.DataFrame(out_rows).sort_values(
            ["fiscal_yearq", "ticker_ibes"]
        ).reset_index(drop=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Public API ──────────────────────────────────────────────────────────────
def bulk_fetch_earnings_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
    *,
    mock_mode:  Optional[bool] = None,
    use_cache:  bool = True,
    cache_path: Optional[Path] = None,
) -> EarningsPanelResult:
    """Bulk-fetch firm-quarter earnings panel (consensus + actual + dispersion + rdq).

    Args:
        tickers:    list of I/B/E/S tickers (typically top-200 SP500 vintage)
        start_date: window start (rdq ≥ start_date)
        end_date:   window end (rdq ≤ end_date)
        mock_mode:  True → synthetic panel; False → real WRDS;
                    None → auto-detect via is_wrds_available()
        use_cache:  load existing cache if present + covers full window
        cache_path: override default cache path (used by tests with tmp_path)

    Returns:
        EarningsPanelResult with `panel` (long-form firm-quarter DataFrame),
        mode, n_firm_quarters, and exclusion_stats.
    """
    if mock_mode is None:
        mock_mode = not is_wrds_available()
    mode_str = "mock" if mock_mode else "wrds"

    path = cache_path if cache_path is not None else _EARNINGS_PANEL_CACHE_PATH
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    # Cache load — gated by sidecar metadata that records the build envelope.
    # rdq is sparse (~4/year per firm) so cannot reconstruct the original
    # window from row data alone; sidecar is authoritative.
    if use_cache and path.exists() and meta_path.exists():
        try:
            import json
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            built_start = datetime.date.fromisoformat(meta["window_start"])
            built_end   = datetime.date.fromisoformat(meta["window_end"])
            built_tickers = set(meta.get("tickers", []))
            wanted_tickers = set(tickers)
            cache_ok = (
                wanted_tickers.issubset(built_tickers)
                and built_start <= start_date
                and built_end   >= end_date
            )
            if cache_ok:
                cached = pd.read_parquet(path)
                if "rdq" in cached.columns:
                    cached["rdq"] = pd.to_datetime(cached["rdq"]).dt.date
                logger.info(
                    "earnings_panel cache HIT: %d firm-quarters from envelope [%s, %s]",
                    len(cached), built_start, built_end,
                )
                filtered = cached[
                    (cached["ticker_ibes"].isin(tickers))
                    & (cached["rdq"] >= start_date)
                    & (cached["rdq"] <= end_date)
                ].reset_index(drop=True)
                return EarningsPanelResult(
                    panel=filtered,
                    mode=mode_str,
                    n_firm_quarters=len(filtered),
                    exclusion_stats={"from_cache": True},
                    window_start=start_date,
                    window_end=end_date,
                )
        except Exception as exc:
            logger.warning("earnings_panel cache load failed: %s — refetching", exc)

    # Cache miss → real fetch
    logger.info(
        "earnings_panel cache MISS — %s-fetching %d tickers [%s, %s]",
        mode_str, len(tickers), start_date, end_date,
    )
    if mock_mode:
        panel = _mock_earnings_panel(tickers, start_date, end_date)
    else:
        panel = _real_earnings_panel(tickers, start_date, end_date)

    # Exclusion stats are computed inside _real_*; for mock we report 0
    exclusion_stats = {
        "no_actual":              0,
        "insufficient_analysts":  0,
        "zero_dispersion":        0,
        "out_of_window":          0,
    }

    # Persist (parquet + sidecar JSON envelope metadata)
    if use_cache and not panel.empty:
        try:
            panel.to_parquet(path)
            import json
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "window_start": start_date.isoformat(),
                    "window_end":   end_date.isoformat(),
                    "tickers":      sorted(set(tickers)),
                    "mode":         mode_str,
                    "n_rows":       int(len(panel)),
                }, fh)
            logger.info(
                "earnings_panel cache persisted: %d firm-quarters → %s (+ meta)",
                len(panel), path,
            )
        except Exception as exc:
            logger.warning("earnings_panel persist failed: %s", exc)

    return EarningsPanelResult(
        panel=panel,
        mode=mode_str,
        n_firm_quarters=len(panel),
        exclusion_stats=exclusion_stats,
        window_start=start_date,
        window_end=end_date,
    )
