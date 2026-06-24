"""
engine/path_c/labor_signal_panel.py — Path C Labor Signal Drift firm-quarter panel builder.

Pre-registration: docs/spec_path_c_labor_signal_drift_v1.md (id=58) §2.2 + §2.3

Pulls + joins:
  - Revelio postings_cosmos (job posting events 2008-2026, 2.4B rows)
  - Revelio layoffs (WARN notices 1989-2027, 61K rows)
  - Revelio company_mapping (rcid <-> ticker/cusip/gvkey, 32M rows)
  - Compustat fundq (rdq announcement date + market_cap inputs)
  - CRSP linkage via crsp.ccmxpf_lnkhist (permno <-> gvkey)

Output schema (firm-quarter long form):
  permno, ticker, gvkey, rcid, fiscal_yearq, rdq,
  l6_postings_count, b12_postings_count, layoff_flag,
  market_cap_at_q

NOTE: labor signal FORMULA composition (LS = (L6 - B12/2) / max(B12/2, 1)
- 0.5 * layoff_flag) lives in engine/path_c/labor_signal.py (Sprint G3),
mirroring earnings_panel.py vs sue_signal.py separation.

Two modes:
  - mock_mode=True: synthetic deterministic panel for tests
  - mock_mode=False: real WRDS query with @with_wrds_retry
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

from engine.universe_singlename.crsp_loader import (
    is_wrds_available as _crsp_is_wrds_available,
    _open_wrds_connection as _crsp_open_wrds_connection,
)
from engine.universe_singlename.wrds_retry import with_wrds_retry

logger = logging.getLogger(__name__)


# Locked from spec §2.3 + §六
L6_WINDOW_MONTHS: int       = 6     # rolling 6mo posting count
B12_WINDOW_MONTHS: int      = 12    # 12mo baseline (pre-L6 window)
LAYOFF_WINDOW_DAYS: int     = 90    # [rdq - 90d, rdq] for layoff flag
MIN_L6_POSTINGS_REQUIRED:   int = 5    # spec §2.3 fallback
MIN_B12_POSTINGS_REQUIRED:  int = 10   # spec §2.3 fallback


# ── Storage ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _REPO_ROOT / "data" / "path_c_labor"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_LABOR_PANEL_CACHE_PATH = _CACHE_DIR / "_labor_signal_panel.parquet"


# ── Public types ────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class LaborSignalPanelResult:
    """Output of bulk_fetch_labor_signal_panel.

    `panel` columns:
      - permno (int): CRSP permanent number
      - ticker (str): primary ticker (US-exchange)
      - gvkey (int): Compustat firm identifier
      - rcid (int): Revelio Labs Company ID
      - fiscal_yearq (str): e.g. "2014Q1"
      - rdq (datetime.date): Compustat quarterly earnings report date
      - l6_postings_count (int): count of postings in [rdq - 6mo, rdq - 30d]
      - b12_postings_count (int): count in [rdq - 18mo, rdq - 6mo]
      - layoff_flag (int): 1 if any WARN layoff in [rdq - 90d, rdq] else 0
      - market_cap_at_q (float): from Compustat (cshoq * prccq), USD millions
    """
    panel:           pd.DataFrame
    mode:            str          # "mock" or "wrds"
    n_firm_quarters: int
    exclusion_stats: dict
    window_start:    datetime.date
    window_end:      datetime.date


# ── WRDS availability ──────────────────────────────────────────────────────
def is_wrds_available() -> bool:
    """Delegate to crsp_loader's WRDS availability check."""
    return _crsp_is_wrds_available()


# ── Mock-mode panel generator ──────────────────────────────────────────────
def _mock_labor_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Deterministic synthetic labor-signal panel.

    Reproducible across runs (ticker hash → RNG seed). Each firm-quarter:
      - l6_postings_count ~ Poisson(baseline_lambda × 1.0)
      - b12_postings_count ~ Poisson(baseline_lambda × 2.0)  (≈ 2× the 6mo)
      - layoff_flag ~ Bernoulli(0.05)
      - market_cap_at_q drifting GBM ~$5B - $50B
    """
    if not tickers:
        return pd.DataFrame()

    rows = []
    q_starts = pd.date_range(start=start_date, end=end_date, freq="QS").to_pydatetime()
    if len(q_starts) == 0:
        return pd.DataFrame()

    for ticker in sorted(set(tickers)):
        seed = int(hashlib.md5(ticker.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        baseline_lambda = float(rng.uniform(20.0, 500.0))   # per-firm hiring rate
        base_mcap       = float(rng.uniform(5_000.0, 50_000.0))
        rcid_synth      = abs(seed) % 9_999_999
        gvkey_synth     = abs(seed) % 999_999
        permno_synth    = (abs(seed) // 1000) % 99_999 + 10_000

        for q_start in q_starts:
            q_start_date = q_start.date()
            quarter_label = f"{q_start_date.year}Q{(q_start_date.month - 1) // 3 + 1}"
            q_end_offset_days = 90 + int(rng.integers(25, 46))
            rdq = q_start_date + datetime.timedelta(days=q_end_offset_days)
            if rdq > end_date:
                continue

            l6_count = int(rng.poisson(lam=baseline_lambda))
            b12_count = int(rng.poisson(lam=baseline_lambda * 2.0))
            layoff_flag = int(rng.random() < 0.05)
            quarter_idx = (q_start_date.year - start_date.year) * 4 + (q_start_date.month - 1) // 3
            mcap = base_mcap * float(np.exp(rng.normal(0.02, 0.10) * quarter_idx / 4.0))

            rows.append({
                "permno":             permno_synth,
                "ticker":             ticker,
                "gvkey":              gvkey_synth,
                "rcid":               rcid_synth,
                "fiscal_yearq":       quarter_label,
                "rdq":                rdq,
                "l6_postings_count":  l6_count,
                "b12_postings_count": b12_count,
                "layoff_flag":        layoff_flag,
                "market_cap_at_q":    mcap,
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["fiscal_yearq", "ticker"]).reset_index(drop=True)


# ── Real WRDS queries ───────────────────────────────────────────────────────
# SQL templates — parameterized via raw_sql params for injection safety.

# Revelio company_mapping: ticker → rcid (filter to US exchanges).
#
# DETERMINISTIC dedupe: probe 2026-05-12 found 50 SP500 sample tickers → 81
# rcid matches (avg 1.65 rcids/ticker due to subsidiaries / historical
# company changes). `DISTINCT ON (ticker) ... ORDER BY ticker, rcid ASC`
# locks selection to lowest rcid per ticker (typically = parent /
# earliest registered Revelio entity), avoiding non-deterministic across-
# run drift.
#
# Exchange filter: 'NASDAQ' + 'New York Stock Exchange' are the actual
# values in the schema for US main listings (verified by probe; my prior
# guesses 'NYSE' / 'NASDAQ Global Select' / 'NYSEArca' / 'NYSE American'
# don't appear in revelio.company_mapping. SP500 firms all use the 2 listed.
_REVELIO_COMPANY_MAPPING_SQL = """
SELECT DISTINCT ON (ticker)
    rcid,
    ticker,
    cusip,
    gvkey
FROM revelio.company_mapping
WHERE ticker IN %(tickers)s
  AND ticker != ''
  AND exchange_name IN ('NASDAQ', 'New York Stock Exchange')
ORDER BY ticker, rcid ASC
"""

# Compustat fundq for rdq + market_cap inputs (same as Path C #1 PEAD)
_COMP_FUNDQ_RDQ_SQL = """
SELECT
    gvkey,
    fyearq,
    fqtr,
    datadate,
    rdq,
    cshoq,
    prccq
FROM comp.fundq
WHERE rdq BETWEEN %(start_date)s AND %(end_date)s
  AND gvkey IN %(gvkeys)s
  AND indfmt = 'INDL'
  AND datafmt = 'STD'
  AND popsrc = 'D'
  AND consol = 'C'
ORDER BY gvkey, fyearq, fqtr
"""

# Revelio postings counts per rcid per month (aggregated server-side to reduce transfer)
# Fetched for the full window + 18mo buffer to support B12 baseline.
_REVELIO_POSTINGS_AGG_SQL = """
SELECT
    rcid,
    DATE_TRUNC('month', post_date)::date AS month,
    COUNT(*) AS postings_count
FROM revelio.postings_cosmos
WHERE rcid IN %(rcids)s
  AND post_date BETWEEN %(start_date)s AND %(end_date)s
GROUP BY rcid, DATE_TRUNC('month', post_date)
"""

# Revelio layoffs for rcid within window (small table 61K rows total)
_REVELIO_LAYOFFS_SQL = """
SELECT
    rcid,
    layoff_date
FROM revelio.layoffs
WHERE rcid IN %(rcids)s
  AND layoff_date BETWEEN %(start_date)s AND %(end_date)s
"""

# CRSP <-> Compustat linkage (same as PEAD)
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
def _real_labor_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Build firm-quarter labor-signal panel from WRDS Revelio.

    Pipeline:
      1. Open WRDS connection
      2. ticker → rcid + gvkey via revelio.company_mapping (US-exchange filter)
      3. Pull comp.fundq for those gvkeys, filtered to rdq in [start, end]
      4. Compute rdq dates per firm-quarter; build per-firm monthly buckets
      5. Pull revelio.postings_cosmos aggregated by rcid×month for window
         [start - 18mo, end] (covers both L6 and B12 lookback)
      6. Pull revelio.layoffs for rcid within [start - 90d, end]
      7. Per firm-quarter, compute:
         - L6 = sum of monthly postings in [rdq - 6mo, rdq - 30d]
         - B12 = sum of monthly postings in [rdq - 18mo, rdq - 6mo]
         - layoff_flag = 1 if any layoff in [rdq - 90d, rdq] else 0
      8. Compute market_cap_at_q = cshoq × prccq
      9. Return firm-quarter DataFrame per LaborSignalPanelResult schema
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
        # Step 1: ticker → rcid + gvkey via revelio.company_mapping
        logger.info("path_c.labor_signal_panel: resolving %d tickers → rcid via revelio.company_mapping",
                    len(needed_tickers))
        cm_df = conn.raw_sql(
            _REVELIO_COMPANY_MAPPING_SQL,
            params={"tickers": tuple(needed_tickers)},
        )
        if cm_df.empty:
            logger.warning("path_c.labor_signal_panel: no rcid linkages found")
            return pd.DataFrame()
        # Take first non-null rcid per ticker (some tickers have multiple Revelio entities)
        cm_df = cm_df.dropna(subset=["rcid"]).drop_duplicates(subset=["ticker"], keep="first")
        rcids = sorted(set(int(r) for r in cm_df["rcid"].dropna().tolist()))
        gvkeys = sorted(set(g for g in cm_df["gvkey"].dropna().astype(str).tolist() if g))

        # Step 2: comp.fundq for rdq + market_cap (only firms with valid gvkey link)
        if not gvkeys:
            logger.warning("path_c.labor_signal_panel: no gvkey linkages from Revelio")
            return pd.DataFrame()
        rdq_buffer_start = start_date - datetime.timedelta(days=L6_WINDOW_MONTHS * 30 + 30)
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
            logger.warning("path_c.labor_signal_panel: no fundq rows")
            return pd.DataFrame()
        fundq_df = fundq_df.dropna(subset=["rdq", "datadate"])
        fundq_df["market_cap_at_q"] = (
            fundq_df["cshoq"].astype(float) * fundq_df["prccq"].astype(float)
        )

        # Step 3: bulk Revelio postings aggregated by rcid × month
        # Window needs to cover [start - 18mo, end] for B12 baseline
        postings_buffer_start = start_date - datetime.timedelta(days=(L6_WINDOW_MONTHS + B12_WINDOW_MONTHS) * 31)
        logger.info("path_c.labor_signal_panel: pulling postings aggregated for %d rcids × [%s, %s]",
                    len(rcids), postings_buffer_start, end_date)
        postings_agg_df = conn.raw_sql(
            _REVELIO_POSTINGS_AGG_SQL,
            params={
                "rcids":      tuple(rcids),
                "start_date": postings_buffer_start.isoformat(),
                "end_date":   end_date.isoformat(),
            },
            date_cols=["month"],
        )
        if postings_agg_df.empty:
            logger.warning("path_c.labor_signal_panel: no postings rows")
            return pd.DataFrame()

        # Step 4: layoffs
        layoff_buffer_start = start_date - datetime.timedelta(days=LAYOFF_WINDOW_DAYS + 30)
        layoffs_df = conn.raw_sql(
            _REVELIO_LAYOFFS_SQL,
            params={
                "rcids":      tuple(rcids),
                "start_date": layoff_buffer_start.isoformat(),
                "end_date":   end_date.isoformat(),
            },
            date_cols=["layoff_date"],
        )
        layoffs_by_rcid: dict[int, list[datetime.date]] = {}
        if not layoffs_df.empty:
            for _, row in layoffs_df.iterrows():
                r = int(row["rcid"])
                d = row["layoff_date"]
                if hasattr(d, "date"):
                    d = d.date()
                layoffs_by_rcid.setdefault(r, []).append(d)

        # Step 5: CRSP linkage for permno
        comp_link_df = conn.raw_sql(
            _CRSP_COMP_LINK_SQL,
            params={
                "permnos":    tuple(),   # placeholder (we'll re-query via gvkey)
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
            },
            date_cols=["linkdt", "linkenddt"],
        ) if False else None  # NOTE: simpler approach below

        # gvkey → permno via reverse lookup (we don't have permnos yet)
        # Use ccmxpf_lnkhist directly with gvkey filter
        gvkey_link_sql = """
        SELECT gvkey, lpermno AS permno, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE gvkey IN %(gvkeys)s
          AND linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
          AND (linkenddt IS NULL OR linkenddt >= %(start_date)s)
          AND linkdt <= %(end_date)s
        """
        gvkey_link_df = conn.raw_sql(
            gvkey_link_sql,
            params={
                "gvkeys":     tuple(gvkeys),
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
            },
            date_cols=["linkdt", "linkenddt"],
        )
        gvkey_to_permno: dict[str, int] = {}
        if not gvkey_link_df.empty:
            for _, row in gvkey_link_df.iterrows():
                if pd.notna(row.get("gvkey")) and pd.notna(row.get("permno")):
                    gvkey_to_permno[str(row["gvkey"])] = int(row["permno"])

        # Step 6: assemble per-firm-quarter rows
        # Build rcid → (ticker, gvkey) lookup from company_mapping
        rcid_to_ticker: dict[int, str] = {}
        rcid_to_gvkey:  dict[int, str] = {}
        for _, row in cm_df.iterrows():
            r = int(row["rcid"])
            rcid_to_ticker[r] = str(row["ticker"]).strip()
            if pd.notna(row.get("gvkey")):
                rcid_to_gvkey[r] = str(row["gvkey"])

        # gvkey → rcid reverse lookup (multi-mapping possible; pick first)
        gvkey_to_rcid: dict[str, int] = {}
        for r, g in rcid_to_gvkey.items():
            if g not in gvkey_to_rcid:
                gvkey_to_rcid[g] = r

        # Build per-firm monthly postings dict: rcid → {month_start_date: count}
        postings_by_rcid: dict[int, dict[datetime.date, int]] = {}
        for _, row in postings_agg_df.iterrows():
            r = int(row["rcid"])
            m = row["month"]
            if hasattr(m, "date"):
                m = m.date()
            postings_by_rcid.setdefault(r, {})[m] = int(row["postings_count"])

        # Now iterate firm-quarter rows
        out_rows = []
        for _, q_row in fundq_df.iterrows():
            gvkey = str(q_row["gvkey"])
            rcid = gvkey_to_rcid.get(gvkey)
            if rcid is None:
                continue
            ticker = rcid_to_ticker.get(rcid, "")
            permno = gvkey_to_permno.get(gvkey)
            if permno is None:
                continue
            rdq = q_row["rdq"]
            if hasattr(rdq, "date"):
                rdq = rdq.date()
            if not (start_date <= rdq <= end_date):
                continue
            datadate = q_row["datadate"]
            if hasattr(datadate, "date"):
                datadate = datadate.date()
            quarter_label = f"{datadate.year}Q{(datadate.month - 1) // 3 + 1}"

            # L6 window: [rdq - 6mo, rdq - 30d]
            l6_start = rdq - datetime.timedelta(days=L6_WINDOW_MONTHS * 30)
            l6_end   = rdq - datetime.timedelta(days=30)
            # B12 window: [rdq - 18mo, rdq - 6mo]
            b12_start = rdq - datetime.timedelta(days=(L6_WINDOW_MONTHS + B12_WINDOW_MONTHS) * 30)
            b12_end   = rdq - datetime.timedelta(days=L6_WINDOW_MONTHS * 30)

            monthly = postings_by_rcid.get(rcid, {})
            l6_count = sum(c for m, c in monthly.items() if l6_start <= m <= l6_end)
            b12_count = sum(c for m, c in monthly.items() if b12_start <= m <= b12_end)

            # Layoff flag: any layoff in [rdq - 90d, rdq]
            layoff_dates = layoffs_by_rcid.get(rcid, [])
            layoff_flag = int(any(
                rdq - datetime.timedelta(days=LAYOFF_WINDOW_DAYS) <= d <= rdq
                for d in layoff_dates
            ))

            mcap = float(q_row["market_cap_at_q"]) if pd.notna(q_row["market_cap_at_q"]) else np.nan

            out_rows.append({
                "permno":             permno,
                "ticker":             ticker,
                "gvkey":              int(gvkey),
                "rcid":               rcid,
                "fiscal_yearq":       quarter_label,
                "rdq":                rdq,
                "l6_postings_count":  l6_count,
                "b12_postings_count": b12_count,
                "layoff_flag":        layoff_flag,
                "market_cap_at_q":    mcap,
            })

        if not out_rows:
            return pd.DataFrame()
        return pd.DataFrame(out_rows).sort_values(
            ["fiscal_yearq", "ticker"]
        ).reset_index(drop=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Public API ──────────────────────────────────────────────────────────────
def bulk_fetch_labor_signal_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
    *,
    mock_mode:  Optional[bool] = None,
    use_cache:  bool = True,
    cache_path: Optional[Path] = None,
) -> LaborSignalPanelResult:
    """Bulk-fetch firm-quarter labor-signal panel (Revelio + Compustat + CRSP linkage).

    Args:
        tickers:    list of US-exchange tickers (top-200 SP500 vintage)
        start_date: window start (rdq >= start_date)
        end_date:   window end (rdq <= end_date)
        mock_mode:  True → synthetic; False → real WRDS; None → auto via is_wrds_available()
        use_cache:  load existing parquet cache + sidecar metadata if covers window
        cache_path: override default cache path (for tests with tmp_path)

    Returns:
        LaborSignalPanelResult with `panel` (long-form firm-quarter DataFrame),
        mode, n_firm_quarters, and exclusion_stats.
    """
    if mock_mode is None:
        mock_mode = not is_wrds_available()
    mode_str = "mock" if mock_mode else "wrds"

    path = cache_path if cache_path is not None else _LABOR_PANEL_CACHE_PATH
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    # Cache load via sidecar metadata
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
                    "labor_signal_panel cache HIT: %d firm-quarters envelope [%s, %s]",
                    len(cached), built_start, built_end,
                )
                filtered = cached[
                    (cached["ticker"].isin(tickers))
                    & (cached["rdq"] >= start_date)
                    & (cached["rdq"] <= end_date)
                ].reset_index(drop=True)
                return LaborSignalPanelResult(
                    panel=filtered,
                    mode=mode_str,
                    n_firm_quarters=len(filtered),
                    exclusion_stats={"from_cache": True},
                    window_start=start_date,
                    window_end=end_date,
                )
        except Exception as exc:
            logger.warning("labor_signal_panel cache load failed: %s — refetching", exc)

    # Cache miss → fetch
    logger.info(
        "labor_signal_panel cache MISS — %s-fetching %d tickers [%s, %s]",
        mode_str, len(tickers), start_date, end_date,
    )
    if mock_mode:
        panel = _mock_labor_panel(tickers, start_date, end_date)
    else:
        panel = _real_labor_panel(tickers, start_date, end_date)

    exclusion_stats = {
        "no_rcid_mapping":     0,
        "thin_l6_postings":    0,   # populated by labor_signal.py downstream
        "thin_b12_postings":   0,
        "out_of_window":       0,
    }

    # Persist
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
            logger.info("labor_signal_panel cache persisted: %d firm-quarters → %s (+ meta)",
                        len(panel), path)
        except Exception as exc:
            logger.warning("labor_signal_panel persist failed: %s", exc)

    return LaborSignalPanelResult(
        panel=panel,
        mode=mode_str,
        n_firm_quarters=len(panel),
        exclusion_stats=exclusion_stats,
        window_start=start_date,
        window_end=end_date,
    )
