"""
engine/path_l/profitability_signal_panel.py — Novy-Marx 2013 Gross Profitability panel builder.

Pre-registration: docs/spec_path_l_profitability_v1.md (id=68 hash 5a2ab1cc) §2.3

Novy-Marx 2013 canonical formulation:
  GP_q  = revtq_q - cogsq_q                     (gross profit, quarterly)
  GP_TTM_q = sum(GP over [q-3, q-2, q-1, q])    (trailing 4Q sum for stability)
  TA_lag4avg_q = avg(atq over [q-4, q-3, q-2, q-1])  (lagged 4Q-avg total assets)
  GPA_q = GP_TTM_q / TA_lag4avg_q               (gross profitability ratio)

Two modes:
  - mock_mode=True (default when WRDS unavailable): deterministic synthetic panel
  - mock_mode=False: real WRDS query via comp.fundq

Disk cache: parquet at data/path_l/_profitability_signal_panel.parquet.
"""
from __future__ import annotations

import dataclasses
import datetime
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
TTM_WINDOW_QUARTERS:       int   = 4     # 4Q trailing sum for GP / 4Q lagged avg for TA
LAG_OFFSET_QUARTERS:       int   = 1     # TA lagged 1Q to avoid look-ahead
GPA_WINSORIZE_LOW:         float = -0.5  # Novy-Marx + AFP 2019 convention
GPA_WINSORIZE_HIGH:        float = +2.0


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _REPO_ROOT / "data" / "path_l"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PROFITABILITY_PANEL_CACHE_PATH = _CACHE_DIR / "_profitability_signal_panel.parquet"


@dataclasses.dataclass(frozen=True)
class ProfitabilitySignalPanelResult:
    """Output of bulk_fetch_profitability_signal_panel.

    `panel` columns:
      - permno (int)
      - ticker (str)
      - gvkey (int)
      - fiscal_yearq (str): "2014Q1"
      - rdq (datetime.date): earnings announcement
      - gp_ttm (float): 4Q-sum gross profit
      - ta_lag4avg (float): 4Q-avg lagged total assets
      - gpa (float): GP_TTM / TA_lag4avg (winsorized [-0.5, +2.0])
      - market_cap_at_q (float): for top-N universe ranking (approximated via atq if cshoq/prccq not pulled)
    """
    panel:           pd.DataFrame
    mode:            str          # "mock" or "wrds"
    n_firm_quarters: int
    exclusion_stats: dict
    window_start:    datetime.date
    window_end:      datetime.date


def is_wrds_available() -> bool:
    return _crsp_is_wrds_available()


# ── Mock-mode generator ────────────────────────────────────────────────────
def _mock_profitability_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Synthetic profitability panel for smoke testing (no WRDS)."""
    rng = np.random.default_rng(20260513)
    rows = []
    # Quarterly grid: 4 rdq per year
    year_range = list(range(start_date.year, end_date.year + 1))
    for ticker in tickers:
        permno = 10000 + abs(hash(ticker)) % 90000
        gvkey  = 100000 + abs(hash(ticker)) % 900000
        # Firm-specific mean GPA
        firm_gpa_mean = rng.normal(0.3, 0.2)  # cross-firm mean varies
        for y in year_range:
            for q in range(1, 5):
                rdq = datetime.date(y, ((q - 1) * 3 + 1), min(15 + int(rng.integers(0, 28)), 28))
                if rdq < start_date or rdq > end_date:
                    continue
                gpa_noise = rng.normal(0, 0.1)
                gpa = max(GPA_WINSORIZE_LOW, min(GPA_WINSORIZE_HIGH, firm_gpa_mean + gpa_noise))
                rows.append({
                    "permno":          permno,
                    "ticker":          ticker,
                    "gvkey":           gvkey,
                    "fiscal_yearq":    f"{y}Q{q}",
                    "rdq":             rdq,
                    "gp_ttm":          rng.normal(500, 200),  # synthetic in $millions
                    "ta_lag4avg":      rng.normal(2000, 800),
                    "gpa":             gpa,
                    "market_cap_at_q": rng.uniform(500, 50000),  # $millions
                })
    return pd.DataFrame(rows)


# ── Real WRDS query ────────────────────────────────────────────────────────
_COMP_FUNDQ_PROFITABILITY_SQL = """
SELECT
    gvkey,
    fyearq,
    fqtr,
    datadate,
    rdq,
    revtq,
    cogsq,
    atq,
    cshoq,
    prccq
FROM comp.fundq
WHERE datadate BETWEEN %(buffer_start)s AND %(end_date)s
  AND gvkey IN %(gvkeys)s
  AND indfmt = 'INDL'
  AND datafmt = 'STD'
  AND popsrc = 'D'
  AND consol = 'C'
ORDER BY gvkey, fyearq, fqtr
"""

_CRSP_MSE_TICKER_SQL = """
SELECT DISTINCT permno, ticker
FROM crsp.msenames
WHERE ticker IN %(tickers)s
  AND nameendt >= %(start_date)s
  AND namedt <= %(end_date)s
"""

_CRSP_COMP_LINK_SQL = """
SELECT gvkey, lpermno AS permno, linkdt, linkenddt
FROM crsp.ccmxpf_lnkhist
WHERE lpermno IN %(permnos)s
  AND linktype IN ('LU', 'LC')
  AND linkprim IN ('P', 'C')
  AND (linkenddt IS NULL OR linkenddt >= %(start_date)s)
  AND linkdt <= %(end_date)s
"""


@with_wrds_retry(max_attempts=3, base_delay=5.0)
def _real_profitability_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Build firm-quarter GPA panel from WRDS Compustat."""
    if not is_wrds_available():
        raise RuntimeError("WRDS not configured. Pass mock_mode=True.")
    if not tickers:
        return pd.DataFrame()

    needed_tickers = sorted(set(tickers))
    conn = _crsp_open_wrds_connection()
    try:
        # Step 1: ticker → permno
        logger.info("path_l: resolving %d tickers → permno via msenames", len(needed_tickers))
        mse_df = conn.raw_sql(
            _CRSP_MSE_TICKER_SQL,
            params={
                "tickers":    tuple(needed_tickers),
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
            },
        )
        if mse_df.empty:
            return pd.DataFrame()
        mse_df = mse_df.sort_values(["ticker", "permno"]).drop_duplicates(
            subset=["ticker"], keep="first"
        )
        permnos = sorted(set(int(p) for p in mse_df["permno"].dropna().tolist()))
        permno_to_ticker = {int(r["permno"]): str(r["ticker"]).strip()
                            for _, r in mse_df.iterrows()}

        # Step 2: permno → gvkey
        link_df = conn.raw_sql(
            _CRSP_COMP_LINK_SQL,
            params={
                "permnos":    tuple(permnos),
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
            },
            date_cols=["linkdt", "linkenddt"],
        )
        if link_df.empty:
            return pd.DataFrame()
        link_df = link_df.dropna(subset=["gvkey", "permno"]).sort_values(
            ["permno", "gvkey"]
        ).drop_duplicates(subset=["permno"], keep="first")
        permno_to_gvkey = {int(r["permno"]): str(r["gvkey"]) for _, r in link_df.iterrows()}
        gvkey_to_permno = {v: k for k, v in permno_to_gvkey.items()}
        gvkeys = sorted(set(permno_to_gvkey.values()))

        # Step 3: pull fundq with 6Q lookback buffer (4Q TTM + 1Q lag + slack)
        buffer_start = start_date - datetime.timedelta(days=6 * 92 + 90)
        logger.info("path_l: pulling fundq for %d gvkeys × [%s, %s]",
                    len(gvkeys), buffer_start, end_date)
        fundq_df = conn.raw_sql(
            _COMP_FUNDQ_PROFITABILITY_SQL,
            params={
                "buffer_start": buffer_start.isoformat(),
                "end_date":     end_date.isoformat(),
                "gvkeys":       tuple(gvkeys),
            },
            date_cols=["datadate", "rdq"],
        )
        if fundq_df.empty:
            return pd.DataFrame()

        fundq_df["gvkey"] = fundq_df["gvkey"].astype(str)
        fundq_df["permno"] = fundq_df["gvkey"].map(gvkey_to_permno)
        fundq_df["ticker"] = fundq_df["permno"].map(permno_to_ticker)
        fundq_df = fundq_df.dropna(subset=["permno", "ticker", "fyearq", "fqtr"])
        if fundq_df.empty:
            return pd.DataFrame()

        # Cast to float
        for col in ["revtq", "cogsq", "atq", "cshoq", "prccq"]:
            fundq_df[col] = pd.to_numeric(fundq_df[col], errors="coerce")

        # Compute quarterly GP
        fundq_df["gp_q"] = fundq_df["revtq"] - fundq_df["cogsq"]

        # Sort by firm + fiscal time
        fundq_df = fundq_df.sort_values(["gvkey", "fyearq", "fqtr"]).reset_index(drop=True)

        # TTM 4Q sum of GP (current + 3 prior quarters)
        def _rolling_4q_sum(s: pd.Series) -> pd.Series:
            return s.rolling(window=TTM_WINDOW_QUARTERS, min_periods=TTM_WINDOW_QUARTERS).sum()

        fundq_df["gp_ttm"] = fundq_df.groupby("gvkey")["gp_q"].transform(_rolling_4q_sum)

        # Lagged 4Q-avg TA (avg over q-4 through q-1)
        def _lagged_4q_avg(s: pd.Series) -> pd.Series:
            return s.shift(LAG_OFFSET_QUARTERS).rolling(
                window=TTM_WINDOW_QUARTERS, min_periods=TTM_WINDOW_QUARTERS
            ).mean()

        fundq_df["ta_lag4avg"] = fundq_df.groupby("gvkey")["atq"].transform(_lagged_4q_avg)

        # GPA = GP_TTM / TA_lag4avg
        with np.errstate(divide="ignore", invalid="ignore"):
            fundq_df["gpa_raw"] = fundq_df["gp_ttm"] / fundq_df["ta_lag4avg"]
        # Drop rows where divisor is invalid
        valid = fundq_df["ta_lag4avg"].notna() & (fundq_df["ta_lag4avg"] > 0) & fundq_df["gp_ttm"].notna()
        fundq_df.loc[~valid, "gpa_raw"] = np.nan

        # Winsorize
        fundq_df["gpa"] = fundq_df["gpa_raw"].clip(GPA_WINSORIZE_LOW, GPA_WINSORIZE_HIGH)

        # Market cap for universe ranking
        fundq_df["market_cap_at_q"] = fundq_df["cshoq"] * fundq_df["prccq"]

        # Filter to rdq in window
        fundq_df = fundq_df.dropna(subset=["rdq"])
        if hasattr(fundq_df["rdq"].iloc[0], "date"):
            fundq_df["rdq"] = fundq_df["rdq"].apply(
                lambda d: d.date() if hasattr(d, "date") else d
            )
        filtered = fundq_df[
            (fundq_df["rdq"] >= start_date) & (fundq_df["rdq"] <= end_date)
        ].copy()
        if filtered.empty:
            return pd.DataFrame()

        # Assemble output
        out = filtered[[
            "permno", "ticker", "gvkey", "fyearq", "fqtr", "rdq",
            "gp_ttm", "ta_lag4avg", "gpa_raw", "gpa", "market_cap_at_q",
        ]].copy()
        out["fiscal_yearq"] = (
            out["fyearq"].astype(int).astype(str)
            + "Q" + out["fqtr"].astype(int).astype(str)
        )
        out = out.drop(columns=["fyearq", "fqtr"])
        out["permno"] = out["permno"].astype(int)
        try:
            out["gvkey"] = out["gvkey"].astype(int)
        except (ValueError, TypeError):
            pass
        for col in ("gp_ttm", "ta_lag4avg", "gpa_raw", "gpa", "market_cap_at_q"):
            out[col] = out[col].astype(float)

        return out.sort_values(["fiscal_yearq", "ticker"]).reset_index(drop=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Public API ──────────────────────────────────────────────────────────────
def bulk_fetch_profitability_signal_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
    *,
    mock_mode:  Optional[bool] = None,
    use_cache:  bool = True,
    cache_path: Optional[Path] = None,
) -> ProfitabilitySignalPanelResult:
    """Bulk-fetch firm-quarter GPA panel with cache."""
    if mock_mode is None:
        mock_mode = not is_wrds_available()
    mode_str = "mock" if mock_mode else "wrds"

    path = cache_path if cache_path is not None else _PROFITABILITY_PANEL_CACHE_PATH
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    # Cache check
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
                and built_end >= end_date
            )
            if cache_ok:
                cached = pd.read_parquet(path)
                if "rdq" in cached.columns:
                    cached["rdq"] = pd.to_datetime(cached["rdq"]).dt.date
                logger.info("path_l: cache HIT — %d firm-quarters envelope [%s, %s]",
                            len(cached), built_start, built_end)
                filtered = cached[
                    (cached["ticker"].isin(tickers))
                    & (cached["rdq"] >= start_date)
                    & (cached["rdq"] <= end_date)
                ].reset_index(drop=True)
                return ProfitabilitySignalPanelResult(
                    panel=filtered, mode=mode_str, n_firm_quarters=len(filtered),
                    exclusion_stats={"from_cache": True},
                    window_start=start_date, window_end=end_date,
                )
        except Exception as exc:
            logger.warning("path_l cache load failed: %s — refetching", exc)

    # Cache miss → fetch
    logger.info("path_l: cache MISS — %s-fetching %d tickers [%s, %s]",
                mode_str, len(tickers), start_date, end_date)
    if mock_mode:
        panel = _mock_profitability_panel(tickers, start_date, end_date)
    else:
        panel = _real_profitability_panel(tickers, start_date, end_date)

    exclusion_stats = {
        "no_gpa":          int(panel["gpa"].isna().sum()) if "gpa" in panel.columns else 0,
        "winsorized_low":  int((panel.get("gpa_raw", pd.Series([])) <= GPA_WINSORIZE_LOW).sum()) if "gpa_raw" in panel.columns else 0,
        "winsorized_high": int((panel.get("gpa_raw", pd.Series([])) >= GPA_WINSORIZE_HIGH).sum()) if "gpa_raw" in panel.columns else 0,
    }

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
            logger.info("path_l: cache persisted: %d firm-quarters → %s", len(panel), path)
        except Exception as exc:
            logger.warning("path_l persist failed: %s", exc)

    return ProfitabilitySignalPanelResult(
        panel=panel, mode=mode_str, n_firm_quarters=len(panel),
        exclusion_stats=exclusion_stats,
        window_start=start_date, window_end=end_date,
    )
