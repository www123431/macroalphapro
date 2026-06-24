"""
engine/path_c/fin_signal_panel.py — Path D FIN factor raw panel builder.

Pre-registration: docs/spec_path_d_dhs_behavioral_2factor_v1.md (id=62) §2.4 + §六

DHS 2020 (Daniel-Hirshleifer-Sun JF) FIN composite — simplified short-horizon:
  - NSI (Net Stock Issues, Daniel-Titman 2006): trailing 4Q share growth
  - ACC (Sloan 1996 working capital accruals): scaled by lagged assets

This module produces RAW per-firm-quarter NSI + ACC_scaled values.
Cross-section z-norm + FIN composite + decile leg live in fin_signal.py.

For each firm-quarter (i, q) with rdq_iq:
  - shares_adj_iq = cshoq_iq × ajexq_iq
  - NSI_iq = log(shares_adj_iq / shares_adj_{i, q-4})
  - Sloan working capital accruals:
      ΔACT  = actq_iq − actq_{i, q-1}
      ΔCASH = cheq_iq − cheq_{i, q-1}      (default 0 if cheq NULL)
      ΔLCT  = lctq_iq − lctq_{i, q-1}
      ΔDLC  = dlcq_iq − dlcq_{i, q-1}      (default 0 if dlcq NULL)
      ΔTXP  = txpq_iq − txpq_{i, q-1}      (default 0 if txpq NULL)
      DEP   = dpq_iq                       (quarterly depreciation expense)
      ACC   = (ΔACT − ΔCASH) − (ΔLCT − ΔDLC − ΔTXP) − DEP
      ACC_scaled = ACC / atq_{i, q-1}
  - Winsorize: NSI to [-0.5, +1.0], ACC_scaled to [-0.3, +0.3]
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


# Locked from spec §2.4 + §六
NSI_LAG_QUARTERS:        int   = 4       # NSI = log Δ adjusted shares over 4Q
NSI_WINSORIZE_LOW:       float = -0.5    # 50% buyback max
NSI_WINSORIZE_HIGH:      float = +1.0    # 100% issuance max
ACC_WINSORIZE_LOW:       float = -0.3
ACC_WINSORIZE_HIGH:      float = +0.3


# Storage
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _REPO_ROOT / "data" / "path_c_dhs"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_FIN_PANEL_CACHE_PATH = _CACHE_DIR / "_fin_signal_panel.parquet"


@dataclasses.dataclass(frozen=True)
class FinSignalPanelResult:
    """Output of bulk_fetch_fin_signal_panel.

    `panel` columns:
      - permno, ticker, gvkey, fiscal_yearq, rdq
      - shares_adj (float): cshoq × ajexq
      - shares_adj_lag4 (float): split-adjusted shares q-4
      - nsi_raw (float): log(shares_adj / shares_adj_lag4)
      - nsi (float): winsorized nsi_raw to [-0.5, +1.0]
      - acc_raw (float): Sloan working capital accruals (USD M)
      - atq_lag1 (float): total assets q-1 (denominator)
      - acc_scaled_raw (float): acc_raw / atq_lag1
      - acc_scaled (float): winsorized acc_scaled_raw to [-0.3, +0.3]
      - market_cap_at_q (float): cshoq × prccq
    """
    panel:           pd.DataFrame
    mode:            str
    n_firm_quarters: int
    exclusion_stats: dict
    window_start:    datetime.date
    window_end:      datetime.date


def is_wrds_available() -> bool:
    return _crsp_is_wrds_available()


# ── Mock-mode panel generator ──────────────────────────────────────────────
def _mock_fin_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Deterministic synthetic FIN panel.

    Per-firm baseline shares ∈ [100M, 5B]; quarterly issuance ~Normal(0.2%, 1.5%).
    Working capital accruals ~Normal(0, 1.5% of lagged atq) — typical industrial range.
    Generates ≥ 8 quarters lookback for NSI/ACC computation feasibility.
    """
    if not tickers:
        return pd.DataFrame()

    rows = []
    buffer_start = start_date - datetime.timedelta(days=365 * 2 + 90)
    q_starts = pd.date_range(start=buffer_start, end=end_date, freq="QS").to_pydatetime()
    if len(q_starts) == 0:
        return pd.DataFrame()

    for ticker in sorted(set(tickers)):
        seed = int(hashlib.md5(ticker.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        base_shares       = float(rng.uniform(100.0, 5000.0))      # M shares
        base_atq          = float(rng.uniform(1000.0, 100000.0))   # M USD
        base_mcap         = float(rng.uniform(5_000.0, 50_000.0))  # M USD
        gvkey_synth       = abs(seed) % 999_999
        permno_synth      = (abs(seed) // 1000) % 99_999 + 10_000

        # Build firm history first
        firm_history = []
        cum_shares = base_shares
        cum_atq    = base_atq
        for idx, q_start in enumerate(q_starts):
            q_start_date = q_start.date()
            quarter_label = f"{q_start_date.year}Q{(q_start_date.month - 1) // 3 + 1}"
            q_end_offset = 90 + int(rng.integers(25, 46))
            rdq = q_start_date + datetime.timedelta(days=q_end_offset)
            # Quarterly share issuance + atq growth
            issuance = float(rng.normal(0.002, 0.015))
            atq_growth = float(rng.normal(0.005, 0.02))
            cum_shares *= (1.0 + issuance)
            cum_atq    *= (1.0 + atq_growth)
            # Working capital accruals raw (USD M, scaled to ~1-2% of atq typical)
            acc_raw = float(rng.normal(0.0, 0.015) * cum_atq)
            firm_history.append({
                "q_idx":          idx,
                "fiscal_yearq":   quarter_label,
                "rdq":            rdq,
                "shares_adj":     cum_shares,
                "atq":            cum_atq,
                "acc_raw":        acc_raw,
                "mcap":           base_mcap * float(np.exp(rng.normal(0.01, 0.04) * idx / 4.0)),
            })

        for j, h in enumerate(firm_history):
            if h["rdq"] < start_date or h["rdq"] > end_date:
                continue
            # Need q-4 for NSI, q-1 for ACC scaling
            if j < max(NSI_LAG_QUARTERS, 1):
                continue
            shares_lag4 = firm_history[j - NSI_LAG_QUARTERS]["shares_adj"]
            atq_lag1    = firm_history[j - 1]["atq"]
            if shares_lag4 <= 0 or atq_lag1 <= 0:
                continue
            nsi_raw = float(np.log(h["shares_adj"] / shares_lag4))
            nsi = float(np.clip(nsi_raw, NSI_WINSORIZE_LOW, NSI_WINSORIZE_HIGH))
            acc_scaled_raw = float(h["acc_raw"] / atq_lag1)
            acc_scaled = float(np.clip(acc_scaled_raw, ACC_WINSORIZE_LOW, ACC_WINSORIZE_HIGH))

            rows.append({
                "permno":           permno_synth,
                "ticker":           ticker,
                "gvkey":            gvkey_synth,
                "fiscal_yearq":     h["fiscal_yearq"],
                "rdq":              h["rdq"],
                "shares_adj":       h["shares_adj"],
                "shares_adj_lag4":  shares_lag4,
                "nsi_raw":          nsi_raw,
                "nsi":              nsi,
                "acc_raw":          h["acc_raw"],
                "atq_lag1":         atq_lag1,
                "acc_scaled_raw":   acc_scaled_raw,
                "acc_scaled":       acc_scaled,
                "market_cap_at_q":  h["mcap"],
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["fiscal_yearq", "ticker"]).reset_index(drop=True)


# ── Real WRDS queries ──────────────────────────────────────────────────────

# Compustat fundq with full balance sheet + share + ajexq fields
_COMP_FUNDQ_FIN_SQL = """
SELECT
    gvkey,
    fyearq,
    fqtr,
    datadate,
    rdq,
    cshoq,
    ajexq,
    atq,
    actq,
    lctq,
    cheq,
    dlcq,
    txpq,
    dpq,
    niq,
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

# CRSP msenames: ticker → permno
_CRSP_MSE_TICKER_SQL = """
SELECT DISTINCT permno, ticker
FROM crsp.msenames
WHERE ticker IN %(tickers)s
  AND nameendt >= %(start_date)s
  AND namedt <= %(end_date)s
"""

# CRSP <-> Compustat linkage
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
def _real_fin_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Build firm-quarter FIN raw panel from WRDS Compustat."""
    if not is_wrds_available():
        raise RuntimeError(
            "WRDS not configured. Pass mock_mode=True for skeleton testing."
        )
    if not tickers:
        return pd.DataFrame()

    needed_tickers = sorted(set(tickers))
    conn = _crsp_open_wrds_connection()
    try:
        logger.info("path_c.fin_signal_panel: resolving %d tickers → permno",
                    len(needed_tickers))
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
        permno_to_gvkey: dict[int, str] = {
            int(r["permno"]): str(r["gvkey"]) for _, r in link_df.iterrows()
        }
        gvkey_to_permno = {v: k for k, v in permno_to_gvkey.items()}
        gvkeys = sorted(set(permno_to_gvkey.values()))

        # 8Q lookback buffer (4Q NSI + 1Q ACC lag + slack)
        buffer_start = start_date - datetime.timedelta(days=730 + 90)
        logger.info("path_c.fin_signal_panel: pulling fundq for %d gvkeys",
                    len(gvkeys))
        fundq_df = conn.raw_sql(
            _COMP_FUNDQ_FIN_SQL,
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
        fundq_df = fundq_df.dropna(subset=["permno", "ticker"])
        fundq_df = fundq_df.dropna(subset=["fyearq", "fqtr"])
        if fundq_df.empty:
            return pd.DataFrame()

        # Cast numerics + Sloan default-zero handling
        for col in ("cshoq", "ajexq", "atq", "actq", "lctq", "cheq",
                    "dlcq", "txpq", "dpq", "niq", "prccq"):
            fundq_df[col] = fundq_df[col].astype(float)
        # Sloan standard: cheq / dlcq / txpq default 0 if NULL (per spec §2.4 fallback)
        fundq_df["cheq"] = fundq_df["cheq"].fillna(0.0)
        fundq_df["dlcq"] = fundq_df["dlcq"].fillna(0.0)
        fundq_df["txpq"] = fundq_df["txpq"].fillna(0.0)
        # ajexq NULL → assume no splits (1.0)
        fundq_df["ajexq"] = fundq_df["ajexq"].fillna(1.0)

        fundq_df["shares_adj"] = fundq_df["cshoq"] * fundq_df["ajexq"]
        fundq_df["market_cap_at_q"] = fundq_df["cshoq"] * fundq_df["prccq"]

        # Sort by gvkey + fiscal time
        fundq_df = fundq_df.sort_values(["gvkey", "fyearq", "fqtr"]).reset_index(drop=True)

        # Per-firm timeseries shifts
        g = fundq_df.groupby("gvkey")
        fundq_df["shares_adj_lag4"] = g["shares_adj"].shift(NSI_LAG_QUARTERS)
        fundq_df["atq_lag1"]        = g["atq"].shift(1)
        fundq_df["actq_lag1"]       = g["actq"].shift(1)
        fundq_df["lctq_lag1"]       = g["lctq"].shift(1)
        fundq_df["cheq_lag1"]       = g["cheq"].shift(1)
        fundq_df["dlcq_lag1"]       = g["dlcq"].shift(1)
        fundq_df["txpq_lag1"]       = g["txpq"].shift(1)

        # NSI
        valid_nsi = (fundq_df["shares_adj_lag4"] > 0) & (fundq_df["shares_adj"] > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            fundq_df["nsi_raw"] = np.where(
                valid_nsi,
                np.log(fundq_df["shares_adj"] / fundq_df["shares_adj_lag4"]),
                np.nan,
            )
        fundq_df["nsi"] = fundq_df["nsi_raw"].clip(NSI_WINSORIZE_LOW, NSI_WINSORIZE_HIGH)

        # Sloan working capital accruals
        d_act  = fundq_df["actq"] - fundq_df["actq_lag1"]
        d_cash = fundq_df["cheq"] - fundq_df["cheq_lag1"]
        d_lct  = fundq_df["lctq"] - fundq_df["lctq_lag1"]
        d_dlc  = fundq_df["dlcq"] - fundq_df["dlcq_lag1"]
        d_txp  = fundq_df["txpq"] - fundq_df["txpq_lag1"]
        fundq_df["acc_raw"] = (d_act - d_cash) - (d_lct - d_dlc - d_txp) - fundq_df["dpq"]

        valid_acc = (fundq_df["atq_lag1"] > 0) & fundq_df["acc_raw"].notna()
        with np.errstate(divide="ignore", invalid="ignore"):
            fundq_df["acc_scaled_raw"] = np.where(
                valid_acc,
                fundq_df["acc_raw"] / fundq_df["atq_lag1"],
                np.nan,
            )
        fundq_df["acc_scaled"] = fundq_df["acc_scaled_raw"].clip(
            ACC_WINSORIZE_LOW, ACC_WINSORIZE_HIGH
        )

        # Drop rows with both signals NaN (no FIN computable)
        fundq_df = fundq_df.dropna(subset=["rdq"])
        both_null = fundq_df["nsi_raw"].isna() & fundq_df["acc_scaled_raw"].isna()
        fundq_df = fundq_df[~both_null]
        if fundq_df.empty:
            return pd.DataFrame()

        if hasattr(fundq_df["rdq"].iloc[0], "date"):
            fundq_df["rdq"] = fundq_df["rdq"].apply(lambda d: d.date() if hasattr(d, "date") else d)
        filtered = fundq_df[
            (fundq_df["rdq"] >= start_date) & (fundq_df["rdq"] <= end_date)
        ].copy()
        if filtered.empty:
            return pd.DataFrame()

        out = filtered[[
            "permno", "ticker", "gvkey", "fyearq", "fqtr", "rdq",
            "shares_adj", "shares_adj_lag4", "nsi_raw", "nsi",
            "acc_raw", "atq_lag1", "acc_scaled_raw", "acc_scaled",
            "market_cap_at_q",
        ]].copy()
        out["fiscal_yearq"] = (
            out["fyearq"].astype(int).astype(str)
            + "Q" + out["fqtr"].astype(int).astype(str)
        )
        out = out.drop(columns=["fyearq", "fqtr"])
        out["permno"] = out["permno"].astype(int)
        out["gvkey"] = out["gvkey"].astype(int, errors="ignore")
        for col in ("shares_adj", "shares_adj_lag4", "nsi_raw", "nsi",
                    "acc_raw", "atq_lag1", "acc_scaled_raw", "acc_scaled",
                    "market_cap_at_q"):
            out[col] = out[col].astype(float)
        return out.sort_values(["fiscal_yearq", "ticker"]).reset_index(drop=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Public API ──────────────────────────────────────────────────────────────
def bulk_fetch_fin_signal_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
    *,
    mock_mode:  Optional[bool] = None,
    use_cache:  bool = True,
    cache_path: Optional[Path] = None,
) -> FinSignalPanelResult:
    """Bulk-fetch firm-quarter FIN raw panel with cache."""
    if mock_mode is None:
        mock_mode = not is_wrds_available()
    mode_str = "mock" if mock_mode else "wrds"

    path = cache_path if cache_path is not None else _FIN_PANEL_CACHE_PATH
    meta_path = path.with_suffix(path.suffix + ".meta.json")

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
                    "fin_signal_panel cache HIT: %d firm-quarters envelope [%s, %s]",
                    len(cached), built_start, built_end,
                )
                filtered = cached[
                    (cached["ticker"].isin(tickers))
                    & (cached["rdq"] >= start_date)
                    & (cached["rdq"] <= end_date)
                ].reset_index(drop=True)
                return FinSignalPanelResult(
                    panel=filtered, mode=mode_str,
                    n_firm_quarters=len(filtered),
                    exclusion_stats={"from_cache": True},
                    window_start=start_date, window_end=end_date,
                )
        except Exception as exc:
            logger.warning("fin_signal_panel cache load failed: %s — refetching", exc)

    logger.info(
        "fin_signal_panel cache MISS — %s-fetching %d tickers [%s, %s]",
        mode_str, len(tickers), start_date, end_date,
    )
    if mock_mode:
        panel = _mock_fin_panel(tickers, start_date, end_date)
    else:
        panel = _real_fin_panel(tickers, start_date, end_date)

    exclusion_stats = {
        "no_rdq":               0,
        "no_shares_lag4":       0,
        "no_atq_lag1":          0,
        "both_signals_null":    0,
        "winsorized":           0,
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
            logger.info("fin_signal_panel cache persisted: %d firm-quarters → %s",
                        len(panel), path)
        except Exception as exc:
            logger.warning("fin_signal_panel persist failed: %s", exc)

    return FinSignalPanelResult(
        panel=panel, mode=mode_str, n_firm_quarters=len(panel),
        exclusion_stats=exclusion_stats,
        window_start=start_date, window_end=end_date,
    )
