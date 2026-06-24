"""
engine/path_c/asset_growth_signal_panel.py — Path J Asset Growth firm-quarter panel.

Pre-registration: docs/spec_path_j_asset_growth_drift_v1.md (id=60) §2.2 + §2.3

Pulls + joins:
  - Compustat fundq (rdq + atq + cshoq + prccq)
  - CRSP msenames (ticker ↔ permno)
  - CRSP linkage via ccmxpf_lnkhist (permno ↔ gvkey)

For each firm-quarter (i, q) with rdq_iq:
  - atq_recent = atq[q] (current quarter total assets)
  - atq_prior  = atq[q-4] (4 quarters / 1 year ago)
  - market_cap_at_q = cshoq × prccq (for top-N filter / size diagnostic)

NOTE: Asset growth signal FORMULA (-1 × (atq_recent - atq_prior) / atq_prior)
lives in asset_growth_signal.py (Sprint J-3).
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
ATQ_LOOKBACK_QUARTERS:        int   = 4       # q vs q-4 (1y lookback)
MIN_ATQ_DOLLAR_M:             float = 100.0   # min $100M atq (edge-effects filter)
MAX_ABSOLUTE_GROWTH:          float = 5.0     # max |growth| = 500% (M&A filter)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _REPO_ROOT / "data" / "path_c_asset_growth"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_AG_PANEL_CACHE_PATH = _CACHE_DIR / "_asset_growth_signal_panel.parquet"


@dataclasses.dataclass(frozen=True)
class AssetGrowthSignalPanelResult:
    """Output of bulk_fetch_asset_growth_signal_panel.

    `panel` columns:
      - permno (int)
      - ticker (str)
      - gvkey (int)
      - fiscal_yearq (str): e.g. "2014Q1"
      - rdq (datetime.date)
      - atq_recent (float): atq at quarter q, USD millions
      - atq_prior (float): atq at quarter q-4 (1y prior), USD millions
      - market_cap_at_q (float): cshoq × prccq, USD millions
    """
    panel:           pd.DataFrame
    mode:            str          # "mock" or "wrds"
    n_firm_quarters: int
    exclusion_stats: dict
    window_start:    datetime.date
    window_end:      datetime.date


def is_wrds_available() -> bool:
    return _crsp_is_wrds_available()


# ── Mock-mode panel generator ──────────────────────────────────────────────
def _mock_asset_growth_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Synthetic deterministic asset-growth panel.

    Each firm: baseline atq ~ Uniform($200M-$5B), quarterly growth ~ Normal(2%, 5%).
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
        base_atq = float(rng.uniform(200.0, 5000.0))
        base_mcap = float(rng.uniform(100.0, 2500.0))
        gvkey_synth  = abs(seed) % 999_999
        permno_synth = (abs(seed) // 1000) % 99_999 + 10_000

        for q_start in q_starts:
            q_start_date = q_start.date()
            quarter_label = f"{q_start_date.year}Q{(q_start_date.month - 1) // 3 + 1}"
            q_end_offset = 90 + int(rng.integers(25, 46))
            rdq = q_start_date + datetime.timedelta(days=q_end_offset)
            if rdq > end_date:
                continue

            quarter_idx = (q_start_date.year - start_date.year) * 4 + (q_start_date.month - 1) // 3
            growth_recent = float(np.exp(rng.normal(0.02, 0.05) * quarter_idx / 4.0))
            growth_prior  = float(np.exp(rng.normal(0.02, 0.05) * (quarter_idx - 4) / 4.0))
            atq_recent = base_atq * growth_recent
            atq_prior  = base_atq * growth_prior if quarter_idx >= 4 else np.nan
            mcap = base_mcap * float(np.exp(rng.normal(0.02, 0.10) * quarter_idx / 4.0))

            rows.append({
                "permno":          permno_synth,
                "ticker":          ticker,
                "gvkey":           gvkey_synth,
                "fiscal_yearq":    quarter_label,
                "rdq":             rdq,
                "atq_recent":      atq_recent,
                "atq_prior":       atq_prior,
                "market_cap_at_q": mcap,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["fiscal_yearq", "ticker"]).reset_index(drop=True)


# ── SQL templates ─────────────────────────────────────────────────────────

# Compustat fundq with atq + market cap inputs
_COMP_FUNDQ_AG_SQL = """
SELECT
    gvkey,
    fyearq,
    fqtr,
    datadate,
    rdq,
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

# CRSP msenames ticker → permno
_CRSP_MSE_TICKER_SQL = """
SELECT DISTINCT permno, ticker
FROM crsp.msenames
WHERE ticker IN %(tickers)s
  AND nameendt >= %(start_date)s
  AND namedt <= %(end_date)s
"""

# CRSP <-> Compustat (gvkey by permno)
_CRSP_COMP_LINK_BY_PERMNO_SQL = """
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
def _real_asset_growth_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Build firm-quarter atq panel from WRDS Compustat."""
    if not is_wrds_available():
        raise RuntimeError("WRDS not configured. Pass mock_mode=True for skeleton testing.")
    if not tickers:
        return pd.DataFrame()

    needed_tickers = sorted(set(tickers))
    conn = _crsp_open_wrds_connection()
    try:
        # Step 1: ticker → permno
        logger.info("path_c.asset_growth_signal_panel: resolving %d tickers → permno via msenames",
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

        # Step 2: permno → gvkey
        link_df = conn.raw_sql(
            _CRSP_COMP_LINK_BY_PERMNO_SQL,
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

        # Step 3: pull fundq with 5Q lookback buffer (q-4 needed; +1Q slack)
        buffer_start = start_date - datetime.timedelta(days=(ATQ_LOOKBACK_QUARTERS + 1) * 92)
        logger.info(
            "path_c.asset_growth_signal_panel: pulling fundq for %d gvkeys × [%s, %s]",
            len(gvkeys), buffer_start, end_date,
        )
        fundq_df = conn.raw_sql(
            _COMP_FUNDQ_AG_SQL,
            params={
                "buffer_start": buffer_start.isoformat(),
                "end_date":     end_date.isoformat(),
                "gvkeys":       tuple(gvkeys),
            },
            date_cols=["datadate", "rdq"],
        )
        if fundq_df.empty:
            return pd.DataFrame()

        # Step 4: compute atq_recent + atq_prior via shift(4)
        fundq_df["gvkey"]  = fundq_df["gvkey"].astype(str)
        fundq_df["permno"] = fundq_df["gvkey"].map(gvkey_to_permno)
        fundq_df["ticker"] = fundq_df["permno"].map(permno_to_ticker)
        fundq_df = fundq_df.dropna(subset=["permno", "ticker"])
        fundq_df["market_cap_at_q"] = (
            fundq_df["cshoq"].astype(float) * fundq_df["prccq"].astype(float)
        )

        # Sort and shift by gvkey
        fundq_df = fundq_df.sort_values(["gvkey", "fyearq", "fqtr"]).reset_index(drop=True)
        fundq_df["atq_recent"] = fundq_df["atq"].astype(float)
        fundq_df["atq_prior"] = (
            fundq_df.groupby("gvkey")["atq_recent"].shift(ATQ_LOOKBACK_QUARTERS)
        )

        # Step 5: drop NA on rdq/fyearq/fqtr (defensive — Compustat may have nulls)
        fundq_df = fundq_df.dropna(subset=["rdq", "fyearq", "fqtr"])
        if fundq_df.empty:
            return pd.DataFrame()
        if hasattr(fundq_df["rdq"].iloc[0], "date"):
            fundq_df["rdq"] = fundq_df["rdq"].apply(
                lambda d: d.date() if hasattr(d, "date") else d
            )

        # Step 6: filter to rdq in window
        filtered = fundq_df[
            (fundq_df["rdq"] >= start_date) & (fundq_df["rdq"] <= end_date)
        ].copy()
        if filtered.empty:
            return pd.DataFrame()

        # Step 7: assemble output
        out = filtered[[
            "permno", "ticker", "gvkey", "fyearq", "fqtr", "rdq",
            "atq_recent", "atq_prior", "market_cap_at_q",
        ]].copy()
        out["fiscal_yearq"] = (
            out["fyearq"].astype(int).astype(str)
            + "Q" + out["fqtr"].astype(int).astype(str)
        )
        out = out.drop(columns=["fyearq", "fqtr"])
        out["permno"] = out["permno"].astype(int)
        out["gvkey"] = out["gvkey"].astype(int, errors="ignore")
        for col in ("atq_recent", "atq_prior", "market_cap_at_q"):
            out[col] = out[col].astype(float)

        return out.sort_values(["fiscal_yearq", "ticker"]).reset_index(drop=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Public API ──────────────────────────────────────────────────────────────
def bulk_fetch_asset_growth_signal_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
    *,
    mock_mode:  Optional[bool] = None,
    use_cache:  bool = True,
    cache_path: Optional[Path] = None,
) -> AssetGrowthSignalPanelResult:
    """Bulk-fetch firm-quarter asset-growth panel.

    Caller supplies ticker list (e.g., from load_russell2000_proxy_at_date).
    Window automatically includes 5Q lookback in real-path query.
    """
    if mock_mode is None:
        mock_mode = not is_wrds_available()
    mode_str = "mock" if mock_mode else "wrds"

    path = cache_path if cache_path is not None else _AG_PANEL_CACHE_PATH
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
                    "asset_growth_signal_panel cache HIT: %d firm-quarters envelope [%s, %s]",
                    len(cached), built_start, built_end,
                )
                filtered = cached[
                    (cached["ticker"].isin(tickers))
                    & (cached["rdq"] >= start_date)
                    & (cached["rdq"] <= end_date)
                ].reset_index(drop=True)
                return AssetGrowthSignalPanelResult(
                    panel=filtered, mode=mode_str,
                    n_firm_quarters=len(filtered),
                    exclusion_stats={"from_cache": True},
                    window_start=start_date, window_end=end_date,
                )
        except Exception as exc:
            logger.warning("asset_growth_signal_panel cache load failed: %s — refetching", exc)

    logger.info(
        "asset_growth_signal_panel cache MISS — %s-fetching %d tickers [%s, %s]",
        mode_str, len(tickers), start_date, end_date,
    )
    if mock_mode:
        panel = _mock_asset_growth_panel(tickers, start_date, end_date)
    else:
        panel = _real_asset_growth_panel(tickers, start_date, end_date)

    exclusion_stats = {
        "no_atq_recent":     0,
        "no_atq_prior":      0,
        "low_atq":           0,
        "abs_growth_extreme": 0,
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
            logger.info("asset_growth_signal_panel persisted: %d rows → %s", len(panel), path)
        except Exception as exc:
            logger.warning("asset_growth_signal_panel persist failed: %s", exc)

    return AssetGrowthSignalPanelResult(
        panel=panel, mode=mode_str, n_firm_quarters=len(panel),
        exclusion_stats=exclusion_stats,
        window_start=start_date, window_end=end_date,
    )
