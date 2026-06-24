"""
engine/path_c/pead_ts_signal_panel.py — Path D PEAD time-series SUE panel builder.

Pre-registration: docs/spec_path_d_dhs_behavioral_2factor_v1.md (id=62) §2.2 + §2.3

B-T 1989 (Bernard-Thomas, JF) time-series standardized unexpected earnings.
Bypasses I/B/E/S blocker (id=57 PARKED) by using Compustat seasonal Δ EPS.

Pulls + joins:
  - Compustat fundq (rdq + epspxq basic EPS + ajexq split adjustment + cshoq + prccq + atq)
  - CRSP linkage via crsp.msenames + ccmxpf_lnkhist (ticker ↔ permno ↔ gvkey)

For each firm-quarter (i, q) with rdq_iq:
  - eps_adj_iq = epspxq_iq × ajexq_iq        (split-adjusted EPS)
  - delta_eps_iq = eps_adj_iq - eps_adj_{i, q-4}    (seasonal Δ, B-T 1989)
  - sigma_8q_iq = std(delta_eps over [q-8..q-1])    (8 PRIOR quarters, excluding current)
  - sue_raw_iq = delta_eps_iq / sigma_8q_iq
  - sue_winsorized_iq = clip(sue_raw_iq, -10, +10)
  - market_cap_at_q = cshoq × prccq            (for top-N universe ranking)

Cross-section rank + decile leg assignment lives in walk-forward orchestrator
(reused from engine.path_c.sue_signal.rank_within_quarter + assign_decile_legs).

Two modes:
  - mock_mode=True (default when WRDS unavailable): deterministic synthetic EPS panel
  - mock_mode=False: real WRDS query, decorated with with_wrds_retry

Disk cache: parquet at data/path_c_dhs/_pead_ts_signal_panel.parquet.
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
SEASONAL_LAG_QUARTERS:     int   = 4     # ΔEPS over q vs q-4 (annual seasonal)
SIGMA_WINDOW_QUARTERS:     int   = 8     # rolling std of seasonal Δ over 8 PRIOR quarters
SIGMA_MIN_PERIODS:         int   = 4     # need ≥ 4 prior seasonal Δ obs to compute σ
SIGMA_MIN_VALUE:           float = 0.01  # σ floor — below this → EXCLUDE thin (too small denominator)
SUE_WINSORIZE_LOW:         float = -10.0
SUE_WINSORIZE_HIGH:        float = +10.0


# Storage
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _REPO_ROOT / "data" / "path_c_dhs"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PEAD_TS_PANEL_CACHE_PATH = _CACHE_DIR / "_pead_ts_signal_panel.parquet"


@dataclasses.dataclass(frozen=True)
class PeadTsSignalPanelResult:
    """Output of bulk_fetch_pead_ts_signal_panel.

    `panel` columns:
      - permno (int): CRSP permno
      - ticker (str): primary US-exchange ticker
      - gvkey (int): Compustat firm identifier
      - fiscal_yearq (str): e.g. "2014Q1"
      - rdq (datetime.date): quarterly earnings announcement date
      - eps_adj (float): split-adjusted EPS for quarter q (epspxq × ajexq)
      - eps_adj_lag4 (float): split-adjusted EPS for quarter q-4
      - delta_eps (float): seasonal change eps_adj − eps_adj_lag4
      - sigma_8q (float): std of delta_eps over 8 PRIOR quarters [q-8..q-1]
      - sue_raw (float): delta_eps / sigma_8q
      - sue (float): winsorized sue_raw to [-10, +10]
      - market_cap_at_q (float): cshoq × prccq USD millions (universe ranking)
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
def _mock_pead_ts_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Deterministic synthetic PEAD-TS panel.

    Per-firm baseline EPS ∈ [$0.10, $5.00], seasonal growth Normal(2%, 8%) annualized.
    Adds occasional earnings surprises (5% chance of ±50% deviation).
    Generates ≥ 12 quarters of synthetic history per ticker for σ computation feasibility.
    """
    if not tickers:
        return pd.DataFrame()

    rows = []
    # Build with 12Q lookback buffer for σ window
    buffer_start = start_date - datetime.timedelta(days=365 * 3 + 90)
    q_starts = pd.date_range(start=buffer_start, end=end_date, freq="QS").to_pydatetime()
    if len(q_starts) == 0:
        return pd.DataFrame()

    for ticker in sorted(set(tickers)):
        seed = int(hashlib.md5(ticker.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        base_eps          = float(rng.uniform(0.10, 5.00))     # quarterly USD
        base_mcap         = float(rng.uniform(5_000.0, 50_000.0))  # USD M
        gvkey_synth       = abs(seed) % 999_999
        permno_synth      = (abs(seed) // 1000) % 99_999 + 10_000

        # Generate per-firm time series first, then derive seasonal Δ + σ
        firm_history = []
        cum_growth = 1.0
        prior_eps = base_eps
        for idx, q_start in enumerate(q_starts):
            q_start_date = q_start.date()
            quarter_label = f"{q_start_date.year}Q{(q_start_date.month - 1) // 3 + 1}"
            q_end_offset = 90 + int(rng.integers(25, 46))
            rdq = q_start_date + datetime.timedelta(days=q_end_offset)

            # Seasonal pattern: same fiscal quarter slightly correlated
            base_for_quarter = base_eps * (1.0 + 0.05 * ((idx % 4) - 1.5))  # mild Q-seasonality
            growth_factor = float(rng.normal(0.005, 0.04))                  # quarterly growth shock
            surprise_factor = 1.0
            if rng.random() < 0.05:
                surprise_factor = 1.0 + float(rng.normal(0.0, 0.40))
            eps_q = max(0.01, base_for_quarter * cum_growth * (1.0 + growth_factor) * surprise_factor)
            cum_growth *= (1.0 + growth_factor * 0.3)
            ajexq = 1.0  # mock: no splits
            firm_history.append({
                "q_idx":            idx,
                "fiscal_yearq":     quarter_label,
                "rdq":              rdq,
                "eps":              float(eps_q),
                "eps_adj":          float(eps_q * ajexq),
                "ajexq":            float(ajexq),
                "mcap":             base_mcap * float(np.exp(rng.normal(0.01, 0.04) * idx / 4.0)),
            })

        # Compute seasonal Δ + σ per firm
        for j, h in enumerate(firm_history):
            if h["rdq"] < start_date or h["rdq"] > end_date:
                continue
            if j < SEASONAL_LAG_QUARTERS:
                continue  # not enough lookback for seasonal diff
            eps_lag4 = firm_history[j - SEASONAL_LAG_QUARTERS]["eps_adj"]
            delta_eps = h["eps_adj"] - eps_lag4
            # σ window: indices [j-8..j-1] = 8 prior quarters
            lo = max(0, j - SIGMA_WINDOW_QUARTERS)
            hi = j  # exclusive upper bound = exclude current
            if hi - lo < SIGMA_MIN_PERIODS:
                continue  # too few prior obs
            prior_deltas = []
            for k in range(lo, hi):
                if k >= SEASONAL_LAG_QUARTERS:
                    prior_deltas.append(firm_history[k]["eps_adj"] - firm_history[k - SEASONAL_LAG_QUARTERS]["eps_adj"])
            if len(prior_deltas) < SIGMA_MIN_PERIODS:
                continue
            sigma_8q = float(np.std(prior_deltas, ddof=1))
            if not np.isfinite(sigma_8q) or sigma_8q < SIGMA_MIN_VALUE:
                continue
            sue_raw = delta_eps / sigma_8q
            sue = float(np.clip(sue_raw, SUE_WINSORIZE_LOW, SUE_WINSORIZE_HIGH))

            rows.append({
                "permno":           permno_synth,
                "ticker":           ticker,
                "gvkey":            gvkey_synth,
                "fiscal_yearq":     h["fiscal_yearq"],
                "rdq":              h["rdq"],
                "eps_adj":          h["eps_adj"],
                "eps_adj_lag4":     eps_lag4,
                "delta_eps":        float(delta_eps),
                "sigma_8q":         sigma_8q,
                "sue_raw":          float(sue_raw),
                "sue":              sue,
                "market_cap_at_q":  h["mcap"],
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["fiscal_yearq", "ticker"]).reset_index(drop=True)


# ── Real WRDS queries ──────────────────────────────────────────────────────

# Compustat fundq with EPS + split adjustment + market cap inputs
# Pulls 12Q lookback buffer for σ window (8Q + 4Q seasonal lag)
_COMP_FUNDQ_PEAD_TS_SQL = """
SELECT
    gvkey,
    fyearq,
    fqtr,
    datadate,
    rdq,
    epspxq,
    ajexq,
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

# CRSP msenames: ticker → permno
_CRSP_MSE_TICKER_SQL = """
SELECT DISTINCT permno, ticker
FROM crsp.msenames
WHERE ticker IN %(tickers)s
  AND nameendt >= %(start_date)s
  AND namedt <= %(end_date)s
"""

# CRSP <-> Compustat linkage (permno ↔ gvkey)
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
def _real_pead_ts_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Build firm-quarter PEAD-TS panel from WRDS Compustat.

    Pipeline:
      1. Open WRDS connection
      2. ticker → permno via crsp.msenames
      3. permno → gvkey via crsp.ccmxpf_lnkhist
      4. Pull comp.fundq for those gvkeys with 12Q buffer (8Q σ + 4Q seasonal)
      5. Per firm, sort by (fyearq, fqtr) and compute eps_adj, seasonal Δ, σ_8q
      6. SUE = ΔEPS / σ_8q; winsorize ±10
      7. Filter to rdq in window
      8. Return long-form firm-quarter DataFrame
    """
    if not is_wrds_available():
        raise RuntimeError(
            "WRDS not configured. Pass mock_mode=True for skeleton testing."
        )
    if not tickers:
        return pd.DataFrame()

    needed_tickers = sorted(set(tickers))
    conn = _crsp_open_wrds_connection()
    try:
        # Step 1: ticker → permno
        logger.info("path_c.pead_ts_signal_panel: resolving %d tickers → permno via msenames",
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
            logger.warning("path_c.pead_ts_signal_panel: no permno matches")
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
            logger.warning("path_c.pead_ts_signal_panel: no gvkey linkages")
            return pd.DataFrame()
        link_df = link_df.dropna(subset=["gvkey", "permno"]).sort_values(
            ["permno", "gvkey"]
        ).drop_duplicates(subset=["permno"], keep="first")
        permno_to_gvkey: dict[int, str] = {
            int(r["permno"]): str(r["gvkey"]) for _, r in link_df.iterrows()
        }
        gvkey_to_permno = {v: k for k, v in permno_to_gvkey.items()}
        gvkeys = sorted(set(permno_to_gvkey.values()))

        # Step 3: pull fundq with 12Q lookback buffer (~3 years)
        # 12 quarters = ~1095 days; add 90 day rdq slack
        buffer_start = start_date - datetime.timedelta(days=1095 + 90)
        logger.info("path_c.pead_ts_signal_panel: pulling fundq for %d gvkeys × [%s, %s]",
                    len(gvkeys), buffer_start, end_date)
        fundq_df = conn.raw_sql(
            _COMP_FUNDQ_PEAD_TS_SQL,
            params={
                "buffer_start": buffer_start.isoformat(),
                "end_date":     end_date.isoformat(),
                "gvkeys":       tuple(gvkeys),
            },
            date_cols=["datadate", "rdq"],
        )
        if fundq_df.empty:
            logger.warning("path_c.pead_ts_signal_panel: no fundq rows")
            return pd.DataFrame()

        # Step 4: per firm timeseries computation
        fundq_df["gvkey"] = fundq_df["gvkey"].astype(str)
        fundq_df["permno"] = fundq_df["gvkey"].map(gvkey_to_permno)
        fundq_df["ticker"] = fundq_df["permno"].map(permno_to_ticker)
        fundq_df = fundq_df.dropna(subset=["permno", "ticker"])

        # Drop NA on critical fields BEFORE timeseries ops (id=59 lesson — fyearq NaN crashes int conversion)
        fundq_df = fundq_df.dropna(subset=["fyearq", "fqtr"])
        if fundq_df.empty:
            return pd.DataFrame()

        fundq_df["epspxq_f"] = fundq_df["epspxq"].astype(float)
        fundq_df["ajexq_f"]  = fundq_df["ajexq"].astype(float)
        # eps_adj = epspxq × ajexq (split-adjusted EPS)
        # If ajexq NULL, default to 1.0 (no splits assumed); disclose in honest_disclose
        fundq_df["ajexq_f"] = fundq_df["ajexq_f"].fillna(1.0)
        fundq_df["eps_adj"] = fundq_df["epspxq_f"] * fundq_df["ajexq_f"]

        fundq_df["market_cap_at_q"] = (
            fundq_df["cshoq"].astype(float) * fundq_df["prccq"].astype(float)
        )

        # Sort by gvkey + fiscal time
        fundq_df = fundq_df.sort_values(["gvkey", "fyearq", "fqtr"]).reset_index(drop=True)

        # Seasonal Δ: eps_adj − eps_adj_{q-4} within firm
        fundq_df["eps_adj_lag4"] = fundq_df.groupby("gvkey")["eps_adj"].shift(SEASONAL_LAG_QUARTERS)
        fundq_df["delta_eps"]    = fundq_df["eps_adj"] - fundq_df["eps_adj_lag4"]

        # σ over 8 PRIOR quarters of delta_eps (exclude current via shift)
        def _rolling_prior_std(s: pd.Series) -> pd.Series:
            # Rolling std over last 8 obs including current, then shift by 1 to drop current
            return s.rolling(window=SIGMA_WINDOW_QUARTERS,
                             min_periods=SIGMA_MIN_PERIODS).std().shift(1)

        fundq_df["sigma_8q"] = (
            fundq_df.groupby("gvkey")["delta_eps"].transform(_rolling_prior_std)
        )

        # SUE = Δ / σ, winsorize, mark thin σ as NaN
        sigma_ok = fundq_df["sigma_8q"].astype(float) >= SIGMA_MIN_VALUE
        fundq_df.loc[~sigma_ok, "sigma_8q"] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            fundq_df["sue_raw"] = fundq_df["delta_eps"] / fundq_df["sigma_8q"]
        fundq_df["sue"] = fundq_df["sue_raw"].clip(SUE_WINSORIZE_LOW, SUE_WINSORIZE_HIGH)

        # Step 5: filter to rdq in window + drop NA on rdq
        fundq_df = fundq_df.dropna(subset=["rdq"])
        if hasattr(fundq_df["rdq"].iloc[0], "date"):
            fundq_df["rdq"] = fundq_df["rdq"].apply(lambda d: d.date() if hasattr(d, "date") else d)
        filtered = fundq_df[
            (fundq_df["rdq"] >= start_date) & (fundq_df["rdq"] <= end_date)
        ].copy()

        if filtered.empty:
            return pd.DataFrame()

        # Step 6: assemble output
        out = filtered[[
            "permno", "ticker", "gvkey", "fyearq", "fqtr", "rdq",
            "eps_adj", "eps_adj_lag4", "delta_eps", "sigma_8q",
            "sue_raw", "sue", "market_cap_at_q",
        ]].copy()
        out["fiscal_yearq"] = (
            out["fyearq"].astype(int).astype(str)
            + "Q" + out["fqtr"].astype(int).astype(str)
        )
        out = out.drop(columns=["fyearq", "fqtr"])
        out["permno"] = out["permno"].astype(int)
        out["gvkey"] = out["gvkey"].astype(int, errors="ignore")
        for col in ("eps_adj", "eps_adj_lag4", "delta_eps", "sigma_8q",
                    "sue_raw", "sue", "market_cap_at_q"):
            out[col] = out[col].astype(float)

        return out.sort_values(["fiscal_yearq", "ticker"]).reset_index(drop=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Public API ──────────────────────────────────────────────────────────────
def bulk_fetch_pead_ts_signal_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
    *,
    mock_mode:  Optional[bool] = None,
    use_cache:  bool = True,
    cache_path: Optional[Path] = None,
) -> PeadTsSignalPanelResult:
    """Bulk-fetch firm-quarter PEAD-TS panel with cache.

    Window auto-extends 12Q lookback in real-WRDS query for σ computation.
    """
    if mock_mode is None:
        mock_mode = not is_wrds_available()
    mode_str = "mock" if mock_mode else "wrds"

    path = cache_path if cache_path is not None else _PEAD_TS_PANEL_CACHE_PATH
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    # Cache load
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
                    "pead_ts_signal_panel cache HIT: %d firm-quarters envelope [%s, %s]",
                    len(cached), built_start, built_end,
                )
                filtered = cached[
                    (cached["ticker"].isin(tickers))
                    & (cached["rdq"] >= start_date)
                    & (cached["rdq"] <= end_date)
                ].reset_index(drop=True)
                return PeadTsSignalPanelResult(
                    panel=filtered, mode=mode_str,
                    n_firm_quarters=len(filtered),
                    exclusion_stats={"from_cache": True},
                    window_start=start_date, window_end=end_date,
                )
        except Exception as exc:
            logger.warning("pead_ts_signal_panel cache load failed: %s — refetching", exc)

    # Cache miss → fetch
    logger.info(
        "pead_ts_signal_panel cache MISS — %s-fetching %d tickers [%s, %s]",
        mode_str, len(tickers), start_date, end_date,
    )
    if mock_mode:
        panel = _mock_pead_ts_panel(tickers, start_date, end_date)
    else:
        panel = _real_pead_ts_panel(tickers, start_date, end_date)

    exclusion_stats = {
        "no_rdq":            0,
        "no_eps_lag4":       0,
        "thin_sigma":        0,
        "winsorized":        0,
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
            logger.info("pead_ts_signal_panel cache persisted: %d firm-quarters → %s",
                        len(panel), path)
        except Exception as exc:
            logger.warning("pead_ts_signal_panel persist failed: %s", exc)

    return PeadTsSignalPanelResult(
        panel=panel, mode=mode_str, n_firm_quarters=len(panel),
        exclusion_stats=exclusion_stats,
        window_start=start_date, window_end=end_date,
    )
