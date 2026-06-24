"""
engine/path_c/rd_signal_panel.py — Path I R&D Premium Drift firm-quarter panel builder.

Pre-registration: docs/spec_path_i_rd_premium_drift_v1.md (id=59) §2.2 + §2.3

Pulls + joins:
  - Compustat fundq (rdq + xrdq quarterly R&D + atq total assets + cshoq + prccq)
  - CRSP linkage via crsp.ccmxpf_lnkhist + msenames (gvkey ↔ permno ↔ ticker)

For each firm-quarter (i, q) with rdq_iq:
  - r_and_d_4q_recent = SUM(xrdq) for q-3..q (4 quarters ending q inclusive)
  - r_and_d_4q_prior  = SUM(xrdq) for q-7..q-4 (4 quarters before recent)
  - n_quarters_recent = COUNT(xrdq NOT NULL) over recent window
  - n_quarters_prior  = COUNT(xrdq NOT NULL) over prior window
  - atq + market_cap_at_q

NOTE: R&D signal FORMULA composition (RD_growth × log(1 + intensity × 100))
lives in engine/path_c/rd_signal.py (Sprint I-3), mirrors labor pattern.
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
R_AND_D_RECENT_QUARTERS:        int   = 4    # trailing 4Q sum (recent window)
R_AND_D_PRIOR_QUARTERS:         int   = 4    # trailing 4Q sum (prior baseline window)
R_AND_D_MIN_DISCLOSED_QUARTERS: int   = 2    # need ≥2 disclosed in each window
R_AND_D_MIN_DOLLAR_M:           float = 1.0  # both recent + prior windows must sum to ≥ $1M


# Storage
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _REPO_ROOT / "data" / "path_c_rd"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_RD_PANEL_CACHE_PATH = _CACHE_DIR / "_rd_signal_panel.parquet"


@dataclasses.dataclass(frozen=True)
class RdSignalPanelResult:
    """Output of bulk_fetch_rd_signal_panel.

    `panel` columns:
      - permno (int): CRSP permno
      - ticker (str): primary US-exchange ticker
      - gvkey (int): Compustat firm identifier
      - fiscal_yearq (str): e.g. "2014Q1"
      - rdq (datetime.date): quarterly earnings announcement date
      - r_and_d_4q_recent (float): trailing-4Q R&D sum USD millions
      - r_and_d_4q_prior (float): prior-4Q R&D sum USD millions (4-7Q ago)
      - n_quarters_recent (int): count of non-null xrdq in recent 4Q window
      - n_quarters_prior (int): count of non-null xrdq in prior 4Q window
      - atq (float): total assets at quarter-end USD millions
      - market_cap_at_q (float): cshoq × prccq USD millions
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
def _mock_rd_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Deterministic synthetic R&D-signal panel.

    Per-firm: baseline R&D ~ Uniform($5M-$500M annual = $1.25M-$125M quarterly),
    quarterly growth ~ Normal(2%, 5%) (firms grow R&D over time).
    n_quarters_recent / prior ~ 4 with 5% chance of dropping a quarter.
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
        base_quarterly_rd = float(rng.uniform(1.25, 125.0))     # USD M
        base_atq          = float(rng.uniform(1000.0, 100000.0))  # USD M total assets
        base_mcap         = float(rng.uniform(5_000.0, 50_000.0)) # USD M market cap
        gvkey_synth       = abs(seed) % 999_999
        permno_synth      = (abs(seed) // 1000) % 99_999 + 10_000

        for q_start in q_starts:
            q_start_date = q_start.date()
            quarter_label = f"{q_start_date.year}Q{(q_start_date.month - 1) // 3 + 1}"
            q_end_offset = 90 + int(rng.integers(25, 46))
            rdq = q_start_date + datetime.timedelta(days=q_end_offset)
            if rdq > end_date:
                continue

            growth_recent = float(rng.normal(0.02, 0.05))       # avg 2% quarterly growth
            growth_prior  = float(rng.normal(0.02, 0.05))
            # 4-quarter sums (with random scaling)
            r_recent = base_quarterly_rd * 4.0 * (1.0 + growth_recent)
            r_prior  = base_quarterly_rd * 4.0 * (1.0 + growth_prior - 0.02)  # prior slightly less on avg
            n_q_recent = 4 if rng.random() > 0.05 else int(rng.integers(2, 4))
            n_q_prior  = 4 if rng.random() > 0.05 else int(rng.integers(2, 4))
            atq_q     = base_atq * float(np.exp(rng.normal(0.01, 0.03)))
            quarter_idx = (q_start_date.year - start_date.year) * 4 + (q_start_date.month - 1) // 3
            mcap = base_mcap * float(np.exp(rng.normal(0.02, 0.10) * quarter_idx / 4.0))

            rows.append({
                "permno":             permno_synth,
                "ticker":             ticker,
                "gvkey":              gvkey_synth,
                "fiscal_yearq":       quarter_label,
                "rdq":                rdq,
                "r_and_d_4q_recent":  r_recent,
                "r_and_d_4q_prior":   r_prior,
                "n_quarters_recent":  n_q_recent,
                "n_quarters_prior":   n_q_prior,
                "atq":                atq_q,
                "market_cap_at_q":    mcap,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["fiscal_yearq", "ticker"]).reset_index(drop=True)


# ── Real WRDS queries ──────────────────────────────────────────────────────

# Compustat fundq with R&D + assets + market cap inputs (subscribed, full 10y)
# Pulls 8Q lookback buffer (recent 4Q + prior 4Q = 8Q total)
_COMP_FUNDQ_RD_SQL = """
SELECT
    gvkey,
    fyearq,
    fqtr,
    datadate,
    rdq,
    xrdq,
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

# CRSP msenames: ticker → permno + gvkey via linkage
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
def _real_rd_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """Build firm-quarter R&D-signal panel from WRDS Compustat.

    Pipeline:
      1. Open WRDS connection
      2. ticker → permno via crsp.msenames
      3. permno → gvkey via crsp.ccmxpf_lnkhist
      4. Pull comp.fundq for those gvkeys with 8Q buffer for R&D lookback
      5. Per firm, sort by (fyearq, fqtr) and compute trailing 4Q + prior 4Q sums
      6. Filter to rdq in [start, end]
      7. Compute market_cap_at_q = cshoq × prccq
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
        logger.info("path_c.rd_signal_panel: resolving %d tickers → permno via msenames",
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
            logger.warning("path_c.rd_signal_panel: no permno matches")
            return pd.DataFrame()
        # Keep first permno per ticker (deterministic by permno asc)
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
            logger.warning("path_c.rd_signal_panel: no gvkey linkages")
            return pd.DataFrame()
        # First gvkey per permno
        link_df = link_df.dropna(subset=["gvkey", "permno"]).sort_values(
            ["permno", "gvkey"]
        ).drop_duplicates(subset=["permno"], keep="first")
        permno_to_gvkey: dict[int, str] = {
            int(r["permno"]): str(r["gvkey"]) for _, r in link_df.iterrows()
        }
        gvkey_to_permno = {v: k for k, v in permno_to_gvkey.items()}
        gvkeys = sorted(set(permno_to_gvkey.values()))

        # Step 3: pull fundq with 8Q lookback buffer (~2 years)
        # 8 quarters = ~730 days; add 90 day rdq slack
        buffer_start = start_date - datetime.timedelta(days=730 + 90)
        logger.info("path_c.rd_signal_panel: pulling fundq for %d gvkeys × [%s, %s]",
                    len(gvkeys), buffer_start, end_date)
        fundq_df = conn.raw_sql(
            _COMP_FUNDQ_RD_SQL,
            params={
                "buffer_start": buffer_start.isoformat(),
                "end_date":     end_date.isoformat(),
                "gvkeys":       tuple(gvkeys),
            },
            date_cols=["datadate", "rdq"],
        )
        if fundq_df.empty:
            logger.warning("path_c.rd_signal_panel: no fundq rows")
            return pd.DataFrame()

        # Step 4: per firm, sort by (fyearq, fqtr) then compute trailing sums
        fundq_df["gvkey"] = fundq_df["gvkey"].astype(str)
        fundq_df["permno"] = fundq_df["gvkey"].map(gvkey_to_permno)
        fundq_df["ticker"] = fundq_df["permno"].map(permno_to_ticker)
        fundq_df = fundq_df.dropna(subset=["permno", "ticker"])
        fundq_df["market_cap_at_q"] = (
            fundq_df["cshoq"].astype(float) * fundq_df["prccq"].astype(float)
        )

        # Sort and compute rolling 4Q sums per firm
        fundq_df = fundq_df.sort_values(["gvkey", "fyearq", "fqtr"]).reset_index(drop=True)
        fundq_df["xrdq_float"] = fundq_df["xrdq"].astype(float)
        # Rolling 4Q sum (including current quarter) using GroupBy.rolling
        fundq_df["r_and_d_4q_recent"] = (
            fundq_df.groupby("gvkey")["xrdq_float"]
                    .rolling(window=R_AND_D_RECENT_QUARTERS, min_periods=1)
                    .sum().reset_index(level=0, drop=True)
        )
        # n_quarters_recent: count of non-null xrdq in window
        fundq_df["xrdq_notnull"] = fundq_df["xrdq_float"].notna().astype(int)
        fundq_df["n_quarters_recent"] = (
            fundq_df.groupby("gvkey")["xrdq_notnull"]
                    .rolling(window=R_AND_D_RECENT_QUARTERS, min_periods=1)
                    .sum().reset_index(level=0, drop=True)
        )
        # Prior 4Q = shift recent by 4 quarters
        fundq_df["r_and_d_4q_prior"] = (
            fundq_df.groupby("gvkey")["r_and_d_4q_recent"].shift(R_AND_D_PRIOR_QUARTERS)
        )
        fundq_df["n_quarters_prior"] = (
            fundq_df.groupby("gvkey")["n_quarters_recent"].shift(R_AND_D_PRIOR_QUARTERS)
        )

        # Step 5: filter to rdq in window
        # Real-data fix 2026-05-12: drop NA on rdq AND fyearq/fqtr (some
        # Compustat fundq rows have missing fiscal year/quarter info,
        # especially for delisted firms / re-organizations / early years).
        # Small 1y smoke didn't surface; 10y window did.
        fundq_df = fundq_df.dropna(subset=["rdq", "fyearq", "fqtr"])
        if fundq_df.empty:
            return pd.DataFrame()
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
            "r_and_d_4q_recent", "r_and_d_4q_prior",
            "n_quarters_recent", "n_quarters_prior",
            "atq", "market_cap_at_q",
        ]].copy()
        # fiscal_yearq from fyearq/fqtr (NA already dropped above)
        out["fiscal_yearq"] = (
            out["fyearq"].astype(int).astype(str)
            + "Q" + out["fqtr"].astype(int).astype(str)
        )
        out = out.drop(columns=["fyearq", "fqtr"])
        out["permno"] = out["permno"].astype(int)
        out["gvkey"] = out["gvkey"].astype(int, errors="ignore")
        out["n_quarters_recent"] = out["n_quarters_recent"].fillna(0).astype(int)
        out["n_quarters_prior"]  = out["n_quarters_prior"].fillna(0).astype(int)
        for col in ("r_and_d_4q_recent", "r_and_d_4q_prior", "atq", "market_cap_at_q"):
            out[col] = out[col].astype(float)

        return out.sort_values(["fiscal_yearq", "ticker"]).reset_index(drop=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Public API ──────────────────────────────────────────────────────────────
def bulk_fetch_rd_signal_panel(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
    *,
    mock_mode:  Optional[bool] = None,
    use_cache:  bool = True,
    cache_path: Optional[Path] = None,
) -> RdSignalPanelResult:
    """Bulk-fetch firm-quarter R&D-signal panel.

    Caller must supply tickers; window includes 8Q lookback buffer automatically
    in the underlying real-WRDS query.
    """
    if mock_mode is None:
        mock_mode = not is_wrds_available()
    mode_str = "mock" if mock_mode else "wrds"

    path = cache_path if cache_path is not None else _RD_PANEL_CACHE_PATH
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
                    "rd_signal_panel cache HIT: %d firm-quarters envelope [%s, %s]",
                    len(cached), built_start, built_end,
                )
                filtered = cached[
                    (cached["ticker"].isin(tickers))
                    & (cached["rdq"] >= start_date)
                    & (cached["rdq"] <= end_date)
                ].reset_index(drop=True)
                return RdSignalPanelResult(
                    panel=filtered, mode=mode_str,
                    n_firm_quarters=len(filtered),
                    exclusion_stats={"from_cache": True},
                    window_start=start_date, window_end=end_date,
                )
        except Exception as exc:
            logger.warning("rd_signal_panel cache load failed: %s — refetching", exc)

    # Cache miss → fetch
    logger.info(
        "rd_signal_panel cache MISS — %s-fetching %d tickers [%s, %s]",
        mode_str, len(tickers), start_date, end_date,
    )
    if mock_mode:
        panel = _mock_rd_panel(tickers, start_date, end_date)
    else:
        panel = _real_rd_panel(tickers, start_date, end_date)

    exclusion_stats = {
        "no_rdq":                0,
        "thin_recent_quarters":  0,   # populated by rd_signal.py downstream
        "thin_prior_quarters":   0,
        "low_dollar":            0,
        "low_atq":               0,
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
            logger.info("rd_signal_panel cache persisted: %d firm-quarters → %s",
                        len(panel), path)
        except Exception as exc:
            logger.warning("rd_signal_panel persist failed: %s", exc)

    return RdSignalPanelResult(
        panel=panel, mode=mode_str, n_firm_quarters=len(panel),
        exclusion_stats=exclusion_stats,
        window_start=start_date, window_end=end_date,
    )
