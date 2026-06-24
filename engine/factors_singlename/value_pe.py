"""
engine/factors_singlename/value_pe.py — Wave B P/E ratio value factor.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.2 Wave B
Literature: Fama-French 1993 — earnings-to-price as classical Value definition.

Status (2026-05-10): SKELETON + MOCK MODE.
  - Real path queries Compustat `comp.fundq` for trailing-12mo EPS via WRDS
  - Real path joins CRSP↔Compustat via `crsp.ccmxpf_lnkhist` (permno↔gvkey)
  - Activated when WRDS account approved (NUS BizFDB application 2026-05-09)
  - Until then, mock mode generates deterministic synthetic E/P z-scores so
    Wave B walk-forward smoke tests + factor-ensemble integration can run
    end-to-end pre-activation

Design mirror:
  - Public API matches `dividend_yield.py` exactly so factor_ensemble.compute_*
    consumes both Wave A (div yield) and Wave B (E/P) without per-factor wiring
  - Cross-section z-score with min-5 universe gate (same as dividend_yield)
  - Mock-real auto-routing via `crsp_loader.is_wrds_available()`

Why "forward 12mo" in spec means trailing-TTM (clarification):
  - Spec phrase "forward 12mo earnings yield (E/P, more robust than B/M for
    non-financials)" anchors on Fama-French 1993 which uses trailing TTM EPS.
  - "Forward 12mo" here is industry shorthand for "trailing 12mo of available
    earnings as of t, used to forecast forward returns" — NOT analyst-consensus
    forward EPS (which would require IBES, separate WRDS dataset).
  - This interpretation is consistent with the FF 1993 anchor.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Locked per spec §2.2 Wave B
EARNINGS_LOOKBACK_QUARTERS_LOCKED: int = 4   # trailing-4Q TTM EPS = "12mo earnings"
MIN_UNIVERSE_FOR_ZSCORE_LOCKED:    int = 5   # mirror dividend_yield gate


# ── Compustat query templates (real-mode stubs, activated on WRDS approval) ─
_COMPUSTAT_TTM_EPS_SQL_TEMPLATE = """
SELECT
    gvkey,
    datadate,
    rdq,        -- report date (point-in-time honesty: only knowable after rdq)
    epspxq,     -- EPS Basic Incl Extraordinary (quarterly)
    epspiq      -- EPS Basic Excl Extraordinary (quarterly)
FROM comp.fundq
WHERE rdq <= %(as_of)s
  AND rdq >= %(as_of_minus_2yr)s
  AND gvkey IN %(gvkeys)s
ORDER BY gvkey, datadate
"""

_CCM_LINK_SQL_TEMPLATE = """
SELECT lpermno, gvkey, linkdt, linkenddt, linktype, linkprim
FROM crsp.ccmxpf_lnkhist
WHERE linktype IN ('LU', 'LC')
  AND linkprim IN ('P', 'C')
  AND linkdt <= %(as_of)s
  AND (linkenddt >= %(as_of)s OR linkenddt IS NULL)
"""


# ── Public API ──────────────────────────────────────────────────────────────
def compute_value_pe_singlestock_signal(
    as_of:         datetime.date,
    universe:      list[str],
    asset_classes: Optional[dict[str, str]] = None,
    panel:         Optional[pd.DataFrame] = None,
    *,
    mock_mode:     Optional[bool] = None,
) -> pd.Series:
    """
    Wave B Value (E/P) cross-section z-score signal.

    Mechanism:
      1. For each ticker: trailing-4Q (TTM) EPS as of as_of
                          via Compustat fundq.epspxq sum, point-in-time by rdq
      2. Earnings yield (E/P) = TTM_EPS / price[as_of]
      3. Cross-section z-score within universe (high E/P → positive = "cheap")

    API mirror of `dividend_yield.compute_dividend_yield_singlestock_signal`
    so factor ensemble consumes Wave A and Wave B factors without bespoke wiring.

    Args:
        as_of:          decision date (no look-ahead — uses rdq ≤ as_of)
        universe:       list of tickers
        asset_classes:  ignored (factor signature consistency)
        panel:          pre-fetched daily price panel (required for price[as_of])
        mock_mode:      None (default) auto-detects via crsp_loader.is_wrds_available();
                        True forces synthetic E/P; False raises if WRDS unavailable.

    Returns:
        pd.Series indexed by ticker, continuous z-score; NaN for missing data.
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)
    if panel is None or panel.empty:
        logger.warning("compute_value_pe_singlestock_signal: panel required → all-NaN")
        return pd.Series(np.nan, index=universe, dtype=float)

    if mock_mode is None:
        try:
            from engine.universe_singlename.crsp_loader import is_wrds_available
            mock_mode = not is_wrds_available()
        except ImportError:
            mock_mode = True

    if mock_mode:
        return _mock_value_pe_signal(as_of, universe, panel)
    return _real_value_pe_signal(as_of, universe, panel)


# ── Mock mode (deterministic E/P synthesis) ─────────────────────────────────
def _mock_value_pe_signal(
    as_of:    datetime.date,
    universe: list[str],
    panel:    pd.DataFrame,
) -> pd.Series:
    """Synthetic E/P z-scores — deterministic per (ticker, as_of) seed.

    Mechanism:
      - Each (ticker, as_of) hashes to a seed → draws TTM_EPS ∈ [0.5, 12.0]
        (representing $0.50 - $12 EPS, realistic large-cap range)
      - Reads price[as_of] from the panel (real shape, real ticker)
      - Computes E/P, then cross-section z-score

    This gives Wave B harness real-shape z-score output (range ~[-2, +2])
    so smoke tests can verify factor-ensemble integration before WRDS is live.
    """
    raw_ep: dict[str, float] = {}
    for ticker in universe:
        if ticker not in panel.columns:
            raw_ep[ticker] = np.nan
            continue
        ts = panel[ticker].dropna()
        before = ts[ts.index <= pd.Timestamp(as_of)]
        if before.empty:
            raw_ep[ticker] = np.nan
            continue
        price = float(before.iloc[-1])
        if price <= 0:
            raw_ep[ticker] = np.nan
            continue

        # Deterministic synthetic TTM EPS: seed from (ticker, as_of)
        seed_str = f"{ticker}|{as_of.isoformat()}"
        seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        ttm_eps = float(rng.uniform(0.5, 12.0))

        raw_ep[ticker] = ttm_eps / price

    return _cross_section_zscore(raw_ep, universe)


# ── Real mode (WRDS Compustat query, activated 2026-05-11) ──────────────────
def _real_value_pe_signal(
    as_of:    datetime.date,
    universe: list[str],
    panel:    pd.DataFrame,
) -> pd.Series:
    """Retry-wrapped entry point (delegate to _real_value_pe_signal_inner).

    Transient WRDS errors (connection drops / EOF prompts) auto-retry per
    `engine.universe_singlename.wrds_retry.with_wrds_retry`.
    """
    from engine.universe_singlename.wrds_retry import with_wrds_retry
    wrapped = with_wrds_retry(max_attempts=3, base_delay=5.0)(
        _real_value_pe_signal_inner
    )
    return wrapped(as_of, universe, panel)


def _real_value_pe_signal_inner(
    as_of:    datetime.date,
    universe: list[str],
    panel:    pd.DataFrame,
) -> pd.Series:
    """Real WRDS-Compustat E/P query.

    Pipeline:
      1. Open WRDS connection (`crsp_loader._open_wrds_connection()`)
      2. ticker → permno via crsp.msenames (point-in-time at as_of)
      3. permno → gvkey via crsp.ccmxpf_lnkhist (linktype LU/LC, linkprim P/C)
      4. comp.fundq → 4 most recent quarters with rdq ≤ as_of per gvkey
         (rdq = report date, the date the earnings became publicly known —
          critical for no-look-ahead)
      5. TTM EPS = sum(epspxq) over 4 most recent quarters; skip ticker if < 4Q
      6. E/P = TTM_EPS / price[as_of]
      7. Cross-section z-score within universe

    Per FF 1993 + spec §2.2: TTM EPS / Price is the canonical earnings-yield
    factor; sign convention raw (high z = high E/P = "cheap"), expected_sign
    +1 in factor library (long high-z = long value).
    """
    try:
        from engine.universe_singlename.crsp_loader import (
            _open_wrds_connection,
            is_wrds_available,
        )
    except ImportError as exc:
        raise RuntimeError(f"crsp_loader unavailable: {exc}")

    if not is_wrds_available():
        raise RuntimeError(
            "WRDS not configured. Pass mock_mode=True for skeleton testing, "
            "or install wrds + configure credentials for real-data path."
        )

    conn = _open_wrds_connection()
    try:
        # Step 1: ticker → permno (point-in-time)
        names = conn.raw_sql(
            """
            SELECT permno, ticker, namedt, nameendt
            FROM crsp.msenames
            WHERE ticker IN %(tickers)s
              AND namedt  <= %(as_of)s
              AND nameendt >= %(as_of)s
            """,
            params={
                "tickers": tuple(universe),
                "as_of":   as_of.isoformat(),
            },
            date_cols=["namedt", "nameendt"],
        )
        if names.empty:
            return pd.Series(np.nan, index=universe, dtype=float)
        permnos = sorted(int(p) for p in names["permno"].dropna().unique())
        permno_to_ticker: dict[int, str] = {}
        for _, row in names.iterrows():
            permno_to_ticker[int(row["permno"])] = str(row["ticker"]).strip()

        # Step 2: permno → gvkey via CRSP-Compustat link table
        # linktype LU=USEDIT, LC=primary-link; linkprim P=primary, C=primary-but-different
        link = conn.raw_sql(
            """
            SELECT lpermno, gvkey, linkdt, linkenddt
            FROM crsp.ccmxpf_lnkhist
            WHERE lpermno IN %(permnos)s
              AND linktype IN ('LU', 'LC')
              AND linkprim IN ('P', 'C')
              AND linkdt <= %(as_of)s
              AND (linkenddt >= %(as_of)s OR linkenddt IS NULL)
            """,
            params={
                "permnos": tuple(permnos),
                "as_of":   as_of.isoformat(),
            },
            date_cols=["linkdt", "linkenddt"],
        )
        permno_to_gvkey: dict[int, str] = {}
        for _, row in link.iterrows():
            permno_to_gvkey[int(row["lpermno"])] = str(row["gvkey"])
        gvkeys = sorted(set(permno_to_gvkey.values()))
        if not gvkeys:
            return pd.Series(np.nan, index=universe, dtype=float)

        # Step 3: comp.fundq → quarters with rdq ≤ as_of, last ~2 years
        two_years_ago = (as_of - datetime.timedelta(days=730)).isoformat()
        fundq = conn.raw_sql(
            """
            SELECT gvkey, datadate, rdq, epspxq
            FROM comp.fundq
            WHERE gvkey IN %(gvkeys)s
              AND rdq IS NOT NULL
              AND rdq <= %(as_of)s
              AND rdq >= %(two_years_ago)s
              AND epspxq IS NOT NULL
            ORDER BY gvkey, rdq DESC, datadate DESC
            """,
            params={
                "gvkeys":         tuple(gvkeys),
                "as_of":          as_of.isoformat(),
                "two_years_ago":  two_years_ago,
            },
            date_cols=["datadate", "rdq"],
        )

        # Step 4: TTM EPS per gvkey = sum of 4 most recent unique quarters
        ttm_eps_by_gvkey: dict[str, float] = {}
        for gvkey, group in fundq.groupby("gvkey"):
            # Deduplicate by datadate (in case of restatements, take most-recent rdq)
            unique_quarters = (
                group.sort_values("rdq")
                     .drop_duplicates(subset=["datadate"], keep="last")
                     .sort_values("datadate", ascending=False)
            )
            if len(unique_quarters) < 4:
                continue   # need 4 quarters for TTM
            ttm = float(unique_quarters.head(4)["epspxq"].sum())
            ttm_eps_by_gvkey[str(gvkey)] = ttm

        # Step 5: query CRSP raw (unadjusted) prc per permno at as_of
        # CRITICAL: Compustat epspxq is in PRE-SPLIT units (as-reported);
        # using split-ADJUSTED panel price for E/P creates ratio inflation
        # whenever a post-as_of split occurred (e.g., NVDA 10:1 in 2024-06
        # inflates 2023 E/P 10x if using adjusted price).
        # Solution: query raw `prc` (unadjusted) for the as_of date.
        as_of_minus_7 = (as_of - datetime.timedelta(days=7)).isoformat()
        raw_prc = conn.raw_sql(
            """
            SELECT permno, date, prc
            FROM crsp.dsf
            WHERE permno IN %(permnos)s
              AND date BETWEEN %(start)s AND %(end)s
            ORDER BY permno, date DESC
            """,
            params={
                "permnos": tuple(permnos),
                "start":   as_of_minus_7,
                "end":     as_of.isoformat(),
            },
            date_cols=["date"],
        )
        permno_to_raw_price: dict[int, float] = {}
        for permno_id, group in raw_prc.groupby("permno"):
            latest = group.sort_values("date").iloc[-1]
            p = float(latest["prc"])
            permno_to_raw_price[int(permno_id)] = abs(p) if p != 0 else 0.0

        # Step 6: ticker → TTM EPS via permno → gvkey lookup; use RAW price
        raw_ep: dict[str, float] = {}
        for ticker in universe:
            raw_ep[ticker] = np.nan
            # Find permno for this ticker
            ticker_permnos = [
                p for p, t in permno_to_ticker.items() if t == ticker
            ]
            if not ticker_permnos:
                continue
            # Pick gvkey via the permno (first valid)
            gvkey = None
            chosen_permno = None
            for p in ticker_permnos:
                if p in permno_to_gvkey:
                    gvkey = permno_to_gvkey[p]
                    chosen_permno = p
                    break
            if gvkey is None or gvkey not in ttm_eps_by_gvkey:
                continue
            ttm = ttm_eps_by_gvkey[gvkey]

            # Use RAW price (not adjusted) to match Compustat EPS units
            price = permno_to_raw_price.get(chosen_permno)
            if price is None or price <= 0:
                continue

            raw_ep[ticker] = ttm / price

        return _cross_section_zscore(raw_ep, universe)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Shared: cross-section z-score (mirror dividend_yield logic) ─────────────
def _cross_section_zscore(
    raw_values: dict[str, float],
    universe:   list[str],
) -> pd.Series:
    """Cross-section z-score within universe; min-5 gate (mirror dividend_yield)."""
    raw_series = pd.Series(raw_values, dtype=float)
    valid = raw_series.dropna()
    if len(valid) < MIN_UNIVERSE_FOR_ZSCORE_LOCKED:
        return pd.Series(np.nan, index=universe, dtype=float)
    mean = float(valid.mean())
    std = float(valid.std(ddof=1))
    if std <= 1e-9:
        return pd.Series(np.nan, index=universe, dtype=float)

    out: dict[str, float] = {}
    for ticker in universe:
        v = raw_series.get(ticker, np.nan)
        out[ticker] = (v - mean) / std if np.isfinite(v) else np.nan
    return pd.Series(out, dtype=float)
