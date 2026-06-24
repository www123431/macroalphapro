"""
engine/etf_holdings_ingestion.py — ETF top-10 holdings ingestion (Sprint Week 1).

Pre-registration: docs/spec_etf_holdings_llm_risk_monitor.md
Spec id (engine.preregistration.SpecRegistry): 49
Spec hash (registered 2026-05-08): 0c3696fc4145

Purpose
-------
Fetch top 10 holdings per equity ETF from yfinance fund_holdings API, with SEC
EDGAR 13F as fallback (deferred — yfinance proves reliable for major US ETFs).
Cache snapshots monthly. Output deduplicated unique names across the 24 equity
ETF screening universe for downstream LLM risk screening.

Boundary invariant (project rule "0-LLM-in-evaluation"):
  - Pure deterministic data ingestion, no LLM in this path.
  - Caller (etf_holdings_risk_monitor) handles LLM screening + aggregation.

Spec §2.2 covers ingestion design.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)

# Storage: data/etf_holdings/holdings_{etf}_{YYYYMM}.json
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "etf_holdings"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Spec §2.2 — Top N holdings per ETF (locked at register time)
TOP_N_HOLDINGS: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def fetch_etf_top10_holdings(
    etf_ticker: str,
    as_of: datetime.date,
    *,
    use_cache: bool = True,
) -> list[dict]:
    """
    Fetch top 10 holdings for ETF as of `as_of` month-end snapshot.

    Returns:
        List of {name, weight, rank} dicts, sorted by weight descending.
        Empty list on fetch failure (caller decides whether to skip).

    Cache:
        data/etf_holdings/{etf_ticker}_{YYYYMM}.json
        Re-fetched once per month-end (`as_of` truncated to YYYYMM).

    Source priority:
        1. yfinance Ticker.funds_data.top_holdings (covers most ETFs incl. SPDR/iShares/Invesco)
        2. SEC EDGAR 13F-HR (deferred — stub for v1; activate if yfinance failure rate > 5%)
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")

    cache_path = _DATA_DIR / f"{etf_ticker}_{as_of.strftime('%Y%m')}.json"

    if use_cache and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return payload.get("holdings", [])
        except Exception as exc:
            logger.warning(
                "etf_holdings_ingestion: cache read failed for %s, refetching: %s",
                cache_path.name, exc,
            )

    holdings = _fetch_via_yfinance(etf_ticker)
    source = "yfinance"

    if not holdings:
        # SEC EDGAR 13F fallback (US issuer ETFs) — stub for v1
        holdings = _fetch_via_sec_edgar_13f(etf_ticker, as_of)
        source = "sec_edgar_13f" if holdings else "fallback_failed"

    if not holdings:
        logger.warning(
            "etf_holdings_ingestion: no holdings returned for %s (yfinance + EDGAR fallback both failed)",
            etf_ticker,
        )
        return []

    # Persist cache snapshot
    payload = {
        "etf_ticker":         etf_ticker,
        "as_of_date":         as_of.isoformat(),
        "holdings":           holdings,
        "top_n":              len(holdings),
        "top_n_weight_share": round(sum(h["weight"] for h in holdings), 6),
        "source":             source,
        "fetched_at":         datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    try:
        cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("etf_holdings_ingestion: cache write failed: %s", exc)

    return holdings


def deduplicate_holdings_to_unique_names(
    holdings_by_etf: dict[str, list[dict]],
) -> set[str]:
    """
    Spec §2.2 — Across all input ETFs, return the set of unique holding names
    (uppercased tickers).

    AAPL appearing in QQQ + SMH + XLK + QUAL + MTUM → counted once.
    Used by caller to determine LLM screening workload (~120-150 unique names
    expected across 24 equity ETFs per spec).
    """
    names: set[str] = set()
    for etf, holdings in holdings_by_etf.items():
        for h in holdings:
            name = h.get("name")
            if name:
                names.add(str(name).upper().strip())
    return names


def fetch_all_equity_etf_holdings(
    as_of: datetime.date,
    *,
    use_cache: bool = True,
) -> dict[str, list[dict]]:
    """
    Spec §2.1 — Fetch top 10 holdings for the 24 equity ETF screening universe.

    Equity ETFs = equity_sector + equity_factor (per engine.universe_manager).
    Excludes commodity / fixed_income / volatility / fx (no corporate fundamentals).

    Returns:
        Dict {etf_ticker: holdings_list}. ETFs with 0 holdings are still
        included (with empty list) so caller can audit ingestion quality.
    """
    from engine.universe_manager import get_active_universe

    universe = get_active_universe(
        asset_classes=["equity_sector", "equity_factor"],
    )

    out: dict[str, list[dict]] = {}
    for sector, ticker in universe.items():
        holdings = fetch_etf_top10_holdings(ticker, as_of, use_cache=use_cache)
        out[ticker] = holdings
        if not holdings:
            logger.warning(
                "etf_holdings_ingestion: %s (%s) returned 0 holdings — quality flag",
                ticker, sector,
            )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Source-specific fetchers
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_via_yfinance(etf_ticker: str) -> list[dict]:
    """
    Primary source — yfinance Ticker.funds_data.top_holdings.

    Returns standardized list[{name, weight, rank}] sorted by weight descending,
    truncated to TOP_N_HOLDINGS (=10 per spec §2.2).

    Empty list on any fetch / parse failure (caller flags and falls back).
    """
    try:
        ticker = yf.Ticker(etf_ticker)
        funds_data = ticker.funds_data
        top_df = funds_data.top_holdings
    except Exception as exc:
        logger.debug(
            "etf_holdings_ingestion: yfinance funds_data.top_holdings failed for %s: %s",
            etf_ticker, exc,
        )
        return []

    if top_df is None:
        return []
    try:
        # DataFrame indexed by Symbol with Name + "Holding Percent" columns
        if not hasattr(top_df, "head") or len(top_df) == 0:
            return []
        df = top_df.head(TOP_N_HOLDINGS)
    except Exception as exc:
        logger.debug(
            "etf_holdings_ingestion: yfinance df shape unexpected for %s: %s",
            etf_ticker, exc,
        )
        return []

    out: list[dict] = []
    for rank, (symbol, row) in enumerate(df.iterrows(), 1):
        try:
            weight_val = row.get("Holding Percent")
            if weight_val is None:
                continue
            weight = float(weight_val)
            # Some yfinance returns weights in percent (e.g. 8.5) vs fraction (0.085);
            # auto-detect: if any weight > 1.0, treat as percent and normalize
            out.append({
                "name":   str(symbol).strip().upper(),
                "weight": weight,  # will normalize after collecting all
                "rank":   rank,
            })
        except Exception as exc:
            logger.debug(
                "etf_holdings_ingestion: row parse failed for %s rank=%d: %s",
                etf_ticker, rank, exc,
            )
            continue

    if not out:
        return []

    # Normalize weights to fraction (in case yfinance returned percent)
    max_w = max(h["weight"] for h in out)
    if max_w > 1.0:
        for h in out:
            h["weight"] = h["weight"] / 100.0

    return out


def _fetch_via_sec_edgar_13f(
    etf_ticker: str,
    as_of: datetime.date,
) -> list[dict]:
    """
    Fallback source — SEC EDGAR 13F-HR filings (US issuer quarterly disclosures).

    Status v1: STUB returning empty list. Full implementation deferred unless
    yfinance failure rate > 5% empirically. Reasoning:
      - Major US sector ETFs (SPDR/iShares/Invesco) reliable on yfinance funds_data.top_holdings
      - SEC 13F implementation requires CIK lookup + XML parsing + duplicate-form
        handling + value-to-weight normalization (~150 lines)
      - Only viable for US-issued ETFs; non-US ETFs (KWEB, ASHR, EWS, EWJ, INDA, VGK,
        EWG, EWC, EWA) excluded from EDGAR scope by definition

    To activate: implement edgar_13f_fetch + uncomment in fetch_etf_top10_holdings.
    """
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Quality audit helpers (used by Tier R + tests)
# ─────────────────────────────────────────────────────────────────────────────


def audit_universe_holdings_coverage(
    as_of: datetime.date,
) -> dict:
    """
    Run full equity ETF universe ingestion + quality stats.
    Used by tests + Tier R rule_etf_holdings_ingestion_quality.
    """
    holdings_by_etf = fetch_all_equity_etf_holdings(as_of)
    n_total_etfs = len(holdings_by_etf)
    n_with_holdings = sum(1 for h in holdings_by_etf.values() if h)
    n_empty = n_total_etfs - n_with_holdings
    unique_names = deduplicate_holdings_to_unique_names(holdings_by_etf)
    return {
        "as_of":             as_of.isoformat(),
        "n_total_etfs":      n_total_etfs,
        "n_with_holdings":   n_with_holdings,
        "n_empty":           n_empty,
        "n_unique_names":    len(unique_names),
        "etfs_empty":        sorted([
            t for t, h in holdings_by_etf.items() if not h
        ]),
        "coverage_pct":      round(n_with_holdings / n_total_etfs * 100, 1) if n_total_etfs else 0.0,
    }
