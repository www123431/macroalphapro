"""
engine/factors/quality.py — Quality factor (Asness-Frazzini-Pedersen 2019, simplified 2-component).

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50) §2.2.3
Spec lock:
  - Equity-only scope: equity_sector + equity_factor (24 ETFs)
  - Top-10 holdings aggregation (reuses ETF Holdings Monitor ingestion pipeline)
  - 2-component simplification of AFP 2019 "Quality Minus Junk":
    1. Profitability: ROE (return on equity) via yfinance .info["returnOnEquity"]
    2. Growth: revenueGrowth via yfinance .info["revenueGrowth"]
  - Excluded sub-components per spec §rule-9 N16:
    - Safety: low-vol overlap with BAB (would double-count)
    - Payout: yfinance .info coverage sparse, defer to v2

  - Cross-section z-score standardization within equity universe at as_of

Literature: AFP 2019 "Quality Minus Junk" Sharpe ~0.4-0.6 on US large-cap equity
1957-2016 (full 4-component version).

NaN protocol (per spec §2.3):
  - Non-equity asset class → NaN (excluded)
  - ETF without holdings data → NaN
  - Holdings without ROE OR revenueGrowth → excluded from aggregate (NaN-aware)
  - All-NaN aggregate → NaN

Honest limitations:
  - Non-US ETF holdings (KWEB / ASHR / EWS / EWG / EWC / EWA / EWJ / INDA / VGK)
    have weaker yfinance fundamental coverage → effective coverage reduced
    on these ETFs (transparently reported in verdict template)
  - Top-10 represents 30-50% ETF weight; holdings 11+ unmonitored (same blind
    spot as ETF Holdings Monitor)
  - 2-component simplification is "Quality-light" not full QMJ
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# Locked scope (per spec §2.2.3)
EQUITY_ASSET_CLASSES: frozenset[str] = frozenset({"equity_sector", "equity_factor"})

# Locked components (2 of 4 from AFP 2019)
QUALITY_SUB_COMPONENTS: tuple[str, ...] = ("profitability", "growth")

# Walk-forward lookahead bias guard (v1 clarification amendment 2026-05-09)
# Rationale: yfinance .info returns CURRENT fundamentals (2026-near-realtime),
# NOT point-in-time historical. For walk-forward backtest at e.g. as_of=2015-01,
# .info would return 2026 ROE/revenueGrowth labeled as 2015 → lookahead.
# Fix: as_of < SPEC_LOCK_DATE → Quality factor returns all-NaN (excluded from
# ensemble per §2.3 NaN protocol). Forward live (as_of >= SPEC_LOCK_DATE) uses
# actual yfinance .info for real contribution measurement (real-time fundamentals
# at decision time = no lookahead).
# Two-stage verdict per spec §3.1 amendment:
#   - Walk-forward 1996-2024 verdict = 3-factor (TSMOM + Carry-eq + BAB) ensemble
#   - Forward live verdict (≥2026-05-09) = full 4-factor ensemble incl. Quality
SPEC_LOCK_DATE: datetime.date = datetime.date(2026, 5, 9)

# Cache for fundamentals (slow yfinance .info calls)
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "factor_quality"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_FUNDAMENTALS_CACHE_DIR = _DATA_DIR / "fundamentals_cache"
_FUNDAMENTALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_quality_signal(
    as_of:          datetime.date,
    universe:       list[str],
    asset_classes:  dict[str, str],
    use_cache:      bool = False,
) -> pd.Series:
    """
    Compute per-ETF Quality signal at as_of.

    Steps:
      1. For each equity ETF: fetch top-10 holdings (reuse Holdings Monitor pipeline)
      2. For each holding: fetch ROE + revenueGrowth from yfinance .info
      3. Aggregate per ETF: weighted average of available holdings' (ROE + revenueGrowth)
      4. Cross-section standardize (z-score) within equity universe
      5. Non-equity ETFs return NaN

    Returns:
        pd.Series indexed by ticker; equity ETF z-scores; non-equity NaN.
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)
    if asset_classes is None:
        raise ValueError("Quality requires asset_classes to enforce equity-only scope")

    # Walk-forward lookahead guard (v1 amendment 2026-05-09)
    # Historical as_of < SPEC_LOCK_DATE → Quality all-NaN (excluded from ensemble)
    # to avoid yfinance .info point-in-time-vs-current mismatch.
    if as_of < SPEC_LOCK_DATE:
        logger.info(
            "quality: walk-forward as_of=%s < SPEC_LOCK_DATE=%s; "
            "returning all-NaN per amendment to avoid yfinance .info lookahead bias",
            as_of, SPEC_LOCK_DATE,
        )
        return pd.Series(np.nan, index=universe, dtype=float)

    # Step 1-3: per-ETF raw aggregate (forward live only)
    raw_quality: dict[str, float] = {}
    for ticker in universe:
        ac = asset_classes.get(ticker)
        if ac not in EQUITY_ASSET_CLASSES:
            raw_quality[ticker] = np.nan
            continue
        try:
            agg = _compute_etf_quality_aggregate(ticker, as_of)
            raw_quality[ticker] = agg if agg is not None else np.nan
        except Exception as exc:
            logger.debug("quality: aggregate failed for %s: %s — NaN", ticker, exc)
            raw_quality[ticker] = np.nan

    # Step 4: cross-section z-score standardization within equity ETFs (non-NaN)
    equity_tickers = [
        t for t in universe
        if asset_classes.get(t) in EQUITY_ASSET_CLASSES
    ]
    equity_raw = pd.Series({t: raw_quality.get(t, np.nan) for t in equity_tickers})
    equity_valid = equity_raw.dropna()

    if len(equity_valid) < 2:
        # Insufficient data for cross-section standardization
        logger.warning(
            "quality: only %d equity ETFs with valid data at %s, skipping z-score",
            len(equity_valid), as_of,
        )
        return pd.Series({t: np.nan for t in universe}, dtype=float)

    mean = equity_valid.mean()
    std = equity_valid.std(ddof=0)
    if std < 1e-9:
        # All identical → z-scores all zero (no signal)
        z_scores = pd.Series({t: 0.0 for t in equity_valid.index})
    else:
        z_scores = (equity_valid - mean) / std

    # Step 5: combine — equity ETFs get z-score, non-equity NaN
    out = pd.Series({t: np.nan for t in universe}, dtype=float)
    for t, z in z_scores.items():
        out[t] = z
    return out


def _compute_etf_quality_aggregate(
    etf_ticker: str,
    as_of:      datetime.date,
) -> Optional[float]:
    """
    Aggregate ETF Quality from top-10 holdings:
      raw_quality_etf = Σ (holding_weight × holding_quality_score)
                       / Σ holding_weight (for holdings with valid data)

    Where holding_quality_score = (ROE + revenueGrowth) / 2
    (equal-weight 2-component per spec).

    Returns None if no holdings have valid quality data.
    """
    # Reuse ETF Holdings Monitor ingestion pipeline (Sprint Week 1-5)
    try:
        from engine.etf_holdings_ingestion import fetch_etf_top10_holdings
        holdings = fetch_etf_top10_holdings(etf_ticker, as_of, use_cache=True)
    except Exception as exc:
        logger.debug("quality: holdings fetch failed for %s: %s", etf_ticker, exc)
        return None

    if not holdings:
        return None

    weighted_quality_sum = 0.0
    weighted_total = 0.0
    n_valid = 0

    for h in holdings:
        name = str(h.get("name", "")).upper()
        weight = float(h.get("weight", 0.0))
        if weight <= 0 or not name:
            continue

        roe = _fetch_holding_metric(name, "returnOnEquity", as_of)
        rev_growth = _fetch_holding_metric(name, "revenueGrowth", as_of)

        # Need BOTH metrics for valid contribution (per spec 2-component lock)
        if roe is None or rev_growth is None:
            continue

        # Sanity: clip extreme values (yfinance occasionally returns outliers)
        roe_clipped = max(-1.0, min(1.0, roe))
        growth_clipped = max(-1.0, min(1.0, rev_growth))

        holding_quality = (roe_clipped + growth_clipped) / 2.0
        weighted_quality_sum += weight * holding_quality
        weighted_total += weight
        n_valid += 1

    if n_valid == 0 or weighted_total < 1e-9:
        return None

    return weighted_quality_sum / weighted_total


def _fetch_holding_metric(
    holding_ticker: str,
    metric_name:    str,
    as_of:          datetime.date,
) -> Optional[float]:
    """
    Fetch a single fundamental metric (ROE or revenueGrowth) from yfinance .info.

    Cached per (ticker, YYYYMM) to avoid repeated slow .info calls.
    Returns None if metric not available.

    Note: yfinance .info returns CURRENT snapshot, not point-in-time; for v1
    walk-forward this introduces minor lookahead (fundamentals at run time, not
    historical). Acceptable for capability MVP per spec §rule-9 N4 (data quality
    honest disclose). v2 candidate: Compustat point-in-time.
    """
    cache_key = f"{holding_ticker.upper()}_{as_of.strftime('%Y%m')}"
    cache_path = _FUNDAMENTALS_CACHE_DIR / f"{cache_key}.json"

    # Cache hit
    if cache_path.exists():
        try:
            import json
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            val = cached.get(metric_name)
            return float(val) if val is not None else None
        except Exception:
            pass  # fall through to fetch

    # Cache miss → fetch
    try:
        t = yf.Ticker(holding_ticker)
        info = t.info or {}
    except Exception as exc:
        logger.debug("quality: yf.Ticker(%s).info failed: %s", holding_ticker, exc)
        return None

    # Persist cache (write all metrics in one go to amortize cost)
    try:
        import json
        cache_data = {
            "returnOnEquity":  info.get("returnOnEquity"),
            "revenueGrowth":   info.get("revenueGrowth"),
            "fetched_at":      datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source":          "yfinance.info",
        }
        cache_path.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    except Exception:
        pass  # cache write failure non-fatal

    val = info.get(metric_name)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
