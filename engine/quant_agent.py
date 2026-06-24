"""
Quant Agent
===========
Pure-quant, non-LLM specialized agent. Wraps engine/signal.py and
engine/regime.py with structured output schemas, composite scoring,
ATR calculation, and vol-parity sizing.

This agent runs first in the daily batch pipeline and feeds
QuantAssessment objects into ResearchAgent prompts and the watchlist
state machine.

Composite score (0-100)
-----------------------
  Delegated entirely to signal.compute_composite_scores() to ensure the gate
  logic (get_quant_gates) and the QuantAssessment record use an identical formula.
  Formula: TSMOM 40% + CSMOM percentile 30% + Sharpe 20% + Carry 10%.

Gate rule
---------
  composite < COMPOSITE_GATE_MIN  → blocked
  regime = risk-off               → blocked (new entries only)

ATR stop-loss convention
------------------------
  stop_price = trailing_high_since_entry - 2 × ATR(21)
  (ATR period 21 ≈ one trading month, aligns with monthly rebalance horizon)
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from engine.history import get_active_sector_etf
from engine.signal import get_signal_dataframe, compute_composite_scores
from engine.regime import get_regime_on
from engine.trading_schema import (
    COMPOSITE_GATE_MIN,
    EntryCondition,
    InvalidationCondition,
    PositionRank,
    QuantAssessment,
    TradeRecommendation,
    WEIGHT_LIMITS,
)

logger = logging.getLogger(__name__)

# ── ATR / SMA helpers ─────────────────────────────────────────────────────────

def _wilder_atr(tr: np.ndarray, period: int) -> float:
    """Wilder EMA of True Range — seed with simple mean, then smooth."""
    if len(tr) < period:
        return 0.0
    atr = float(np.mean(tr[:period]))
    for t in tr[period:]:
        atr = (atr * (period - 1) + float(t)) / period
    return atr


def _fetch_price_context(
    ticker: str,
    as_of: datetime.date,
    atr_period: int = 21,
) -> tuple[float, float]:
    """
    Single-ticker price context for patrol/stop-loss checks.
    Returns (atr, ann_vol) where ann_vol is annualised realised vol
    over the same window used by the ATR calculation.
    Returns (0.0, 0.0) on any data failure.
    """
    lookback = atr_period * 3 + 10
    start = as_of - datetime.timedelta(days=lookback * 2)
    end   = as_of + datetime.timedelta(days=1)
    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False,
                          multi_level_index=False)
        if raw.empty or len(raw) < atr_period + 2:
            return 0.0, 0.0
        sub = raw[["Close", "High", "Low"]].dropna()
        closes = sub["Close"].values.astype(float)
        highs  = sub["High"].values.astype(float)
        lows   = sub["Low"].values.astype(float)
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:]  - closes[:-1]),
            ),
        )
        atr     = _wilder_atr(tr, atr_period)
        ret_arr = np.diff(closes) / closes[:-1]
        ann_vol = float(np.std(ret_arr[-atr_period:]) * np.sqrt(252)) if len(ret_arr) >= atr_period else 0.0
        return atr, ann_vol
    except Exception as exc:
        logger.debug("_fetch_price_context(%s): %s", ticker, exc)
        return 0.0, 0.0


def _batch_fetch_price_context(
    sector_etf: dict[str, str],
    as_of: datetime.date,
    sma_period: int = 200,
) -> dict[str, tuple[float, float, float]]:
    """
    Single yf.download for all tickers; compute ATR(21), ATR(63), SMA(200) ratio.

    Returns {sector: (atr_21, atr_63, sma_ratio)}.
    (0.0, 0.0, 0.0) per sector on any data failure.

    ATR(21) ≈ monthly  — use for tactical entries / early warning
    ATR(63) ≈ quarterly — use for stop-loss on 3-6 month positions
    """
    tickers = list(sector_etf.values())
    sectors_by_ticker = {v: k for k, v in sector_etf.items()}
    result: dict[str, tuple[float, float, float]] = {s: (0.0, 0.0, 0.0) for s in sector_etf}

    lookback = sma_period * 2
    start = as_of - datetime.timedelta(days=lookback)
    end   = as_of + datetime.timedelta(days=1)

    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return result

        # yfinance returns multi-level columns (field, ticker) for multiple tickers;
        # flat columns for a single ticker — normalise to multi-level.
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([raw.columns, [tickers[0]]])

        for ticker in tickers:
            sector = sectors_by_ticker.get(ticker)
            if sector is None:
                continue
            try:
                sub = raw.xs(ticker, level=1, axis=1).dropna(subset=["Close", "High", "Low"])
                if len(sub) < 22:
                    continue

                closes = sub["Close"].values.astype(float)
                highs  = sub["High"].values.astype(float)
                lows   = sub["Low"].values.astype(float)

                tr = np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(
                        np.abs(highs[1:] - closes[:-1]),
                        np.abs(lows[1:]  - closes[:-1]),
                    ),
                )

                atr_21 = _wilder_atr(tr, 21)
                atr_63 = _wilder_atr(tr, 63)

                sma_ratio = 0.0
                if len(closes) >= sma_period:
                    sma = float(np.mean(closes[-sma_period:]))
                    sma_ratio = (closes[-1] / sma) - 1.0 if sma > 0 else 0.0

                result[sector] = (atr_21, atr_63, sma_ratio)
            except Exception as exc:
                logger.debug("_batch_fetch_price_context(%s): %s", ticker, exc)

    except Exception as exc:
        logger.debug("_batch_fetch_price_context bulk download failed: %s", exc)

    return result


# ── Vol-parity weight ─────────────────────────────────────────────────────────

def _vol_parity_weight(ann_vol: float, universe_vols: list[float]) -> float:
    if ann_vol <= 0:
        return 1.0 / max(len(universe_vols), 1)
    inv_vols = [1.0 / v for v in universe_vols if v > 0]
    return (1.0 / ann_vol) / sum(inv_vols) if inv_vols else 1.0 / max(len(universe_vols), 1)


# ── CSMOM ordinal ranks ───────────────────────────────────────────────────────

def _csmom_ranks(signal_df: pd.DataFrame) -> dict[str, int]:
    if "raw_return" in signal_df.columns:
        ordered = signal_df["raw_return"].sort_values(ascending=False).index.tolist()
    else:
        ordered = signal_df.index.tolist()
    return {sector: i + 1 for i, sector in enumerate(ordered)}


# ── Public API ────────────────────────────────────────────────────────────────

def run_quant_assessment(
    as_of:   datetime.date,
    sectors: Optional[list[str]] = None,
    fetch_price_context: bool = True,
) -> list[QuantAssessment]:
    """
    Produce QuantAssessment for all (or a subset of) sectors as of `as_of`.

    Args:
        as_of                : reference date (T-day)
        sectors              : optional filter; None = full universe
        fetch_price_context  : set False to skip ATR/SMA fetches (faster, for tests)

    Returns:
        List of QuantAssessment, one per sector with available signal data.
        Empty list if signal data unavailable.
    """
    sector_etf = get_active_sector_etf()
    if sectors:
        sector_etf = {k: v for k, v in sector_etf.items() if k in sectors}

    signal_df = get_signal_dataframe(as_of=as_of)
    if signal_df.empty:
        logger.warning("QuantAgent: empty signal_df for %s", as_of)
        return []

    regime_result = get_regime_on(as_of)
    regime_label  = getattr(regime_result, "regime",    "transition")
    p_risk_on     = getattr(regime_result, "p_risk_on", 0.5)

    # Unified composite scores — same formula used by get_quant_gates() in signal.py
    scores_df = compute_composite_scores(as_of)

    universe_vols = [
        float(signal_df.loc[s, "ann_vol"])
        for s in sector_etf
        if s in signal_df.index and signal_df.loc[s, "ann_vol"] > 0
    ]
    ranks      = _csmom_ranks(signal_df)   # ordinal rank still needed for QuantAssessment field
    n_sectors  = len(sector_etf)

    # Single batch download for all tickers — eliminates N serial HTTP calls
    if fetch_price_context:
        price_ctx = _batch_fetch_price_context(sector_etf, as_of)
    else:
        price_ctx = {s: (0.0, 0.0, 0.0) for s in sector_etf}

    results: list[QuantAssessment] = []

    for sector, ticker in sector_etf.items():
        if sector not in signal_df.index:
            continue

        row      = signal_df.loc[sector]
        tsmom    = int(row.get("tsmom", 0))
        ann_vol  = float(row.get("ann_vol", 0.20))
        raw_ret  = float(row.get("raw_return", 0.0))
        rank     = ranks.get(sector, n_sectors // 2 + 1)

        # Use unified score from signal.compute_composite_scores() — identical to gate formula
        score = float(scores_df.loc[sector, "composite_score"]) \
                if (not scores_df.empty and sector in scores_df.index) else 50.0
        gate  = "blocked" if (score < COMPOSITE_GATE_MIN or regime_label == "risk-off") else "open"

        vol_wt = _vol_parity_weight(ann_vol, universe_vols)
        # Use satellite caps as pre-ranking proxy; final cap is applied per-rank in build_trade_recommendation
        regime_cap = WEIGHT_LIMITS["satellite"].get(regime_label, 0.10)

        atr_21, atr_63, sma_ratio = price_ctx.get(sector, (0.0, 0.0, 0.0))

        results.append(QuantAssessment(
            sector=sector, ticker=ticker, as_of_date=as_of,
            tsmom_signal=tsmom, tsmom_raw_return=raw_ret,
            csmom_rank=rank, ann_vol=ann_vol,
            composite_score=score, gate_status=gate,
            regime_label=regime_label, p_risk_on=float(p_risk_on),
            vol_parity_weight=vol_wt, regime_weight_cap=regime_cap,
            atr_14=atr_21, atr_63=atr_63, price_vs_sma_200=sma_ratio,
        ))

    return results


def build_trade_recommendation(
    assessment:       QuantAssessment,
    position_rank:    PositionRank          = "satellite",
    llm_adjustment:   float                 = 0.0,
    entry_condition:  Optional[EntryCondition] = None,
    extra_invalidation: Optional[list[InvalidationCondition]] = None,
    decision_log_id:  Optional[int]         = None,
    source_agent:     str                   = "quant_agent",
) -> TradeRecommendation:
    """
    Construct a TradeRecommendation from a QuantAssessment.

    Called by QuantAgent for automated watchlist additions and by
    ResearchAgent after applying LLM soft-override adjustments.

    The suggested_weight is clipped to the regime × rank weight cap.
    """
    direction = "long" if assessment.tsmom_signal >= 0 else "short"
    if assessment.tsmom_signal == 0:
        direction = "neutral"

    baseline   = assessment.vol_parity_weight
    raw_weight = baseline + llm_adjustment
    cap        = WEIGHT_LIMITS[position_rank].get(assessment.regime_label, 0.15)
    suggested  = float(np.clip(raw_weight, 0.0, cap))

    # Default invalidation: TSMOM flip + price below SMA200
    default_inv: list[InvalidationCondition] = [
        InvalidationCondition(
            type="quant", rule="tsmom_flipped",
            entry_value=assessment.tsmom_signal,
        ),
        InvalidationCondition(
            type="quant", rule="price_below_sma", sma_period=200,
        ),
    ]
    invalidation = default_inv + (extra_invalidation or [])

    if entry_condition is None:
        entry_condition = EntryCondition(type="price_breakout", n_days=20)

    return TradeRecommendation(
        sector=assessment.sector, ticker=assessment.ticker,
        direction=direction, position_rank=position_rank,
        quant_baseline_weight=baseline,
        llm_adjustment_pct=llm_adjustment,
        suggested_weight=suggested,
        regime_label=assessment.regime_label,
        tsmom_signal=assessment.tsmom_signal,
        csmom_rank=assessment.csmom_rank,
        composite_score=assessment.composite_score,
        ann_vol=assessment.ann_vol,
        gate_status=assessment.gate_status,
        source_agent=source_agent,
        confidence=assessment.composite_score,
        decision_log_id=decision_log_id,
        entry_condition=entry_condition,
        invalidation_conditions=invalidation,
        as_of_date=assessment.as_of_date,
    )
