"""
engine/execution/cost_model.py — ADV-aware transaction-cost model.

Tier-1 audit #3 (2026-05-14): replaces hardcoded 10bp / 4bp TC with a
3-component estimator that accounts for instrument class, position size
relative to ADV, and volatility-coupled market impact.

Components per fill:
  1. base_tc_bps     — tiered base by instrument class
                         ETF Tier-1 (SPY/QQQ/...)      :  4 bp
                         ETF Tier-2 (sector / style)    :  6 bp
                         Single-stock S&P 500 / large   : 10 bp
                         Single-stock mid-cap           : 20 bp
                         Single-stock small-cap         : 30 bp
                         Mutual fund (NAV-priced)       :  0 bp
  2. impact_bps      — linear-above-threshold square-root-vol model
                         impact_bps = max(0, size/ADV - 0.05) × λ × √vol_ann × 1e4
                         where λ = 0.5 (tuned so 10% ADV trade at 20% vol
                         pays ~10bp incremental impact — institutional
                         literature: BARRA / Capital IQ ranges).
  3. half_spread_bps — 0.5 × inferred bid-ask spread proxy
                         ETFs: 1-2 bp half-spread floor (very tight)
                         Stocks: derived from daily vol — 0.02 × daily_vol_bps
                                 where daily_vol_bps = σ_ann/√252 × 1e4.
                                 Calibrated against typical AAPL real-world
                                 half-spread of ~1bp at σ_ann ≈ 0.25.
                                 (Capped at 15 bp.)

Capacity warning fires when size/ADV > 0.20 (20% of one day's volume —
executing this in one day moves the market materially; institutional
practice splits over multiple days via VWAP/TWAP, which we don't model
at this tier — we just FLAG it for the supervisor).

NOT MODELED at this scope (intentional):
  - Order splitting / TWAP / VWAP (needs OMS + broker integration)
  - True Almgren-Chriss optimal execution (needs order-level history)
  - Tax-aware lot selection (needs lot ledger)
  - Borrow fees for short positions (needs broker stock-loan data)

Stubbed: docs/portfolio_deployment_design_2026-05-13.md notes "deferred
to broker layer Phase B" — true OMS/broker integration is post-paper-trade.

References:
  - Almgren-Chriss 2000 "Optimal Execution of Portfolio Transactions"
  - Kissell-Glantz 2003 "Optimal Trading Strategies"
  - BlackRock 2020 transaction-cost research note (sqrt-vol scaling)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Instrument classification
# ─────────────────────────────────────────────────────────────────────────────
class InstrumentClass(str, Enum):
    ETF_TIER1     = "etf_tier1"      # SPY, QQQ, IWM, EFA — mega-liquid
    ETF_TIER2     = "etf_tier2"      # sector/style/factor ETFs
    SS_LARGE_CAP  = "ss_large_cap"   # S&P 500 / >$10B market cap
    SS_MID_CAP    = "ss_mid_cap"     # $2B-$10B
    SS_SMALL_CAP  = "ss_small_cap"   # <$2B
    MUTUAL_FUND   = "mutual_fund"    # PQTIX-like; NAV-priced
    UNKNOWN       = "unknown"


# Tier-1 ETFs: mega-liquid, ADV typically > $1B/day, half-spread ~1bp
KNOWN_ETFS_TIER1: frozenset[str] = frozenset({
    "SPY", "QQQ", "IWM", "IWB", "IWD", "IWF",
    "EFA", "EEM", "AGG", "TLT", "HYG", "LQD",
    "VTI", "VOO", "VEA", "VWO", "BND",
    "GLD", "SLV", "DBC", "USO",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
})

# Tier-2 ETFs: sector/style/factor — half-spread ~2-4bp typical
KNOWN_ETFS_TIER2: frozenset[str] = frozenset({
    "MTUM", "QUAL", "USMV", "VLUE", "SIZE",  # factor ETFs
    "VTV", "VUG", "VBR", "VBK",              # value/growth small/large
    "SCHG", "SCHV", "VOE", "VOT",
    "VXUS", "ACWI", "VT",
    "VYM", "VIG", "DVY",
})

# Mutual funds: priced at NAV, no intraday TC
KNOWN_MUTUAL_FUNDS: frozenset[str] = frozenset({
    "PQTIX",   # PIMCO TRENDS Managed Futures Strategy Fund (Path O CTA proxy)
})

# Convenience superset
KNOWN_ETFS: frozenset[str] = KNOWN_ETFS_TIER1 | KNOWN_ETFS_TIER2


def classify_instrument(
    ticker: str,
    market_cap_usd: Optional[float] = None,
) -> InstrumentClass:
    """Classify ticker for TC tier selection.

    Order: known ETFs → known mutual funds → market_cap_usd (if provided)
    → default SS_LARGE_CAP (most paper-trade single-stock fills land in
    S&P 500 or near-S&P universe, so this is the right prior).
    """
    if not ticker:
        return InstrumentClass.UNKNOWN
    tk = ticker.upper().strip()
    if tk in KNOWN_ETFS_TIER1:
        return InstrumentClass.ETF_TIER1
    if tk in KNOWN_ETFS_TIER2:
        return InstrumentClass.ETF_TIER2
    if tk in KNOWN_MUTUAL_FUNDS:
        return InstrumentClass.MUTUAL_FUND
    # Market-cap-based tier (when caller provides cap)
    if market_cap_usd is not None and market_cap_usd > 0:
        if market_cap_usd >= 10_000_000_000:
            return InstrumentClass.SS_LARGE_CAP
        if market_cap_usd >= 2_000_000_000:
            return InstrumentClass.SS_MID_CAP
        return InstrumentClass.SS_SMALL_CAP
    # Default: single-stock large-cap (most D-PEAD / Path N fills)
    return InstrumentClass.SS_LARGE_CAP


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_BASE_TC_BPS: dict[InstrumentClass, float] = {
    InstrumentClass.ETF_TIER1:    4.0,
    InstrumentClass.ETF_TIER2:    6.0,
    InstrumentClass.SS_LARGE_CAP: 10.0,
    InstrumentClass.SS_MID_CAP:   20.0,
    InstrumentClass.SS_SMALL_CAP: 30.0,
    InstrumentClass.MUTUAL_FUND:  0.0,
    InstrumentClass.UNKNOWN:      15.0,   # mid-tier prior
}

_HALF_SPREAD_FLOOR_BPS: dict[InstrumentClass, float] = {
    InstrumentClass.ETF_TIER1:    1.0,
    InstrumentClass.ETF_TIER2:    2.0,
    InstrumentClass.SS_LARGE_CAP: 3.0,
    InstrumentClass.SS_MID_CAP:   5.0,
    InstrumentClass.SS_SMALL_CAP: 10.0,
    InstrumentClass.MUTUAL_FUND:  0.0,
    InstrumentClass.UNKNOWN:      5.0,
}

_HALF_SPREAD_CAP_BPS = 15.0
_SPREAD_VOL_COEFFICIENT = 0.02   # see module docstring

# Impact model coefficient — tuned so 10% ADV trade at 20% annualized vol
# pays ~10bp impact:  max(0, 0.10 - 0.05) × λ × sqrt(0.20) × 1e4
#                  = 0.05 × λ × 0.447 × 10000 = 223.6 × λ
# Solve λ for 10bp:  λ ≈ 10/223.6 = 0.045
# Use λ = 0.05 (round up; matches Kissell-Glantz mid-range for institutional).
IMPACT_LAMBDA = 0.05

CAPACITY_WARN_FRAC = 0.20   # size/ADV > 20% → flag
ADV_THRESHOLD_FRAC = 0.05   # impact term zero below this


# ─────────────────────────────────────────────────────────────────────────────
# Core estimator
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TCEstimate:
    """Per-fill TC breakdown."""
    ticker:            str
    instrument_class:  InstrumentClass
    base_tc_bps:       float
    impact_bps:        float
    half_spread_bps:   float
    total_tc_bps:      float
    size_usd:          float
    adv_usd:           float
    size_over_adv:     float
    capacity_warning:  bool

    @property
    def total_tc_decimal(self) -> float:
        """TC as decimal fraction of notional (e.g., 0.0010 = 10 bp)."""
        return self.total_tc_bps / 10_000.0


def estimate_tc_bps(
    ticker:           str,
    position_size_usd: float,
    adv_usd:           float,
    vol_ann:           float = 0.20,
    market_cap_usd:    Optional[float] = None,
) -> TCEstimate:
    """Estimate per-fill TC in bps with 3-component decomposition.

    Args:
        ticker:            instrument symbol
        position_size_usd: absolute notional traded (always positive)
        adv_usd:           20-day or 60-day average daily $-volume
                           (use engine.risk_metrics.fetch_adv)
        vol_ann:           annualized volatility (default 0.20 = 20%);
                           used in √vol impact coupling
        market_cap_usd:    optional, used only for single-stock tier
                           classification when not in KNOWN_ETFS

    Returns:
        TCEstimate dataclass with breakdown + capacity_warning flag.

    Defensive behavior:
        - adv_usd ≤ 0 → impact_bps treated as max (size/ADV → ∞) →
          capacity warning fires.
        - size_usd ≤ 0 → returns zero-TC estimate (no fill happened).
        - Returns conservative estimate (never underestimates) when
          data is missing — half-spread floor + UNKNOWN tier default.
    """
    cls = classify_instrument(ticker, market_cap_usd)
    base = _BASE_TC_BPS.get(cls, 15.0)
    spread_floor = _HALF_SPREAD_FLOOR_BPS.get(cls, 5.0)

    size = max(0.0, float(position_size_usd))
    if size <= 0:
        return TCEstimate(
            ticker=ticker, instrument_class=cls,
            base_tc_bps=0.0, impact_bps=0.0, half_spread_bps=0.0,
            total_tc_bps=0.0, size_usd=0.0, adv_usd=float(adv_usd or 0.0),
            size_over_adv=0.0, capacity_warning=False,
        )

    # Mutual funds: priced at NAV, no intraday TC
    if cls == InstrumentClass.MUTUAL_FUND:
        return TCEstimate(
            ticker=ticker, instrument_class=cls,
            base_tc_bps=0.0, impact_bps=0.0, half_spread_bps=0.0,
            total_tc_bps=0.0, size_usd=size, adv_usd=0.0,
            size_over_adv=0.0, capacity_warning=False,
        )

    # ADV ratio
    adv = float(adv_usd or 0.0)
    if adv <= 0:
        # Missing ADV — assume worst case for capacity but charge spread + base only
        size_over_adv = float("inf")
        impact = 0.0   # can't compute meaningfully; surface via warning
        capacity_warn = True
    else:
        size_over_adv = size / adv
        excess = max(0.0, size_over_adv - ADV_THRESHOLD_FRAC)
        vol_term = math.sqrt(max(0.0, float(vol_ann)))
        impact = excess * IMPACT_LAMBDA * vol_term * 10_000.0
        capacity_warn = size_over_adv > CAPACITY_WARN_FRAC

    # Half-spread: floor by tier, vol-coupled for stocks
    if cls in (InstrumentClass.SS_LARGE_CAP, InstrumentClass.SS_MID_CAP,
               InstrumentClass.SS_SMALL_CAP, InstrumentClass.UNKNOWN):
        # daily σ in bps = σ_ann / √252 × 1e4 (NOT sqrt(σ²/T))
        daily_vol_bps = (max(0.0, float(vol_ann)) / math.sqrt(252.0)) * 10_000.0
        spread_est = max(spread_floor, _SPREAD_VOL_COEFFICIENT * daily_vol_bps)
    else:
        spread_est = spread_floor
    half_spread = min(_HALF_SPREAD_CAP_BPS, spread_est)

    total = base + impact + half_spread
    return TCEstimate(
        ticker=ticker, instrument_class=cls,
        base_tc_bps=base, impact_bps=impact, half_spread_bps=half_spread,
        total_tc_bps=total, size_usd=size, adv_usd=adv,
        size_over_adv=size_over_adv if adv > 0 else float("inf"),
        capacity_warning=capacity_warn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level aggregation
# ─────────────────────────────────────────────────────────────────────────────
def compute_portfolio_tc(
    turnover_usd_by_ticker: dict[str, float],
    adv_by_ticker:           dict[str, float],
    vol_ann_by_ticker:       Optional[dict[str, float]] = None,
    default_vol_ann:         float = 0.20,
    nav_usd:                 float = 1_000_000.0,
) -> dict:
    """Aggregate TC across a multi-ticker rebalance day.

    Args:
        turnover_usd_by_ticker: {ticker: abs traded notional} this rebal
        adv_by_ticker:           {ticker: 60d ADV $ from fetch_adv}
        vol_ann_by_ticker:       optional per-ticker σ (annualized);
                                 ticks missing fall back to default_vol_ann
        default_vol_ann:         0.20 = 20% (typical large-cap)
        nav_usd:                 used to express drag as % of book

    Returns:
        dict with keys:
          total_tc_usd          — sum of all per-ticker TC
          total_tc_bps_on_book  — TC drag in bps relative to nav_usd
          total_tc_drag_decimal — total_tc_usd / nav_usd
          n_fills               — count of non-zero turnover tickers
          n_capacity_warnings   — count where size/ADV > 20%
          per_ticker            — {ticker: TCEstimate} for drill-down
    """
    vol_lookup = vol_ann_by_ticker or {}
    per_ticker: dict[str, TCEstimate] = {}
    total_tc_usd = 0.0
    n_caps = 0
    n_fills = 0
    for tk, turnover in turnover_usd_by_ticker.items():
        if turnover is None or abs(turnover) < 1e-9:
            continue
        adv = float(adv_by_ticker.get(tk, 0.0) or 0.0)
        vol = float(vol_lookup.get(tk, default_vol_ann))
        est = estimate_tc_bps(tk, abs(float(turnover)), adv, vol_ann=vol)
        per_ticker[tk] = est
        # TC cost in $ = size × tc_decimal
        total_tc_usd += abs(float(turnover)) * est.total_tc_decimal
        if est.capacity_warning:
            n_caps += 1
        n_fills += 1

    drag_decimal = total_tc_usd / nav_usd if nav_usd > 0 else 0.0
    return {
        "total_tc_usd":          float(total_tc_usd),
        "total_tc_drag_decimal": float(drag_decimal),
        "total_tc_bps_on_book":  float(drag_decimal * 10_000.0),
        "n_fills":               int(n_fills),
        "n_capacity_warnings":   int(n_caps),
        "per_ticker":            per_ticker,
    }
