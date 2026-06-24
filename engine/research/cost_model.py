"""engine/research/cost_model.py — production-grade transaction cost
modeling beyond a single bps-per-side constant.

Senior gap ② per [[project-end-to-end-vision-2026-05-30]]: current
templates apply `apply_round_trip_cost(returns, bps_per_side, turnover=1.0)`
which is a single number per period regardless of trade size or universe
liquidity. At institutional AUM this seriously under-estimates real cost
for low-cap / high-impact trades.

THREE MODELS, escalating realism:

  simple_bps         — current behavior. Constant bps × 2 sides × turnover.
                        Reasonable for very small portfolios.

  linear_spread       — half-spread × |traded notional|. Captures the
                        bid-ask cost component, varies with trade size.
                        Good baseline for liquid mega-cap portfolios.

  almgren_chriss     — half-spread + temporary impact (σ × √(participation))
                        per Almgren-Chriss 2000. The canonical institutional
                        cost model. Captures size-dependent slippage,
                        critical for AUM > $50M or thin-liquidity universes.

All three return cost in BPS of AUM per period, so they're plug-compatible
with how existing templates already subtract bps from gross returns.

INPUTS (where applicable):
  weights_t / weights_tminus1: portfolio weights per (date, asset)
    — used to compute |Δw| = trade fraction per asset
  sigma_daily:    rolling daily vol per asset (from return panel)
  adv_dollars:    average daily $ volume per asset (∝ prc × volume)
  portfolio_aum:  total assets under management ($) — drives notional sizing
  spread_bps_per_side: half-spread baseline (e.g. 2-5 bps for mega-cap,
                        15-30 bps for small-cap)
  impact_coef:    multiplier on the sqrt impact term, calibrated per
                  asset class (literature default ~0.5 for US equities)

KEY DESIGN: returns BPS not dollars, so it's portfolio-AUM-invariant in
the same way as the existing 'cost_bps_per_side' parameter. The model
just makes the bps dynamic instead of constant.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CostModel = Literal["simple_bps", "linear_spread", "almgren_chriss"]


@dataclasses.dataclass
class CostModelParams:
    """Parameters for the cost model. All optional with sensible defaults
    for US equity mega-cap universes."""
    model:                CostModel = "simple_bps"
    cost_bps_per_side:    float = 12.0     # simple_bps only
    spread_bps_per_side:  float = 2.5      # half-spread (linear + almgren)
    impact_coef:          float = 0.5      # almgren only
    portfolio_aum:        float = 100_000_000   # $100M default
    # Fallback when ADV / sigma not available — assume mega-cap defaults
    default_adv_dollars:  float = 200_000_000   # $200M typical mega-cap
    default_daily_sigma:  float = 0.015         # 1.5% daily vol

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ── Model 1: simple bps (back-compat with existing templates) ────────────

def simple_bps_cost(
    turnover_series: pd.Series,
    params: CostModelParams,
) -> pd.Series:
    """Constant bps × 2 sides × turnover. Identical to existing
    primitives.apply_round_trip_cost behavior."""
    # cost per period in bps of AUM
    return (turnover_series.fillna(0) *
            2.0 * params.cost_bps_per_side)


# ── Model 2: linear spread (size-aware bid-ask) ──────────────────────────

def linear_spread_cost(
    weights_t: pd.DataFrame,
    weights_tminus1: pd.DataFrame,
    params: CostModelParams,
) -> pd.Series:
    """Half-spread × Σ |Δw_i| per period.

    Cost in BPS of AUM = sum across assets of (half_spread × |Δw|).
    Because Δw is FRACTION of AUM, multiplying by spread_bps gives bps.
    """
    delta_w = (weights_t - weights_tminus1).abs()
    # Total absolute turnover per period (fraction of AUM)
    period_turnover = delta_w.sum(axis=1)
    # bps cost = half-spread per side × 2 sides × turnover
    cost_bps = period_turnover * 2.0 * params.spread_bps_per_side
    return cost_bps


# ── Model 3: Almgren-Chriss (linear + sqrt impact) ───────────────────────

def almgren_chriss_cost(
    weights_t:        pd.DataFrame,
    weights_tminus1:  pd.DataFrame,
    sigma_daily:      pd.DataFrame | None,
    adv_dollars:      pd.DataFrame | None,
    params:           CostModelParams,
) -> pd.Series:
    """Almgren-Chriss 2000 cost model.

    cost_bps_t = half_spread × Σ|Δw_i| + impact_coef × Σ_i (|Δw_i| × σ_i × √(notional_i/ADV_i))

    Where:
      Δw_i = w_i,t - w_i,t-1 (fraction of AUM into asset i this period)
      σ_i = daily volatility of asset i (rolling, e.g. 60-day)
      notional_i = |Δw_i| × portfolio_aum (dollars traded in asset i)
      ADV_i = average daily $ volume in asset i

    The sqrt term captures TEMPORARY market impact (Almgren 2003 calibration).
    σ × √(participation_rate) has the dimension of bps directly when
    participation < 1.

    When sigma_daily or adv_dollars are None or missing for an asset, we
    fall back to default_daily_sigma + default_adv_dollars.
    """
    delta_w = (weights_t - weights_tminus1).abs()
    # Linear spread term
    spread_cost = delta_w.sum(axis=1) * 2.0 * params.spread_bps_per_side

    # Sqrt impact term
    # notional_dollars per asset per period
    notional_dollars = delta_w * params.portfolio_aum
    # ADV dollars per asset (with fallback)
    if adv_dollars is None or adv_dollars.empty:
        adv = pd.DataFrame(
            params.default_adv_dollars,
            index=delta_w.index, columns=delta_w.columns,
        )
    else:
        adv = adv_dollars.reindex_like(delta_w).fillna(params.default_adv_dollars)
        adv = adv.where(adv > 0, params.default_adv_dollars)
    # σ daily per asset (with fallback)
    if sigma_daily is None or sigma_daily.empty:
        sig = pd.DataFrame(
            params.default_daily_sigma,
            index=delta_w.index, columns=delta_w.columns,
        )
    else:
        sig = sigma_daily.reindex_like(delta_w).fillna(params.default_daily_sigma)

    # Participation rate per asset (capped at 1.0)
    participation = (notional_dollars / adv).clip(upper=1.0)
    # Per-asset impact cost in DOLLARS:
    # impact_$ = notional × σ × √(participation) × impact_coef
    impact_dollars_per_asset = (
        notional_dollars * sig * np.sqrt(participation) * params.impact_coef
    )
    # Sum across assets, convert to bps of AUM
    impact_bps = (impact_dollars_per_asset.sum(axis=1)
                    / params.portfolio_aum * 10_000.0)

    return spread_cost + impact_bps


# ── Helpers: compute σ and ADV from raw panels ───────────────────────────

def rolling_daily_sigma(
    return_panel_daily: pd.DataFrame,
    *,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling 60-day daily-volatility per asset. Returns a panel
    indexed the same as return_panel_daily."""
    return return_panel_daily.rolling(window=window, min_periods=20).std()


def adv_dollars_from_price_volume(
    price_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    *,
    window: int = 60,
) -> pd.DataFrame:
    """ADV in dollars = rolling mean of (prc × volume)."""
    notional = price_panel * volume_panel
    return notional.rolling(window=window, min_periods=20).mean()


# ── Unified entrypoint ───────────────────────────────────────────────────

def compute_cost_bps(
    *,
    model: CostModel = "simple_bps",
    weights_t:        pd.DataFrame | None = None,
    weights_tminus1:  pd.DataFrame | None = None,
    turnover_series:  pd.Series | None = None,
    sigma_daily:      pd.DataFrame | None = None,
    adv_dollars:      pd.DataFrame | None = None,
    params:           CostModelParams | None = None,
) -> pd.Series:
    """Dispatch on model name. Returns cost in bps of AUM per period.

    Caller subtracts cost (after converting bps→fraction) from gross
    returns to get net returns.
    """
    p = params or CostModelParams(model=model)
    if model == "simple_bps":
        if turnover_series is None:
            if weights_t is not None and weights_tminus1 is not None:
                turnover_series = (weights_t - weights_tminus1).abs().sum(axis=1)
            else:
                raise ValueError("simple_bps needs turnover_series or weights")
        return simple_bps_cost(turnover_series, p)
    if model == "linear_spread":
        if weights_t is None or weights_tminus1 is None:
            raise ValueError("linear_spread needs weights_t + weights_tminus1")
        return linear_spread_cost(weights_t, weights_tminus1, p)
    if model == "almgren_chriss":
        if weights_t is None or weights_tminus1 is None:
            raise ValueError("almgren_chriss needs weights_t + weights_tminus1")
        return almgren_chriss_cost(
            weights_t, weights_tminus1, sigma_daily, adv_dollars, p,
        )
    raise ValueError(f"unknown cost model {model!r}")


def apply_cost_to_returns(
    gross_returns: pd.Series,
    cost_bps: pd.Series,
) -> pd.Series:
    """Subtract cost (bps of AUM) from gross returns to get net.

    bps → fractional: divide by 10_000. Aligns on index.
    """
    cost_fraction = cost_bps.reindex(gross_returns.index).fillna(0) / 10_000.0
    return (gross_returns - cost_fraction).rename(
        getattr(gross_returns, "name", None)
    )
