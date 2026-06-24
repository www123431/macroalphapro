"""engine/research/templates/equity_xsmom.py — equity cross-sectional momentum.

Generic template for the Jegadeesh-Titman 1993 / Fama-French style equity
cross-sectional momentum. Used by library mechanisms:
  - equity_xsmom_jt (JT 1993 canonical)
  - residual_momentum (with FF3 residualization upstream)
  - quality variants that overlay quality on momentum ranking

The template is a PURE COMPOSITION of Layer 0 primitives — does not import
pandas operations directly. A bug here affects only equity_xsmom-class
mechanisms; bugs in primitives.py affect all templates.

Binding schema (required fields):
  universe              — string identifier for data loader to fetch
  lookback_months       — int (e.g. 12)
  skip_months           — int (e.g. 1 for canonical 12-1)
  top_frac              — float (e.g. 0.1 for decile)
  bottom_frac           — float (default top_frac)
  weighting             — "equal_weight" | "value_weight" (v1 supports EW only)
  rebal_freq            — "monthly" (v1 only)
  cost_bps_per_side     — float (e.g. 12.0)
  microcap_price_threshold — float (e.g. 5.0, in USD)
  vol_target            — float | null (e.g. 0.10 for 10% annualized)
  vol_target_lookback   — int (default 36, only if vol_target set)

Inputs (required as part of binding OR loaded externally):
  price_panel:    wide DataFrame monthly close prices, dates × tickers
  return_panel:   wide DataFrame monthly returns, dates × tickers
                  (if not provided, computed from price_panel)

Returns:
  pd.Series of monthly L/S net-of-cost returns ready for run_gate.
"""
from __future__ import annotations

import pandas as pd

from engine.research import primitives as P

# Per [[project-gate-production-redesign-2026-05-30]]:
# Monthly rebal → HAC lags=6. Equity universe → pead_control=True.
# Grid: (lookback × skip × decile_frac) ≈ 30 trials.
GATE_PROFILE = {
    "hac_lags":         6,
    "cost_bps_default": 12,
    "pead_control":     True,
    "n_trials_base":    30,
}


def warmup_months(binding: dict) -> int:
    """Months of NaN-warmup this template produces given a binding.

    For equity_xsmom: rolling_sum(lookback) + apply_lag(1) + optional
    vol_target_normalize lookback. Used by protocol designer to compute
    effective sample range so split_first_half / split_second_half land
    in REAL data not warmup region.
    """
    b = binding or {}
    warmup = int(b.get("lookback_months", 12)) + 1
    if b.get("vol_target") is not None:
        warmup += int(b.get("vol_target_lookback", 36))
    return warmup


def run_equity_xsmom(*,
                      price_panel: pd.DataFrame,
                      return_panel: pd.DataFrame | None = None,
                      universe: str | None = None,    # informational
                      lookback_months: int = 12,
                      skip_months: int = 1,
                      top_frac: float = 0.1,
                      bottom_frac: float | None = None,
                      weighting: str = "equal_weight",
                      rebal_freq: str = "monthly",
                      cost_bps_per_side: float = 12.0,
                      microcap_price_threshold: float = 5.0,
                      vol_target: float | None = 0.10,
                      vol_target_lookback: int = 36,
                      ) -> pd.Series:
    """Compose primitives into a monthly L/S net-of-cost return series.

    Pipeline:
    1. exclude_microcap on price_panel
    2. compute_log_return → log returns
    3. rolling_sum(lookback, skip) → momentum signal (anti-look-ahead built-in)
    4. cross_sectional_rank → percent ranks
    5. top_bottom_membership → long/short masks
    6. equal_weight_long_short_returns
    7. vol_target_normalize (if vol_target set)
    8. apply_round_trip_cost
    """
    if weighting != "equal_weight":
        raise NotImplementedError(
            "v1 supports equal_weight only; value_weight needs mcap panel "
            "(add to binding schema first)"
        )
    if rebal_freq != "monthly":
        raise NotImplementedError("v1 supports monthly rebal only")
    if bottom_frac is None:
        bottom_frac = top_frac

    # 1. Microcap exclusion
    cleaned_prices = P.exclude_microcap(price_panel,
                                          threshold=microcap_price_threshold)

    # 2. Log returns
    if return_panel is None:
        ret_panel = P.compute_log_return(cleaned_prices)
    else:
        # Use provided returns but mask via cleaned prices (microcap NaN propagates)
        ret_panel = return_panel.where(cleaned_prices.notna())

    # 3. Momentum signal
    signal = P.rolling_sum(ret_panel,
                             window=lookback_months,
                             skip=skip_months)

    # 4. Cross-sectional rank
    rank_panel = P.cross_sectional_rank(signal)

    # 5. Long/short masks
    long_mask, short_mask = P.top_bottom_membership(
        rank_panel, top_frac=top_frac, bottom_frac=bottom_frac
    )

    # 6. Equal-weight L/S returns
    # Use NEXT-period returns (rank at t, holds at t+1) — built-in via the
    # rolling_sum anti-look-ahead lag. But for clarity: shift the masks
    # forward by 1 so position at t reflects rank at t-1.
    long_mask_lagged = P.apply_lag(long_mask.astype(float), n_periods=1).fillna(0).astype(bool)
    short_mask_lagged = P.apply_lag(short_mask.astype(float), n_periods=1).fillna(0).astype(bool)
    ls = P.equal_weight_long_short_returns(long_mask_lagged, short_mask_lagged, ret_panel)

    # 7. Vol target (optional)
    if vol_target is not None:
        ls = P.vol_target_normalize(ls, target_vol=vol_target,
                                       lookback=vol_target_lookback,
                                       periods_per_year=12)

    # 8. Cost
    # Assume monthly rebalance with ~100% turnover for decile L/S
    ls = P.apply_round_trip_cost(ls, bps_per_side=cost_bps_per_side, turnover=1.0)

    return ls.rename("equity_xsmom_ls")
