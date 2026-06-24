"""engine/research/templates/factor_quartile.py — generic single-factor decile L/S.

Generic template for any single-factor cross-sectional long-short:
  - Quality (QMJ / gross profitability)
  - Low-volatility / BAB (low-beta-leveraged minus high-beta-deleveraged)
  - Value (HML / book-to-market)
  - Investment / asset growth
  - Anything else where you have a per-ticker factor score and want
    long-top-quantile minus short-bottom-quantile

Distinct from equity_xsmom (which computes signal from price history).
factor_quartile takes the FACTOR PANEL as an INPUT (the caller provides
the factor score; this template handles the cross-sectional sorting
and L/S construction).

Pure composition of Layer 0 primitives.

Binding schema:
  weighting             — "equal_weight" (v1 only)
  rebal_freq            — "monthly" (v1 only)
  top_frac              — float (e.g. 0.1 for decile)
  bottom_frac           — float (default top_frac; can be asymmetric)
  cost_bps_per_side     — float (e.g. 12.0)
  microcap_price_threshold — float (e.g. 5.0)
  vol_target            — float | null (e.g. 0.10)
  vol_target_lookback   — int (default 36)
  factor_sign           — +1 means "long high-factor / short low-factor" (default);
                           -1 inverts (e.g. low-vol = long-low-vol, sign=-1)

Inputs (provided by caller in data_kwargs):
  factor_panel:  wide DataFrame, dates × tickers, factor score
                  (caller is responsible for pre-computing this; e.g. for
                  QMJ this is the composite quality score, for BAB it's
                  the beta).
  price_panel:   wide DataFrame, dates × tickers, monthly close prices
                  (used for microcap filter and to compute returns if
                  return_panel not provided).
  return_panel:  wide DataFrame, dates × tickers, monthly returns (optional;
                  computed from price_panel if not given).

Returns:
  pd.Series of monthly L/S net-of-cost returns.
"""
from __future__ import annotations

import pandas as pd

from engine.research import primitives as P

# Per [[project-gate-production-redesign-2026-05-30]]:
# Monthly rebal → HAC lags=6. Equity universe → pead_control=True
# (caller can override via run_gate kwarg for non-equity factor panels).
# Grid: (top_frac × bottom_frac) ≈ 10 trials.
GATE_PROFILE = {
    "hac_lags":         6,
    "cost_bps_default": 12,
    "pead_control":     True,
    "n_trials_base":    10,
}


def warmup_months(binding: dict) -> int:
    """Months of NaN-warmup. apply_lag(1) + optional vol_target lookback."""
    b = binding or {}
    warmup = 1
    if b.get("vol_target") is not None:
        warmup += int(b.get("vol_target_lookback", 36))
    return warmup


def run_factor_quartile(*,
                          factor_panel: pd.DataFrame,
                          price_panel: pd.DataFrame,
                          return_panel: pd.DataFrame | None = None,
                          universe: str | None = None,
                          top_frac: float = 0.1,
                          bottom_frac: float | None = None,
                          weighting: str = "equal_weight",
                          rebal_freq: str = "monthly",
                          cost_bps_per_side: float = 12.0,
                          microcap_price_threshold: float = 5.0,
                          vol_target: float | None = 0.10,
                          vol_target_lookback: int = 36,
                          factor_sign: int = 1,
                          ) -> pd.Series:
    """Generic single-factor decile L/S monthly returns.

    Pipeline:
    1. exclude_microcap → cleaned price panel
    2. compute returns (from prices if return_panel not given)
    3. align factor_panel to cleaned tickers (microcap NaN propagates)
    4. apply factor_sign (flips for "lower factor is better" mechanisms)
    5. apply_lag the factor (anti-look-ahead)
    6. cross_sectional_rank → percent ranks
    7. top_bottom_membership → long/short masks
    8. equal_weight_long_short_returns
    9. vol_target_normalize (optional)
    10. apply_round_trip_cost
    """
    if weighting != "equal_weight":
        raise NotImplementedError(
            "v1 supports equal_weight only; value_weight needs mcap panel"
        )
    if rebal_freq != "monthly":
        raise NotImplementedError("v1 supports monthly rebal only")
    if factor_sign not in (-1, 1):
        raise ValueError(f"factor_sign must be -1 or +1; got {factor_sign}")
    if bottom_frac is None:
        bottom_frac = top_frac

    cleaned_prices = P.exclude_microcap(price_panel,
                                          threshold=microcap_price_threshold)
    if return_panel is None:
        ret_panel = P.compute_log_return(cleaned_prices)
    else:
        ret_panel = return_panel.where(cleaned_prices.notna())

    # Align factor panel to cleaned universe
    factor_aligned = (factor_panel
                       .reindex(index=ret_panel.index, columns=ret_panel.columns)
                       .where(cleaned_prices.notna()))

    # Anti-look-ahead lag on factor BEFORE ranking
    factor_lagged = P.apply_lag(factor_aligned, n_periods=1)

    # Rank canonically (high factor → high rank), then SWAP masks for
    # factor_sign=-1. This guarantees exact anti-correlation between
    # factor_sign=+1 and factor_sign=-1 outputs (avoiding the pandas
    # pct rank tie-break asymmetry that would arise from negating the
    # factor before ranking).
    rank_panel = P.cross_sectional_rank(factor_lagged)
    long_mask, short_mask = P.top_bottom_membership(
        rank_panel, top_frac=top_frac, bottom_frac=bottom_frac
    )
    if factor_sign == -1:
        long_mask, short_mask = short_mask, long_mask

    ls = P.equal_weight_long_short_returns(long_mask, short_mask, ret_panel)

    if vol_target is not None:
        ls = P.vol_target_normalize(ls, target_vol=vol_target,
                                       lookback=vol_target_lookback,
                                       periods_per_year=12)

    ls = P.apply_round_trip_cost(ls, bps_per_side=cost_bps_per_side, turnover=1.0)
    return ls.rename("factor_quartile_ls")
