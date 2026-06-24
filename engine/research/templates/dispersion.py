"""engine/research/templates/dispersion.py — cross-sectional dispersion L/S template.

DESIGN PER [[project-senior-pipeline-roadmap-2026-05-30]]:
  Distinct from factor_quartile because the SIGNAL is a per-firm
  measure of UNCERTAINTY (cross-section std of a noisy proxy at
  firm-quarter level), and the prediction is that HIGH dispersion
  underperforms — Diether-Malloy-Scherbina 2002 ("Differences of
  Opinion and the Cross-Section of Stock Returns").

PARAMETERIZATION (CRITICAL — per senior roadmap):
  Two distinct modes — these have OPPOSITE economic logic and
  must not be conflated:

  mode='level' — current dispersion level itself is the signal
    Long low-dispersion, short high-dispersion (DMS 2002 mean reversion).
    Use when underlying proxy is a noisy estimate (analyst forecasts,
    earnings surprise, IVOL).

  mode='change' — change in dispersion is the signal
    Long firms with NEGATIVE change in dispersion (uncertainty resolving),
    short firms with POSITIVE change (uncertainty rising).
    Theoretical basis: resolution of disagreement → return realization
    (Banerjee-Kremer 2010, Garfinkel-Sokobin 2006).

Binding schema:
  mode                  — "level" | "change" (required)
  signal_lookback       — int, months for change calc (mode=change only)
  top_frac              — float (e.g. 0.2 for quintile)
  bottom_frac           — float | null
  weighting             — "equal_weight" | "value_weight" (v1 EW only)
  cost_bps_per_side     — float
  vol_target            — float | null
  vol_target_lookback   — int (default 36)

Inputs:
  signal_panel:  monthly wide DataFrame (dates × tickers) of the
                  per-firm dispersion proxy (NOT cross-section std)
  return_panel:  monthly returns wide DataFrame

Returns:
  pd.Series of monthly L/S net-of-cost returns
"""
from __future__ import annotations

import pandas as pd

from engine.research import primitives as P

# Per [[project-gate-production-redesign-2026-05-30]]:
# Monthly rebal → HAC lags=6. Equity universe → pead_control=True.
# Grid: (mode × top_frac × signal_lookback) ≈ 15 trials.
GATE_PROFILE = {
    "hac_lags":         6,
    "cost_bps_default": 15,
    "pead_control":     True,
    "n_trials_base":    15,
}


def warmup_months(binding: dict) -> int:
    b = binding or {}
    warmup = 1
    if b.get("mode") == "change":
        warmup += int(b.get("signal_lookback", 3))
    if b.get("vol_target") is not None:
        warmup += int(b.get("vol_target_lookback", 36))
    return warmup


def run_dispersion(
    *,
    signal_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    mode: str = "level",
    signal_lookback: int = 3,
    top_frac: float = 0.2,
    bottom_frac: float | None = None,
    weighting: str = "equal_weight",
    cost_bps_per_side: float = 15.0,
    vol_target: float | None = 0.10,
    vol_target_lookback: int = 36,
) -> pd.Series:
    """Compose primitives into monthly dispersion-trade L/S series."""
    if mode not in ("level", "change"):
        raise ValueError(f"mode must be 'level' or 'change', got {mode!r}")
    if bottom_frac is None:
        bottom_frac = top_frac
    if weighting != "equal_weight":
        raise NotImplementedError("v1 supports equal_weight only")

    # 1. Form trading signal
    if mode == "level":
        # Level: high signal = high dispersion → predict UNDERPERFORMANCE.
        # We want long LOW dispersion, short HIGH dispersion.
        # Implement by negating signal so high signal_for_rank = low dispersion.
        signal_for_rank = -signal_panel
        signal_label = "level (negated for long-low-dispersion)"
    else:
        # Change: positive change = uncertainty rising → predict UNDER-
        # PERFORMANCE. Long NEGATIVE change.
        change = signal_panel - signal_panel.shift(signal_lookback)
        signal_for_rank = -change
        signal_label = f"change over {signal_lookback}m (negated)"

    # 2. Apply lag to prevent same-month look-ahead
    signal_lagged = P.apply_lag(signal_for_rank, n_periods=1)

    # 3. Cross-sectional rank
    rank_panel = P.cross_sectional_rank(signal_lagged)

    # 4. Membership masks
    long_mask, short_mask = P.top_bottom_membership(
        rank_panel, top_frac=top_frac, bottom_frac=bottom_frac,
    )

    # 5. L/S returns (primitive signature: long_mask, short_mask, return_panel)
    gross_returns = P.equal_weight_long_short_returns(
        long_mask, short_mask, return_panel,
    )

    # 6. Cost (default monthly turnover assumed; primitives apply 2×bps)
    net_returns = P.apply_round_trip_cost(
        gross_returns, bps_per_side=cost_bps_per_side,
    )

    # 7. Vol target
    if vol_target is not None:
        net_returns = P.vol_target_normalize(
            net_returns, target_vol=vol_target, lookback_months=vol_target_lookback,
        )

    return net_returns.dropna()
