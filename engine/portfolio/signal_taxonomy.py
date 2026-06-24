"""engine/portfolio/signal_taxonomy.py — Phase 5.6: signal-type
taxonomy + signal-to-expected-return calibration interface.

⚠ SCOPE NOTE (2026-06-01 fitness review): the CA filter abstraction
this module supports is only fitting for cross_asset_carry among
deployed sleeves. The 4 other deployed sleeves (PEAD, tsmom, crisis_
hedge, mom_hedge_overlay) + 2 PENDING_DEPLOY don't fit CA filter
semantics for sleeve-shape reasons; their library YAMLs declare
ca_filter_k_method: not_applicable with per-sleeve explanations. The
taxonomy + calibration still drives the carry sleeve's CA filter and
remains useful for any FUTURE high-turnover rank-based L/S sleeve.
See [[project-multi-asset-ca-filter-gap-2026-06-01]] +
[[feedback-pre-implementation-fitness-check-2026-06-01]].


WHY THIS EXISTS (per senior audit
[[project-paper-borrow-ml-btc-costs-2026-06-01]] §5.6):

  The paper's CA filter formula `|expected_return| > k × tcost`
  assumes the signal IS a point estimate of next-period return. Our
  5 deployed sleeves use very different signal types — only carry
  gives direct expected return. The others need CALIBRATION before
  CA filter logic can apply.

SignalType taxonomy:
  - POINT_FORECAST   : signal IS the predicted return (carry sleeve)
  - CROSS_SECT_RANK  : signal is a rank/decile within a universe; need
                       historical rank → forward return mapping
  - REGIME_INDICATOR : signal triggers a regime; expected return is
                       E[return | regime] historical avg
  - VOL_NORM_ZSCORE  : signal is z-score of a beta-residual or
                       vol-normalized series; need z → return mapping
  - BINARY_TRIGGER   : signal is 0/1 (or +/-1); expected return is
                       the unconditional avg in the active state

Each type has a calibrator: (signal_value, historical_panel) →
expected_return. Calibrators default to identity / unconditional mean
when no historical panel is supplied — gives a usable answer with
clear "no calibration" diagnostic flag.

Registry of deployed sleeves with their signal type lets 5.7 CA
filter pick the right calibrator without per-sleeve special cases.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    POINT_FORECAST   = "point_forecast"
    CROSS_SECT_RANK  = "cross_sect_rank"
    REGIME_INDICATOR = "regime_indicator"
    VOL_NORM_ZSCORE  = "vol_norm_zscore"
    BINARY_TRIGGER   = "binary_trigger"


@dataclass(frozen=True)
class SignalCalibration:
    """Output of a calibrator: expected forward return for the signal
    value provided, plus a diagnostic flag."""
    expected_return:    float
    n_calibration_obs:  int
    method:             str            # "identity" / "decile_map" / ...
    confident:          bool           # True iff enough data was available


# ── Per-type calibrators ───────────────────────────────────────────────


def calibrate_point_forecast(
    signal_value: float,
    historical_panel: Optional[pd.DataFrame] = None,
) -> SignalCalibration:
    """Carry, dividend yield, etc. Signal already IS expected return.
    Identity passthrough."""
    return SignalCalibration(
        expected_return=float(signal_value),
        n_calibration_obs=0,
        method="identity",
        confident=True,
    )


def calibrate_cross_sect_rank(
    signal_value: float,
    historical_panel: Optional[pd.DataFrame] = None,
) -> SignalCalibration:
    """Cross-sectional rank score (e.g. SUE decile in PEAD, momentum
    decile). Map signal → historical forward-return for that decile.

    historical_panel must have columns: ['signal', 'forward_return'].
    Falls back to identity * 0.01 (assume signal is in 'std' units
    of return) if no panel.
    """
    if historical_panel is None or "forward_return" not in historical_panel.columns:
        return SignalCalibration(
            expected_return=float(signal_value) * 0.01,
            n_calibration_obs=0,
            method="fallback-identity-scaled",
            confident=False,
        )
    # Bucket signals into 10 deciles by historical signal value, then
    # average forward_return per decile; assign incoming signal to its
    # decile and return that average.
    s = historical_panel.dropna(subset=["signal", "forward_return"]).copy()
    if len(s) < 30:
        return SignalCalibration(
            expected_return=float(signal_value) * 0.01,
            n_calibration_obs=len(s),
            method="fallback-identity-scaled",
            confident=False,
        )
    s["decile"] = pd.qcut(s["signal"], 10, labels=False, duplicates="drop")
    decile_means = s.groupby("decile")["forward_return"].mean()
    # Bucket the incoming signal by the same edges
    quantiles = np.quantile(s["signal"].values, np.linspace(0, 1, 11))
    decile_idx = int(np.clip(
        np.searchsorted(quantiles[1:-1], signal_value), 0, 9,
    ))
    fwd = float(decile_means.get(decile_idx, decile_means.mean()))
    return SignalCalibration(
        expected_return=fwd,
        n_calibration_obs=len(s),
        method=f"decile_map(decile={decile_idx})",
        confident=True,
    )


def calibrate_regime_indicator(
    signal_value: float,
    historical_panel: Optional[pd.DataFrame] = None,
) -> SignalCalibration:
    """Regime indicator (e.g. crisis_hedge VIX regime). Expected return
    is the historical mean of strategy returns IN that regime.

    historical_panel must have columns: ['regime', 'forward_return'].
    Falls back to 0 if no panel (regime trigger but no data).
    """
    if historical_panel is None or "regime" not in historical_panel.columns:
        return SignalCalibration(
            expected_return=0.0,
            n_calibration_obs=0,
            method="no-calibration",
            confident=False,
        )
    regime_id = int(round(signal_value))
    s = historical_panel.dropna(subset=["regime", "forward_return"])
    in_regime = s[s["regime"] == regime_id]
    if len(in_regime) < 10:
        return SignalCalibration(
            expected_return=float(s["forward_return"].mean()),
            n_calibration_obs=len(in_regime),
            method="unconditional-mean",
            confident=False,
        )
    return SignalCalibration(
        expected_return=float(in_regime["forward_return"].mean()),
        n_calibration_obs=len(in_regime),
        method=f"regime_mean(regime={regime_id})",
        confident=True,
    )


def calibrate_vol_norm_zscore(
    signal_value: float,
    historical_panel: Optional[pd.DataFrame] = None,
) -> SignalCalibration:
    """Vol-normalized z-score (e.g. mom hedge beta-residual z, tsmom
    z-score). Map z → forward return via historical conditional mean.

    Without panel, falls back to z * historical-volatility * scaling;
    when no panel even for vol, returns z * 0.005 (~50bp per z-unit
    is a defensible scale for monthly equity strategies)."""
    if historical_panel is None or "forward_return" not in historical_panel.columns:
        return SignalCalibration(
            expected_return=float(signal_value) * 0.005,
            n_calibration_obs=0,
            method="fallback-z-scaled",
            confident=False,
        )
    s = historical_panel.dropna(subset=["signal", "forward_return"])
    if len(s) < 30:
        return SignalCalibration(
            expected_return=float(signal_value) * 0.005,
            n_calibration_obs=len(s),
            method="fallback-z-scaled",
            confident=False,
        )
    # Linear regression of forward_return on signal
    x = s["signal"].values
    y = s["forward_return"].values
    n = len(x)
    x_mean, y_mean = x.mean(), y.mean()
    cov = np.sum((x - x_mean) * (y - y_mean)) / max(1, n - 1)
    var_x = np.sum((x - x_mean) ** 2) / max(1, n - 1)
    if var_x <= 0:
        return SignalCalibration(
            expected_return=y_mean,
            n_calibration_obs=n,
            method="degenerate-mean",
            confident=False,
        )
    beta = cov / var_x
    alpha = y_mean - beta * x_mean
    fwd = float(alpha + beta * signal_value)
    return SignalCalibration(
        expected_return=fwd,
        n_calibration_obs=n,
        method=f"ols(alpha={alpha:.4f},beta={beta:.4f})",
        confident=True,
    )


def calibrate_binary_trigger(
    signal_value: float,
    historical_panel: Optional[pd.DataFrame] = None,
) -> SignalCalibration:
    """Binary or sign trigger (0/1 or +/-1). Expected return is the
    historical avg when signal active."""
    active = abs(signal_value) > 0
    if historical_panel is None or "forward_return" not in historical_panel.columns:
        return SignalCalibration(
            expected_return=0.0 if not active else 0.005,
            n_calibration_obs=0,
            method="fallback-onoff",
            confident=False,
        )
    s = historical_panel.dropna(subset=["signal", "forward_return"])
    side = np.sign(signal_value)
    in_state = s[np.sign(s["signal"]) == side] if side != 0 else s.iloc[:0]
    if len(in_state) < 10:
        return SignalCalibration(
            expected_return=float(s["forward_return"].mean()),
            n_calibration_obs=len(in_state),
            method="unconditional-mean",
            confident=False,
        )
    return SignalCalibration(
        expected_return=float(in_state["forward_return"].mean()),
        n_calibration_obs=len(in_state),
        method=f"trigger_mean(side={side})",
        confident=True,
    )


# ── Dispatch ───────────────────────────────────────────────────────────


_CALIBRATORS: dict[SignalType, Callable[[float, Optional[pd.DataFrame]], SignalCalibration]] = {
    SignalType.POINT_FORECAST:   calibrate_point_forecast,
    SignalType.CROSS_SECT_RANK:  calibrate_cross_sect_rank,
    SignalType.REGIME_INDICATOR: calibrate_regime_indicator,
    SignalType.VOL_NORM_ZSCORE:  calibrate_vol_norm_zscore,
    SignalType.BINARY_TRIGGER:   calibrate_binary_trigger,
}


def calibrate(
    signal_type:        SignalType,
    signal_value:       float,
    historical_panel:   Optional[pd.DataFrame] = None,
) -> SignalCalibration:
    """Dispatch to per-type calibrator."""
    return _CALIBRATORS[signal_type](signal_value, historical_panel)


# ── Sleeve registry ────────────────────────────────────────────────────
# Senior-classified per [[project-paper-borrow-ml-btc-costs-2026-06-01]]


@dataclass(frozen=True)
class SleeveSignalSpec:
    sleeve_id:        str
    signal_type:      SignalType
    typical_horizon:  str               # "monthly" / "quarterly" / "event-4d"
    notes:            str = ""


DEPLOYED_SLEEVE_SIGNALS: dict[str, SleeveSignalSpec] = {
    "cross_asset_carry": SleeveSignalSpec(
        sleeve_id="cross_asset_carry",
        signal_type=SignalType.POINT_FORECAST,
        typical_horizon="monthly",
        notes=("Roll-yield carry IS the forward expected return "
               "(KMPV 2018). Direct identity calibration."),
    ),
    "post_earnings_drift": SleeveSignalSpec(
        sleeve_id="post_earnings_drift",
        signal_type=SignalType.CROSS_SECT_RANK,
        typical_horizon="event-4d",
        notes=("SUE decile within FF12 sector. Needs decile→forward-"
               "return calibration on the SUE panel history."),
    ),
    "crisis_hedge_tlt_gld": SleeveSignalSpec(
        sleeve_id="crisis_hedge_tlt_gld",
        signal_type=SignalType.REGIME_INDICATOR,
        typical_horizon="monthly",
        notes=("VIX 1y z-score regime (CALM/NORMAL/STRESS). Expected "
               "return is E[strategy_return | regime]."),
    ),
    "mom_hedge_overlay": SleeveSignalSpec(
        sleeve_id="mom_hedge_overlay",
        signal_type=SignalType.VOL_NORM_ZSCORE,
        typical_horizon="monthly",
        notes=("Beta-residual z-score vs MTUM. Needs OLS calibration "
               "z → forward return."),
    ),
    "time_series_momentum": SleeveSignalSpec(
        sleeve_id="time_series_momentum",
        signal_type=SignalType.VOL_NORM_ZSCORE,
        typical_horizon="monthly",
        notes=("MOP 2012 12-1 z-score per asset. OLS calibration. "
               "Canonical library id; 'tsmom' is the legacy short alias."),
    ),
}

# Legacy short ids → canonical library ids
_SLEEVE_ALIASES = {
    "tsmom": "time_series_momentum",
}


def get_sleeve_spec(sleeve_id: str) -> Optional[SleeveSignalSpec]:
    canonical = _SLEEVE_ALIASES.get(sleeve_id, sleeve_id)
    return DEPLOYED_SLEEVE_SIGNALS.get(canonical)


def list_sleeve_specs() -> list[SleeveSignalSpec]:
    return list(DEPLOYED_SLEEVE_SIGNALS.values())
