"""engine/research/templates/term_structure.py — yield-curve / vol-surface / commodity-curve.

DESIGN PER [[project-senior-pipeline-roadmap-2026-05-30]]:
  Reusable across multiple asset classes — anywhere there's a TERM
  STRUCTURE (rates / vol surface / commodity contango):
    - Bond yield curves (UST / Bund / JGB / OAT / Gilt)
    - VIX term structure (1m / 3m / 6m / 1y)
    - Commodity futures curves (front-month / 6m / 12m / back-end)
    - FX implied vol surface

  Trading signal extracted via 3 alternative parameterizations:

  mode='slope' — simple long-end minus short-end spread
    Standard for rates (10y-2y), VIX (3m-1m), commodity (12m-2m).
    Robust, no fitting required.

  mode='nelson_siegel_3factor' — fitted β0 (level), β1 (slope),
    β2 (curvature) via Nelson-Siegel parametric form
    y(τ) = β0 + β1·[1 - e^(-τ/λ)]/(τ/λ) + β2·{[1 - e^(-τ/λ)]/(τ/λ) - e^(-τ/λ)}
    More structurally informative; requires λ choice (default 18m for
    bonds, 3m for vol).

  mode='curvature_only' — fitted Nelson-Siegel curvature β2 alone.
    Predicts ~5y point vs avg(2y, 10y); negative curvature = humped
    curve = late-cycle / recession signal.

Binding schema:
  mode               — "slope" | "nelson_siegel_3factor" | "curvature_only"
  asset_class        — "rates" | "vol" | "commodity" | "fx" (affects λ default)
  long_tenor_months  — int (e.g. 120 for 10y)
  short_tenor_months — int (e.g. 24 for 2y)
  nelson_siegel_lambda — float (only if mode != slope)
  cost_bps_per_side  — float
  vol_target         — float | null
  vol_target_lookback — int (default 36)

Inputs:
  yield_panel:  monthly wide DataFrame (dates × tenors_months)
                column names = tenor in months (int)

Returns:
  pd.Series of monthly L/S net-of-cost returns based on the chosen signal
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from engine.research import primitives as P

# Per [[project-gate-production-redesign-2026-05-30]]:
# Monthly rebal of curve-spread → HAC lags=6. Cross-asset (rates/vol/
# commodity/fx) → pead_control=False (PEAD doesn't span these).
# Grid: (mode × NS_lambda × tenor choices) ≈ 20 trials.
GATE_PROFILE = {
    "hac_lags":         6,
    "cost_bps_default": 8,
    "pead_control":     False,
    "n_trials_base":    20,
}


def warmup_months(binding: dict) -> int:
    b = binding or {}
    warmup = 1
    if b.get("vol_target") is not None:
        warmup += int(b.get("vol_target_lookback", 36))
    return warmup


def _default_lambda(asset_class: str) -> float:
    """Nelson-Siegel decay parameter — short-tenor humps reflect short-end
    activity. Defaults from Diebold-Li 2006 for bonds; shorter for vol."""
    defaults = {
        "rates":     18.0,    # 18m — humps near short-medium end
        "commodity": 12.0,
        "vol":        3.0,    # vol surface decays faster
        "fx":         6.0,
    }
    return defaults.get(asset_class, 18.0)


def _nelson_siegel_basis(tenors: np.ndarray, lam: float) -> np.ndarray:
    """Build (n_tenors, 3) design matrix of NS basis functions.

    NS: y(τ) = β0·1 + β1·X1(τ) + β2·X2(τ)
      X1(τ) = (1 - e^(-τ/λ)) / (τ/λ)
      X2(τ) = X1(τ) - e^(-τ/λ)
    """
    t_lam = tenors / lam
    # Avoid div by 0 at τ=0
    safe = np.where(t_lam > 1e-8, t_lam, 1e-8)
    exp_neg = np.exp(-safe)
    x1 = (1 - exp_neg) / safe
    x2 = x1 - exp_neg
    return np.column_stack([np.ones_like(tenors, dtype=float), x1, x2])


def _fit_nelson_siegel_panel(
    yield_panel: pd.DataFrame, lam: float,
) -> pd.DataFrame:
    """For each row (date), fit NS to the tenor curve and return
    DataFrame of (date, [beta0, beta1, beta2])."""
    tenors = np.array([float(c) for c in yield_panel.columns])
    basis = _nelson_siegel_basis(tenors, lam)
    out = []
    for date, row in yield_panel.iterrows():
        y = row.values.astype(float)
        valid = ~np.isnan(y)
        if valid.sum() < 3:    # need at least 3 tenors to fit 3 betas
            out.append((date, np.nan, np.nan, np.nan))
            continue
        # OLS: β = (X'X)^{-1} X'y
        X = basis[valid]
        y_v = y[valid]
        try:
            beta, *_ = np.linalg.lstsq(X, y_v, rcond=None)
        except np.linalg.LinAlgError:
            beta = (np.nan, np.nan, np.nan)
        out.append((date, beta[0], beta[1], beta[2]))
    return pd.DataFrame(out, columns=["date", "beta0", "beta1", "beta2"]).set_index("date")


def run_term_structure(
    *,
    yield_panel: pd.DataFrame,
    mode: str = "slope",
    asset_class: str = "rates",
    long_tenor_months: int = 120,
    short_tenor_months: int = 24,
    nelson_siegel_lambda: float | None = None,
    signal_lookback: int = 1,
    cost_bps_per_side: float = 8.0,
    vol_target: float | None = 0.10,
    vol_target_lookback: int = 36,
) -> pd.Series:
    """Extract signal from term structure → form L/S returns.

    The CONCRETE TRADING is: hold a position in the slope-spread itself.
    For rates: pay-fixed-receive-floating swap when slope is steepening.
    For commodity: roll positions when contango is increasing.

    This template returns the signal-driven monthly returns assuming
    the slope spread (or β2 in NS mode) is itself the tradable.
    """
    if mode not in ("slope", "nelson_siegel_3factor", "curvature_only"):
        raise ValueError(f"unknown mode: {mode!r}")

    if mode == "slope":
        if long_tenor_months not in yield_panel.columns:
            raise KeyError(f"long_tenor_months {long_tenor_months} not in yield_panel columns")
        if short_tenor_months not in yield_panel.columns:
            raise KeyError(f"short_tenor_months {short_tenor_months} not in yield_panel columns")
        signal = yield_panel[long_tenor_months] - yield_panel[short_tenor_months]
    else:
        lam = nelson_siegel_lambda or _default_lambda(asset_class)
        betas = _fit_nelson_siegel_panel(yield_panel, lam)
        if mode == "nelson_siegel_3factor":
            # Use slope (β1) as primary signal
            signal = betas["beta1"]
        else:    # curvature_only
            signal = betas["beta2"]

    # Generate monthly returns from signal CHANGES — buy when signal
    # rises, sell when it falls. This is the simplest tradable.
    # Lag by 1 to avoid look-ahead.
    signal_lagged = signal.shift(1)
    delta = signal_lagged.diff(signal_lookback)
    # Position sign: long if delta > 0 (steepening), short if delta < 0
    position = np.sign(delta).fillna(0)

    # Realized return: position × next-period signal change
    next_change = signal.diff()
    gross = position * next_change

    # Cost = |position change| × bps
    pos_change = position.diff().abs().fillna(0)
    cost = pos_change * (cost_bps_per_side / 10000.0)
    net = gross - cost

    if vol_target is not None:
        net = P.vol_target_normalize(
            net, target_vol=vol_target, lookback_months=vol_target_lookback,
        )

    return net.dropna()
