"""engine.research.ablation.weighting — 5 weighting methods.

Each method takes a DataFrame slice for ONE leg of ONE month (subset of
events with same "leg" ∈ {long, short}) and returns weights summing to 1.0.

Theoretical decomposition (per Phase A v3 rigor item #14):

  equal: w_i = 1/N.
    Theory: maximum-entropy prior on stock weights when signal precision
    is unknown. Robust to alpha estimation error (DeMiguel-Garlappi-Uppal
    2009 RFS). The DEPLOYED method; the null hypothesis.

  signal_magnitude: w_i ∝ |signal_i|.
    Theory: linearizes the mapping from signal → conviction. Equivalent to
    Markowitz mean-variance with diagonal covariance and unbiased
    information ratio. Asness-Frazzini-Pedersen 2014 (JFE) "Quality minus
    Junk" uses this in QMJ construction.

  rank: w_i ∝ rank(signal_i).
    Theory: signal-magnitude robust to outliers (winsorization implicit).
    Used in Daniel-Grinblatt-Titman-Wermers 1997 (DGTW) momentum scoring,
    Carhart 1997 four-factor model construction.

  inverse_vol: w_i ∝ 1/σ_idio_i.
    Theory: risk parity within the leg. Equalizes ex-ante per-stock risk
    contribution. Asness-Frazzini 2013 "Leverage Aversion and Risk Parity",
    Roncalli textbook "Risk Parity". Maximum diversification under Pythagoras
    assumption (zero pairwise correlation).

  signal_x_inverse_vol: w_i ∝ |signal_i| × 1/σ_idio_i.
    Theory: combined Markowitz-style optimization with diagonal cov +
    information ratio scaling. Equivalent to maximizing Σ(signal_i² /
    σ_idio_i²) per Grinold-Kahn 2000 §6 "Information Ratio". The
    "naive Markowitz" benchmark; expected best in cases where signal is
    truly proportional to alpha.

Winsorization: signal is clipped to ±WINSORIZE_LIMIT before weighting
to bound the effect of extreme z-scores (Bali-Cakici-Whitelaw 2011 standard).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


WINSORIZE_LIMIT = 3.0


def _winsorize(s: pd.Series, lim: float = WINSORIZE_LIMIT) -> pd.Series:
    return s.clip(-lim, lim)


def _safe_norm(w: pd.Series, index: pd.Index) -> pd.Series:
    """Normalize so weights sum to 1.0. If all zero / NaN, return uniform."""
    w = w.fillna(0)
    s = w.sum()
    if s > 0:
        return w / s
    return pd.Series(1.0 / len(index), index=index)


def weight_equal(g: pd.DataFrame, signal_col: str) -> pd.Series:
    return pd.Series(1.0 / len(g), index=g.index)


def weight_signal_magnitude(g: pd.DataFrame, signal_col: str) -> pd.Series:
    w = _winsorize(g[signal_col]).abs()
    return _safe_norm(w, g.index)


def weight_rank(g: pd.DataFrame, signal_col: str) -> pd.Series:
    n = len(g)
    if n < 2:
        return pd.Series(1.0 / n, index=g.index)
    # Rank ascending then center at 1 (so smallest rank doesn't get 0 weight)
    r = g[signal_col].abs().rank()
    return _safe_norm(r, g.index)


def weight_inverse_vol(g: pd.DataFrame, signal_col: str) -> pd.Series:
    inv = 1.0 / g["sigma_idio"]
    inv = inv.replace([np.inf, -np.inf], 0).fillna(0)
    return _safe_norm(inv, g.index)


def weight_signal_x_inverse_vol(g: pd.DataFrame, signal_col: str) -> pd.Series:
    sig = _winsorize(g[signal_col]).abs()
    inv = 1.0 / g["sigma_idio"]
    inv = inv.replace([np.inf, -np.inf], 0).fillna(0)
    w = sig * inv
    return _safe_norm(w, g.index)


WEIGHTING_METHODS = {
    "equal":                weight_equal,
    "signal_magnitude":     weight_signal_magnitude,
    "rank":                 weight_rank,
    "inverse_vol":          weight_inverse_vol,
    "signal_x_inverse_vol": weight_signal_x_inverse_vol,
}


WEIGHTING_THEORY = {
    "equal": (
        "Maximum-entropy prior. The null hypothesis (DeMiguel-Garlappi-Uppal "
        "2009 RFS). Robust to alpha estimation error."
    ),
    "signal_magnitude": (
        "Linear conviction mapping. Markowitz MV with diagonal cov + "
        "unbiased IR. Asness-Frazzini-Pedersen 2014 QMJ analog."
    ),
    "rank": (
        "Outlier-robust signal-magnitude. DGTW 1997 / Carhart 1997 construction."
    ),
    "inverse_vol": (
        "Risk parity within leg. Asness-Frazzini 2013, Roncalli textbook. "
        "Maximum diversification under zero-correlation assumption."
    ),
    "signal_x_inverse_vol": (
        "Naive Markowitz with diagonal cov. Maximizes Σ(signal²/σ²) per "
        "Grinold-Kahn §6 IR formula. Expected best when signal ∝ alpha."
    ),
}
