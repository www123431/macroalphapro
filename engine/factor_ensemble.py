"""
engine/factor_ensemble.py — Vol-parity single-stage ensemble combiner.

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50, hash 1665945d2ca5)
Spec section: §2.3 Ensemble Combiner — Vol-Parity Single-Stage

Methodology (AMP 2013 *FAJ* "Value and Momentum Everywhere" framework):
  Step 1: NaN-aware — per ticker, only consider available factor signals
          (insufficient history / non-applicable factor → NaN excluded)
  Step 2: Vol-parity via cross-sectional z-score per factor — each factor's
          signal centered + scaled to unit cross-section std, so each
          factor contributes equal RISK (not equal numerical weight)
  Step 3: Single-stage — equal-weighted (1/N) NaN-aware average across
          available z-scored factors per ticker; portfolio-level vol-target
          handled downstream by engine/portfolio.py (no per-factor pre-norm)

HARKing-safe properties (per spec §rule-9 N6 + N7):
  - No tunable parameters (cross-section z-score is deterministic from data)
  - Single-stage normalization (no opportunity to "tweak intermediate steps")
  - NaN protocol locked (insufficient → exclude; consistent across factors)
  - Equal-weight 1/N over available factors (vs IC-weighted which is tunable)

Boundary invariant (project rule "0-LLM-in-evaluation"):
  Pure deterministic combination logic. No LLM in this path.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Locked factor enumeration (per spec §2.2)
ENSEMBLE_FACTORS: tuple[str, ...] = ("tsmom", "carry_equity", "quality", "bab")
N_FACTORS: int = len(ENSEMBLE_FACTORS)

# Numerical tolerance for cross-section std degeneracy detection
_EPS_STD: float = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional z-score per factor (Step 2 vol-parity proxy)
# ─────────────────────────────────────────────────────────────────────────────


def _cross_section_z_score(raw_signal: pd.Series) -> pd.Series:
    """
    AMP 2013 cross-section z-score for vol-parity:
        z_i = (raw_i - mean(raw)) / std(raw)
    over the cross-section at as_of.

    NaN preserved (per spec §2.3 NaN protocol — non-applicable factor for ticker).

    Edge cases:
      - len(valid) < 2 → all-NaN output (insufficient cross-section for z-score)
      - std < EPS → all-zero output (no signal dispersion → no information)
    """
    if raw_signal is None or raw_signal.empty:
        return pd.Series(dtype=float)

    valid = raw_signal.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=raw_signal.index, dtype=float)

    mean = float(valid.mean())
    std = float(valid.std(ddof=0))

    if std < _EPS_STD:
        # Degenerate: all factor signals identical → no cross-sectional information
        result = pd.Series(0.0, index=raw_signal.index, dtype=float)
        # Preserve NaN for inputs that were NaN
        result[raw_signal.isna()] = np.nan
        return result

    return (raw_signal - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# NaN-aware ensemble average (Step 1 + Step 3)
# ─────────────────────────────────────────────────────────────────────────────


def _nan_aware_factor_average(
    z_signals_by_factor: dict[str, pd.Series],
    universe:            list[str],
) -> pd.Series:
    """
    Per ticker, average ONLY available (non-NaN) factor z-scores.

    All-NaN ticker → 0.0 (neutral, no trade) per spec §2.3 NaN protocol.

    Returns:
        pd.Series indexed by universe, ensemble signal value or 0.0.
    """
    out = pd.Series(0.0, index=universe, dtype=float)

    for ticker in universe:
        contributions = []
        for factor_name, z_series in z_signals_by_factor.items():
            if z_series is None or z_series.empty:
                continue
            z = z_series.get(ticker, np.nan)
            if pd.notna(z):
                contributions.append(float(z))

        if contributions:
            # Equal-weight 1/n over available factors per spec
            out[ticker] = sum(contributions) / len(contributions)
        # else: stays 0.0 (neutral fallback)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API: compute ensemble signal at as_of
# ─────────────────────────────────────────────────────────────────────────────


def compute_ensemble_signal(
    as_of:          datetime.date,
    universe:       list[str],
    asset_classes:  dict[str, str],
    use_cache:      bool = False,
) -> pd.Series:
    """
    Spec §2.3 — Vol-parity single-stage ensemble of TSMOM + Carry-equity +
    Quality + BAB.

    Returns per-ticker raw ensemble signal (NOT yet portfolio-level vol-target
    normalized — that's portfolio.py Step 4's responsibility downstream).

    Args:
        as_of:         signal computation date (no look-ahead, ≤ t-1 data)
        universe:      list of ETF tickers
        asset_classes: {ticker: asset_class} for per-factor scope enforcement
        use_cache:     pass-through to factor modules

    Returns:
        pd.Series indexed by ticker; ensemble signal value (NaN-aware average).
        All-NaN ticker → 0.0 (neutral).
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)
    if asset_classes is None:
        raise ValueError("asset_classes required for ensemble (factor scope enforcement)")

    # Step 1: compute each factor signal at as_of
    raw_signals = _compute_all_factor_signals(
        as_of=as_of,
        universe=universe,
        asset_classes=asset_classes,
        use_cache=use_cache,
    )

    # Step 2: cross-section z-score per factor (vol-parity proxy)
    z_signals = {
        factor: _cross_section_z_score(raw_signals.get(factor))
        for factor in ENSEMBLE_FACTORS
    }

    # Step 3: NaN-aware equal-weight 1/N average per ticker
    ensemble = _nan_aware_factor_average(z_signals, universe=universe)

    return ensemble


def _compute_all_factor_signals(
    as_of:          datetime.date,
    universe:       list[str],
    asset_classes:  dict[str, str],
    use_cache:      bool,
) -> dict[str, pd.Series]:
    """Compute all 4 factor signals; resilient — failed factor → all-NaN series."""
    from engine.factors import (
        compute_tsmom_signal,
        compute_carry_equity_signal,
        compute_quality_signal,
        compute_bab_signal,
    )

    factor_fns = {
        "tsmom":         compute_tsmom_signal,
        "carry_equity":  compute_carry_equity_signal,
        "quality":       compute_quality_signal,
        "bab":           compute_bab_signal,
    }

    signals: dict[str, pd.Series] = {}
    for factor_name, fn in factor_fns.items():
        try:
            sig = fn(
                as_of=as_of,
                universe=universe,
                asset_classes=asset_classes,
                use_cache=use_cache,
            )
            signals[factor_name] = sig
        except Exception as exc:
            logger.warning(
                "factor_ensemble: factor %s failed at %s: %s — all-NaN fallback",
                factor_name, as_of, exc,
            )
            signals[factor_name] = pd.Series(np.nan, index=universe, dtype=float)
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: cross-factor correlation matrix (verdict template + Tier R)
# ─────────────────────────────────────────────────────────────────────────────


def compute_cross_factor_correlation(
    as_of:          datetime.date,
    universe:       list[str],
    asset_classes:  dict[str, str],
    use_cache:      bool = False,
) -> pd.DataFrame:
    """
    Compute pairwise cross-section correlation between factor signals at as_of.

    Used for §五 Validation Gate 2 (|ρ| < 0.7 between any pair) + verdict
    template diagnostic.

    Returns DataFrame indexed by factor name, columns same; pairwise Pearson ρ.
    NaN-aware: only tickers with both factors non-NaN counted in pair correlation.
    """
    raw_signals = _compute_all_factor_signals(
        as_of=as_of,
        universe=universe,
        asset_classes=asset_classes,
        use_cache=use_cache,
    )

    df = pd.DataFrame(raw_signals)
    return df.corr(method="pearson")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: per-factor coverage stats (verdict template, NaN reporting)
# ─────────────────────────────────────────────────────────────────────────────


def compute_per_factor_coverage(
    as_of:          datetime.date,
    universe:       list[str],
    asset_classes:  dict[str, str],
    use_cache:      bool = False,
) -> dict[str, dict]:
    """
    Per-factor coverage stats for transparency reporting.

    Returns:
        {factor: {n_total, n_valid, n_nan, coverage_pct, applicable_classes}}
    """
    raw_signals = _compute_all_factor_signals(
        as_of=as_of,
        universe=universe,
        asset_classes=asset_classes,
        use_cache=use_cache,
    )

    out = {}
    for factor_name, sig in raw_signals.items():
        n_total = len(sig) if sig is not None else 0
        n_valid = int(sig.notna().sum()) if sig is not None and not sig.empty else 0
        n_nan = n_total - n_valid
        out[factor_name] = {
            "n_total":       n_total,
            "n_valid":       n_valid,
            "n_nan":         n_nan,
            "coverage_pct":  round(100.0 * n_valid / n_total, 1) if n_total > 0 else 0.0,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Risk contribution diagnostic (vol-parity verification, §五 Gate 6)
# ─────────────────────────────────────────────────────────────────────────────


def compute_factor_risk_contribution(
    as_of:          datetime.date,
    universe:       list[str],
    asset_classes:  dict[str, str],
    use_cache:      bool = False,
) -> dict[str, float]:
    """
    Vol-parity verification: compute each factor's contribution to ensemble
    cross-section variance.

    Per spec §五 Gate 6: each factor's risk contribution should be ~equal
    (within ±10% of 1/N). Deviation > 10% indicates vol-parity assumption
    violation (e.g., one factor's z-score has systematically different
    variance pattern in the universe at this t).

    Returns:
        {factor: risk_share_fraction} (sums to ~1.0)
    """
    raw_signals = _compute_all_factor_signals(
        as_of=as_of,
        universe=universe,
        asset_classes=asset_classes,
        use_cache=use_cache,
    )

    z_signals = {
        factor: _cross_section_z_score(raw_signals.get(factor))
        for factor in ENSEMBLE_FACTORS
    }

    # Per-factor cross-section variance over the available (non-NaN) tickers
    variances: dict[str, float] = {}
    for factor_name, z in z_signals.items():
        if z is None or z.empty:
            variances[factor_name] = 0.0
            continue
        valid = z.dropna()
        if len(valid) < 2:
            variances[factor_name] = 0.0
            continue
        variances[factor_name] = float(valid.var(ddof=0))

    total = sum(variances.values())
    if total < 1e-12:
        return {f: 0.0 for f in ENSEMBLE_FACTORS}

    return {f: v / total for f, v in variances.items()}
