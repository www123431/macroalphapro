"""
engine/path_c/labor_signal.py — Path C Labor Signal Drift signal composition.

Pre-registration: docs/spec_path_c_labor_signal_drift_v1.md (id=58) §2.3 + §2.4

Consumes the firm-quarter panel from engine.path_c.labor_signal_panel and adds:
  - `labor_signal` column: (L6 - B12/2) / max(B12/2, 1) - 0.5 × layoff_flag
    (Choi-Lochstoer-Sosyura 2024 style growth rate, normalized to 6mo)
  - `sue_rank_pct` column: cross-section mid-rank percentile (REUSED from
    engine.path_c.sue_signal.rank_within_quarter via sue_col="labor_signal";
    column NAMED sue_rank_pct for code reuse, semantically signal_rank_pct)
  - `leg` column: ≥ 0.90 long, ≤ 0.10 short, else flat; NaN → excluded
    (REUSED from engine.path_c.sue_signal.assign_decile_legs)

This module is SIGNAL LAYER. Backtest aggregation lives in
engine/path_c/pead_backtest.py (REUSED). Verdict in
engine/path_c/verdict.py (REUSED).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.path_c.labor_signal_panel import (
    MIN_L6_POSTINGS_REQUIRED,
    MIN_B12_POSTINGS_REQUIRED,
)
from engine.path_c.sue_signal import (
    rank_within_quarter as _rank_within_quarter,
    assign_decile_legs as _assign_decile_legs,
)
from engine.path_c import (
    DECILE_LONG_THRESHOLD,
    DECILE_SHORT_THRESHOLD,
)

logger = logging.getLogger(__name__)


# Locked from spec §2.3
LAYOFF_WEIGHT_LOCKED: float = 0.5   # half-sigma penalty per layoff_flag


# ── Signal composition ────────────────────────────────────────────────────
def compute_labor_signal(
    panel:         pd.DataFrame,
    *,
    layoff_weight: float = LAYOFF_WEIGHT_LOCKED,
) -> pd.DataFrame:
    """Add `labor_signal` column to firm-quarter panel.

    Formula (spec §2.3):
        L6_norm = L6 / (B12 / 2)   if B12/2 ≥ 1 else nan
        growth = L6_norm - 1.0     (centered: 0 = no growth, +1 = doubled hiring)
        labor_signal = growth - layoff_weight × layoff_flag

    Equivalent to: (L6 - B12/2) / max(B12/2, 1) - λ × layoff_flag, with the
    max() preventing div-by-zero (firms with B12/2 < 1 are flagged NaN below).

    Fallback NaN (excluded downstream by rank step):
      - L6 < MIN_L6_POSTINGS_REQUIRED
      - B12 < MIN_B12_POSTINGS_REQUIRED
      - layoff_flag missing / non-binary

    Returns a copy; does not mutate input.
    """
    if panel.empty:
        out = panel.copy()
        out["labor_signal"] = pd.Series(dtype=float)
        return out

    required = {"l6_postings_count", "b12_postings_count", "layoff_flag"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"compute_labor_signal: panel missing columns {missing}")

    out = panel.copy()
    l6  = out["l6_postings_count"].astype(float)
    b12 = out["b12_postings_count"].astype(float)
    lf  = out["layoff_flag"].fillna(0).astype(float)

    half_b12 = b12 / 2.0
    # Growth rate with safe div: where half_b12 < 1, NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.where(
            (half_b12 >= 1.0) & np.isfinite(half_b12),
            (l6 - half_b12) / half_b12,
            np.nan,
        )

    # Fallback thresholds: thin coverage → NaN
    thin_mask = (
        (l6 < MIN_L6_POSTINGS_REQUIRED)
        | (b12 < MIN_B12_POSTINGS_REQUIRED)
        | ~np.isfinite(l6)
        | ~np.isfinite(b12)
    )
    signal = np.where(thin_mask, np.nan, growth - layoff_weight * lf)

    out["labor_signal"] = signal
    return out


# ── One-shot pipeline (compose + rank + decile) ──────────────────────────
def build_labor_signal_panel(
    panel:           pd.DataFrame,
    *,
    layoff_weight:   float = LAYOFF_WEIGHT_LOCKED,
    long_threshold:  float = DECILE_LONG_THRESHOLD,
    short_threshold: float = DECILE_SHORT_THRESHOLD,
) -> pd.DataFrame:
    """Pipeline: compute_labor_signal → rank_within_quarter → assign_decile_legs.

    Reuses `engine.path_c.sue_signal.rank_within_quarter` (with
    sue_col="labor_signal", tie_break_col="ticker") and
    `engine.path_c.sue_signal.assign_decile_legs` (with rank_col="sue_rank_pct").
    Column NAMES "sue_rank_pct" + "leg" inherited; semantically the
    rank-and-leg-assignment logic is signal-agnostic.

    Returns input augmented with: labor_signal, sue_rank_pct, leg.
    """
    p1 = compute_labor_signal(panel, layoff_weight=layoff_weight)
    p2 = _rank_within_quarter(
        p1,
        quarter_col="fiscal_yearq",
        sue_col="labor_signal",
        tie_break_col="ticker",
    )
    p3 = _assign_decile_legs(
        p2,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        rank_col="sue_rank_pct",
    )
    if not p3.empty:
        logger.info(
            "build_labor_signal_panel: %d firm-quarters → long=%d short=%d flat=%d excluded=%d",
            len(p3),
            int((p3["leg"] == "long").sum()),
            int((p3["leg"] == "short").sum()),
            int((p3["leg"] == "flat").sum()),
            int((p3["leg"] == "excluded").sum()),
        )
    return p3
