"""
engine/path_c/asset_growth_signal.py — Path J Asset Growth signal composition.

Pre-registration: docs/spec_path_j_asset_growth_drift_v1.md (id=60) §2.3 + §2.4

Composes:
  - asset_growth_yoy = (atq_recent - atq_prior) / atq_prior
  - asset_growth_signal = -1 × asset_growth_yoy
    (CGS BEARISH direction inverted to preserve LONG-top-decile semantic;
     HIGH signal = LOW asset growth = LONG; LOW signal = HIGH asset growth = SHORT)

Then REUSES sue_signal.rank_within_quarter + assign_decile_legs.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.path_c.asset_growth_signal_panel import (
    MIN_ATQ_DOLLAR_M,
    MAX_ABSOLUTE_GROWTH,
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


def compute_asset_growth_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `asset_growth_signal` column to firm-quarter atq panel.

    Formula:
        asset_growth_yoy = (atq_recent - atq_prior) / atq_prior
        signal = -1 × asset_growth_yoy
    (Top decile signal = LOW growth firms = LONG, per CGS 2008 BEARISH inverted)

    Fallback to NaN (excluded by rank step):
      - atq_recent missing OR ≤ MIN_ATQ_DOLLAR_M ($100M)
      - atq_prior missing OR ≤ 0 (no growth computable)
      - |asset_growth_yoy| > MAX_ABSOLUTE_GROWTH (5.0 = 500%; filters M&A)

    Returns copy with signal column appended; does not mutate input.
    """
    if panel.empty:
        out = panel.copy()
        out["asset_growth_signal"] = pd.Series(dtype=float)
        return out

    required = {"atq_recent", "atq_prior"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"compute_asset_growth_signal: panel missing columns {missing}")

    out = panel.copy()
    recent = out["atq_recent"].astype(float)
    prior  = out["atq_prior"].astype(float)

    # Growth = (recent - prior) / prior, safe div (NaN if prior ≤ 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.where(
            np.isfinite(prior) & (prior > 0),
            (recent - prior) / prior,
            np.nan,
        )

    # Apply fallback masks
    exclude_mask = (
        ~np.isfinite(recent)
        | (recent < MIN_ATQ_DOLLAR_M)
        | ~np.isfinite(prior)
        | (prior <= 0)
        | (np.abs(growth) > MAX_ABSOLUTE_GROWTH)
    )

    # Inverted signal: HIGH signal = LOW growth = LONG per CGS BEARISH
    signal = np.where(exclude_mask, np.nan, -1.0 * growth)

    out["asset_growth_signal"] = signal
    return out


def build_asset_growth_signal_panel(
    panel:           pd.DataFrame,
    *,
    long_threshold:  float = DECILE_LONG_THRESHOLD,
    short_threshold: float = DECILE_SHORT_THRESHOLD,
) -> pd.DataFrame:
    """Pipeline: compute_asset_growth_signal → rank_within_quarter → assign_decile_legs.

    Reuses sue_signal rank+leg with sue_col="asset_growth_signal", tie_break "ticker".
    Returns input augmented with: asset_growth_signal, sue_rank_pct, leg.
    """
    p1 = compute_asset_growth_signal(panel)
    p2 = _rank_within_quarter(
        p1,
        quarter_col="fiscal_yearq",
        sue_col="asset_growth_signal",
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
            "build_asset_growth_signal_panel: %d firm-quarters → long=%d short=%d flat=%d excluded=%d",
            len(p3),
            int((p3["leg"] == "long").sum()),
            int((p3["leg"] == "short").sum()),
            int((p3["leg"] == "flat").sum()),
            int((p3["leg"] == "excluded").sum()),
        )
    return p3
