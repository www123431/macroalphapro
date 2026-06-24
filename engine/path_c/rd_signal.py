"""
engine/path_c/rd_signal.py — Path I R&D Premium Drift signal composition.

Pre-registration: docs/spec_path_i_rd_premium_drift_v1.md (id=59) §2.3 + §2.4

Composes:
  - `rd_growth`: (recent_4Q - prior_4Q) / max(prior_4Q, 1.0) per spec §2.3
  - `rd_intensity`: recent_4Q / atq
  - `intensity_weight`: log(1 + intensity × 100) per spec §2.3 (dampens scale)
  - `rd_signal`: rd_growth × intensity_weight

Then REUSES `engine.path_c.sue_signal.rank_within_quarter` (sue_col="rd_signal",
tie_break_col="ticker") + `assign_decile_legs` for rank + leg assignment.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.path_c.rd_signal_panel import (
    R_AND_D_MIN_DISCLOSED_QUARTERS,
    R_AND_D_MIN_DOLLAR_M,
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


# Locked from spec §2.3 — intensity weight uses log(1 + x × 100) dampener
INTENSITY_WEIGHT_SCALE: float = 100.0


def compute_rd_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `rd_signal` column to firm-quarter R&D panel.

    Formula (spec §2.3):
        rd_growth  = (r_and_d_4q_recent - r_and_d_4q_prior) / max(r_and_d_4q_prior, 1.0)
        intensity  = r_and_d_4q_recent / atq
        weight     = log(1 + intensity × 100)         # log dampens scale
        rd_signal  = rd_growth × weight

    Fallback to NaN (excluded by rank step downstream):
      - n_quarters_recent < MIN_DISCLOSED (=2)
      - n_quarters_prior  < MIN_DISCLOSED (=2)
      - r_and_d_4q_recent < MIN_DOLLAR_M ($1.0)
      - r_and_d_4q_prior  < MIN_DOLLAR_M ($1.0)
      - atq missing or ≤ 0

    Returns a copy with `rd_signal` column appended; does not mutate input.
    """
    if panel.empty:
        out = panel.copy()
        out["rd_signal"] = pd.Series(dtype=float)
        return out

    required = {
        "r_and_d_4q_recent", "r_and_d_4q_prior",
        "n_quarters_recent", "n_quarters_prior", "atq",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"compute_rd_signal: panel missing columns {missing}")

    out = panel.copy()
    recent = out["r_and_d_4q_recent"].astype(float)
    prior  = out["r_and_d_4q_prior"].astype(float)
    n_r    = out["n_quarters_recent"].fillna(0).astype(int)
    n_p    = out["n_quarters_prior"].fillna(0).astype(int)
    atq    = out["atq"].astype(float)

    # R&D growth = (recent - prior) / max(prior, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rd_growth = np.where(
            np.isfinite(prior) & (prior >= R_AND_D_MIN_DOLLAR_M),
            (recent - prior) / np.maximum(prior, R_AND_D_MIN_DOLLAR_M),
            np.nan,
        )

    # Intensity = recent / atq (in $ per $), then log(1 + intensity × 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = np.where(
            np.isfinite(atq) & (atq > 0),
            recent / atq,
            np.nan,
        )
        intensity_weight = np.where(
            np.isfinite(intensity) & (intensity > 0),
            np.log1p(intensity * INTENSITY_WEIGHT_SCALE),
            np.nan,
        )

    raw_signal = rd_growth * intensity_weight

    # Apply thin-coverage + low-dollar exclusions → NaN
    exclude_mask = (
        (n_r < R_AND_D_MIN_DISCLOSED_QUARTERS)
        | (n_p < R_AND_D_MIN_DISCLOSED_QUARTERS)
        | (recent < R_AND_D_MIN_DOLLAR_M)
        | (prior < R_AND_D_MIN_DOLLAR_M)
        | ~np.isfinite(atq)
        | (atq <= 0)
    )
    signal = np.where(exclude_mask, np.nan, raw_signal)

    out["rd_signal"] = signal
    return out


def build_rd_signal_panel(
    panel:           pd.DataFrame,
    *,
    long_threshold:  float = DECILE_LONG_THRESHOLD,
    short_threshold: float = DECILE_SHORT_THRESHOLD,
) -> pd.DataFrame:
    """Pipeline: compute_rd_signal → rank_within_quarter → assign_decile_legs.

    Reuses sue_signal rank+leg functions with sue_col="rd_signal" + tie_break
    "ticker". Returns input augmented with: rd_signal, sue_rank_pct, leg.
    """
    p1 = compute_rd_signal(panel)
    p2 = _rank_within_quarter(
        p1,
        quarter_col="fiscal_yearq",
        sue_col="rd_signal",
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
            "build_rd_signal_panel: %d firm-quarters → long=%d short=%d flat=%d excluded=%d",
            len(p3),
            int((p3["leg"] == "long").sum()),
            int((p3["leg"] == "short").sum()),
            int((p3["leg"] == "flat").sum()),
            int((p3["leg"] == "excluded").sum()),
        )
    return p3
